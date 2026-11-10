"""Предпроверка маршрута и 12 независимых реальных запросов OpenRouter.

Запуск: .venv/bin/python -m scripts.lab1_experiment
Полные ответы остаются в игнорируемом Git каталоге .lab1/.
"""

import argparse
import asyncio
import json
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path

from app.config import ConfigError, Settings
from app.db import create_pool
from app.llm import LLMError, LLMResult, OpenRouterChatClient
from app.prompts import MODE_PROMPTS
from app.service import AssistantService
from app.store import PostgresStore

STUDY_QUERY = (
    "Объясни, как с помощью словаря Python посчитать частоту слов в списке. "
    "Приведи короткий пример и разбери одну типичную ошибку"
)
COMPARISON_QUERY = (
    "Объясни, почему изменяемый аргумент по умолчанию в функции Python может вызвать "
    "ошибку. Приведи пример исправления"
)
BASELINE_PROMPT = "Ты помощник по программированию. Ответь на вопрос."
TEMPERATURES = (0.0, 0.3, 0.7, 1.0)
EXPERIMENT_CHAT_ID = 0  # Синтетический ID, которого нет у личного Telegram-чата.
VERIFIED_ENDPOINTS = {
    "google/gemma-4-26b-a4b-it:free": ("google-ai-studio", "Google AI Studio"),
    "nvidia/nemotron-3.5-lightning:free": ("nvidia/nvfp4", "Nvidia"),
    "dots-studio/dots-3-note-preview:free": ("atlas-cloud/fp8", "AtlasCloud"),
}
_ENDPOINT_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)?\Z")
_PREFLIGHT_DIALOGUE = [
    {"role": "system", "content": "Ответь одним словом: готово."},
    {"role": "user", "content": "Проверка первой реплики"},
    {"role": "assistant", "content": "готово"},
    {"role": "user", "content": "Проверка второй реплики"},
]


class ExperimentRouteError(LLMError):
    code = "route_changed"
    user_message = "OpenRouter сменил модель или провайдера. Серия остановлена."


def _write(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _route(
    settings: Settings, model: str | None, endpoint: str | None
) -> tuple[str, str, str | None]:
    selected_model = model or settings.openrouter_models[0]
    if selected_model not in settings.openrouter_models:
        raise ConfigError("--model: модель должна входить в OPENROUTER_MODELS.")
    known_route = VERIFIED_ENDPOINTS.get(selected_model)
    selected_endpoint = endpoint or (known_route[0] if known_route else None)
    if selected_endpoint is None:
        raise ConfigError("--endpoint: для этой модели укажите проверенный эндпоинт.")
    if not _ENDPOINT_SLUG.fullmatch(selected_endpoint):
        raise ConfigError("--endpoint: нужен полный slug эндпоинта провайдера.")
    expected_provider = (
        known_route[1] if known_route and selected_endpoint == known_route[0] else None
    )
    return selected_model, selected_endpoint, expected_provider


async def run(
    settings: Settings,
    path: Path,
    *,
    model: str | None = None,
    endpoint: str | None = None,
    preflight_only: bool = False,
) -> None:
    selected_model, selected_endpoint, expected_provider = _route(settings, model, endpoint)
    result = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "api": "OpenRouter Chat Completions",
        "api_version": "/api/v1",
        "model": selected_model,
        "endpoint": selected_endpoint,
        "expected_provider": expected_provider,
        "query": STUDY_QUERY,
        "system_prompt": MODE_PROMPTS["study"],
        "constant_settings": {
            "history": "reset before every request",
            "timeout_seconds": settings.llm_timeout_seconds,
            "provider_fallbacks": False,
        },
        "criteria": ["формат", "полнота", "понятность", "фактическая корректность"],
        "preflight": [],
        "runs": [],
        "comparison": [],
    }
    _write(path, result)

    observed_provider = expected_provider
    async with OpenRouterChatClient(settings) as llm:

        async def pinned(messages: list[dict[str, str]], temperature: float) -> LLMResult:
            nonlocal observed_provider
            response = await llm.generate(
                messages,
                temperature,
                secrets.token_hex(8),
                model=selected_model,
                endpoint=selected_endpoint,
            )
            if (
                response.model != selected_model
                or response.provider is None
                or (observed_provider is not None and response.provider != observed_provider)
            ):
                result["route_mismatch"] = {
                    "expected_model": selected_model,
                    "expected_provider": observed_provider,
                    "requested_endpoint": selected_endpoint,
                    "actual_model": response.model,
                    "actual_provider": response.provider,
                }
                _write(path, result)
                raise ExperimentRouteError
            observed_provider = response.provider
            return response

        # Это реальные запросы с синтетическим диалогом, но не строки эксперимента.
        for temperature in TEMPERATURES:
            response = await pinned(_PREFLIGHT_DIALOGUE, temperature)
            result["preflight"].append(
                {
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "temperature": temperature,
                    "request_id": response.request_id,
                    "model": response.model,
                    "provider": response.provider,
                    "usage": response.usage,
                }
            )
            _write(path, result)
        if preflight_only:
            result["completed_at_utc"] = datetime.now(UTC).isoformat()
            _write(path, result)
            return

        pool = await create_pool(settings)
        try:
            store = PostgresStore(pool)
            await store.initialize()
            await store.set_mode(EXPERIMENT_CHAT_ID, "study")
            service = AssistantService(store, llm, settings)
            for temperature in TEMPERATURES:
                await store.set_temperature(EXPERIMENT_CHAT_ID, temperature)
                for repeat in range(1, 4):
                    await service.reset(EXPERIMENT_CHAT_ID)
                    response = await pinned(
                        [
                            {"role": "system", "content": MODE_PROMPTS["study"]},
                            {"role": "user", "content": STUDY_QUERY},
                        ],
                        temperature,
                    )
                    result["runs"].append(
                        {
                            "number": len(result["runs"]) + 1,
                            "completed_at_utc": datetime.now(UTC).isoformat(),
                            "temperature": temperature,
                            "repeat": repeat,
                            "request_id": response.request_id,
                            "model": response.model,
                            "provider": response.provider,
                            "response": response.text,
                            "length_chars": len(response.text),
                            "usage": response.usage,
                            "format_ok": None,
                            "observations": "Заполнить после ручной проверки",
                        }
                    )
                    _write(path, result)
            await store.set_temperature(EXPERIMENT_CHAT_ID, 0.3)
            for label, system in (
                ("неполная", BASELINE_PROMPT),
                ("улучшенная", MODE_PROMPTS["study"]),
            ):
                await service.reset(EXPERIMENT_CHAT_ID)
                response = await pinned(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": COMPARISON_QUERY},
                    ],
                    0.3,
                )
                result["comparison"].append(
                    {
                        "instruction": label,
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                        "system_prompt": system,
                        "query": COMPARISON_QUERY,
                        "temperature": 0.3,
                        "request_id": response.request_id,
                        "model": response.model,
                        "provider": response.provider,
                        "response": response.text,
                        "length_chars": len(response.text),
                        "usage": response.usage,
                    }
                )
                _write(path, result)
            result["completed_at_utc"] = datetime.now(UTC).isoformat()
            _write(path, result)
        finally:
            await pool.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Эксперимент лабораторной № 1")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", help="Модель из OPENROUTER_MODELS; по умолчанию первая")
    parser.add_argument("--endpoint", help="Полный slug эндпоинта выбранной модели")
    parser.add_argument(
        "--preflight-only", action="store_true", help="Проверить роли и температуры"
    )
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        print(str(exc))
        return 1
    directory = Path(".lab1")
    directory.mkdir(exist_ok=True)
    path = directory / f"experiment-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}.json"
    try:
        asyncio.run(
            run(
                settings,
                path,
                model=args.model,
                endpoint=args.endpoint,
                preflight_only=args.preflight_only,
            )
        )
    except (ConfigError, LLMError) as exc:
        message = exc.user_message if isinstance(exc, LLMError) else str(exc)
        print(f"Эксперимент прерван: {message} Частичные результаты: {path}")
        return 1
    except Exception:
        print(f"Эксперимент прерван: проверьте БД и настройки. Частичные результаты: {path}")
        return 1
    print(f"Результаты сохранены: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
