"""Асинхронный клиент бесплатных моделей OpenRouter Chat Completions."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import aiohttp

from app.config import OPENROUTER_API_BASE_URL, ConfigError, Settings, parse_openrouter_models

logger = logging.getLogger("app.llm")


class LLMError(Exception):
    code = "api_error"
    user_message = "Сервис модели временно недоступен. Попробуйте позже."


class LLMTimeout(LLMError):
    code = "timeout"
    user_message = "Модель не ответила вовремя. Попробуйте ещё раз."


class LLMInvalidResponse(LLMError):
    code = "invalid_response"
    user_message = "Модель вернула некорректный ответ. Попробуйте ещё раз."


class LLMUnavailable(LLMError):
    code = "free_pool_unavailable"
    user_message = "Бесплатные модели сейчас недоступны. Попробуйте позже."


class LLMQuotaError(LLMError):
    code = "account_quota"
    user_message = "Лимит бесплатных запросов OpenRouter исчерпан. Попробуйте позже."


class LLMAuthError(LLMError):
    code = "authorization"
    user_message = "Доступ к OpenRouter не настроен. Проверьте ключ API."


class LLMRequestError(LLMError):
    code = "request_rejected"
    user_message = "OpenRouter отклонил запрос. Измените формулировку или настройки."


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    provider: str | None
    usage: dict[str, int] | None
    request_id: str


@dataclass(frozen=True)
class _RetryableError(Exception):
    status: int | None = None
    retry_after: float = 0.0


_STOP_TYPES = {
    "authentication",
    "permission_denied",
    "payment_required",
    "invalid_request",
    "invalid_prompt",
    "content_policy_violation",
    "refusal",
}
_RETRY_TYPES = {
    "rate_limit_exceeded",
    "provider_overloaded",
    "provider_unavailable",
    "timeout",
    "server",
}


def _error_type(data: object) -> str | None:
    if not isinstance(data, dict) or not isinstance(data.get("error"), dict):
        return None
    error = data["error"]
    metadata = error.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("error_type"), str):
        return metadata["error_type"].lower()
    if isinstance(error.get("type"), str):
        return error["type"].lower()
    return None


def _retry_after(headers: Any) -> float:
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return 0.0
    try:
        seconds = float(raw)
        return max(0.0, seconds) if math.isfinite(seconds) else 0.0
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(raw))
            return max(0.0, (target - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


def _is_account_limit(headers: Any) -> bool:
    lower = {str(key).lower(): value for key, value in headers.items()}
    if "x-ratelimit-limit" not in lower:
        return False
    try:
        remaining = float(lower["x-ratelimit-remaining"])
        return math.isfinite(remaining) and remaining <= 0
    except (KeyError, TypeError, ValueError):
        return False


def _no_providers_available(data: object, model: str) -> bool:
    if not isinstance(data, dict):
        return False
    metadata = data.get("openrouter_metadata")
    if not isinstance(metadata, dict):
        return False
    endpoints = metadata.get("endpoints")
    return (
        metadata.get("requested") == model
        and metadata.get("attempt") == 0
        and isinstance(endpoints, dict)
        and isinstance(endpoints.get("total"), int)
        and endpoints["total"] > 0
        and isinstance(endpoints.get("available"), list)
        and bool(endpoints["available"])
    )


def _provider(data: dict[str, Any]) -> str | None:
    metadata = data.get("openrouter_metadata")
    if not isinstance(metadata, dict):
        return None
    endpoints = metadata.get("endpoints")
    if not isinstance(endpoints, dict) or not isinstance(endpoints.get("available"), list):
        return None
    for endpoint in endpoints["available"]:
        if (
            isinstance(endpoint, dict)
            and endpoint.get("selected") is True
            and isinstance(endpoint.get("provider"), str)
        ):
            return endpoint["provider"]
    return None


def _parse_result(data: object, request_id: str) -> LLMResult:
    if not isinstance(data, dict):
        raise LLMInvalidResponse
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMInvalidResponse
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise LLMInvalidResponse
    content = message["content"]
    model = data.get("model")
    if (
        not content.strip()
        or choice.get("finish_reason") not in (None, "stop")
        or not isinstance(model, str)
        or not model
    ):
        raise LLMInvalidResponse
    usage_data = data.get("usage")
    usage = None
    if isinstance(usage_data, dict):
        usage = {
            key: usage_data[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(usage_data.get(key), int)
            and not isinstance(usage_data[key], bool)
            and usage_data[key] >= 0
        }
    return LLMResult(content, model, _provider(data), usage, request_id)


def _classify_error(status: int, data: object, headers: Any, model: str) -> None:
    if status == 200 and isinstance(data, dict) and isinstance(data.get("error"), dict):
        code = data["error"].get("code")
        if isinstance(code, int) and not isinstance(code, bool):
            status = code
    error_type = _error_type(data)
    if status == 429 and _is_account_limit(headers):
        raise LLMQuotaError
    if error_type in _STOP_TYPES:
        if error_type == "authentication":
            raise LLMAuthError
        if error_type == "payment_required":
            raise LLMQuotaError
        raise LLMRequestError
    if status == 401:
        raise LLMAuthError
    if status == 402:
        raise LLMQuotaError
    if status == 404 and _no_providers_available(data, model):
        raise _RetryableError(status, _retry_after(headers))
    if status in {400, 403, 404, 422}:
        raise LLMRequestError
    if status in {408, 429} or status >= 500 or error_type in _RETRY_TYPES:
        raise _RetryableError(status, _retry_after(headers))
    raise LLMError


class OpenRouterChatClient:
    def __init__(self, settings: Settings):
        if settings.openrouter_api_base_url != OPENROUTER_API_BASE_URL:
            raise ConfigError("OPENROUTER_API_BASE_URL: нужен https://openrouter.ai/api/v1.")
        if not settings.openrouter_api_key.strip():
            raise ConfigError("OPENROUTER_API_KEY: укажите ключ OpenRouter.")
        if (
            parse_openrouter_models(",".join(settings.openrouter_models))
            != settings.openrouter_models
        ):
            raise ConfigError("OPENROUTER_MODELS: неверный список моделей.")
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "OpenRouterChatClient":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.settings.llm_timeout_seconds),
            trust_env=False,
        )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._session is not None:
            await self._session.close()

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        request_id: str,
        *,
        model: str | None = None,
        endpoint: str | None = None,
    ) -> LLMResult:
        if self._session is None:
            raise RuntimeError("LLM-клиент не открыт.")
        if model is not None and model not in self.settings.openrouter_models:
            raise ValueError("Экспериментальная модель не входит в бесплатный пул.")
        if endpoint is not None and model is None:
            raise ValueError("Для закрепления эндпоинта укажите модель.")
        models = (model,) if model else self.settings.openrouter_models
        deadline = time.monotonic() + self.settings.llm_timeout_seconds
        try:
            async with asyncio.timeout(self.settings.llm_timeout_seconds):
                for candidate in models:
                    started = time.monotonic()
                    outcome = "cancelled"
                    try:
                        provider: dict[str, Any] = {
                            "require_parameters": True,
                            "max_price": {"prompt": 0, "completion": 0, "request": 0},
                            "allow_fallbacks": False,
                        }
                        if endpoint is not None:
                            provider["order"] = [endpoint]
                            provider["only"] = [endpoint]
                        async with self._session.post(
                            f"{OPENROUTER_API_BASE_URL}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                                "Content-Type": "application/json",
                                "X-OpenRouter-Metadata": "enabled",
                            },
                            json={
                                "model": candidate,
                                "messages": messages,
                                "temperature": temperature,
                                "provider": provider,
                            },
                            allow_redirects=False,
                        ) as response:
                            status = response.status
                            headers = response.headers
                            try:
                                data = await response.json(content_type=None)
                            except (ValueError, TypeError, UnicodeDecodeError):
                                data = None
                        if (
                            status < 200
                            or status >= 300
                            or (isinstance(data, dict) and "error" in data)
                        ):
                            _classify_error(status, data, headers, candidate)
                        result = _parse_result(data, request_id)
                        outcome = "ok"
                        logger.info(
                            "Ответ модели: id=%s model=%s provider=%s tokens=%s",
                            request_id,
                            result.model,
                            result.provider or "unknown",
                            result.usage.get("total_tokens") if result.usage else "unknown",
                        )
                        return result
                    except _RetryableError as exc:
                        outcome = f"http_{exc.status}" if exc.status else "api_error"
                        remaining = deadline - time.monotonic()
                        if 0 < exc.retry_after < remaining and candidate != models[-1]:
                            await asyncio.sleep(exc.retry_after)
                    except aiohttp.ClientError:
                        outcome = "api_error"
                    except TimeoutError:
                        outcome = "endpoint_timeout"
                        if candidate == models[-1] or time.monotonic() >= deadline:
                            raise LLMTimeout from None
                    except LLMError as exc:
                        outcome = exc.code
                        raise
                    finally:
                        logger.info(
                            "Попытка модели: id=%s model=%s duration_ms=%d result=%s",
                            request_id,
                            candidate,
                            round((time.monotonic() - started) * 1000),
                            outcome,
                        )
                raise LLMUnavailable
        except TimeoutError:
            raise LLMTimeout from None
