"""Серия реальных решений маршрутизатора без выполнения действий."""

import argparse
import asyncio
import hashlib
import json
import secrets
import sys
import unicodedata
from datetime import UTC, date, datetime
from pathlib import Path

from mcp import Client, StdioServerParameters

from app.agent_router import AgentRouter, RouteContext, RoutingError
from app.config import ConfigError, Settings
from app.context import build_messages
from app.llm import LLMError, OpenRouterChatClient
from app.prompts import MODE_PROMPTS

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "docs/evaluation/lab2-routing.json"
MODEL = "nvidia/nemotron-3.5-lightning:free"
VERSION = "nvidia/nemotron-3.5-lightning-20260807:free"
ENDPOINT = "nvidia/nvfp4"
PROVIDER = "Nvidia"
TEMPERATURE = 0.0
_CITY_NAMES = {
    "saint petersburg": "санкт петербург",
    "st petersburg": "санкт петербург",
    "sankt peterburg": "санкт петербург",
    "kazan": "казань",
    "tbilisi": "тбилиси",
}
_COUNTRY_NAMES = {
    "russia": "россия",
    "российская федерация": "россия",
    "russian federation": "россия",
    "georgia": "грузия",
}
_CITY_COUNTRIES = {"санкт петербург": "россия", "казань": "россия", "тбилиси": "грузия"}


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _city(text: str) -> tuple[str, str]:
    name, _, country = text.partition(",")
    for punctuation in ("-", "–", "—", "."):
        name = name.replace(punctuation, " ")
    normalized_name = _normalized(name)
    normalized_country = _normalized(country)
    return (
        _CITY_NAMES.get(normalized_name, normalized_name),
        _COUNTRY_NAMES.get(normalized_country, normalized_country),
    )


def arguments_match(action: str, expected: dict, actual: dict) -> bool:
    if expected.keys() != actual.keys():
        return False
    if action == "get_weather":
        expected_city, expected_country = _city(expected["city"])
        actual_city, actual_country = _city(actual["city"])
        if expected_country:
            return (actual_city, actual_country) == (expected_city, expected_country)
        return actual_city == expected_city and actual_country in {
            "",
            _CITY_COUNTRIES.get(expected_city),
        }
    if action == "get_schedule":
        return date.fromisoformat(actual["date"]) == date.fromisoformat(expected["date"])
    if action == "add_reminder":
        actual_at = datetime.fromisoformat(actual["remind_at"])
        expected_at = datetime.fromisoformat(expected["remind_at"])
        actual_text = _normalized(actual["text"]).rstrip(".!?").replace("ё", "е")
        expected_text = _normalized(expected["text"]).rstrip(".!?").replace("ё", "е")
        return actual_text == expected_text and actual_at == expected_at
    return False


def _write(path: Path, result: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _metrics(rows: list[dict]) -> dict:
    tool_rows = [row for row in rows if row["arguments_correct"] is not None]
    return {
        "routing_accuracy": {
            "correct": sum(row["routing_correct"] for row in rows),
            "total": len(rows),
        },
        "argument_accuracy": {
            "correct": sum(row["arguments_correct"] for row in tool_rows),
            "total": len(tool_rows),
        },
    }


def _case_result(case: dict, *, action: str, arguments: dict, status: str) -> dict:
    expected = case["expected_action"]
    action_match = action == expected
    argument_match = None
    if case["expected_arguments"]:
        argument_match = action_match and arguments_match(
            expected, case["expected_arguments"], arguments
        )
    return {
        "id": case["id"],
        "query": case["query"],
        "expected_action": expected,
        "expected_arguments": case["expected_arguments"],
        "actual_action": action,
        "actual_arguments": arguments,
        "routing_correct": action_match,
        "arguments_correct": argument_match,
        "result": status,
    }


async def run(settings: Settings, output: Path, prior: Path | None, variant: str) -> bool:
    dataset_bytes = await asyncio.to_thread(DATASET.read_bytes)
    dataset = json.loads(dataset_bytes)
    evaluation = dataset["evaluation_context"]
    context = RouteContext(datetime.fromisoformat(evaluation["now"]), evaluation["timezone"])
    cases = dataset["cases"]
    constants = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "evaluation_context": evaluation,
        "api": "OpenRouter /api/v1/chat/completions",
        "model": MODEL,
        "model_version": VERSION,
        "provider": PROVIDER,
        "endpoint": ENDPOINT,
        "temperature": TEMPERATURE,
        "timeout_seconds": settings.llm_timeout_seconds,
        "provider_fallbacks": False,
        "context_max_chars": settings.context_max_chars,
        "tool_execution": False,
        "mode": "study",
    }
    if prior is not None:
        baseline = json.loads(await asyncio.to_thread(prior.read_text, encoding="utf-8"))
        if (
            baseline["status"] != "completed"
            or baseline["instruction_variant"] != "baseline"
            or baseline["constants"] != constants
        ):
            raise ValueError("Повтор требует завершённой исходной серии с одинаковыми настройками.")
        misses = {
            row["id"]
            for row in baseline["rows"]
            if not row["routing_correct"] or row["arguments_correct"] is False
        }
        cases = [case for case in cases if case["id"] in misses]
        if not cases:
            raise ValueError("В исходной серии нет ошибок для повторного прогона.")

    result = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "constants": constants,
        "instruction_variant": variant,
        "baseline_file": prior.name if prior else None,
        "case_ids": [case["id"] for case in cases],
        "rows": [],
    }
    await asyncio.to_thread(_write, output, result)
    async with Client(
        StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_tools.server"], cwd=ROOT)
    ) as mcp:
        tools = tuple((await mcp.list_tools()).tools)
    async with OpenRouterChatClient(settings) as llm:
        router = AgentRouter(llm)
        for case in cases:
            messages = build_messages(
                MODE_PROMPTS["study"],
                [],
                case["query"],
                max_history_messages=settings.history_max_messages,
                max_context_chars=settings.context_max_chars,
            )
            try:
                choice = await router.decide(
                    messages,
                    TEMPERATURE,
                    secrets.token_hex(8),
                    tools=tools,
                    context=context,
                    model=MODEL,
                    endpoint=ENDPOINT,
                    instruction_variant=variant,
                )
            except (RoutingError, LLMError) as exc:
                actual = (exc.name or "error") if isinstance(exc, RoutingError) else "error"
                observed = exc.result if isinstance(exc, RoutingError) else None
                row = _case_result(case, action=actual, arguments={}, status=exc.code)
            else:
                observed = choice.result
                row = _case_result(
                    case,
                    action=choice.name if choice.action == "call_tool" else choice.action,
                    arguments=choice.arguments,
                    status="validated",
                )
                if choice.timezone is not None:
                    row["normalized_timezone"] = choice.timezone
            if observed is not None:
                row.update(
                    model=observed.model,
                    model_version=observed.version,
                    provider=observed.provider,
                    reported_endpoint=observed.endpoint,
                    usage=observed.usage,
                    cost=observed.cost,
                )
                if (
                    (observed.model, observed.version, observed.provider)
                    != (MODEL, VERSION, PROVIDER)
                    or observed.endpoint not in (None, ENDPOINT)
                    or observed.cost not in (None, 0)
                ):
                    row["result"] = "route_changed"
                    result["rows"].append(row)
                    result["status"] = "route_changed"
                    await asyncio.to_thread(_write, output, result)
                    return False
            result["rows"].append(row)
            await asyncio.to_thread(_write, output, result)

    result["status"] = "completed"
    result["completed_at_utc"] = datetime.now(UTC).isoformat()
    result["metrics"] = _metrics(result["rows"])
    await asyncio.to_thread(_write, output, result)
    for name, counts in result["metrics"].items():
        print(f"{name} = {counts['correct']}/{counts['total']}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Эксперимент маршрутизации лабораторной № 2")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, help="JSON завершённой исходной серии")
    parser.add_argument("--variant", default="baseline", help="Версия инструкции маршрутизатора")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Укажите новый файл для результатов.")
    if args.rerun is not None and args.variant == "baseline":
        parser.error("Для повтора укажите --rerun и изменённый --variant.")
    try:
        settings = Settings.load(args.env_file, require_runtime=False)
        completed = asyncio.run(run(settings, args.output, args.rerun, args.variant))
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"Проверьте файлы эксперимента ({type(exc).__name__}).", file=sys.stderr)
        return 1
    print(args.output)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
