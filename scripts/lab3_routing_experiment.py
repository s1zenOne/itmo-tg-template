"""Run fixed lab-3 cases with a pinned model and provider; no Telegram or DB writes."""

import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from app.agent import AgentService
from app.agent_router import AgentRouter
from app.config import Settings
from app.llm import OpenRouterChatClient
from app.mcp_tools.client import MCPClient
from app.reminders import PreparedReminder
from app.supervisor import SpecialistRouter

ROOT = Path(__file__).resolve().parents[1]


class MemoryStore:
    def __init__(self, context):
        if context["city"] != "Санкт-Петербург, Россия":
            raise ValueError("Для города эксперимента нет подтверждённой тестовой фикстуры")
        self.city = {
            "city_name": "Санкт-Петербург",
            "city_country": "Россия",
            "city_region": None,
            "latitude": 59.9343,
            "longitude": 30.3351,
        }
        self.timezone = context["timezone"]

    async def reminder_for_update(self, *_args):
        return None

    async def get_timezone(self, *_args):
        return self.timezone

    async def get_confirmed_city(self, *_args):
        return self.city.copy()

    async def write_audit(self, *_args):
        return None

    async def prepare_reminder(self, _owner, _update, text, remind_at, timezone):
        return PreparedReminder(
            uuid4(), text, remind_at, timezone, remind_at - timedelta(minutes=1), "pending"
        )


class PinnedLLM:
    def __init__(self, inner, model, endpoint):
        self.inner, self.model, self.endpoint = inner, model, endpoint
        self.responses = []

    async def generate(self, messages, temperature, request_id, **kwargs):
        kwargs.update(model=self.model, endpoint=self.endpoint)
        result = await self.inner.generate(messages, temperature, request_id, **kwargs)
        self.responses.append(result)
        return result


class ObservedMCP:
    def __init__(self, inner):
        self.inner = inner
        self.tools_called = []
        self.resources_read = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def call(self, name, arguments, **kwargs):
        self.tools_called.append(name)
        return await self.inner.call(name, arguments, **kwargs)

    async def read_resource(self, uri):
        self.resources_read.append(uri)
        return await self.inner.read_resource(uri)


class ObservedRouter(SpecialistRouter):
    decision = None

    async def decide(self, text, request_id, history=None):
        self.decision = await super().decide(text, request_id, history)
        return self.decision


async def run(args):
    settings = Settings.load(args.env_file)
    dataset = json.loads((ROOT / "docs/evaluation/lab3-routing.json").read_text(encoding="utf-8"))
    now = datetime.fromisoformat(dataset["evaluation_context"]["now"])
    results = []
    async with OpenRouterChatClient(settings) as llm, MCPClient(settings) as raw_mcp:
        pinned = PinnedLLM(llm, args.model, args.endpoint)
        mcp = ObservedMCP(raw_mcp)
        router = ObservedRouter(pinned)
        agent = AgentService(
            MemoryStore(dataset["evaluation_context"]),
            AgentRouter(pinned),
            mcp,
            settings,
            clock=lambda: now,
            specialist_router=router,
        )
        cases = dataset["cases"]
        if args.case:
            requested = set(args.case)
            cases = [case for case in cases if case["id"] in requested]
            if len(cases) != len(requested):
                raise ValueError("Неизвестный идентификатор случая")
        for index, case in enumerate(cases, 1):
            pinned.responses.clear()
            mcp.tools_called.clear()
            mcp.resources_read.clear()
            router.decision = None
            started = time.monotonic()
            failure = None
            try:
                await agent.answer(
                    123456789,
                    index,
                    [
                        {"role": "system", "content": "Учебный помощник."},
                        {"role": "user", "content": case["query"]},
                    ],
                    0.0,
                )
            except Exception as exc:
                failure = type(exc).__name__
            decision = router.decision
            results.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected_target": case["expected_target"],
                    "actual_target": decision.target if decision else None,
                    "confidence": decision.confidence if decision else None,
                    "tools_called": list(mcp.tools_called),
                    "resources_read": list(mcp.resources_read),
                    "duration_ms": round((time.monotonic() - started) * 1000),
                    "model_metadata": [
                        {
                            "model": x.model,
                            "version": x.version,
                            "endpoint": x.endpoint,
                            "usage": x.usage,
                        }
                        for x in pinned.responses
                    ],
                    "error": failure or (None if decision else "router_unavailable"),
                }
            )
    output = {
        "kind": "lab3_routing",
        "pinned_model": args.model,
        "pinned_endpoint": args.endpoint,
        "temperature": 0.0,
        "cases": results,
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Модель из разрешённого пула")
    parser.add_argument("--endpoint", required=True, help="Закреплённый провайдер OpenRouter")
    parser.add_argument(
        "--output", required=True, help="Новый JSON-файл; существующий не перезаписывается"
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--case", action="append", help="ID промаха для отдельного повтора")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Выходной файл уже существует")
    output = asyncio.run(run(args))
    with Path(args.output).open("x", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    count = sum(item["actual_target"] is not None for item in output["cases"])
    print(f"Сохранено случаев: {len(output['cases'])}; маршрутов получено: {count}")


if __name__ == "__main__":
    main()
