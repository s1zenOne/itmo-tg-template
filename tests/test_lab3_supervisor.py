import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.llm import LLMResult, NativeToolCall
from app.mcp_tools.contracts import WEATHER, ToolError, tool_definitions
from app.supervisor import (
    AgentRequest,
    KnowledgeFacts,
    RouteDecision,
    ScheduleFacts,
    SpecialistRouter,
    WeatherFacts,
    allowed_tools,
    require_resource,
    require_tool,
)


def test_specialist_facts_reject_invalid_dates_and_nonfinite_temperature():
    with pytest.raises(ValidationError):
        ScheduleFacts(dates=["not-a-date"])
    with pytest.raises(ValidationError):
        ScheduleFacts(dates=["20261123"])
    with pytest.raises(ValidationError):
        WeatherFacts(temperature_c=float("nan"))
    for invalid in ("", "../private", "x" * 101):
        with pytest.raises(ValidationError):
            KnowledgeFacts(source_ids=[invalid])


def proposal(value):
    return LLMResult("", "test", None, None, "test", (NativeToolCall("1", "select_route", value),))


@pytest.mark.parametrize(
    "value",
    [
        '{"target":"intruder","reason":"x","confidence":1.0}',
        '{"target":"schedule","reason":"x","confidence":2.0}',
        '{"target":"schedule","target":"lifestyle","reason":"x","confidence":1.0}',
        '{"target":"schedule","reason":"x","confidence":0.2}',
    ],
)
async def test_invalid_or_uncertain_route_clarifies(value):
    llm = SimpleNamespace(generate=AsyncMock(return_value=proposal(value)))
    decision = await SpecialistRouter(llm).decide("текст", "id")
    assert decision.target == "clarify"


async def test_route_model_receives_no_tools_other_than_choice():
    llm = SimpleNamespace(
        generate=AsyncMock(
            return_value=proposal(
                json.dumps({"target": "schedule", "reason": "schedule", "confidence": 0.9})
            )
        )
    )
    decision = await SpecialistRouter(llm).decide("расписание", "id")
    assert decision == RouteDecision(target="schedule", reason="schedule", confidence=0.9)
    assert [tool["function"]["name"] for tool in llm.generate.await_args.kwargs["tools"]] == [
        "select_route"
    ]


def test_authorization_intersects_discovery_and_exact_component():
    tools = tool_definitions()
    assert {t.name for t in allowed_tools("lifestyle", tools)} == {"get_weather"}
    require_tool("lifestyle", WEATHER, tools)
    require_tool("briefing_lifestyle", WEATHER, tools)
    assert not allowed_tools("router", tools)
    assert not allowed_tools("schedule-extra", tools)
    for component, name in [
        ("lifestyle", "add_reminder"),
        ("knowledge", "get_weather"),
        ("briefing_schedule", "add_reminder"),
        ("schedule", WEATHER),
        ("schedule", "unknown"),
    ]:
        with pytest.raises(ToolError):
            require_tool(component, name, tools)
    with pytest.raises(ToolError):
        require_tool("schedule", "get_schedule", ())
    with pytest.raises(ToolError):
        require_resource(
            "lifestyle", "materials://catalog", (SimpleNamespace(uri="materials://catalog"),)
        )


def test_request_contract_rejects_untrusted_fields_and_naive_time():
    fields = dict(
        request_id=uuid4(),
        trace_id=uuid4(),
        actor_id=123,
        text="x",
        intent="user_query",
        occurred_at=datetime.now(UTC),
        timezone="Europe/Moscow",
    )
    AgentRequest(**fields)
    with pytest.raises(ValidationError):
        AgentRequest(**fields, destination_chat_id=456)
    with pytest.raises(ValidationError):
        AgentRequest(**{**fields, "occurred_at": datetime(2026, 1, 1)})


@pytest.mark.parametrize(
    "query,text,proposed,expected",
    [
        (
            "Напомни завтра в 9:00 отправить отчёт",
            "отправить отчёт",
            "2026-10-13T09:00:00+00:00",
            True,
        ),
        (
            "Напомни завтра в 10:00 отправить отчёт, а не послезавтра",
            "отправить отчёт",
            "2026-10-14T10:00:00+00:00",
            False,
        ),
        (
            "В 10:00 я учусь. Напомни завтра в 12:00 отправить отчёт",
            "отправить отчёт",
            "2026-10-13T10:00:00+00:00",
            False,
        ),
        (
            "Что означает фраза «Напомни завтра в 10:00 отправить отчёт»?",
            "отправить отчёт",
            "2026-10-13T10:00:00+00:00",
            False,
        ),
        (
            "Не напомни завтра в 10:00 отправить отчёт",
            "отправить отчёт",
            "2026-10-13T10:00:00+00:00",
            False,
        ),
    ],
)
def test_reminder_is_bound_to_user_request(query, text, proposed, expected):
    from app.reminder_intent import matches_reminder_request

    assert (
        matches_reminder_request(
            query, text, proposed, datetime(2026, 10, 12, 6, tzinfo=UTC), "UTC"
        )
        is expected
    )


def test_reminder_accepts_explicit_utc_offset():
    from app.reminder_intent import matches_reminder_request

    assert matches_reminder_request(
        "Напомни 2026-10-13 в 17:00 UTC+03:00 отправить отчёт",
        "отправить отчёт",
        "2026-10-13T14:00:00+00:00",
        datetime(2026, 10, 12, 6, tzinfo=UTC),
        "UTC+03:00",
    )
