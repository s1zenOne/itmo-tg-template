from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.briefing import BriefingBuilder
from app.mcp_tools.contracts import (
    CATALOG_URI,
    MATERIALS,
    SCHEDULE,
    WEATHER,
    ToolError,
    tool_definitions,
)
from app.supervisor import AgentRequest

WEATHER_RESULT = {
    "status": "ok",
    "city": "Санкт-Петербург",
    "country": "Россия",
    "region": None,
    "temperature_c": 3.1,
    "conditions": "слабый дождь",
    "wind_speed": 18.0,
    "wind_unit": "km/h",
    "observed_at": "2026-11-23T08:00:00+00:00",
    "timezone": "UTC",
    "source": "Open-Meteo",
}


def settings(latitude, longitude):
    return {
        "timezone": "UTC",
        "city_name": "Санкт-Петербург",
        "city_country": "Россия",
        "city_region": None,
        "latitude": latitude,
        "longitude": longitude,
    }


def mock_mcp(call):
    return SimpleNamespace(
        tools=tool_definitions(),
        resources=[SimpleNamespace(uri=CATALOG_URI)],
        call=AsyncMock(side_effect=call),
        read_resource=AsyncMock(return_value={"documents": []}),
    )


async def test_briefing_calls_weather_with_confirmed_coordinates_and_traces_it():
    async def call(name, args):
        if name == WEATHER:
            assert args == {
                "city": "Санкт-Петербург",
                "country": "Россия",
                "region": None,
                "latitude": 59.9,
                "longitude": 30.3,
            }
            return WEATHER_RESULT
        if name == SCHEDULE:
            return {
                "status": "ok",
                "date": "2026-11-23",
                "lessons": [],
                "source": "Учебное расписание",
            }
        return {"status": "ok", "documents": []}

    mcp = mock_mcp(call)
    now = datetime(2026, 11, 23, 8, tzinfo=UTC)
    events = []
    result = await BriefingBuilder(mcp)(
        1,
        now.date(),
        now,
        settings(59.9, 30.3),
        trace_events=events,
        clock=lambda: now,
    )
    assert "Погода: 3.1 °C, слабый дождь" in result
    assert "Занятия: нет" in result
    assert ("lifestyle", "tool", "ok", WEATHER) in {
        (e.component, e.operation, e.status, e.tool_name) for e in events
    }


async def test_briefing_keeps_other_sections_on_weather_failure():
    async def call(name, args):
        if name == WEATHER:
            raise TimeoutError()
        if name == SCHEDULE:
            return {
                "status": "ok",
                "date": "2026-11-23",
                "lessons": [],
                "source": "Учебное расписание",
            }
        return {"status": "ok", "documents": []}

    mcp = mock_mcp(call)
    result = await BriefingBuilder(mcp)(
        1, date(2026, 11, 23), datetime.now(UTC), settings(59.9, 30.3)
    )
    assert "Занятия: нет" in result
    assert "Погода: данные недоступны" in result
    assert "подходящий материал не найден" in result


async def test_briefing_records_only_executed_mcp_calls_with_safe_failures():
    secret = "private text that must not be recorded"

    async def call(name, args):
        if name == SCHEDULE:
            raise RuntimeError(secret)
        return WEATHER_RESULT if name == WEATHER else {"status": "ok", "documents": []}

    mcp = mock_mcp(call)
    events = []
    fixed = datetime(2026, 11, 23, 8, tzinfo=UTC)
    text = await BriefingBuilder(mcp)(
        1,
        fixed.date(),
        fixed,
        settings(59.9, 30.3),
        trace_events=events,
        clock=lambda: fixed,
    )
    assert "Занятия: данные недоступны" in text
    assert {(x.component, x.operation, x.status, x.tool_name, x.error_code) for x in events} == {
        ("schedule", "execute", "error", None, "internal_error"),
        ("knowledge", "resource", "ok", None, None),
        ("knowledge", "tool", "ok", MATERIALS, None),
        ("lifestyle", "tool", "ok", WEATHER, None),
    }
    assert secret not in repr(events)
    assert all(x.finished_at >= x.started_at for x in events)


async def test_denied_policy_creates_safe_error_span_and_never_calls_mcp():
    mcp = SimpleNamespace(tools=(), resources=[], call=AsyncMock(), read_resource=AsyncMock())
    events = []
    now = datetime(2026, 11, 23, 8, tzinfo=UTC)
    await BriefingBuilder(mcp)(
        1, now.date(), now, settings(1, 2), trace_events=events, clock=lambda: now
    )
    assert {(event.component, event.error_code) for event in events if event.status == "error"} == {
        ("schedule", "unauthorized"),
        ("knowledge", "unauthorized"),
        ("lifestyle", "unauthorized"),
    }
    assert len([event for event in events if event.status == "error"]) == 3
    assert not any(event.operation in {"tool", "resource"} for event in events)
    mcp.call.assert_not_awaited()
    mcp.read_resource.assert_not_awaited()


async def test_unauthorized_from_mcp_creates_only_one_error_span():
    async def call(name, args):
        if name == SCHEDULE:
            raise ToolError("unauthorized")
        if name == WEATHER:
            return WEATHER_RESULT
        return {"status": "ok", "documents": []}

    mcp = mock_mcp(call)
    now = datetime(2026, 11, 23, 8, tzinfo=UTC)
    events = []
    await BriefingBuilder(mcp)(
        1, now.date(), now, settings(1, 2), trace_events=events, clock=lambda: now
    )
    schedule_errors = [event for event in events if event.component == "schedule"]
    assert len(schedule_errors) == 1
    assert schedule_errors[0].error_code == "unauthorized"


async def test_unavailable_mcp_is_not_reported_as_server_tool_call():
    from app.mcp_tools.client import MCPUnavailable

    mcp = mock_mcp(MCPUnavailable())
    mcp.read_resource.side_effect = MCPUnavailable()
    events = []
    now = datetime(2026, 11, 23, 8, tzinfo=UTC)
    await BriefingBuilder(mcp)(
        1, now.date(), now, settings(1, 2), trace_events=events, clock=lambda: now
    )
    assert not any(event.operation in {"tool", "resource"} for event in events)
    assert {event.error_code for event in events if event.status == "error"} == {"unavailable"}


async def test_timeout_before_mcp_dispatch_is_not_claimed_as_tool_call():
    import asyncio

    mcp = mock_mcp(asyncio.CancelledError())
    mcp.read_resource.side_effect = asyncio.CancelledError()
    events = []
    now = datetime(2026, 11, 23, 8, tzinfo=UTC)
    with pytest.raises(asyncio.CancelledError):
        await BriefingBuilder(mcp)(
            1,
            now.date(),
            now,
            settings(1, 2),
            trace_events=events,
            clock=lambda: now,
        )
    assert not any(event.operation in {"tool", "resource"} for event in events)
    assert any(event.error_code == "timeout" for event in events)


def request(now):
    return AgentRequest(
        request_id=uuid4(),
        trace_id=uuid4(),
        actor_id=1,
        text="Ежедневная сводка",
        intent="daily_briefing",
        occurred_at=now,
        timezone="UTC",
    )


async def test_specialist_results_record_verified_calls_and_partial_failure():
    async def call(name, args):
        if name == SCHEDULE:
            return {
                "status": "ok",
                "date": "2026-11-23",
                "lessons": [],
                "source": "Учебное расписание",
            }
        if name == WEATHER:
            raise TimeoutError()
        return {"status": "ok", "documents": []}

    mcp = mock_mcp(call)
    now = datetime(2026, 11, 23, 8, tzinfo=UTC)
    results = await BriefingBuilder(mcp)._sections(
        request(now), now.date(), settings(1, 2), None, lambda: now
    )
    assert [(x.status, x.used_tools) for x in results] == [
        ("success", [SCHEDULE]),
        ("failed", []),
        ("not_found", [MATERIALS]),
    ]
    assert results[1].error_code == "timeout"
    assert results[0].facts.dates == ["2026-11-23"]


async def test_specialist_rejects_unvalidated_request_before_mcp():
    now = datetime(2026, 11, 23, 8, tzinfo=UTC)
    forged = request(now).model_copy(update={"actor_id": 0})
    mcp = mock_mcp(None)
    with pytest.raises(ValidationError):
        await BriefingBuilder(mcp)._sections(forged, now.date(), settings(1, 2), None, lambda: now)
    mcp.call.assert_not_awaited()
    mcp.read_resource.assert_not_awaited()
