from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

from app.agent import AgentReply, AgentService, SpecialistResponse, _Outcome
from app.config import Settings
from app.supervisor import AgentRequest, AgentResult, EmptyFacts, WeatherFacts


@pytest.mark.parametrize("target", ["schedule", "knowledge", "lifestyle"])
async def test_specialist_receives_trusted_request_without_trace(target):
    now = datetime(2026, 11, 23, 9, tzinfo=UTC)
    store = Mock(
        reminder_for_update=AsyncMock(return_value=None),
        get_timezone=AsyncMock(return_value="Europe/Moscow"),
        write_audit=AsyncMock(),
    )
    mcp = Mock(connect=AsyncMock(return_value=True))
    router = Mock(decide=AsyncMock(return_value=SimpleNamespace(target=target)))
    service = AgentService(
        store,
        Mock(),
        mcp,
        Settings(bot_token="", postgres_password=""),
        clock=lambda: now,
        specialist_router=router,
    )
    outcome = _Outcome(AgentReply("Ответ"), "ответ", "Тест")
    specialist = AsyncMock(return_value=service._specialist_response(outcome, []))
    setattr(service, f"_{target}_specialist", specialist)
    reply = await service.answer(42, 9, [{"role": "user", "content": "проверочный запрос"}], 0.0)
    request = specialist.await_args.args[0]
    assert isinstance(request, AgentRequest)
    assert request.actor_id == 42
    assert request.text == "проверочный запрос"
    assert request.occurred_at == now
    assert request.timezone == "Europe/Moscow"
    assert isinstance(request.trace_id, UUID)
    assert reply.text == "Ответ"


async def test_invalid_specialist_result_is_not_returned_to_user():
    store = Mock(
        reminder_for_update=AsyncMock(return_value=None),
        get_timezone=AsyncMock(return_value="UTC"),
        write_audit=AsyncMock(),
    )
    router = Mock(decide=AsyncMock(return_value=SimpleNamespace(target="schedule")))
    service = AgentService(
        store,
        Mock(),
        Mock(connect=AsyncMock(return_value=True)),
        Settings(bot_token="", postgres_password=""),
        clock=lambda: datetime(2026, 11, 23, tzinfo=UTC),
        specialist_router=router,
    )
    invalid = AgentResult.model_construct(
        status="success",
        message="подменённый результат",
        confidence=3.0,
        facts=EmptyFacts(),
        used_tools=[],
        error_code=None,
    )
    service._schedule_specialist = AsyncMock(
        return_value=SpecialistResponse(
            invalid, _Outcome(AgentReply("подменённый результат"), "ответ", "Тест"), ()
        )
    )
    reply = await service.answer(42, 9, [{"role": "user", "content": "запрос"}], 0.0)
    assert reply.text == "Не удалось проверить результат специалиста."
    assert "подменённый результат" not in reply.text


@pytest.mark.parametrize("mismatch", ["status", "error", "confirmation", "failed_code", "facts"])
async def test_inconsistent_specialist_envelope_is_rejected(mismatch):
    from uuid import uuid4

    store = Mock(
        reminder_for_update=AsyncMock(return_value=None),
        get_timezone=AsyncMock(return_value="UTC"),
        write_audit=AsyncMock(),
    )
    router = Mock(decide=AsyncMock(return_value=SimpleNamespace(target="schedule")))
    service = AgentService(
        store,
        Mock(),
        Mock(connect=AsyncMock(return_value=True)),
        Settings(bot_token="", postgres_password=""),
        clock=lambda: datetime(2026, 11, 23, tzinfo=UTC),
        specialist_router=router,
    )
    reply = AgentReply(
        "Неверный ответ", confirmation=uuid4() if mismatch == "confirmation" else None
    )
    result = AgentResult(
        status="failed" if mismatch in {"status", "failed_code"} else "success",
        message=reply.text,
        confidence=1.0,
        facts=WeatherFacts(temperature_c=8.0) if mismatch == "facts" else EmptyFacts(),
        used_tools=[],
        error_code="unavailable" if mismatch in {"status", "error", "failed_code"} else None,
    )
    service._schedule_specialist = AsyncMock(
        return_value=SpecialistResponse(
            result,
            _Outcome(
                reply,
                "отказ" if mismatch == "failed_code" else "чтение",
                "Тест",
                error_code="unauthorized" if mismatch == "failed_code" else None,
            ),
            (),
        )
    )
    response = await service.answer(42, 9, [{"role": "user", "content": "запрос"}], 0.0)
    assert response.text == "Не удалось проверить результат специалиста."
    assert response.confirmation is None


async def test_dispatch_signal_distinguishes_pre_and_post_sdk_failures(monkeypatch):
    from anyio import EndOfStream

    from app.mcp_tools.client import MCPClient, MCPUnavailable, observe_tool_dispatch
    from app.mcp_tools.contracts import SCHEDULE, ToolError

    events = []
    async with MCPClient(Settings(bot_token="", postgres_password="unused")) as mcp:
        with observe_tool_dispatch(events.append), pytest.raises(ToolError):
            await mcp.call(SCHEDULE, {"date": "not-a-date"})
        assert events == []
        send = AsyncMock(side_effect=EndOfStream)
        monkeypatch.setattr(mcp._client.session, "call_tool", send)
        with observe_tool_dispatch(events.append), pytest.raises(MCPUnavailable):
            await mcp.call(SCHEDULE, {"date": "2026-10-12"})
        assert events == [SCHEDULE]
        send.assert_awaited_once()


async def test_weather_trace_and_used_tools_follow_sdk_dispatch(monkeypatch):
    from uuid import uuid4

    from anyio import EndOfStream

    from app.mcp_tools.client import MCPClient
    from app.mcp_tools.contracts import WEATHER

    settings = Settings(bot_token="", postgres_password="unused")
    place = {
        "city_name": "Helsinki",
        "city_country": "Finland",
        "city_region": None,
        "latitude": 60.17,
        "longitude": 24.94,
    }
    store = Mock(get_confirmed_city=AsyncMock(return_value=place))
    trace = Mock(span=AsyncMock())
    async with MCPClient(settings) as mcp:
        service = AgentService(store, Mock(), mcp, settings, traces=trace)
        discovered = mcp.tools
        mcp.tools = ()
        pre_tools = []
        await service._confirmed_city_weather(42, "погода", uuid4(), pre_tools)
        assert pre_tools == []
        assert trace.span.await_args.kwargs["tool_name"] is None
        assert trace.span.await_args.args[2] == "execute"
        trace.span.reset_mock()
        mcp.tools = discovered
        monkeypatch.setattr(mcp._client.session, "call_tool", AsyncMock(side_effect=EndOfStream))
        post_tools = []
        outcome = await service._confirmed_city_weather(42, "погода", uuid4(), post_tools)
        assert outcome.action == "отказ"
        assert outcome.facts is None
        assert post_tools == [WEATHER]
        assert trace.span.await_args.kwargs["tool_name"] == WEATHER
        assert trace.span.await_args.args[2] == "tool"
        assert trace.span.await_args.kwargs["error_code"] == "unavailable"


async def test_briefing_trace_records_failed_dispatched_tool(monkeypatch):
    from uuid import uuid4

    from anyio import EndOfStream

    from app.briefing import BriefingBuilder
    from app.mcp_tools.client import MCPClient
    from app.mcp_tools.contracts import WEATHER

    settings = Settings(bot_token="", postgres_password="unused")
    async with MCPClient(settings) as mcp:
        mcp.tools = tuple(tool for tool in mcp.tools if tool.name == WEATHER)
        mcp.resources = ()
        monkeypatch.setattr(mcp._client.session, "call_tool", AsyncMock(side_effect=EndOfStream))
        events = []
        now = datetime(2026, 11, 23, tzinfo=UTC)
        request = AgentRequest(
            request_id=uuid4(),
            trace_id=uuid4(),
            actor_id=42,
            text="Ежедневная сводка",
            intent="daily_briefing",
            occurred_at=now,
            timezone="UTC",
        )
        results = await BriefingBuilder(mcp)._sections(
            request,
            now.date(),
            {
                "city_name": "Helsinki",
                "city_country": "Finland",
                "city_region": None,
                "latitude": 60.17,
                "longitude": 24.94,
                "timezone": "UTC",
            },
            events,
            lambda: now,
        )
        assert results[1].message == "Погода: данные недоступны."
        assert results[1].used_tools == [WEATHER]
        assert results[1].facts == EmptyFacts()
        assert any(
            e.component == "lifestyle"
            and e.operation == "tool"
            and e.status == "error"
            and e.tool_name == WEATHER
            for e in events
        )


async def test_policy_rejection_has_unauthorized_code_and_no_used_tool():
    from uuid import uuid4

    from app.agent_router import RouteContext
    from app.mcp_tools.contracts import WEATHER, tool_definitions

    now = datetime(2026, 11, 23, tzinfo=UTC)
    decision = SimpleNamespace(action="call_tool", name=WEATHER)
    model_router = Mock(decide=AsyncMock(return_value=decision))
    mcp = Mock(available=True, tools=tool_definitions(), call=AsyncMock())
    service = AgentService(Mock(), model_router, mcp, Settings(bot_token="", postgres_password=""))
    request = AgentRequest(
        request_id=uuid4(),
        trace_id=uuid4(),
        actor_id=42,
        text="расписание",
        intent="user_query",
        occurred_at=now,
        timezone="UTC",
    )
    response = await service._schedule_specialist(
        request,
        1,
        [{"role": "user", "content": request.text}],
        0.0,
        RouteContext(now, "UTC", "UTC"),
        None,
    )
    assert response.result.status == "failed"
    assert response.result.error_code == "unauthorized"
    assert response.result.used_tools == []
    mcp.call.assert_not_awaited()


async def test_ambiguous_confirmed_weather_is_failed_before_success_span(monkeypatch):
    from uuid import uuid4

    from app.briefing import BriefingBuilder
    from app.mcp_tools.client import MCPClient
    from app.mcp_tools.contracts import WEATHER

    settings = Settings(bot_token="", postgres_password="unused")
    async with MCPClient(settings) as mcp:
        mcp.tools = tuple(tool for tool in mcp.tools if tool.name == WEATHER)
        mcp.resources = ()
        ambiguous = {
            "status": "ambiguous",
            "source": "Open-Meteo",
            "options": [
                {"city": "Helsinki", "country": "Finland", "region": None},
                {"city": "Helsinki", "country": "Other", "region": None},
            ],
        }
        monkeypatch.setattr(
            mcp._client.session,
            "call_tool",
            AsyncMock(return_value=SimpleNamespace(structured_content=ambiguous, is_error=False)),
        )
        now = datetime(2026, 11, 23, tzinfo=UTC)
        request = AgentRequest(
            request_id=uuid4(),
            trace_id=uuid4(),
            actor_id=42,
            text="Ежедневная сводка",
            intent="daily_briefing",
            occurred_at=now,
            timezone="UTC",
        )
        events = []
        sections = await BriefingBuilder(mcp)._sections(
            request,
            now.date(),
            {
                "city_name": "Helsinki",
                "city_country": "Finland",
                "city_region": None,
                "latitude": 60.17,
                "longitude": 24.94,
                "timezone": "UTC",
            },
            events,
            lambda: now,
        )
        assert sections[1].status == "failed"
        assert sections[1].error_code == "invalid_result"
        assert sections[1].facts == EmptyFacts()
        assert sections[1].used_tools == [WEATHER]
        weather_events = [e for e in events if e.component == "lifestyle"]
        assert [(e.status, e.operation, e.tool_name, e.error_code) for e in weather_events] == [
            ("error", "tool", WEATHER, "invalid_result")
        ]
