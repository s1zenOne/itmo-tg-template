from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.agent import AgentService
from app.config import Settings
from app.db import create_pool
from app.mcp_tools.client import MCPUnavailable, _dispatch_observer
from app.mcp_tools.contracts import CATALOG_URI, REMINDER, WEEK_URI, tool_definitions
from app.migrate import migrate
from app.reminders import PreparedReminder, ReminderError
from app.tracing import TraceRecorder

NOW = datetime(2026, 10, 12, 6, tzinfo=UTC)
TRACE = uuid4()
ACTION = uuid4()


def build():
    traces = Mock(start=AsyncMock(return_value=TRACE), span=AsyncMock(), close=AsyncMock())
    store = Mock(confirm_reminder=AsyncMock(), cancel_reminder=AsyncMock(), write_audit=AsyncMock())
    mcp = Mock(
        tools=tool_definitions(),
        resources=[SimpleNamespace(uri=WEEK_URI), SimpleNamespace(uri=CATALOG_URI)],
        read_resource=AsyncMock(),
        call=AsyncMock(),
    )
    agent = AgentService(
        store,
        Mock(),
        mcp,
        Settings(bot_token="", postgres_password=""),
        clock=lambda: NOW,
        specialist_router=Mock(),
        traces=traces,
    )
    return agent


@pytest.mark.parametrize("resource", ["week", "catalog"])
async def test_resource_failure_records_safe_error_without_claiming_a_tool(resource):
    agent = build()
    agent.mcp.read_resource.side_effect = MCPUnavailable()
    if resource == "week":
        result = await agent._current_week(NOW, TRACE)
        component = "schedule"
    else:
        result = await agent._knowledge(SimpleNamespace(text="личное сообщение"), TRACE, [])
        component = "knowledge"
    assert result.action == "отказ"
    args = agent.traces.span.await_args
    assert args.args == (TRACE, component, "resource", NOW, NOW, "error")
    assert args.kwargs == {"error_code": "unavailable"}
    agent.mcp.call.assert_not_awaited()


async def test_missing_week_resource_is_policy_error_before_read():
    agent = build()
    agent.mcp.resources = []
    result = await agent._current_week(NOW, TRACE)
    assert result.action == "отказ"
    agent.mcp.read_resource.assert_not_awaited()
    assert agent.traces.span.await_args.args[1:3] == ("schedule", "execute")
    assert agent.traces.span.await_args.kwargs == {"error_code": "unauthorized"}


def prepared():
    return PreparedReminder(
        ACTION, "личный текст задачи", NOW + timedelta(hours=1), "UTC", NOW, "confirmed"
    )


@pytest.mark.parametrize(
    "case", ["success", "owner", "before_dispatch", "after_dispatch", "invalid_result"]
)
async def test_confirmation_trace_distinguishes_denial_and_sdk_dispatch(case):
    agent = build()
    agent.store.confirm_reminder.return_value = prepared()
    if case == "owner":
        agent.store.confirm_reminder.side_effect = ReminderError("unavailable")
    elif case == "before_dispatch":
        agent.mcp.call.side_effect = MCPUnavailable()
    elif case == "after_dispatch":

        async def fail_after_dispatch(*args, **kwargs):
            _dispatch_observer.get()(REMINDER)
            raise MCPUnavailable()

        agent.mcp.call.side_effect = fail_after_dispatch
    else:
        agent.mcp.call.return_value = {
            "status": "ok",
            "id": 7,
            "remind_at": (
                prepared().remind_at + timedelta(hours=1)
                if case == "invalid_result"
                else prepared().remind_at
            ).isoformat(),
            "timezone": "UTC",
            "created": True,
        }

    result = await agent.confirm(101, ACTION, accepted=True)
    calls = agent.traces.span.await_args_list
    tools = [call for call in calls if call.args[2] == "tool"]
    assert len(tools) == (1 if case in {"success", "after_dispatch", "invalid_result"} else 0)
    if tools:
        assert tools[0].kwargs["tool_name"] == REMINDER
        assert tools[0].args[5] == ("ok" if case == "success" else "error")
    assert calls[-1].args[1:3] == ("schedule", "confirm")
    assert calls[-1].args[5] == ("ok" if case == "success" else "error")
    assert agent.traces.close.await_args.args == (TRACE, "ok" if case == "success" else "error")
    assert "личный текст задачи" not in str(calls)
    if case == "owner":
        agent.mcp.call.assert_not_awaited()
        assert calls[-1].kwargs == {"error_code": "unavailable"}
    assert ("сохранено" in result.text) == (case == "success")


async def test_cancel_records_confirmation_without_calling_mcp():
    agent = build()
    result = await agent.confirm(101, ACTION, accepted=False)
    agent.store.cancel_reminder.assert_awaited_once_with(101, ACTION)
    agent.mcp.call.assert_not_awaited()
    assert "отменено" in result.text
    assert len(agent.traces.span.await_args_list) == 1
    assert agent.traces.span.await_args.args[1:3] == ("schedule", "confirm")
    agent.traces.close.assert_awaited_once_with(TRACE, "ok")


@pytest.mark.integration
async def test_confirmation_trace_persists_for_owner_on_postgres(lab2_database):
    async with await create_pool(lab2_database) as pool:
        await migrate(pool)
        agent = build()
        agent.traces = TraceRecorder(pool, clock=lambda: NOW)
        agent.store.confirm_reminder.return_value = prepared()
        agent.mcp.call.return_value = {
            "status": "ok",
            "id": 7,
            "remind_at": prepared().remind_at.isoformat(),
            "timezone": "UTC",
            "created": True,
        }
        await agent.confirm(101, ACTION, accepted=True)
        trace = await agent.traces.render(101, "last")
        assert "add_reminder" in trace and "confirm" in trace
        assert "личный текст задачи" not in trace
        assert str(ACTION) not in trace
        assert await agent.traces.render(202, "last") == "Трасса не найдена."
