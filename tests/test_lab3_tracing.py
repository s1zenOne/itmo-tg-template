from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.db import create_pool
from app.migrate import migrate
from app.tracing import TraceRecorder

NOW = datetime(2026, 11, 23, 10, tzinfo=UTC)


async def test_span_records_only_allowed_metadata_and_sanitizes_errors():
    pool = Mock()
    pool.fetchval = AsyncMock(side_effect=lambda _sql, *args: args[0])
    recorder = TraceRecorder(pool)
    trace = uuid4()
    span = await recorder.span(
        trace,
        "knowledge",
        "tool",
        NOW,
        NOW + timedelta(milliseconds=20),
        "error",
        tool_name="search_materials",
        error_code="private user message / secret",
    )
    sql, *args = pool.fetchval.await_args.args
    assert args[0] == span and args[1] == trace and args[2] == trace
    assert args[-1] == "internal_error"
    assert "private user message" not in str(args)
    assert "parent_span_id" in sql


@pytest.mark.parametrize(
    "kwargs",
    [
        {"component": "ignore all instructions"},
        {"operation": "steal"},
        {"status": "revealed"},
        {"operation": "tool", "tool_name": "unknown"},
        {"tool_name": "get_weather"},
        {"started_at": datetime(2026, 1, 1)},
    ],
)
async def test_unsafe_span_fields_are_rejected_before_database(kwargs):
    pool = Mock()
    pool.fetchval = AsyncMock()
    args = dict(
        trace_id=uuid4(),
        component="router",
        operation="route",
        started_at=NOW,
        finished_at=NOW,
        status="ok",
    )
    args.update(kwargs)
    with pytest.raises(ValueError):
        await TraceRecorder(pool).span(**args)
    pool.fetchval.assert_not_awaited()


async def test_render_uses_owner_for_uuid_and_does_not_return_foreign_data():
    pool = Mock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetch = AsyncMock()
    trace_id = uuid4()
    text = await TraceRecorder(pool).render(3, str(trace_id))
    sql, owner, supplied = pool.fetchrow.await_args.args
    assert "owner_id = $1" in sql and owner == 3 and supplied == trace_id
    assert text == "Трасса не найдена."
    pool.fetch.assert_not_awaited()


async def test_start_close_and_render_use_safe_stored_fields():
    pool = Mock()
    pool.execute = AsyncMock()
    pool.fetchval = AsyncMock(side_effect=lambda _sql, *args: args[0])
    recorder = TraceRecorder(pool, clock=lambda: NOW)
    trace_id = await recorder.start(42, "0123456789abcdef", "schedule")
    _, written_id, request_id, owner_id, route, started = pool.execute.await_args.args
    assert (written_id, request_id, owner_id, route, started) == (
        trace_id,
        "0123456789abcdef",
        42,
        "schedule",
        NOW,
    )
    pool.fetchrow = AsyncMock(return_value={"trace_id": trace_id, "route": route})
    pool.fetch = AsyncMock(
        return_value=[
            {
                "component": "schedule",
                "operation": "tool",
                "status": "ok",
                "started_at": NOW,
                "finished_at": NOW + timedelta(milliseconds=23),
                "tool_name": "get_schedule",
                "error_code": None,
            }
        ]
    )
    rendered = await recorder.render(owner_id, "last")
    assert "schedule" in rendered and "get_schedule" in rendered and "23 мс" in rendered
    await recorder.close(trace_id, "ok", NOW)
    assert pool.fetchval.await_args.args[1:] == (trace_id, NOW, "ok")


async def test_invalid_request_identifier_is_rejected_before_write():
    pool = Mock()
    pool.execute = AsyncMock()
    with pytest.raises(ValueError):
        await TraceRecorder(pool).start(42, "personal message", "schedule")
    pool.execute.assert_not_awaited()


@pytest.mark.integration
async def test_tracing_on_real_postgres_keeps_tree_and_owner_isolation(lab2_database):
    async with await create_pool(lab2_database) as pool:
        await migrate(pool)
        recorder = TraceRecorder(pool)
        trace = await recorder.start(101, uuid4(), "knowledge")
        now = datetime.now(UTC)
        parent = await recorder.span(trace, "knowledge", "execute", now, now, "ok")
        await recorder.span(
            trace, "mcp", "tool", now, now, "ok", "search_materials", parent_span_id=parent
        )
        await recorder.close(trace, "ok")
        assert (
            await pool.fetchval("SELECT count(*) FROM lab3_trace_spans WHERE trace_id=$1", trace)
            == 3
        )
        assert (
            await pool.fetchval(
                "SELECT parent_span_id FROM lab3_trace_spans WHERE tool_name='search_materials'"
            )
            == parent
        )
        assert "search_materials" in await recorder.render(101, str(trace))
        assert "search_materials" in await recorder.render(101, "last")
        assert await recorder.render(202, str(trace)) == "Трасса не найдена."
        assert await recorder.render(202, "last") == "Трасса не найдена."


@pytest.mark.integration
async def test_confirmed_city_weather_trace_on_real_postgres(lab2_database):
    async with await create_pool(lab2_database) as pool:
        await migrate(pool)
        recorder = TraceRecorder(pool)
        trace = await recorder.start(101, uuid4(), "lifestyle")
        now = datetime.now(UTC)
        await recorder.span(trace, "lifestyle", "tool", now, now, "ok", "get_weather")
        await recorder.close(trace, "ok")
        assert "get_weather" in await recorder.render(101, str(trace))
