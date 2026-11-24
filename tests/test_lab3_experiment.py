from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.mcp_tools.client import MCPClient, MCPUnavailable, observe_tool_dispatch
from app.mcp_tools.contracts import SCHEDULE, ToolError
from scripts.lab3_routing_experiment import MemoryStore, ObservedMCP, PinnedLLM, case_error


async def test_experiment_pins_model_and_endpoint_and_records_provider_metadata():
    response = SimpleNamespace(
        model="test/model:free",
        version="v1",
        usage={"total_tokens": 12},
        tool_calls=[SimpleNamespace(name="select_route"), SimpleNamespace(name="get_schedule")],
    )
    inner = SimpleNamespace(generate=AsyncMock(return_value=response))
    client = PinnedLLM(inner, "test/model:free", "test-provider")
    await client.generate([], 0.0, "id", tools=[])
    assert inner.generate.await_args.kwargs["model"] == "test/model:free"
    assert inner.generate.await_args.kwargs["endpoint"] == "test-provider"
    assert client.responses == [response]
    assert client.tool_proposals == ["select_route", "get_schedule"]


async def test_experiment_records_attempts_without_claiming_sdk_dispatch():
    inner = SimpleNamespace(
        call=AsyncMock(return_value={}), read_resource=AsyncMock(return_value={})
    )
    client = ObservedMCP(inner)
    await client.call("get_schedule", {"date": "2026-10-12"})
    await client.read_resource("materials://catalog")
    inner.call.assert_awaited_once_with("get_schedule", {"date": "2026-10-12"})
    assert client.tool_attempts == [
        {"name": "get_schedule", "sdk_dispatch": False, "outcome": "result"}
    ]
    assert client.resource_attempts == [{"uri": "materials://catalog", "outcome": "result"}]


async def test_experiment_measures_pre_dispatch_rejection_and_sdk_outcomes(monkeypatch):
    # The SDK is stubbed after real discovery, so dispatch still passes through MCPClient.
    async with MCPClient(Settings(bot_token="", postgres_password="unused")) as mcp:
        client = ObservedMCP(mcp)
        outer = []
        with observe_tool_dispatch(outer.append), pytest.raises(ToolError):
            await client.call(SCHEDULE, {"date": "invalid"})
        assert client.tool_attempts[-1] == {
            "name": SCHEDULE,
            "sdk_dispatch": False,
            "outcome": "error",
            "error_type": "ToolError",
        }
        assert outer == []

        payload = {"status": "error", "code": "unavailable"}
        send = AsyncMock(return_value=SimpleNamespace(structured_content=payload, is_error=True))
        monkeypatch.setattr(mcp._client.session, "call_tool", send)
        with observe_tool_dispatch(outer.append), pytest.raises(ToolError):
            await client.call(SCHEDULE, {"date": "2026-10-12"})
        assert outer == [SCHEDULE]
        assert client.tool_attempts[-1] == {
            "name": SCHEDULE,
            "sdk_dispatch": True,
            "outcome": "error",
            "error_type": "ToolError",
        }
        send.assert_awaited_once()

        send.return_value = SimpleNamespace(
            structured_content={
                "status": "ok",
                "date": "2026-10-12",
                "lessons": [],
                "source": "Учебное расписание",
            },
            is_error=False,
        )
        result = await client.call(SCHEDULE, {"date": "2026-10-12"})
        assert result["status"] == "ok"
        assert client.tool_attempts[-1]["sdk_dispatch"] is True
        assert client.tool_attempts[-1]["outcome"] == "result"
        assert client.tool_attempts[-1]["result_status"] == "ok"


async def test_experiment_records_handled_resource_failure_and_error_payload():
    inner = SimpleNamespace(
        call=AsyncMock(return_value={"status": "error"}),
        read_resource=AsyncMock(side_effect=MCPUnavailable("private details")),
    )
    client = ObservedMCP(inner)
    await client.call(SCHEDULE, {})
    with pytest.raises(MCPUnavailable):
        await client.read_resource("materials://catalog")
    assert client.tool_attempts[0]["outcome"] == "error_result"
    assert client.tool_attempts[0]["result_status"] == "error"
    assert client.resource_attempts == [
        {"uri": "materials://catalog", "outcome": "error", "error_type": "MCPUnavailable"}
    ]
    assert case_error(None, object(), client) == "tool_error"
    client.tool_attempts.clear()
    assert case_error(None, object(), client) == "resource_error"
    assert case_error("RuntimeError", object(), client) == "RuntimeError"


async def test_experiment_uses_explicit_confirmed_city_fixture():
    store = MemoryStore({"city": "Санкт-Петербург, Россия", "timezone": "Europe/Moscow"})
    assert (await store.get_confirmed_city(1))["latitude"] == 59.9343
    with pytest.raises(ValueError):
        MemoryStore({"city": "Другой город", "timezone": "Europe/Moscow"})
