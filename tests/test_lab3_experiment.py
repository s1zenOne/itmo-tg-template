from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scripts.lab3_routing_experiment import MemoryStore, ObservedMCP, PinnedLLM


async def test_experiment_pins_model_and_endpoint_and_records_provider_metadata():
    response = SimpleNamespace(model="test/model:free", version="v1", usage={"total_tokens": 12})
    inner = SimpleNamespace(generate=AsyncMock(return_value=response))
    client = PinnedLLM(inner, "test/model:free", "test-provider")
    await client.generate([], 0.0, "id", tools=[])
    assert inner.generate.await_args.kwargs["model"] == "test/model:free"
    assert inner.generate.await_args.kwargs["endpoint"] == "test-provider"
    assert client.responses == [response]


async def test_experiment_records_actual_mcp_calls_and_resource_reads():
    inner = SimpleNamespace(
        call=AsyncMock(return_value={}), read_resource=AsyncMock(return_value={})
    )
    client = ObservedMCP(inner)
    await client.call("get_schedule", {"date": "2026-10-12"})
    await client.read_resource("materials://catalog")
    inner.call.assert_awaited_once_with("get_schedule", {"date": "2026-10-12"})
    assert client.tools_called == ["get_schedule"]
    assert client.resources_read == ["materials://catalog"]


async def test_experiment_uses_explicit_confirmed_city_fixture():
    store = MemoryStore({"city": "Санкт-Петербург, Россия", "timezone": "Europe/Moscow"})
    assert (await store.get_confirmed_city(1))["latitude"] == 59.9343
    with pytest.raises(ValueError):
        MemoryStore({"city": "Другой город", "timezone": "Europe/Moscow"})
