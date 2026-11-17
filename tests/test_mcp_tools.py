import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from anyio import EndOfStream
from mcp import Client

from app.config import Settings
from app.db import create_pool
from app.mcp_tools.auth import GRANT_META, make_grant
from app.mcp_tools.client import MCPClient, MCPUnavailable
from app.mcp_tools.contracts import (
    REMINDER,
    SCHEDULE,
    WEATHER,
    WEEK_URI,
    ToolError,
    validate_arguments,
    validate_result,
)
from app.mcp_tools.schedule import FileSchedule
from app.mcp_tools.server import create_server
from app.mcp_tools.weather import FORECAST_URL, GEOCODING_URL, OpenMeteoWeather
from app.store import PostgresStore

HELSINKI = {
    "name": "Helsinki",
    "country": "Finland",
    "admin1": "Uusimaa",
    "latitude": 60.1695,
    "longitude": 24.9354,
}
FORECAST = {
    "timezone": "Europe/Helsinki",
    "current": {
        "time": 1794819600,
        "temperature_2m": 3.1,
        "wind_speed_10m": 18,
        "weather_code": 61,
    },
    "current_units": {"time": "unixtime", "temperature_2m": "°C", "wind_speed_10m": "km/h"},
}


def _response(status, data):
    @asynccontextmanager
    async def response():
        partial = data if isinstance(data, bytes) else json.dumps(data).encode()
        yield SimpleNamespace(
            status=status,
            content=SimpleNamespace(
                readexactly=AsyncMock(side_effect=asyncio.IncompleteReadError(partial, 1))
            ),
        )

    return response()


def _weather(*responses):
    session = SimpleNamespace(get=Mock(side_effect=responses))
    return OpenMeteoWeather(session)


async def test_weather_structured_result_through_mcp():
    weather = _weather(
        _response(200, {"generationtime_ms": 0.2, "results": [HELSINKI]}),
        _response(200, FORECAST),
    )
    async with Client(create_server(weather, FileSchedule())) as client:
        await client.list_tools()
        result = await client.call_tool(WEATHER, {"city": "  Helsinki, Finland  "})
    assert not result.is_error
    assert result.structured_content == {
        "status": "ok",
        "city": "Helsinki",
        "country": "Finland",
        "region": "Uusimaa",
        "temperature_c": 3.1,
        "conditions": "слабый дождь",
        "wind_speed": 18.0,
        "wind_unit": "km/h",
        "observed_at": "2026-11-16T11:00:00+02:00",
        "timezone": "Europe/Helsinki",
        "source": "Open-Meteo",
    }
    geocoding, forecast = weather.session.get.call_args_list
    assert geocoding.args == (GEOCODING_URL,)
    assert geocoding.kwargs["params"]["name"] == "Helsinki, Finland"
    assert geocoding.kwargs["allow_redirects"] is False
    assert forecast.args == (FORECAST_URL,)
    assert forecast.kwargs["params"]["latitude"] == HELSINKI["latitude"]
    assert forecast.kwargs["params"]["longitude"] == HELSINKI["longitude"]


async def test_weather_unknown_city():
    weather = _weather(_response(200, {"generationtime_ms": 0.1}))
    with pytest.raises(ToolError, match="city_not_found"):
        await weather.get_weather("Неизвестный город")
    assert weather.session.get.call_count == 1


async def test_weather_skips_incomplete_geocoding_results():
    incomplete = {key: value for key, value in HELSINKI.items() if key != "country"}
    weather = _weather(
        _response(200, {"generationtime_ms": 0.2, "results": [HELSINKI, incomplete]}),
        _response(200, FORECAST),
    )
    async with Client(create_server(weather, FileSchedule())) as client:
        result = await client.call_tool(WEATHER, {"city": "Helsinki"})
    assert not result.is_error
    assert result.structured_content["city"] == "Helsinki"
    assert result.structured_content["temperature_c"] == 3.1


async def test_weather_rejects_geocoding_with_no_valid_results():
    incomplete = {key: value for key, value in HELSINKI.items() if key != "country"}
    weather = _weather(_response(200, {"generationtime_ms": 0.2, "results": [incomplete]}))
    with pytest.raises(ToolError, match="weather_invalid_response"):
        await weather.get_weather("Helsinki")
    assert weather.session.get.call_count == 1


@pytest.mark.parametrize("kind", ["timeout", "http_503", "malformed_json", "invalid_data"])
async def test_weather_errors_and_bounded_retries(kind):
    match kind:
        case "timeout":
            responses, code = (TimeoutError(), TimeoutError()), "weather_timeout"
        case "http_503":
            responses, code = (_response(503, {}), _response(503, {})), "weather_unavailable"
        case "malformed_json":
            responses, code = (_response(200, b"{"),), "weather_invalid_response"
        case _:
            responses, code = (_response(200, {"results": "bad"}),), "weather_invalid_response"
    weather = _weather(*responses)
    async with Client(create_server(weather, FileSchedule())) as client:
        await client.list_tools()
        result = await client.call_tool(WEATHER, {"city": "Helsinki"})
    assert result.is_error and result.structured_content == {"status": "error", "code": code}
    assert weather.session.get.call_count == len(responses)


async def test_weather_homonyms_return_regions_without_forecast():
    weather = _weather(
        _response(
            200,
            {
                "generationtime_ms": 0.1,
                "results": [
                    dict(HELSINKI, name="Paris", country="France", admin1="Île-de-France"),
                    dict(HELSINKI, name="Paris", country="United States", admin1="Texas"),
                    dict(HELSINKI, name="Parish", country="United States", admin1="New York"),
                ],
            },
        )
    )
    result = await weather.get_weather("Paris")
    assert result.model_dump() == {
        "status": "ambiguous",
        "options": [
            {"city": "Paris", "country": "France", "region": "Île-de-France"},
            {"city": "Paris", "country": "United States", "region": "Texas"},
        ],
        "source": "Open-Meteo",
    }
    assert weather.session.get.call_count == 1


@pytest.mark.parametrize("day, has_lesson", [("2026-12-31", True), ("2027-01-01", False)])
async def test_schedule_day(tmp_path, day, has_lesson):
    lesson = {"start": "10:00", "end": "11:00", "title": "Алгоритмы", "location": "Ауд. 1"}
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"2026-12-31": [lesson]}), encoding="utf-8")
    async with Client(create_server(_weather(), FileSchedule(path))) as client:
        await client.list_tools()
        result = await client.call_tool(SCHEDULE, {"date": day})
    assert not result.is_error
    assert result.structured_content == {
        "status": "ok",
        "date": day,
        "lessons": [lesson] if has_lesson else [],
        "source": "Учебное расписание",
    }


@pytest.mark.parametrize(
    "name, arguments, code",
    [
        (SCHEDULE, {"date": "2026-02-29"}, "invalid_arguments"),
        (SCHEDULE, {"date": "12.10.2026"}, "invalid_arguments"),
        (SCHEDULE, {"date": "1999-12-31"}, "invalid_arguments"),
        (WEATHER, {"city": "Moscow", "url": "https://example.com"}, "invalid_arguments"),
        (WEATHER, {"city": " "}, "invalid_arguments"),
        (WEATHER, {"city": "a" * 101}, "invalid_arguments"),
        (REMINDER, {"text": " ", "remind_at": "2099-10-01T10:00:00+03:00"}, "invalid_arguments"),
        (REMINDER, {"text": "Test", "remind_at": "2099-10-01T10:00:00"}, "invalid_arguments"),
        ("system_shell", {"command": "ls"}, "unknown_tool"),
    ],
)
async def test_contract_rejection_at_both_boundaries(name, arguments, code):
    weather = _weather()
    async with Client(create_server(weather, FileSchedule())) as client:
        await client.list_tools()
        result = await client.call_tool(name, arguments)
    assert result.is_error and result.structured_content == {"status": "error", "code": code}
    with pytest.raises(ToolError, match=code):
        validate_arguments(name, arguments)
    assert weather.session.get.call_count == 0


async def test_schedule_invalid_source(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text('{"2026-02-29": []}', encoding="utf-8")
    with pytest.raises(ToolError, match="schedule_unavailable"):
        await FileSchedule(path).get_schedule("2026-02-28")


async def test_current_week_resource_uses_timezone_at_year_boundary(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text("{}", encoding="utf-8")

    def clock():
        # In Moscow it is already Monday 28 December.
        return datetime(2026, 12, 27, 22, 30, tzinfo=UTC)

    schedule = FileSchedule(path, "Europe/Moscow", clock=clock)
    async with Client(create_server(_weather(), schedule)) as client:
        resources = (await client.list_resources()).resources
        result = await client.read_resource(WEEK_URI)
    assert len(resources) == 1
    assert str(resources[0].uri) == WEEK_URI
    assert resources[0].mime_type == "application/json"
    week = json.loads(result.contents[0].text)
    assert week["timezone"] == "Europe/Moscow"
    assert [day["date"] for day in week["days"]] == [
        "2026-12-28",
        "2026-12-29",
        "2026-12-30",
        "2026-12-31",
        "2027-01-01",
        "2027-01-02",
        "2027-01-03",
    ]
    assert all(day["lessons"] == [] for day in week["days"])


async def test_reminder_rejects_unsigned_forged_and_changed_arguments():
    arguments = {"text": "Проверить отчёт", "remind_at": "2099-10-01T10:00:00+03:00"}
    action_id = uuid4()
    key = b"s" * 32
    valid_grant = make_grant(key, REMINDER, arguments, 5, action_id)
    forged = {GRANT_META: {**valid_grant[GRANT_META], "owner_id": 10}}
    writer = AsyncMock()
    async with Client(
        create_server(_weather(), FileSchedule(), grant_key=key, write_reminder=writer)
    ) as client:
        await client.list_tools()
        for payload, meta in (
            (arguments, None),
            (arguments, forged),
            (dict(arguments, text="Подмена"), valid_grant),
        ):
            result = await client.call_tool(REMINDER, payload, meta=meta)
            assert result.is_error
            assert result.structured_content == {"status": "error", "code": "unauthorized"}
    writer.assert_not_awaited()


def test_missing_or_fabricated_output_is_invalid():
    for data in (None, {"status": "ok"}, {"status": "ok", "lessons": "yes"}):
        with pytest.raises(ToolError, match="invalid_result"):
            validate_result(SCHEDULE, data)


async def test_stdio_mcp_reconnects_after_lost_response_without_replay(monkeypatch):
    async with MCPClient(Settings(bot_token="", postgres_password="unused")) as mcp:
        assert mcp.available
        previous = mcp._client
        lost = AsyncMock(side_effect=EndOfStream)
        monkeypatch.setattr(previous.session, "call_tool", lost)
        with pytest.raises(MCPUnavailable):
            await mcp.call(SCHEDULE, {"date": "2026-10-12"})
        lost.assert_awaited_once()
        assert not mcp.available and not mcp.tools
        result = await mcp.call(SCHEDULE, {"date": "2026-10-12"})
        assert mcp._client is not previous
        assert result["date"] == "2026-10-12"
        assert len(result["lessons"]) == 2


@pytest.mark.integration
async def test_stdio_mcp_signed_write_to_temporary_postgres(lab2_database):
    pool = await create_pool(lab2_database)
    try:
        store = PostgresStore(pool)
        await store.initialize()
        owner_id = 123
        instant = (datetime.now(UTC) + timedelta(days=1)).replace(microsecond=0)
        prepared = await store.prepare_reminder(
            owner_id, 456, "Проверить отчёт", instant, "UTC+03:00"
        )
        args = {"text": prepared.text, "remind_at": instant.isoformat()}
        async with MCPClient(lab2_database) as mcp:
            assert mcp.available
            with pytest.raises(ToolError, match="unconfirmed"):
                await mcp.call(REMINDER, args, owner_id=owner_id, action_id=prepared.action_id)
            await store.confirm_reminder(owner_id, prepared.action_id)
            with pytest.raises(ToolError, match="unavailable"):
                await mcp.call(REMINDER, args, owner_id=owner_id + 1, action_id=prepared.action_id)
            results = await asyncio.gather(
                mcp.call(REMINDER, args, owner_id=owner_id, action_id=prepared.action_id),
                mcp.call(REMINDER, args, owner_id=owner_id, action_id=prepared.action_id),
            )
        assert {result.pop("created") for result in results} == {True, False}
        assert (
            results[0]
            == results[1]
            == {
                "status": "ok",
                "id": 1,
                "remind_at": instant.isoformat(),
                "timezone": "UTC+03:00",
            }
        )
        assert await pool.fetchval("SELECT count(*) FROM reminder_actions") == 1
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 1
    finally:
        await pool.close()
