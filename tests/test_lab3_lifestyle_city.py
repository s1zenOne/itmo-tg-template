from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.agent import AgentService, _confirmed_city_requested
from app.agent_router import AgentRouter
from app.config import Settings
from app.mcp_tools.contracts import WEATHER, tool_definitions


def agent_for(city):
    store = Mock(
        reminder_for_update=AsyncMock(return_value=None),
        get_timezone=AsyncMock(return_value="Europe/Moscow"),
        get_confirmed_city=AsyncMock(return_value=city),
        write_audit=AsyncMock(),
    )
    mcp = Mock(
        available=True,
        tools=tool_definitions(),
        connect=AsyncMock(),
        call=AsyncMock(
            return_value={
                "status": "ok",
                "city": "Казань",
                "country": "Россия",
                "region": "Татарстан",
                "temperature_c": 12.0,
                "conditions": "слабый дождь",
                "wind_speed": 8.0,
                "wind_unit": "km/h",
                "observed_at": "2026-10-12T08:55:00+03:00",
                "timezone": "Europe/Moscow",
                "source": "Open-Meteo",
            }
        ),
    )
    llm = Mock(generate=AsyncMock())
    return AgentService(
        store,
        AgentRouter(llm),
        mcp,
        Settings(bot_token="", postgres_password=""),
        clock=lambda: datetime(2026, 10, 12, 6, tzinfo=UTC),
        specialist_router=Mock(decide=AsyncMock(return_value=SimpleNamespace(target="lifestyle"))),
    )


def messages(text):
    return [{"role": "system", "content": "Учебный помощник."}, {"role": "user", "content": text}]


@pytest.mark.parametrize(
    "query",
    [
        "Какая погода в моём городе?",
        "Нужен зонт в моем городе?",
        "Стоит ли сегодня брать зонт?",
        "Сильный ли сейчас ветер?",
        "Как одеться с учётом текущей погоды?",
    ],
)
async def test_confirmed_city_weather_uses_owner_coordinates(query):
    city = {
        "city_name": "Казань",
        "city_country": "Россия",
        "city_region": "Татарстан",
        "latitude": 55.8,
        "longitude": 49.1,
    }
    agent = agent_for(city)
    result = await agent.answer(5, 9, messages(query), 0.3)
    agent.store.get_confirmed_city.assert_awaited_once_with(5)
    agent.mcp.call.assert_awaited_once_with(
        WEATHER,
        {
            "city": "Казань",
            "country": "Россия",
            "region": "Татарстан",
            "latitude": 55.8,
            "longitude": 49.1,
        },
    )
    agent.router.llm.generate.assert_not_awaited()
    assert "Казань, Россия" in result.text
    if "зонт" in query:
        assert "стоит взять зонт" in result.text


async def test_city_must_be_confirmed_before_weather_call():
    agent = agent_for(None)
    result = await agent.answer(5, 9, messages("Погода в моём городе"), 0.3)
    assert "/city" in result.text
    agent.mcp.call.assert_not_awaited()
    agent.router.llm.generate.assert_not_awaited()


async def test_coordinates_tool_must_be_discovered():
    agent = agent_for(
        {"city_name": "Казань", "city_country": "Россия", "latitude": 55.8, "longitude": 49.1}
    )
    agent.mcp.tools = tuple(tool for tool in agent.mcp.tools if tool.name != WEATHER)
    result = await agent.answer(5, 9, messages("Погода в моём городе"), 0.3)
    assert "недоступна" in result.text
    agent.mcp.call.assert_not_awaited()


@pytest.mark.parametrize("query", ["Нужен зонт в Москве?", "Как одеться в Казани?"])
def test_explicit_city_is_not_replaced_with_saved_city(query):
    assert not _confirmed_city_requested(query)
