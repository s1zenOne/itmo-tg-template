import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.agent import AgentService
from app.agent_router import AgentRouter
from app.config import Settings
from app.llm import LLMResult, NativeToolCall
from app.mcp_tools.contracts import WEATHER, WEEK_URI, tool_definitions

NOW = datetime(2026, 10, 12, 6, tzinfo=UTC)


def build(target, responses=()):
    store = Mock(
        reminder_for_update=AsyncMock(return_value=None),
        get_timezone=AsyncMock(return_value="Europe/Moscow"),
        write_audit=AsyncMock(),
    )
    mcp = Mock(
        available=True,
        tools=tool_definitions(),
        resources=[SimpleNamespace(uri=WEEK_URI)],
        connect=AsyncMock(return_value=True),
        read_resource=AsyncMock(),
        call=AsyncMock(),
    )
    llm = Mock(generate=AsyncMock(side_effect=responses))
    specialist = Mock(decide=AsyncMock(return_value=SimpleNamespace(target=target)))
    return AgentService(
        store,
        AgentRouter(llm),
        mcp,
        Settings(bot_token="", postgres_password=""),
        clock=lambda: NOW,
        specialist_router=specialist,
    )


def messages(query):
    return [{"role": "system", "content": "Учебный помощник."}, {"role": "user", "content": query}]


def week():
    return {
        "timezone": "Europe/Moscow",
        "source": "Учебное расписание",
        "days": [
            {
                "status": "ok",
                "date": (NOW.date() + timedelta(days=i)).isoformat(),
                "lessons": [],
                "source": "Учебное расписание",
            }
            for i in range(7)
        ],
    }


async def test_schedule_current_week_reads_discovered_resource_and_all_seven_days():
    agent = build("schedule")
    agent.mcp.read_resource.return_value = week()
    result = await agent.answer(5, 9, messages("Покажи расписание занятий на эту неделю."), 0.3)
    agent.mcp.read_resource.assert_awaited_once_with(WEEK_URI)
    agent.router.llm.generate.assert_not_awaited()
    agent.mcp.call.assert_not_awaited()
    assert all(day["date"] in result.text for day in week()["days"])
    assert result.text.count("Занятий в учебном расписании нет") == 7


async def test_week_resource_must_be_discovered_and_match_current_week():
    agent = build("schedule")
    agent.mcp.resources = []
    result = await agent.answer(5, 9, messages("Расписание на эту неделю"), 0.3)
    assert "недоступно" in result.text
    agent.mcp.read_resource.assert_not_awaited()
    agent.mcp.resources = [SimpleNamespace(uri=WEEK_URI)]
    corrupt = week()
    corrupt["days"][6]["date"] = "2026-10-30"
    agent.mcp.read_resource.return_value = corrupt
    result = await agent.answer(5, 10, messages("Расписание на эту неделю"), 0.3)
    assert "недоступно" in result.text


@pytest.mark.parametrize(
    "query, conditions, temperature, wind, expected",
    [
        ("Стоит ли брать зонт в Казани?", "слабый дождь", 12.0, 8.0, "стоит взять зонт"),
        ("Как одеться в Казани?", "ясно", 2.0, 40.0, "тёплую одежду"),
        ("Нужен ли зонт в Казани?", "слабая ледяная морось", 0.0, 8.0, "стоит взять зонт"),
        ("Нужен ли зонт в Казани?", "сильная ледяная морось", 0.0, 8.0, "стоит взять зонт"),
        ("Нужен ли зонт в Казани?", "ясно", 12.0, 8.0, "дождь не указан"),
    ],
)
async def test_lifestyle_advice_uses_validated_weather_without_tool_text_in_model(
    query, conditions, temperature, wind, expected
):
    call = NativeToolCall("call-1", WEATHER, json.dumps({"city": "Казань"}))
    model_call = LLMResult("", "test", "test", None, "test", tool_calls=(call,))
    model_answer = LLMResult("ответ", "test", "test", None, "test", tool_calls=())
    agent = build("lifestyle", [model_call, model_answer])
    agent.mcp.call.return_value = {
        "status": "ok",
        "city": "Казань",
        "country": "Россия",
        "region": "Татарстан",
        "temperature_c": temperature,
        "conditions": conditions,
        "wind_speed": wind,
        "wind_unit": "km/h",
        "observed_at": "2026-10-12T08:55:00+03:00",
        "timezone": "Europe/Moscow",
        "source": "Open-Meteo",
    }
    result = await agent.answer(5, 9, messages(query), 0.3)
    assert expected in result.text
    if wind >= 35:
        assert "ветер" in result.text
    sent_to_model = agent.router.llm.generate.await_args.args[0][-1]["content"]
    assert conditions not in sent_to_model
    assert "Татарстан" not in sent_to_model
    agent.mcp.call.assert_awaited_once_with(WEATHER, {"city": "Казань"})


async def test_model_cannot_supply_coordinates_even_if_server_accepts_them():
    call = NativeToolCall(
        "call-1",
        WEATHER,
        json.dumps(
            {
                "city": "Казань",
                "country": "Россия",
                "latitude": 55.8,
                "longitude": 49.1,
            }
        ),
    )
    agent = build("lifestyle", [LLMResult("", "test", "test", None, "test", tool_calls=(call,))])
    result = await agent.answer(5, 9, messages("Какая погода в Казани?"), 0.3)
    assert "параметры" in result.text
    agent.mcp.call.assert_not_awaited()
    schemas = agent.router.llm.generate.await_args.kwargs["tools"]
    weather = next(item for item in schemas if item["function"]["name"] == WEATHER)
    assert set(weather["function"]["parameters"]["properties"]) == {"city"}
