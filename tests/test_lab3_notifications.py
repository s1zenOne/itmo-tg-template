from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from app.handlers.assistant import create_router
from app.mcp_tools.contracts import ToolError
from app.mcp_tools.weather import GeocodedPlace, OpenMeteoWeather
from app.notifications import CityProposal, NotificationService, geocode_city

NOW = datetime(2026, 10, 12, 6, tzinfo=UTC)
CITY = GeocodedPlace(
    name="Казань", country="Россия", admin1="Татарстан", latitude=55.8, longitude=49.1
)


class Context:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_):
        return False


class FakePool:
    def __init__(self, rows):
        self.fetchrow = AsyncMock(side_effect=rows)
        self.execute = AsyncMock()

    def acquire(self):
        return Context(self)

    def transaction(self):
        return Context(self)


async def test_geocoder_checks_location_and_ambiguity(monkeypatch):
    places = [CITY.model_dump()]
    get = AsyncMock(return_value={"generationtime_ms": 1.0, "results": places})
    monkeypatch.setattr(OpenMeteoWeather, "_get_json", get)
    assert await geocode_city("Казань, Россия") == CITY
    assert get.await_args.args[1]["name"] == "Казань"
    places.append({**CITY.model_dump(), "admin1": "Другая область"})
    with pytest.raises(ToolError, match="city_ambiguous"):
        await geocode_city("Казань, Россия")
    assert await geocode_city("Казань, Татарстан, Россия") == CITY
    with pytest.raises(ToolError, match="invalid_city"):
        await geocode_city("Казань")


async def test_confirmation_checks_owner_destination_and_expiry():
    action = uuid4()
    row = {
        "status": "pending",
        "expires_at": NOW + timedelta(minutes=1),
        "city_name": CITY.name,
        "city_country": CITY.country,
        "city_region": CITY.admin1,
        "latitude": CITY.latitude,
        "longitude": CITY.longitude,
    }
    pool = FakePool([None, row, {"destination_chat_id": 202}])
    service = NotificationService(pool, clock=lambda: NOW)
    assert not await service.confirm_city(303, 202, action, accepted=True)
    pool.execute.assert_not_awaited()
    assert await service.confirm_city(101, 202, action, accepted=True)
    assert pool.fetchrow.await_args_list[0].args[1:] == (303, 202, action)
    assert pool.execute.await_count == 2
    assert pool.execute.await_args_list[1].args[1:] == (
        101,
        CITY.name,
        CITY.country,
        CITY.admin1,
        CITY.latitude,
        CITY.longitude,
        202,
    )
    expired_pool = FakePool([{**row, "expires_at": NOW}])
    assert not await NotificationService(expired_pool, clock=lambda: NOW).confirm_city(
        101, 202, action, accepted=True
    )
    expired_pool.execute.assert_not_awaited()
    moved_pool = FakePool([row, {"destination_chat_id": 404}])
    assert not await NotificationService(moved_pool, clock=lambda: NOW).confirm_city(
        101, 202, action, accepted=True
    )
    moved_pool.execute.assert_not_awaited()


@pytest.mark.parametrize(
    ("zone", "city", "expected", "updates"),
    [
        (None, CITY.name, "/timezone", 0),
        ("Europe/Moscow", None, "/city", 0),
        ("Europe/Moscow", CITY.name, "включена", 1),
    ],
)
async def test_briefing_requires_both_zone_and_confirmed_city(zone, city, expected, updates):
    pool = FakePool([{"timezone": zone, "city_name": city, "latitude": 55.8, "longitude": 49.1}])
    result = await NotificationService(pool).set_briefing(101, True)
    assert expected in result
    assert pool.execute.await_count == updates


async def test_private_update_and_callback_use_trusted_actor_and_chat():
    action = uuid4()
    notifications = SimpleNamespace(
        register_private=AsyncMock(),
        prepare_city=AsyncMock(return_value=CityProposal(action, CITY)),
        confirm_city=AsyncMock(side_effect=[False, True]),
    )
    bot = Bot("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk")
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(
        create_router(SimpleNamespace(agent=None), notifications=notifications)
    )
    owner = User(id=101, is_bot=False, first_name="Студент")
    message = Message(
        message_id=1,
        date=NOW,
        chat=Chat(id=202, type="private"),
        from_user=owner,
        text="/city Казань, Россия",
    )
    await dispatcher.feed_update(bot, Update(update_id=1, message=message))
    notifications.register_private.assert_awaited_once_with(101, 202)
    notifications.prepare_city.assert_awaited_once_with(101, 202, 1, "Казань, Россия")
    sent = [x.args[1] for x in bot.session.call_args_list if isinstance(x.args[1], SendMessage)]
    button = sent[-1].reply_markup.inline_keyboard[0][0]
    for uid, actor in ((2, User(id=303, is_bot=False, first_name="Другой")), (3, owner)):
        callback = CallbackQuery(
            id=str(uid),
            from_user=actor,
            chat_instance="test",
            message=message,
            data=button.callback_data,
        )
        await dispatcher.feed_update(bot, Update(update_id=uid, callback_query=callback))
    assert notifications.confirm_city.await_args_list[0].args == (303, 202, action)
    assert notifications.confirm_city.await_args_list[1].args == (101, 202, action)
    group = message.model_copy(update={"chat": Chat(id=-100, type="group")})
    await dispatcher.feed_update(bot, Update(update_id=4, message=group))
    assert notifications.register_private.await_count == 1
