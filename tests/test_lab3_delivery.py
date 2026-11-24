import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aiogram.exceptions import (
    TelegramEntityTooLarge,
    TelegramForbiddenError,
    TelegramNotFound,
    TelegramRetryAfter,
)
from aiogram.methods import SendMessage

from app.db import create_pool
from app.delivery import DeliveryService
from app.store import PostgresStore

pytestmark = pytest.mark.integration
NOW = datetime(2026, 11, 16, 8, tzinfo=UTC)


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


class Bot:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def send_message(self, chat_id, text):
        self.calls.append((chat_id, text))
        if self.error:
            raise self.error


async def setup(pool, *, due=NOW, enabled=False, destination=200):
    await PostgresStore(pool).initialize()
    await pool.execute("INSERT INTO chat_settings (chat_id, timezone) VALUES (100, 'UTC')")
    await pool.execute(
        """INSERT INTO notification_settings
           (owner_id, destination_chat_id, briefing_enabled, city_name, latitude, longitude)
           VALUES (100, $1, $2, 'Город', 59.9, 30.3)""",
        destination,
        enabled,
    )
    action = uuid4()
    await pool.execute(
        """INSERT INTO reminder_actions
           (action_id, owner_id, update_id, text, remind_at, timezone, expires_at, status)
           VALUES ($1, 100, 1, 'Проверить лабораторную', $2, 'UTC', $2, 'executed')""",
        action,
        due,
    )
    await pool.execute("INSERT INTO reminders(action_id, created_at) VALUES ($1, $2)", action, due)


async def test_preparation_and_delivery_are_idempotent_on_real_postgres(lab2_database):
    async with await create_pool(lab2_database) as pool:
        await setup(pool)
        clock = Clock()
        bot = Bot()
        first = DeliveryService(pool, bot, clock=clock)
        second = DeliveryService(pool, bot, clock=clock)
        await first.prepare(NOW)
        await first.prepare(NOW)
        await asyncio.gather(first.prepare(NOW), second.prepare(NOW))
        assert await pool.fetchval("SELECT count(*) FROM notification_outbox") == 1
        await asyncio.gather(first.tick(), second.tick())
        await first.tick()
        assert len(bot.calls) == 1
        assert bot.calls[0][0] == 200
        assert await pool.fetchval("SELECT status FROM notification_outbox") == "sent"
        trace = await pool.fetchval("SELECT trace_id FROM notification_outbox")
        spans = await pool.fetch(
            "SELECT component, operation, status FROM lab3_trace_spans "
            "WHERE trace_id=$1 ORDER BY started_at, operation",
            trace,
        )
        assert ("schedule", "prepare", "ok") in [tuple(span) for span in spans]
        assert ("telegram", "send", "ok") in [tuple(span) for span in spans]
        assert await pool.fetchval("SELECT count(*) FROM lab3_traces") == 1


@pytest.mark.parametrize(
    "offset,status,sends", [(0, "sent", 1), (15, "sent", 1), (15.01, "missed", 0), (-1, None, 0)]
)
async def test_reminder_time_boundaries(lab2_database, offset, status, sends):
    async with await create_pool(lab2_database) as pool:
        await setup(pool, due=NOW - timedelta(minutes=offset))
        bot = Bot()
        await DeliveryService(pool, bot, clock=Clock()).tick()
        assert await pool.fetchval("SELECT status FROM notification_outbox") == status
        assert len(bot.calls) == sends
        if sends and offset > 0:
            assert "задержкой" in bot.calls[0][1]


async def test_retry_limit_permanent_failure_and_restart(lab2_database):
    async with await create_pool(lab2_database) as pool:
        await setup(pool)
        clock = Clock()
        bot = Bot(ConnectionError())
        for delay in (0, 30, 60):
            clock.now += timedelta(seconds=delay)
            await DeliveryService(pool, bot, clock=clock).tick()
        row = await pool.fetchrow("SELECT status, attempts, error_code FROM notification_outbox")
        assert tuple(row) == ("failed", 3, "temporary")
        assert len(bot.calls) == 3
        await pool.execute("DELETE FROM notification_outbox")
        method = SendMessage(chat_id=200, text="test")
        bot = Bot(TelegramForbiddenError(method=method, message="forbidden"))
        await DeliveryService(pool, bot, clock=clock).tick()
        row = await pool.fetchrow("SELECT status, attempts, error_code FROM notification_outbox")
        assert tuple(row) == ("failed", 1, "permanent")
        await pool.execute("DELETE FROM notification_outbox")
        entered = asyncio.Event()

        class InterruptedBot:
            async def send_message(self, chat_id, text):
                entered.set()
                await asyncio.Future()

        task = asyncio.create_task(DeliveryService(pool, InterruptedBot(), clock=clock).tick())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        bot = Bot()
        await DeliveryService(pool, bot, clock=clock).tick()
        assert len(bot.calls) == 1
        assert await pool.fetchval("SELECT status FROM notification_outbox") == "sent"


@pytest.mark.parametrize("minutes,status,sends", [(60, "sent", 1), (61, "missed", 0)])
async def test_briefing_grace_and_partial_failure(lab2_database, minutes, status, sends):
    async with await create_pool(lab2_database) as pool:
        await setup(pool, due=NOW + timedelta(days=1), enabled=True)
        clock = Clock(NOW + timedelta(minutes=minutes))
        bot = Bot()

        async def broken(*args):
            raise ConnectionError()

        service = DeliveryService(pool, bot, broken, clock)
        await asyncio.gather(service.tick(), service.tick())
        await service.tick()
        assert await pool.fetchval("SELECT status FROM notification_outbox") == status
        assert len(bot.calls) == sends
        if sends:
            assert "недоступны" in bot.calls[0][1]
            assert "Напоминания:" in bot.calls[0][1]
            assert "UTC" in bot.calls[0][1]


class Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeConnection(Transaction):
    def __init__(self, due, settings):
        self.row = {
            "event_key": "reminder:1",
            "kind": "reminder",
            "owner_id": 100,
            "reminder_id": 1,
            "due_at": due,
            "attempts": 0,
            "message_text": None,
            "next_offset": 0,
        }
        self.settings = settings
        self.updates = []

    def transaction(self):
        return Transaction()

    async def fetchrow(self, sql, *args):
        if "notification_outbox" in sql:
            return self.row
        return self.settings

    async def fetchval(self, sql, *args):
        if "SELECT EXISTS" in sql:
            return False
        return "Выполнить работу"

    async def execute(self, sql, *args):
        self.updates.append((sql, args))
        if "SET message_text=$2" in sql:
            self.row["message_text"] = args[1]
        if "SET next_offset=$2" in sql:
            self.row["next_offset"] = args[1]
        if "SET status=$2, attempts=$3" in sql:
            self.row["attempts"] = args[2]


class FakePool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return self.connection


@pytest.mark.parametrize(
    "minutes,settings,expected_status,expected_calls",
    [
        (0, {"destination_chat_id": 200}, "sent", 1),
        (15, {"destination_chat_id": 200}, "sent", 1),
        (15.01, {"destination_chat_id": 200}, "missed", 0),
        (0, None, "destination_missing", 0),
    ],
)
async def test_delivery_boundary_and_destination_with_fake_pool(
    minutes, settings, expected_status, expected_calls
):
    connection = FakeConnection(NOW - timedelta(minutes=minutes), settings)
    bot = Bot()
    await DeliveryService(FakePool(connection), bot, clock=Clock())._deliver_one(NOW)
    assert len(bot.calls) == expected_calls
    assert expected_status in connection.updates[-1][0]
    if expected_calls:
        assert bot.calls[0][0] == 200
        assert ("задержкой" in bot.calls[0][1]) == (minutes > 0)


async def test_temporary_failure_is_bounded_with_fake_pool():
    connection = FakeConnection(NOW, {"destination_chat_id": 200})
    connection.row["attempts"] = 2
    bot = Bot(ConnectionError())
    await DeliveryService(FakePool(connection), bot, clock=Clock())._deliver_one(NOW)
    assert len(bot.calls) == 1
    assert connection.updates[-1][1][1:3] == ("failed", 3)


@pytest.mark.parametrize("exception", [TelegramNotFound, TelegramEntityTooLarge])
async def test_permanent_telegram_errors_are_not_retried(exception):
    connection = FakeConnection(NOW, {"destination_chat_id": 200})
    bot = Bot(exception(method=SendMessage(chat_id=200, text="x"), message="failure"))
    await DeliveryService(FakePool(connection), bot, clock=Clock())._deliver_one(NOW)
    assert "status='failed'" in connection.updates[-1][0]
    assert "error_code='permanent'" in connection.updates[-1][0]
    assert connection.updates[-1][1][1] == 1


async def test_retry_after_uses_provider_delay_and_limit():
    connection = FakeConnection(NOW, {"destination_chat_id": 200})
    error = TelegramRetryAfter(
        method=SendMessage(chat_id=200, text="x"), message="rate limit", retry_after=90
    )
    await DeliveryService(FakePool(connection), Bot(error), clock=Clock())._deliver_one(NOW)
    assert connection.updates[-1][1][1:] == ("pending", 1, NOW + timedelta(seconds=90))


async def test_long_delivery_resumes_after_committed_part():
    connection = FakeConnection(NOW, {"destination_chat_id": 200})
    connection.row["message_text"] = "A" * 4096 + "B" * 100
    bot = Bot()
    service = DeliveryService(FakePool(connection), bot, clock=Clock())
    await service._deliver_one(NOW)
    assert connection.row["next_offset"] == 4096
    assert bot.calls == [(200, "A" * 4096)]
    bot.error = ConnectionError()
    await service._deliver_one(NOW)
    bot.error = None
    await service._deliver_one(NOW)
    assert [text for _, text in bot.calls] == ["A" * 4096, "B" * 100, "B" * 100]
    assert "status='sent'" in connection.updates[-1][0]


async def test_timezone_change_reschedules_before_delivery():
    settings = {
        "destination_chat_id": 200,
        "briefing_enabled": True,
        "timezone": "Europe/Moscow",
        "city_name": "City",
        "latitude": 59.9,
        "longitude": 30.3,
    }
    connection = FakeConnection(NOW, settings)
    connection.row.update(kind="briefing", local_date=NOW.date(), timezone="UTC")
    bot = Bot()
    await DeliveryService(FakePool(connection), bot, clock=Clock())._deliver_one(NOW)
    sql, args = connection.updates[-1]
    assert "timezone=$2" in sql
    assert args[1:] == ("Europe/Moscow", NOW - timedelta(hours=3))
    assert bot.calls == []


async def test_missing_destination_remains_pending_until_expiry():
    connection = FakeConnection(NOW, None)
    bot = Bot()
    service = DeliveryService(FakePool(connection), bot, clock=Clock())
    await service._deliver_one(NOW)
    assert "destination_missing" in connection.updates[-1][0]
    assert "cancelled" not in connection.updates[-1][0]
    await service._deliver_one(NOW + timedelta(minutes=15, seconds=1))
    assert "status='missed'" in connection.updates[-1][0]
    assert bot.calls == []


async def test_temporary_failure_limit_carries_over_across_parts():
    connection = FakeConnection(NOW, {"destination_chat_id": 200})
    connection.row["message_text"] = "A" * 4096 + "B"
    connection.row["attempts"] = 2
    bot = Bot()
    service = DeliveryService(FakePool(connection), bot, clock=Clock())
    await service._deliver_one(NOW)
    assert connection.row["attempts"] == 2
    bot.error = ConnectionError()
    await service._deliver_one(NOW)
    assert connection.updates[-1][1][1:3] == ("failed", 3)


async def test_partial_briefing_blocks_second_local_date_event():
    settings = {
        "destination_chat_id": 200,
        "briefing_enabled": True,
        "timezone": "UTC",
        "city_name": "City",
        "latitude": 59.9,
        "longitude": 30.3,
    }

    class PartialConnection(FakeConnection):
        first_sent = False

        async def fetchval(self, sql, *args):
            if "SELECT EXISTS" in sql:
                assert "sent_at IS NOT NULL" in sql
                return self.first_sent
            return await super().fetchval(sql, *args)

        async def execute(self, sql, *args):
            await super().execute(sql, *args)
            if "SET next_offset=$2" in sql:
                assert "sent_at=COALESCE(sent_at,$3)" in sql
                self.first_sent = True

    connection = PartialConnection(NOW, settings)
    connection.row.update(
        kind="briefing", local_date=NOW.date(), timezone="UTC", message_text="A" * 4097
    )
    bot = Bot()
    service = DeliveryService(FakePool(connection), bot, clock=Clock())
    await service._deliver_one(NOW)
    assert len(bot.calls) == 1
    connection.row.update(event_key="briefing:second", next_offset=0)
    await service._deliver_one(NOW)
    assert len(bot.calls) == 1
    assert "already_sent" in connection.updates[-1][0]


@pytest.mark.parametrize(
    "error,expected_status,expected_code,terminal",
    [
        (None, "ok", None, True),
        (ConnectionError("sensitive message"), "error", "temporary", False),
        (
            TelegramForbiddenError(
                method=SendMessage(chat_id=200, text="sensitive content"), message="secret"
            ),
            "error",
            "permanent",
            True,
        ),
    ],
)
async def test_actual_send_is_traced_with_safe_metadata(
    monkeypatch, error, expected_status, expected_code, terminal
):
    trace_id = uuid4()
    recorded = []

    class Recorder:
        def __init__(self, connection, *, clock):
            assert isinstance(connection, FakeConnection)

        async def span(self, *args, **kwargs):
            recorded.append(("span", args, kwargs))

        async def close(self, *args):
            recorded.append(("close", args))

    monkeypatch.setattr("app.delivery.TraceRecorder", Recorder)
    connection = FakeConnection(NOW, {"destination_chat_id": 200})
    connection.row["trace_id"] = trace_id
    bot = Bot(error)
    await DeliveryService(FakePool(connection), bot, clock=Clock())._deliver_one(NOW)
    assert len(bot.calls) == 1
    span = recorded[0]
    # Check every field instead of searching a UUID's arbitrary string representation.
    assert span == (
        "span",
        (trace_id, "telegram", "send", NOW, NOW, expected_status),
        {"error_code": expected_code},
    )
    assert (len(recorded) == 2) == terminal
    if terminal:
        assert recorded[1] == ("close", (trace_id, expected_status, NOW))


async def test_preparation_registers_trace_against_event(monkeypatch):
    trace_id = uuid4()
    recorded = []

    class Recorder:
        def __init__(self, connection, *, clock):
            assert isinstance(connection, FakeConnection)

        async def start(self, owner_id, request_id, route):
            recorded.append(("start", owner_id, request_id, route))
            return trace_id

        async def span(self, *args):
            recorded.append(("span", *args))

    monkeypatch.setattr("app.delivery.TraceRecorder", Recorder)
    connection = FakeConnection(NOW, {"destination_chat_id": 200})
    service = DeliveryService(FakePool(connection), Bot(), clock=Clock())
    await service._register_trace(connection, connection.row, NOW)
    assert connection.updates[0][1] == ("reminder:1", trace_id)
    assert recorded[0][1] == 100
    assert recorded[0][3] == "schedule"
    assert recorded[1][1:4] == (trace_id, "schedule", "prepare")
    assert recorded[1][-1] == "ok"
    assert "Выполнить работу" not in repr(recorded)


async def test_briefing_flushes_concurrent_read_events_sequentially_in_delivery_transaction(
    monkeypatch,
):
    from types import SimpleNamespace

    from app.briefing import BriefingBuilder
    from app.mcp_tools.contracts import (
        CATALOG_URI,
        MATERIALS,
        SCHEDULE,
        WEATHER,
        tool_definitions,
    )

    trace_id = uuid4()
    active = False
    recorded = []

    class Recorder:
        def __init__(self, connection, *, clock):
            assert connection is fake

        async def span(self, *args, **kwargs):
            nonlocal active
            assert not active, "concurrent SQL on one connection"
            active = True
            await asyncio.sleep(0)
            recorded.append((args, kwargs))
            active = False

    monkeypatch.setattr("app.delivery.TraceRecorder", Recorder)

    async def call(name, args):
        if name == SCHEDULE:
            return {
                "status": "ok",
                "date": NOW.date().isoformat(),
                "lessons": [],
                "source": "Учебное расписание",
            }
        if name == WEATHER:
            assert args == {
                "city": "Город",
                "country": "Страна",
                "region": None,
                "latitude": 59.9,
                "longitude": 30.3,
            }
            return {
                "status": "ok",
                "city": "Город",
                "country": "Страна",
                "region": None,
                "temperature_c": 3.0,
                "conditions": "ясно",
                "wind_speed": 1.0,
                "wind_unit": "km/h",
                "observed_at": NOW.isoformat(),
                "timezone": "UTC",
                "source": "Open-Meteo",
            }
        assert name == MATERIALS
        return {"status": "ok", "documents": []}

    mcp = SimpleNamespace(
        tools=tool_definitions(),
        resources=[SimpleNamespace(uri=CATALOG_URI)],
        call=AsyncMock(side_effect=call),
        read_resource=AsyncMock(return_value={"documents": []}),
    )
    fake = FakeConnection(NOW, {})
    fake.fetch = AsyncMock(return_value=[])
    row = {"local_date": NOW.date(), "owner_id": 100, "trace_id": trace_id}
    settings = {
        "timezone": "UTC",
        "city_name": "Город",
        "city_country": "Страна",
        "city_region": None,
        "latitude": 59.9,
        "longitude": 30.3,
    }
    text = await DeliveryService(
        FakePool(fake), Bot(), BriefingBuilder(mcp), clock=Clock()
    )._briefing(fake, row, settings, NOW)
    assert "Занятия: нет" in text
    assert len(recorded) == 4
    assert all(args[0] == trace_id for args, _ in recorded)
    assert {kwargs["tool_name"] for _, kwargs in recorded} == {
        None,
        SCHEDULE,
        MATERIALS,
        WEATHER,
    }
    assert all(kwargs["error_code"] is None for _, kwargs in recorded)


async def test_register_briefing_uses_dedicated_trace_route(monkeypatch):
    captured = []

    class Recorder:
        def __init__(self, connection, *, clock):
            pass

        async def start(self, owner, request_id, route):
            captured.append(route)
            return uuid4()

        async def span(self, *args, **kwargs):
            pass

    monkeypatch.setattr("app.delivery.TraceRecorder", Recorder)
    connection = FakeConnection(NOW, {})
    await DeliveryService(FakePool(connection), Bot(), clock=Clock())._register_trace(
        connection, {"event_key": "briefing:1", "kind": "briefing", "owner_id": 100}, NOW
    )
    assert captured == ["briefing"]
