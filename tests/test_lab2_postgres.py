import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.db import create_pool
from app.migrate import MIGRATIONS_DIR
from app.reminders import ReminderError, SavedReminder
from app.store import AuditDecision, HistoryMessage, PostgresStore

pytestmark = pytest.mark.integration
NOW = datetime(2026, 11, 16, 12, tzinfo=UTC)
FUTURE = NOW + timedelta(days=1)


async def test_real_lab2_migrations_and_concurrent_reminder(lab2_database):
    async with await create_pool(lab2_database) as pool, await create_pool(lab2_database) as peer:
        # В ЛР 1 таблицы создавались при старте, журнала версий ещё не было.
        await pool.execute((MIGRATIONS_DIR / "0001_lab1.sql").read_text(encoding="utf-8"))
        first = PostgresStore(pool)
        second = PostgresStore(peer)
        state = await first.set_mode(101, "quiz")
        assert await first.save_result(101, state.generation, "Учебная тема", "Тестовый вопрос")

        await asyncio.gather(first.initialize(), second.initialize())
        assert await pool.fetchval("SELECT count(*) FROM schema_migrations") >= 2
        preserved, history = await first.get_context(101, 10)
        assert preserved == state
        assert history == [
            HistoryMessage("user", "Учебная тема"),
            HistoryMessage("assistant", "Тестовый вопрос"),
        ]
        assert await second.get_timezone(101) is None
        await first.set_timezone(101, "Europe/Moscow")
        assert await second.get_timezone(101) == "Europe/Moscow"
        assert await second.get_timezone(202) is None

        arguments = (101, 10001, "  Сдать работу  ", FUTURE, "Europe/Moscow")
        prepared, duplicate = await asyncio.gather(
            first.prepare_reminder(*arguments, now=NOW),
            second.prepare_reminder(*arguments, now=NOW),
        )
        assert prepared == duplicate
        assert (prepared.text, prepared.status, prepared.expires_at) == (
            "Сдать работу",
            "pending",
            NOW + timedelta(minutes=5),
        )
        assert await first.reminder_for_update(101, 10001, now=NOW) == prepared
        assert await first.reminder_for_update(202, 10001, now=NOW) is None

        async def confirmed_call(store):
            approved = await store.confirm_reminder(101, prepared.action_id, now=NOW)
            result = await store.execute_reminder(
                101, approved.action_id, approved.text, approved.remind_at, now=NOW
            )
            if result.created:
                raise ConnectionError("Ответ потерян после записи")
            return result

        # Первый ответ потерян после записи, конкурентный повтор вернул ту же запись.
        results = await asyncio.gather(
            confirmed_call(first), confirmed_call(second), return_exceptions=True
        )
        assert sum(isinstance(result, ConnectionError) for result in results) == 1
        (saved,) = [result for result in results if isinstance(result, SavedReminder)]
        assert (saved.created, saved.remind_at, saved.timezone) == (False, FUTURE, "Europe/Moscow")
        assert saved.id == await pool.fetchval("SELECT id FROM reminders")
        assert await pool.fetchval("SELECT count(*) FROM reminder_actions") == 1
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 1

        async with await create_pool(lab2_database) as reopened_pool:
            reopened = PostgresStore(reopened_pool)
            later = FUTURE + timedelta(minutes=1)
            approved = await reopened.confirm_reminder(101, prepared.action_id, now=later)
            assert approved.status == "executed"
            retry = await reopened.execute_reminder(
                101, approved.action_id, approved.text, approved.remind_at, now=later
            )
            assert (retry.id, retry.created, retry.remind_at, retry.timezone) == (
                saved.id,
                False,
                FUTURE,
                "Europe/Moscow",
            )


async def test_real_lab2_owner_consent_expiration_cancellation_and_audit(lab2_database):
    async with await create_pool(lab2_database) as pool:
        store = PostgresStore(pool)
        await store.initialize()
        prepared = await store.prepare_reminder(
            101, 10001, "Учебная задача", FUTURE, "UTC", now=NOW
        )
        action = prepared.action_id
        for operation in (
            store.get_prepared_reminder,
            store.confirm_reminder,
            store.cancel_reminder,
        ):
            with pytest.raises(ReminderError, match="unavailable"):
                await operation(202, action, now=NOW)
        with pytest.raises(ReminderError, match="unavailable"):
            await store.execute_reminder(202, action, prepared.text, FUTURE, now=NOW)
        with pytest.raises(ReminderError, match="unconfirmed"):
            await store.execute_reminder(101, action, prepared.text, FUTURE, now=NOW)

        approved = await store.confirm_reminder(101, action, now=NOW)
        assert approved.status == "confirmed"
        with pytest.raises(ReminderError, match="mismatched"):
            await store.execute_reminder(101, action, "Другой текст", FUTURE, now=NOW)
        with pytest.raises(ReminderError, match="mismatched"):
            await store.execute_reminder(
                101, action, prepared.text, FUTURE + timedelta(minutes=1), now=NOW
            )
        expiry = NOW + timedelta(minutes=5)
        assert (await store.get_prepared_reminder(101, action, now=expiry)).status == "expired"
        with pytest.raises(ReminderError, match="expired"):
            await store.execute_reminder(101, action, prepared.text, FUTURE, now=expiry)

        to_cancel = await store.prepare_reminder(101, 10002, "Другое время", FUTURE, "UTC", now=NOW)
        cancelled = await store.cancel_reminder(101, to_cancel.action_id, now=NOW)
        assert cancelled.status == "cancelled"
        assert await store.cancel_reminder(101, to_cancel.action_id, now=NOW) == cancelled
        with pytest.raises(ReminderError, match="cancelled"):
            await store.confirm_reminder(101, to_cancel.action_id, now=NOW)
        with pytest.raises(ReminderError, match="past"):
            await store.prepare_reminder(101, 10003, "Прошло", NOW, "UTC", now=NOW)
        assert await pool.fetchval("SELECT count(*) FROM reminder_actions") == 2
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 0

        assert await store.latest_audit(101) is None
        original = AuditDecision(NOW, "respond", None, "", "ok", "ok", 20, "Обычный ответ")
        latest = AuditDecision(expiry, "clarify", None, "", "ok", "ok", 32, "Нужно уточнение")
        await store.write_audit(101, latest)
        await store.write_audit(101, original)
        await store.write_audit(202, original)
        assert await store.latest_audit(101) == latest
        assert await store.latest_audit(202) == original
        assert await store.latest_audit(303) is None
