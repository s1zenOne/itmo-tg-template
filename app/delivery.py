"""Persistent, bounded delivery of confirmed reminders and daily briefings.

Rows stay locked across sendMessage. A process crash rolls back the row and makes it
eligible again; a crash after Telegram accepts the message can therefore duplicate it.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramEntityTooLarge,
    TelegramForbiddenError,
    TelegramMigrateToChat,
    TelegramNotFound,
    TelegramRetryAfter,
    TelegramUnauthorizedError,
)

from app.briefing import BriefingBuilder as ProductionBriefingBuilder
from app.tracing import TraceRecorder

logger = logging.getLogger(__name__)
BriefingBuilder = Callable[[int, date, datetime, Any], Awaitable[str]]
Clock = Callable[[], datetime]


class DeliveryService:
    def __init__(
        self,
        pool: Any,
        bot: Any,
        briefing_builder: BriefingBuilder | None = None,
        clock: Clock | None = None,
        interval: float = 30,
    ) -> None:
        self.pool = pool
        self.bot = bot
        self.briefing_builder = briefing_builder
        self.clock = clock or (lambda: datetime.now(UTC))
        self.interval = interval

    async def _register_trace(self, connection: Any, row: Any, started: datetime) -> None:
        recorder = TraceRecorder(connection, clock=self.clock)
        route = "schedule" if row["kind"] == "reminder" else "briefing"
        trace_id = await recorder.start(row["owner_id"], uuid4(), route)
        await connection.execute(
            "UPDATE notification_outbox SET trace_id=$2 WHERE event_key=$1",
            row["event_key"],
            trace_id,
        )
        component = "schedule" if row["kind"] == "reminder" else "briefing"
        await recorder.span(trace_id, component, "prepare", started, self.clock(), "ok")

    async def prepare(self, now: datetime) -> None:
        """Persist each event and its trace atomically, including concurrent preparation."""
        async with self.pool.acquire() as connection, connection.transaction():
            started = self.clock()
            rows = await connection.fetch(
                """
                INSERT INTO notification_outbox
                    (event_key, kind, owner_id, reminder_id, due_at, next_attempt_at)
                SELECT 'reminder:' || r.id, 'reminder', a.owner_id, r.id,
                       a.remind_at, a.remind_at
                FROM reminders r JOIN reminder_actions a ON a.action_id = r.action_id
                WHERE a.status = 'executed' AND a.remind_at <= $1
                ON CONFLICT DO NOTHING RETURNING event_key, owner_id, kind
                """,
                now,
            )
            for row in rows:
                await self._register_trace(connection, row, started)
            users = await connection.fetch(
                """
                SELECT n.owner_id, c.timezone FROM notification_settings n
                JOIN chat_settings c ON c.chat_id = n.owner_id
                WHERE n.briefing_enabled AND n.latitude IS NOT NULL AND n.longitude IS NOT NULL
                  AND n.city_name IS NOT NULL AND c.timezone IS NOT NULL
                """
            )
            for user in users:
                try:
                    zone = ZoneInfo(user["timezone"])
                except (ZoneInfoNotFoundError, ValueError):
                    continue
                local_day = now.astimezone(zone).date()
                due = datetime.combine(local_day, time(8), zone).astimezone(UTC)
                if now < due:
                    continue
                started = self.clock()
                row = await connection.fetchrow(
                    """
                    INSERT INTO notification_outbox
                        (event_key, kind, owner_id, local_date, due_at, next_attempt_at, timezone)
                    VALUES ($1, 'briefing', $2, $3, $4, $4, $5) ON CONFLICT DO NOTHING
                    RETURNING event_key, owner_id, kind
                    """,
                    f"briefing:{user['owner_id']}:{local_day.isoformat()}",
                    user["owner_id"],
                    local_day,
                    due,
                    user["timezone"],
                )
                if row is not None:
                    await self._register_trace(connection, row, started)

    async def _trace_event(
        self,
        connection: Any,
        row: Any,
        component: str,
        operation: str,
        started: datetime,
        status: str,
        error: str | None = None,
        *,
        terminal: bool = False,
    ) -> None:
        # Legacy outbox rows can predate trace integration. Fake adapters may omit this field.
        if row.get("trace_id") is None:
            return
        recorder = TraceRecorder(connection, clock=self.clock)
        finished = self.clock()
        await recorder.span(
            row["trace_id"], component, operation, started, finished, status, error_code=error
        )
        if terminal:
            await recorder.close(row["trace_id"], status, finished)

    async def _briefing(self, connection: Any, row: Any, settings: Any, now: datetime) -> str:
        zone = ZoneInfo(settings["timezone"])
        local_day = row["local_date"]
        start = datetime.combine(local_day, time.min, zone).astimezone(UTC)
        end = datetime.combine(local_day + timedelta(days=1), time.min, zone).astimezone(UTC)
        reminders = await connection.fetch(
            """
            SELECT a.text, a.remind_at FROM reminders r
            JOIN reminder_actions a ON a.action_id = r.action_id
            WHERE a.owner_id = $1 AND a.status = 'executed' AND a.remind_at >= $2
              AND a.remind_at < $3 ORDER BY a.remind_at
            """,
            row["owner_id"],
            start,
            end,
        )
        items = [f"• {r['remind_at'].astimezone(zone):%H:%M} — {r['text']}" for r in reminders]
        reminder_section = "\n".join(items) if items else "На сегодня напоминаний нет."
        if self.briefing_builder is None:
            details = (
                "Занятия: данные недоступны.\nПогода: данные недоступны.\n"
                "Учебный фокус: материалы недоступны."
            )
        else:
            events = []
            try:
                async with asyncio.timeout(10):
                    if isinstance(self.briefing_builder, ProductionBriefingBuilder):
                        details = await self.briefing_builder(
                            row["owner_id"],
                            local_day,
                            now,
                            settings,
                            trace_events=events,
                            clock=self.clock,
                            trace_id=row.get("trace_id"),
                        )
                    else:
                        details = await self.briefing_builder(
                            row["owner_id"], local_day, now, settings
                        )
                if not isinstance(details, str) or not details.strip():
                    raise ValueError("empty briefing")
            except Exception:
                details = (
                    "Занятия: данные недоступны.\nПогода: данные недоступны.\n"
                    "Учебный фокус: материалы недоступны."
                )
            if row.get("trace_id") is not None:
                recorder = TraceRecorder(connection, clock=self.clock)
                # Calls to specialists ran concurrently, but one SQL connection cannot.
                for event in sorted(events, key=lambda item: item.started_at):
                    await recorder.span(
                        row["trace_id"],
                        event.component,
                        event.operation,
                        event.started_at,
                        event.finished_at,
                        event.status,
                        tool_name=event.tool_name,
                        error_code=event.error_code,
                    )
        prepared = now.astimezone(zone).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"Сводка на {local_day:%Y-%m-%d}\nПодготовлена: {prepared} ({zone.key})"
            f"\n\n{details}\n\nНапоминания:\n{reminder_section}"
        )

    async def _deliver_one(self, now: datetime) -> bool:
        async with self.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """
                SELECT * FROM notification_outbox
                WHERE status = 'pending' AND next_attempt_at <= $1
                ORDER BY due_at, event_key FOR UPDATE SKIP LOCKED LIMIT 1
                """,
                now,
            )
            if row is None:
                return False
            grace = timedelta(minutes=15 if row["kind"] == "reminder" else 60)
            settings = await connection.fetchrow(
                """
                SELECT n.*, c.timezone FROM notification_settings n
                JOIN chat_settings c ON c.chat_id=n.owner_id
                WHERE n.owner_id=$1 FOR UPDATE OF n, c
                """,
                row["owner_id"],
            )
            if row["kind"] == "briefing" and settings is not None:
                if not settings["briefing_enabled"]:
                    await self._trace_event(
                        connection,
                        row,
                        "briefing",
                        "prepare",
                        now,
                        "error",
                        "unauthorized",
                        terminal=True,
                    )
                    await connection.execute(
                        "UPDATE notification_outbox SET status='cancelled', "
                        "error_code='consent_or_destination' WHERE event_key=$1",
                        row["event_key"],
                    )
                    return True
                try:
                    zone = ZoneInfo(settings["timezone"])
                except (ZoneInfoNotFoundError, ValueError, TypeError):
                    zone = None
                if zone is None or now.astimezone(zone).date() != row["local_date"]:
                    await self._trace_event(
                        connection,
                        row,
                        "briefing",
                        "prepare",
                        now,
                        "error",
                        "unavailable",
                        terminal=True,
                    )
                    await connection.execute(
                        "UPDATE notification_outbox SET status='cancelled', "
                        "error_code='timezone_changed' WHERE event_key=$1",
                        row["event_key"],
                    )
                    return True
                already_sent = await connection.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM notification_outbox "
                    "WHERE owner_id=$1 AND kind='briefing' AND sent_at IS NOT NULL "
                    "AND event_key<>$2 AND (sent_at AT TIME ZONE $3)::date=$4)",
                    row["owner_id"],
                    row["event_key"],
                    settings["timezone"],
                    row["local_date"],
                )
                if already_sent:
                    await self._trace_event(
                        connection,
                        row,
                        "briefing",
                        "prepare",
                        now,
                        "error",
                        "unavailable",
                        terminal=True,
                    )
                    await connection.execute(
                        "UPDATE notification_outbox SET status='cancelled', "
                        "error_code='already_sent' WHERE event_key=$1",
                        row["event_key"],
                    )
                    return True
                if settings["timezone"] != row["timezone"]:
                    if row["next_offset"]:
                        await self._trace_event(
                            connection,
                            row,
                            "briefing",
                            "prepare",
                            now,
                            "error",
                            "unavailable",
                            terminal=True,
                        )
                        await connection.execute(
                            "UPDATE notification_outbox SET status='cancelled', "
                            "error_code='timezone_changed' WHERE event_key=$1",
                            row["event_key"],
                        )
                    else:
                        due = datetime.combine(row["local_date"], time(8), zone).astimezone(UTC)
                        await connection.execute(
                            "UPDATE notification_outbox SET timezone=$2, due_at=$3, "
                            "next_attempt_at=$3, message_text=NULL WHERE event_key=$1",
                            row["event_key"],
                            settings["timezone"],
                            due,
                        )
                    return True
            if now - row["due_at"] > grace:
                await self._trace_event(
                    connection,
                    row,
                    "briefing" if row["kind"] == "briefing" else "schedule",
                    "prepare",
                    now,
                    "missed",
                    terminal=True,
                )
                await connection.execute(
                    "UPDATE notification_outbox SET status='missed', error_code='expired' "
                    "WHERE event_key=$1",
                    row["event_key"],
                )
                return True
            if settings is None or settings["destination_chat_id"] is None:
                # Existing users may not have interacted since the destination was introduced.
                await self._trace_event(
                    connection,
                    row,
                    "briefing" if row["kind"] == "briefing" else "schedule",
                    "prepare",
                    now,
                    "error",
                    "unavailable",
                )
                await connection.execute(
                    "UPDATE notification_outbox SET next_attempt_at=$2, "
                    "error_code='destination_missing' WHERE event_key=$1",
                    row["event_key"],
                    now + timedelta(seconds=30),
                )
                return True
            if row["kind"] == "briefing" and not (
                settings["city_name"]
                and settings["latitude"] is not None
                and settings["longitude"] is not None
            ):
                await self._trace_event(
                    connection,
                    row,
                    "briefing",
                    "prepare",
                    now,
                    "error",
                    "unauthorized",
                    terminal=True,
                )
                await connection.execute(
                    "UPDATE notification_outbox SET status='cancelled', "
                    "error_code='consent_or_destination' WHERE event_key=$1",
                    row["event_key"],
                )
                return True
            message = row["message_text"]
            if message is None:
                if row["kind"] == "reminder":
                    text = await connection.fetchval(
                        "SELECT a.text FROM reminders r "
                        "JOIN reminder_actions a ON a.action_id=r.action_id "
                        "WHERE r.id=$1 AND a.owner_id=$2 AND a.status='executed'",
                        row["reminder_id"],
                        row["owner_id"],
                    )
                    if text is None:
                        await self._trace_event(
                            connection,
                            row,
                            "schedule",
                            "prepare",
                            now,
                            "error",
                            "not_found",
                            terminal=True,
                        )
                        await connection.execute(
                            "UPDATE notification_outbox SET status='cancelled', "
                            "error_code='source_missing' WHERE event_key=$1",
                            row["event_key"],
                        )
                        return True
                    message = f"Напоминание: {text}"
                    if now > row["due_at"]:
                        message += "\nОтправлено с задержкой."
                else:
                    message = await self._briefing(connection, row, settings, now)
                await connection.execute(
                    "UPDATE notification_outbox SET message_text=$2 WHERE event_key=$1",
                    row["event_key"],
                    message,
                )
            # Recheck the deadline after potentially slow data collection.
            send_time = self.clock()
            if send_time - row["due_at"] > grace:
                await self._trace_event(
                    connection,
                    row,
                    "briefing" if row["kind"] == "briefing" else "schedule",
                    "prepare",
                    now,
                    "missed",
                    terminal=True,
                )
                await connection.execute(
                    "UPDATE notification_outbox SET status='missed', error_code='expired' "
                    "WHERE event_key=$1",
                    row["event_key"],
                )
                return True
            attempts = row["attempts"] + 1
            offset = row["next_offset"]
            if row["kind"] == "reminder" and offset == 0 and send_time > row["due_at"]:
                if not message.endswith("\nОтправлено с задержкой."):
                    message += "\nОтправлено с задержкой."
                    await connection.execute(
                        "UPDATE notification_outbox SET message_text=$2 WHERE event_key=$1",
                        row["event_key"],
                        message,
                    )
            part = message[offset : offset + 4096]
            attempt_started = self.clock()
            try:
                async with asyncio.timeout(30):
                    await self.bot.send_message(settings["destination_chat_id"], part)
            except (
                TelegramBadRequest,
                TelegramForbiddenError,
                TelegramUnauthorizedError,
                TelegramNotFound,
                TelegramEntityTooLarge,
                TelegramMigrateToChat,
            ):
                await self._trace_event(
                    connection,
                    row,
                    "telegram",
                    "send",
                    attempt_started,
                    "error",
                    "permanent",
                    terminal=True,
                )
                await connection.execute(
                    "UPDATE notification_outbox SET status='failed', attempts=$2, "
                    "error_code='permanent' WHERE event_key=$1",
                    row["event_key"],
                    attempts,
                )
            except Exception as error:
                status = "failed" if attempts >= 3 else "pending"
                await self._trace_event(
                    connection,
                    row,
                    "telegram",
                    "send",
                    attempt_started,
                    "error",
                    "temporary",
                    terminal=status == "failed",
                )
                delay = 30 * attempts
                if isinstance(error, TelegramRetryAfter):
                    delay = max(delay, error.retry_after)
                await connection.execute(
                    "UPDATE notification_outbox SET status=$2, attempts=$3, "
                    "error_code='temporary', next_attempt_at=$4 WHERE event_key=$1",
                    row["event_key"],
                    status,
                    attempts,
                    send_time + timedelta(seconds=delay),
                )
            else:
                next_offset = offset + len(part)
                await self._trace_event(
                    connection,
                    row,
                    "telegram",
                    "send",
                    attempt_started,
                    "ok",
                    terminal=next_offset >= len(message),
                )
                if next_offset < len(message):
                    # Commit progress before sending another part; a retry resumes at this offset.
                    await connection.execute(
                        "UPDATE notification_outbox SET next_offset=$2, "
                        "sent_at=COALESCE(sent_at,$3), error_code=NULL, "
                        "next_attempt_at=$3 WHERE event_key=$1",
                        row["event_key"],
                        next_offset,
                        send_time,
                    )
                else:
                    await connection.execute(
                        "UPDATE notification_outbox SET status='sent', attempts=$2, "
                        "sent_at=COALESCE(sent_at,$3), "
                        "next_offset=$4, error_code=NULL WHERE event_key=$1",
                        row["event_key"],
                        attempts,
                        send_time,
                        next_offset,
                    )
            return True

    async def tick(self) -> None:
        now = self.clock()
        if now.utcoffset() is None:
            raise ValueError("clock must return an aware datetime")
        await self.prepare(now)
        while await self._deliver_one(self.clock()):
            pass

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("delivery_tick_failed")
            await asyncio.sleep(self.interval)
