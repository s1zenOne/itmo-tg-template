"""Постоянные настройки, история, напоминания и аудит отдельных пользователей."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import asyncpg

from app.migrate import migrate
from app.reminders import (
    PreparedReminder,
    ReminderError,
    SavedReminder,
    get_user_zone,
    normalize_reminder_text,
    reminder_display_time,
)

CONFIRMATION_TTL = timedelta(minutes=5)
ACTIVE_STATUSES = {"pending", "confirmed"}


@dataclass(frozen=True)
class UserState:
    mode: str = "study"
    temperature: float = 0.3
    generation: int = 0


@dataclass(frozen=True)
class HistoryMessage:
    role: str
    content: str


@dataclass(frozen=True)
class AuditDecision:
    """Резюме из безопасных значений, сформированных кодом приложения."""

    requested_at: datetime
    action: str
    tool: str | None
    arguments: str
    validation: str
    execution: str
    duration_ms: int
    reason: str


class PostgresStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def initialize(self) -> None:
        await migrate(self.pool)

    @staticmethod
    async def _ensure_user(connection: asyncpg.Connection, chat_id: int) -> None:
        await connection.execute(
            "INSERT INTO chat_settings (chat_id) VALUES ($1) ON CONFLICT DO NOTHING",
            chat_id,
        )

    @staticmethod
    def _state(row: asyncpg.Record) -> UserState:
        return UserState(
            mode=row["mode"],
            temperature=float(row["temperature"]),
            generation=row["generation"],
        )

    async def get_context(self, chat_id: int, limit: int) -> tuple[UserState, list[HistoryMessage]]:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, chat_id)
            row = await connection.fetchrow(
                "SELECT mode, temperature, generation FROM chat_settings "
                "WHERE chat_id = $1 FOR UPDATE",
                chat_id,
            )
            recent = await connection.fetch(
                "SELECT role, content FROM chat_history WHERE chat_id = $1 "
                "ORDER BY id DESC LIMIT $2",
                chat_id,
                limit,
            )
        return self._state(row), [
            HistoryMessage(role=item["role"], content=item["content"]) for item in reversed(recent)
        ]

    async def set_mode(self, chat_id: int, mode: str) -> UserState:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, chat_id)
            row = await connection.fetchrow(
                "UPDATE chat_settings SET mode = $2, generation = generation + 1 "
                "WHERE chat_id = $1 RETURNING mode, temperature, generation",
                chat_id,
                mode,
            )
            await connection.execute("DELETE FROM chat_history WHERE chat_id = $1", chat_id)
        return self._state(row)

    async def reset_history(self, chat_id: int) -> UserState:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, chat_id)
            row = await connection.fetchrow(
                "UPDATE chat_settings SET generation = generation + 1 "
                "WHERE chat_id = $1 RETURNING mode, temperature, generation",
                chat_id,
            )
            await connection.execute("DELETE FROM chat_history WHERE chat_id = $1", chat_id)
        return self._state(row)

    async def set_temperature(self, chat_id: int, temperature: float) -> UserState:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, chat_id)
            row = await connection.fetchrow(
                "UPDATE chat_settings SET temperature = $2 "
                "WHERE chat_id = $1 RETURNING mode, temperature, generation",
                chat_id,
                Decimal(str(temperature)),
            )
        return self._state(row)

    async def save_result(
        self, chat_id: int, generation: int, user_text: str, assistant_text: str | None
    ) -> bool:
        """Не восстанавливать историю после конкурентного сброса или смены режима."""
        async with self.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT generation FROM chat_settings WHERE chat_id = $1 FOR UPDATE",
                chat_id,
            )
            if row is None or row["generation"] != generation:
                return False
            await connection.execute(
                "INSERT INTO chat_history (chat_id, role, content) VALUES ($1, 'user', $2)",
                chat_id,
                user_text,
            )
            if assistant_text is not None:
                await connection.execute(
                    "INSERT INTO chat_history (chat_id, role, content) "
                    "VALUES ($1, 'assistant', $2)",
                    chat_id,
                    assistant_text,
                )
        return True

    async def get_timezone(self, owner_id: int) -> str | None:
        return await self.pool.fetchval(
            "SELECT timezone FROM chat_settings WHERE chat_id = $1", owner_id
        )

    async def get_confirmed_city(self, owner_id: int):
        """Return only the city confirmed by this user, never a pending proposal."""
        return await self.pool.fetchrow(
            "SELECT city_name, city_country, city_region, latitude, longitude "
            "FROM notification_settings WHERE owner_id = $1 AND city_name IS NOT NULL "
            "AND city_country IS NOT NULL AND latitude IS NOT NULL AND longitude IS NOT NULL",
            owner_id,
        )

    async def set_timezone(self, owner_id: int, zone: str) -> None:
        get_user_zone(zone)
        await self.pool.execute(
            "INSERT INTO chat_settings (chat_id, timezone) VALUES ($1, $2) "
            "ON CONFLICT (chat_id) DO UPDATE SET timezone = EXCLUDED.timezone",
            owner_id,
            zone,
        )

    async def write_audit(self, owner_id: int, decision: AuditDecision) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, owner_id)
            await connection.execute(
                """
                INSERT INTO agent_audit
                    (owner_id, requested_at, action, tool, arguments, validation, execution,
                     duration_ms, reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                owner_id,
                decision.requested_at,
                decision.action,
                decision.tool,
                decision.arguments,
                decision.validation,
                decision.execution,
                decision.duration_ms,
                decision.reason,
            )

    async def latest_audit(self, owner_id: int) -> AuditDecision | None:
        row = await self.pool.fetchrow(
            """
            SELECT requested_at, action, tool, arguments, validation, execution, duration_ms, reason
            FROM agent_audit WHERE owner_id = $1 ORDER BY requested_at DESC, id DESC LIMIT 1
            """,
            owner_id,
        )
        return AuditDecision(**dict(row)) if row else None

    @staticmethod
    def _prepared(row: asyncpg.Record, now: datetime) -> PreparedReminder:
        status = row["status"]
        if status in ACTIVE_STATUSES and row["expires_at"] <= now:
            status = "expired"
        return PreparedReminder(
            row["action_id"],
            row["text"],
            row["remind_at"],
            row["timezone"],
            row["expires_at"],
            status,
        )

    @staticmethod
    async def _now(connection: asyncpg.Connection, now: datetime | None) -> datetime:
        return now if now is not None else await connection.fetchval("SELECT clock_timestamp()")

    @staticmethod
    async def _locked_action(
        connection: asyncpg.Connection, owner_id: int, action_id: UUID
    ) -> asyncpg.Record:
        row = await connection.fetchrow(
            "SELECT * FROM reminder_actions WHERE owner_id = $1 AND action_id = $2 FOR UPDATE",
            owner_id,
            action_id,
        )
        if row is None:
            raise ReminderError("unavailable")
        return row

    async def reminder_for_update(
        self, owner_id: int, update_id: int, *, now: datetime | None = None
    ) -> PreparedReminder | None:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM reminder_actions WHERE owner_id = $1 AND update_id = $2",
                owner_id,
                update_id,
            )
            return self._prepared(row, await self._now(connection, now)) if row else None

    async def get_prepared_reminder(
        self, owner_id: int, action_id: UUID, *, now: datetime | None = None
    ) -> PreparedReminder:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM reminder_actions WHERE owner_id = $1 AND action_id = $2",
                owner_id,
                action_id,
            )
            if row is None:
                raise ReminderError("unavailable")
            return self._prepared(row, await self._now(connection, now))

    async def prepare_reminder(
        self,
        owner_id: int,
        update_id: int,
        text: str,
        remind_at: datetime,
        timezone: str,
        *,
        now: datetime | None = None,
    ) -> PreparedReminder:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._ensure_user(connection, owner_id)
            current = await self._now(connection, now)
            lookup = "SELECT * FROM reminder_actions WHERE owner_id = $1 AND update_id = $2"
            existing = await connection.fetchrow(lookup, owner_id, update_id)
            if existing is not None:
                return self._prepared(existing, current)
            text = normalize_reminder_text(text)
            if remind_at.utcoffset() is None:
                raise ReminderError("invalid_time")
            remind_at = remind_at.astimezone(UTC)
            if remind_at <= current:
                raise ReminderError("past")
            reminder_display_time(remind_at, timezone)
            row = await connection.fetchrow(
                """
                INSERT INTO reminder_actions
                    (action_id, owner_id, update_id, text, remind_at, timezone, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (owner_id, update_id) DO NOTHING RETURNING *
                """,
                uuid4(),
                owner_id,
                update_id,
                text,
                remind_at,
                timezone,
                current + CONFIRMATION_TTL,
            )
            if row is None:
                row = await connection.fetchrow(lookup, owner_id, update_id)
            return self._prepared(row, await self._now(connection, now))

    @staticmethod
    def _require_active(prepared: PreparedReminder, now: datetime) -> None:
        if prepared.status not in ACTIVE_STATUSES:
            raise ReminderError(prepared.status)
        if prepared.remind_at <= now:
            raise ReminderError("past")

    async def confirm_reminder(
        self, owner_id: int, action_id: UUID, *, now: datetime | None = None
    ) -> PreparedReminder:
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._locked_action(connection, owner_id, action_id)
            current = await self._now(connection, now)
            prepared = self._prepared(row, current)
            if prepared.status == "executed":
                return prepared
            self._require_active(prepared, current)
            if prepared.status == "pending":
                row = await connection.fetchrow(
                    "UPDATE reminder_actions SET status = 'confirmed' "
                    "WHERE action_id = $1 RETURNING *",
                    action_id,
                )
            return self._prepared(row, current)

    async def cancel_reminder(
        self, owner_id: int, action_id: UUID, *, now: datetime | None = None
    ) -> PreparedReminder:
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._locked_action(connection, owner_id, action_id)
            current = await self._now(connection, now)
            prepared = self._prepared(row, current)
            if prepared.status == "cancelled":
                return prepared
            self._require_active(prepared, current)
            row = await connection.fetchrow(
                "UPDATE reminder_actions SET status = 'cancelled' WHERE action_id = $1 RETURNING *",
                action_id,
            )
            return self._prepared(row, current)

    async def execute_reminder(
        self,
        owner_id: int,
        action_id: UUID,
        text: str,
        remind_at: datetime,
        *,
        now: datetime | None = None,
    ) -> SavedReminder:
        if remind_at.utcoffset() is None:
            raise ReminderError("invalid_time")
        async with self.pool.acquire() as connection, connection.transaction():
            row = await self._locked_action(connection, owner_id, action_id)
            current = await self._now(connection, now)
            prepared = self._prepared(row, current)
            if text != prepared.text or remind_at.astimezone(UTC) != prepared.remind_at:
                raise ReminderError("mismatched")
            if prepared.status == "executed":
                reminder_id = await connection.fetchval(
                    "SELECT id FROM reminders WHERE action_id = $1", action_id
                )
                return SavedReminder(reminder_id, prepared.remind_at, prepared.timezone, False)
            self._require_active(prepared, current)
            if prepared.status != "confirmed":
                raise ReminderError("unconfirmed")
            reminder_id = await connection.fetchval(
                "INSERT INTO reminders (action_id, created_at) VALUES ($1, $2) RETURNING id",
                action_id,
                current,
            )
            await connection.execute(
                "UPDATE reminder_actions SET status = 'executed' WHERE action_id = $1", action_id
            )
            return SavedReminder(reminder_id, prepared.remind_at, prepared.timezone, True)
