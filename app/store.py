"""Постоянные настройки и история отдельных чатов."""

from dataclasses import dataclass
from decimal import Decimal

import asyncpg


@dataclass(frozen=True)
class UserState:
    mode: str = "study"
    temperature: float = 0.3
    generation: int = 0


@dataclass(frozen=True)
class HistoryMessage:
    role: str
    content: str


class PostgresStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def initialize(self) -> None:
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_settings (
                    chat_id BIGINT PRIMARY KEY,
                    mode TEXT NOT NULL DEFAULT 'study'
                        CHECK (mode IN ('study', 'translate', 'quiz')),
                    temperature NUMERIC(2,1) NOT NULL DEFAULT 0.3
                        CHECK (temperature IN (0.0, 0.3, 0.7, 1.0)),
                    generation BIGINT NOT NULL DEFAULT 0
                )
                """
            )
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id BIGSERIAL PRIMARY KEY,
                    chat_id BIGINT NOT NULL REFERENCES chat_settings(chat_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL
                )
                """
            )
            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS chat_history_chat_id_id_idx
                ON chat_history (chat_id, id DESC)
                """
            )

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
