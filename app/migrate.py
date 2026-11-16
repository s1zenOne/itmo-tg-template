"""Применение версионированных SQL-миграций при запуске."""

from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).with_name("migrations")


async def migrate(pool: asyncpg.Pool) -> None:
    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migrations:
        raise RuntimeError("SQL-миграции отсутствуют в сборке приложения.")
    async with pool.acquire() as connection, connection.transaction():
        # Один процесс применяет миграции, даже если бот и MCP стартуют одновременно.
        await connection.execute("SELECT pg_advisory_xact_lock(1740673246)")
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY)"
        )
        applied = {
            row["version"] for row in await connection.fetch("SELECT * FROM schema_migrations")
        }
        for migration in migrations:
            if migration.stem in applied:
                continue
            await connection.execute(migration.read_text(encoding="utf-8"))
            await connection.execute("INSERT INTO schema_migrations VALUES ($1)", migration.stem)
