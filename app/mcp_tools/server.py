import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from uuid import UUID

import aiohttp
import asyncpg
from mcp import MCPError, stdio_server
from mcp.server import Server
from mcp.types import (
    CallToolResult,
    ErrorData,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
)
from pydantic import ValidationError

from app.config import Settings
from app.db import create_pool
from app.mcp_tools.auth import GRANT_KEY_ENV, verify_grant
from app.mcp_tools.contracts import (
    REMINDER,
    SCHEDULE,
    WEATHER,
    WEEK_URI,
    ReminderResult,
    ToolError,
    tool_definitions,
    validate_arguments,
)
from app.mcp_tools.schedule import DEFAULT_PATH, DEFAULT_TIMEZONE, FileSchedule
from app.mcp_tools.weather import OpenMeteoWeather
from app.reminders import ReminderError, SavedReminder
from app.store import PostgresStore

logger = logging.getLogger(__name__)
ReminderWriter = Callable[[int, UUID, str, datetime], Awaitable[SavedReminder]]


def _tool_result(payload: dict) -> CallToolResult:
    return CallToolResult(
        is_error=payload["status"] == "error",
        structured_content=payload,
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
    )


def create_server(
    weather: OpenMeteoWeather,
    schedule: FileSchedule,
    *,
    grant_key: bytes | None = None,
    write_reminder: ReminderWriter | None = None,
) -> Server:
    tools = tool_definitions()
    resource = Resource(
        uri=WEEK_URI,
        name="current-week",
        description="Учебное расписание на текущую неделю, с понедельника по воскресенье.",
        mime_type="application/json",
    )

    async def list_tools(_context, _params):
        return ListToolsResult(tools=list(tools))

    async def list_resources(_context, _params):
        return ListResourcesResult(resources=[resource])

    async def read_resource(_context, params):
        if str(params.uri) != WEEK_URI:
            raise MCPError(ErrorData(code=-32002, message="resource_not_found"))
        try:
            result = await schedule.current_week()
        except ToolError as exc:
            raise MCPError(ErrorData(code=-32603, message=exc.code)) from None
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=resource.uri, mime_type=resource.mime_type, text=result.model_dump_json()
                )
            ]
        )

    async def call_tool(_context, params):
        try:
            name = params.name
            arguments = validate_arguments(name, params.arguments)
            if name == WEATHER:
                result = await weather.get_weather(arguments["city"])
            elif name == SCHEDULE:
                result = await schedule.get_schedule(arguments["date"])
            elif name == REMINDER:
                owner_id, action_id = verify_grant(grant_key, name, arguments, params.meta)
                if write_reminder is None:
                    raise ToolError("reminder_unavailable")
                saved = await write_reminder(
                    owner_id,
                    action_id,
                    arguments["text"],
                    datetime.fromisoformat(arguments["remind_at"]),
                )
                result = ReminderResult(
                    status="ok",
                    id=saved.id,
                    remind_at=saved.remind_at.isoformat(),
                    timezone=saved.timezone,
                    created=saved.created,
                )
            return _tool_result(result.model_dump(mode="json"))
        except (ToolError, ReminderError) as error:
            return _tool_result({"status": "error", "code": error.code})
        except (
            ValidationError,
            asyncpg.PostgresError,
            asyncpg.InterfaceError,
            OSError,
            TimeoutError,
        ) as error:
            logger.warning("MCP tool failed (%s)", type(error).__name__)
            return _tool_result({"status": "error", "code": "execution_failed"})

    return Server(
        "itmo-tg-tools",
        version="1.0.0",
        on_list_tools=list_tools,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
        on_call_tool=call_tool,
    )


def _grant_key() -> bytes | None:
    encoded = os.environ.get(GRANT_KEY_ENV, "")
    if len(encoded) != 64:
        return None
    try:
        return bytes.fromhex(encoded)
    except ValueError:
        return None


async def run_stdio() -> None:
    schedule = FileSchedule(
        Path(os.environ.get("SCHEDULE_PATH", DEFAULT_PATH)),
        os.environ.get("SCHEDULE_TIMEZONE", DEFAULT_TIMEZONE),
    )
    grant_key = _grant_key()
    pg_password = os.environ.get("POSTGRES_PASSWORD")
    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(aiohttp.ClientSession())
        store = None
        connect_lock = asyncio.Lock()

        async def write_reminder(owner_id: int, action_id: UUID, text: str, remind_at: datetime):
            nonlocal store
            async with connect_lock:
                if store is None:
                    pg_settings = Settings(
                        bot_token="",
                        postgres_password=pg_password or "",
                        postgres_host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
                        postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
                        postgres_db=os.environ.get("POSTGRES_DB", "bot"),
                        postgres_user=os.environ.get("POSTGRES_USER", "bot"),
                    )
                    pool = await create_pool(pg_settings)
                    stack.push_async_callback(_close_pool, pool)
                    store = PostgresStore(pool)
            return await store.execute_reminder(owner_id, action_id, text, remind_at)

        server = create_server(
            OpenMeteoWeather(session),
            schedule,
            grant_key=grant_key,
            write_reminder=write_reminder if grant_key and pg_password else None,
        )
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


async def _close_pool(pool: asyncpg.Pool) -> None:
    try:
        async with asyncio.timeout(5):
            await pool.close()
    except TimeoutError:
        pool.terminate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("mcp").disabled = True
    logging.getLogger("mcp").setLevel(logging.CRITICAL + 1)
    try:
        asyncio.run(run_stdio())
    except Exception as error:
        logger.error("MCP server stopped (%s)", type(error).__name__)
        raise SystemExit(1) from None
