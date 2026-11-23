import asyncio
import json
import logging
import secrets
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from uuid import UUID

from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from mcp import Client, MCPError, StdioServerParameters
from mcp.types import Resource, Tool
from pydantic import ValidationError

from app.config import Settings
from app.mcp_tools.auth import GRANT_KEY_ENV, make_grant
from app.mcp_tools.contracts import (
    CATALOG_URI,
    REMINDER,
    SCHEDULE,
    WEEK_URI,
    MaterialsCatalog,
    ToolError,
    WeekSchedule,
    tool_definitions,
    validate_arguments,
    validate_result,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_dispatch_observer: ContextVar[Callable[[str], None] | None] = ContextVar(
    "mcp_dispatch_observer", default=None
)


@contextmanager
def observe_tool_dispatch(observer: Callable[[str], None]) -> Iterator[None]:
    """Observe calls handed to the MCP SDK in the current asynchronous task."""
    token = _dispatch_observer.set(observer)
    try:
        yield
    finally:
        _dispatch_observer.reset(token)


class MCPUnavailable(Exception):
    pass


class MCPClient:
    def __init__(self, settings: Settings):
        self._key = secrets.token_bytes(32)
        self._parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_tools.server"],
            cwd=PROJECT_ROOT,
            env={
                GRANT_KEY_ENV: self._key.hex(),
                "POSTGRES_HOST": settings.postgres_host,
                "POSTGRES_PORT": str(settings.postgres_port),
                "POSTGRES_DB": settings.postgres_db,
                "POSTGRES_USER": settings.postgres_user,
                "POSTGRES_PASSWORD": settings.postgres_password,
                "SCHEDULE_PATH": str(settings.schedule_path.resolve()),
                "SCHEDULE_TIMEZONE": settings.schedule_timezone,
            },
        )
        self._client: Client | None = None
        self._runner: asyncio.Task | None = None
        self._attempt: asyncio.Future[bool] | None = None
        self._open_requested = asyncio.Event()
        self._session_lost = asyncio.Event()
        self.tools: tuple[Tool, ...] = ()
        self.resources: tuple[Resource, ...] = ()

    @property
    def available(self) -> bool:
        return self._client is not None

    async def __aenter__(self) -> "MCPClient":
        # The SDK can log validation details containing untrusted data.
        logging.getLogger("mcp").setLevel(logging.CRITICAL + 1)
        self._runner = asyncio.create_task(self._serve_connection())
        try:
            await self.connect()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
        if self._attempt is not None and not self._attempt.done():
            self._attempt.set_result(False)

    async def connect(self) -> bool:
        if self._runner is None or self._runner.done():
            return False
        if self.available:
            return True
        if self._attempt is None or self._attempt.done():
            self._attempt = asyncio.get_running_loop().create_future()
            self._open_requested.set()
        return await asyncio.shield(self._attempt)

    async def _discover(self, client: Client) -> None:
        tool_list = await client.list_tools()
        resource_list = await client.list_resources()
        allowed = {tool.name: tool for tool in tool_definitions()}
        # Unknown capabilities are never exposed. A changed known schema fails closed.
        if tool_list.next_cursor or resource_list.next_cursor:
            raise ToolError("unexpected_capabilities")
        announced = {tool.name: tool for tool in tool_list.tools}
        if len(announced) != len(tool_list.tools):
            raise ToolError("unexpected_capabilities")
        self.tools = tuple(
            tool for name, tool in announced.items() if name in allowed and tool == allowed[name]
        )
        uris = {WEEK_URI, CATALOG_URI}
        discovered = [resource for resource in resource_list.resources if str(resource.uri) in uris]
        if len({str(resource.uri) for resource in discovered}) != len(discovered):
            raise ToolError("unexpected_capabilities")
        self.resources = tuple(discovered)
        logger.info("MCP discovery: tools=%d resources=%d", len(self.tools), len(self.resources))

    def _clear_discovery(self) -> None:
        self._client = None
        self.tools = ()
        self.resources = ()

    async def _serve_connection(self) -> None:
        # stdio owns anyio cancellation scopes. The same task opens and closes every session.
        while True:
            await self._open_requested.wait()
            self._open_requested.clear()
            self._session_lost.clear()
            attempt = self._attempt
            try:
                async with Client(self._parameters, read_timeout_seconds=6) as client:
                    await self._discover(client)
                    self._client = client
                    attempt.set_result(True)
                    await self._session_lost.wait()
            except Exception as error:
                reason = error.code if isinstance(error, ToolError) else type(error).__name__
                phase = "shutdown" if attempt.done() else "discovery"
                logger.warning("MCP %s failed (%s)", phase, reason)
            finally:
                self._clear_discovery()
                if not attempt.done():
                    attempt.set_result(False)

    async def call(
        self,
        name: str,
        arguments: dict,
        *,
        owner_id: int | None = None,
        action_id: UUID | None = None,
    ) -> dict:
        if not await self.connect() or self._client is None:
            raise MCPUnavailable()
        client = self._client
        if name not in {tool.name for tool in self.tools}:
            raise ToolError("unknown_tool")
        arguments = validate_arguments(name, arguments)
        meta = None
        if name == REMINDER:
            if owner_id is None or action_id is None:
                raise ToolError("unauthorized")
            meta = make_grant(self._key, name, arguments, owner_id, action_id)

        try:
            # Use the SDK session directly to avoid automatic resubmission or input rounds.
            observer = _dispatch_observer.get()
            if observer is not None:
                observer(name)
            result = await client.session.call_tool(
                name, arguments, meta=meta, read_timeout_seconds=20
            )
        except (ValidationError, RuntimeError):
            raise ToolError("invalid_result") from None
        except (
            MCPError,
            TimeoutError,
            OSError,
            EOFError,
            BrokenResourceError,
            ClosedResourceError,
            EndOfStream,
        ) as error:
            if self._client is client:
                self._clear_discovery()
                self._session_lost.set()
            logger.warning("MCP call failed (%s)", type(error).__name__)
            raise MCPUnavailable() from None
        if not isinstance(result.structured_content, dict) or result.is_error != (
            result.structured_content.get("status") == "error"
        ):
            raise ToolError("invalid_result")
        payload = validate_result(name, result.structured_content)
        if name == SCHEDULE and payload["date"] != arguments["date"]:
            raise ToolError("invalid_result")
        if name == REMINDER:
            actual_instant = datetime.fromisoformat(payload["remind_at"])
            if actual_instant != datetime.fromisoformat(arguments["remind_at"]):
                raise ToolError("invalid_result")
        return payload

    async def read_resource(self, uri: str) -> dict:
        if uri not in {WEEK_URI, CATALOG_URI}:
            raise ToolError("unknown_resource")
        if not await self.connect() or self._client is None:
            raise MCPUnavailable()
        if uri not in {str(resource.uri) for resource in self.resources}:
            raise ToolError("unknown_resource")
        client = self._client
        try:
            result = await client.session.read_resource(uri)
            if len(result.contents) != 1:
                raise ToolError("invalid_result")
            content = result.contents[0]
            if str(content.uri) != uri or content.mime_type != "application/json":
                raise ToolError("invalid_result")
            data = json.loads(content.text)
            model = MaterialsCatalog if uri == CATALOG_URI else WeekSchedule
            return model.model_validate(data).model_dump()
        except (ValidationError, ValueError, AttributeError, TypeError):
            raise ToolError("invalid_result") from None
        except (
            MCPError,
            TimeoutError,
            OSError,
            EOFError,
            BrokenResourceError,
            ClosedResourceError,
            EndOfStream,
        ) as error:
            if self._client is client:
                self._clear_discovery()
                self._session_lost.set()
            logger.warning("MCP resource read failed (%s)", type(error).__name__)
            raise MCPUnavailable() from None
