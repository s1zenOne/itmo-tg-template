"""Минимальные структурированные трассы без содержимого пользовательских запросов."""

import re
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

ROUTES = frozenset(
    {
        "schedule",
        "briefing",
        "knowledge",
        "lifestyle",
        "clarify",
        "general",
        "study",
        "translate",
        "quiz",
        "unknown",
    }
)
COMPONENTS = frozenset(
    {"supervisor", "router", "schedule", "knowledge", "lifestyle", "briefing", "telegram", "mcp"}
)
OPERATIONS = frozenset(
    {"request", "route", "execute", "tool", "resource", "prepare", "send", "confirm"}
)
STATUSES = frozenset({"ok", "error", "clarify", "missed"})
TOOLS = frozenset({"get_schedule", "add_reminder", "get_weather", "search_materials"})
ERRORS = frozenset(
    {
        "invalid_route",
        "invalid_result",
        "unauthorized",
        "unavailable",
        "timeout",
        "temporary",
        "permanent",
        "internal_error",
        "not_found",
    }
)
REQUEST_ID = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{32})\Z")


def _uuid(value: UUID | str) -> UUID:
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("invalid trace id") from None


def _time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("aware timestamp required")
    return value.astimezone(UTC)


def _owner(value: int) -> int:
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError("invalid owner")
    return value


class TraceRecorder:
    def __init__(self, pool, *, clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.pool = pool
        self.clock = clock

    async def start(self, owner_id: int, request_id: UUID | str, route: str) -> UUID:
        owner_id = _owner(owner_id)
        if route not in ROUTES:
            raise ValueError("invalid route")
        if isinstance(request_id, UUID):
            request_id = request_id.hex
        if not isinstance(request_id, str) or REQUEST_ID.fullmatch(request_id) is None:
            raise ValueError("invalid request id")
        trace_id = uuid4()
        now = _time(self.clock())
        await self.pool.execute(
            """WITH new_trace AS (
                INSERT INTO lab3_traces (trace_id, request_id, owner_id, route, started_at)
                VALUES ($1, $2, $3, $4, $5) RETURNING trace_id
            ) INSERT INTO lab3_trace_spans
                (span_id, trace_id, parent_span_id, component, operation, started_at, status)
                SELECT trace_id, trace_id, NULL, 'supervisor', 'request', $5, 'running'
                FROM new_trace""",
            trace_id,
            request_id,
            owner_id,
            route,
            now,
        )
        return trace_id

    async def set_route(self, trace_id: UUID | str, route: str) -> None:
        if route not in ROUTES:
            raise ValueError("invalid route")
        await self.pool.execute(
            "UPDATE lab3_traces SET route=$2 WHERE trace_id=$1", _uuid(trace_id), route
        )

    async def span(
        self,
        trace_id: UUID | str,
        component: str,
        operation: str,
        started_at: datetime,
        finished_at: datetime,
        status: str,
        tool_name: str | None = None,
        error_code: str | None = None,
        *,
        parent_span_id: UUID | str | None = None,
    ) -> UUID:
        trace_id = _uuid(trace_id)
        parent = _uuid(parent_span_id) if parent_span_id is not None else trace_id
        if component not in COMPONENTS or operation not in OPERATIONS or status not in STATUSES:
            raise ValueError("invalid span metadata")
        if (operation == "tool") != (tool_name is not None) or (
            tool_name is not None and tool_name not in TOOLS
        ):
            raise ValueError("invalid tool")
        if error_code is not None and error_code not in ERRORS:
            error_code = "internal_error"
        started, finished = _time(started_at), _time(finished_at)
        if finished < started:
            raise ValueError("invalid duration")
        span_id = uuid4()
        inserted = await self.pool.fetchval(
            """INSERT INTO lab3_trace_spans
                (span_id, trace_id, parent_span_id, component, operation, started_at,
                 finished_at, status, tool_name, error_code)
                SELECT $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
                WHERE EXISTS (SELECT 1 FROM lab3_trace_spans p
                              WHERE p.span_id = $3 AND p.trace_id = $2)
                RETURNING span_id""",
            span_id,
            trace_id,
            parent,
            component,
            operation,
            started,
            finished,
            status,
            tool_name,
            error_code,
        )
        if inserted is None:
            raise ValueError("unknown parent span")
        return span_id

    async def close(
        self, trace_id: UUID | str, status: str, finished_at: datetime | None = None
    ) -> None:
        trace_id = _uuid(trace_id)
        if status not in STATUSES:
            raise ValueError("invalid status")
        finished = _time(finished_at if finished_at is not None else self.clock())
        updated = await self.pool.fetchval(
            """UPDATE lab3_trace_spans SET finished_at = $2, status = $3
               WHERE trace_id = $1 AND span_id = $1 AND status = 'running'
                 AND started_at <= $2 RETURNING span_id""",
            trace_id,
            finished,
            status,
        )
        if updated is None:
            raise ValueError("unknown or closed trace")

    async def render(self, owner_id: int, argument: str) -> str:
        owner_id = _owner(owner_id)
        if argument == "last":
            trace = await self.pool.fetchrow(
                """SELECT trace_id, route FROM lab3_traces WHERE owner_id = $1
                   ORDER BY started_at DESC, trace_id DESC LIMIT 1""",
                owner_id,
            )
        else:
            try:
                trace_id = _uuid(argument)
            except ValueError:
                return "Укажите /trace last или /trace <идентификатор>."
            trace = await self.pool.fetchrow(
                "SELECT trace_id, route FROM lab3_traces WHERE owner_id = $1 AND trace_id = $2",
                owner_id,
                trace_id,
            )
        if trace is None:
            return "Трасса не найдена."
        spans = await self.pool.fetch(
            """SELECT component, operation, status, started_at, finished_at, tool_name, error_code
               FROM lab3_trace_spans WHERE trace_id = $1 ORDER BY started_at, span_id""",
            trace["trace_id"],
        )
        lines = [f"Трасса {trace['trace_id']}", f"Маршрут: {trace['route']}"]
        for item in spans:
            duration = "—"
            if item["finished_at"] is not None:
                elapsed = item["finished_at"] - item["started_at"]
                duration = f"{round(elapsed.total_seconds() * 1000)} мс"
            tool = f", инструмент {item['tool_name']}" if item["tool_name"] else ""
            error = f", ошибка {item['error_code']}" if item["error_code"] else ""
            lines.append(
                f"• {item['component']}: {item['operation']}, {item['status']}, "
                f"{duration}{tool}{error}"
            )
        return "\n".join(lines)
