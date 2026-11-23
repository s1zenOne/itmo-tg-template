"""Bounded read-only collection of daily briefing sections."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from uuid import UUID, uuid4

from app.mcp_tools.client import MCPUnavailable, observe_tool_dispatch
from app.mcp_tools.contracts import (
    CATALOG_URI,
    MATERIALS,
    SCHEDULE,
    WEATHER,
    MaterialsCatalog,
    ToolError,
    validate_result,
)
from app.supervisor import (
    AgentRequest,
    AgentResult,
    EmptyFacts,
    KnowledgeFacts,
    ScheduleFacts,
    WeatherFacts,
    require_resource,
    require_tool,
)


@dataclass(frozen=True)
class BriefingSpan:
    component: str
    operation: str
    started_at: datetime
    finished_at: datetime
    status: str
    tool_name: str | None = None
    error_code: str | None = None


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, MCPUnavailable):
        return "unavailable"
    if isinstance(exc, (TimeoutError, asyncio.CancelledError)):
        return "timeout"
    if isinstance(exc, ToolError) and exc.code in {"invalid_result", "unauthorized", "unavailable"}:
        return exc.code
    return "internal_error"


class BriefingBuilder:
    def __init__(self, mcp):
        self.mcp = mcp

    async def __call__(
        self,
        owner_id: int,
        day: date,
        now: datetime,
        settings,
        *,
        trace_events: list[BriefingSpan] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        trace_id: UUID | None = None,
    ) -> str:
        request = AgentRequest(
            request_id=uuid4(),
            trace_id=trace_id or uuid4(),
            actor_id=owner_id,
            text="Ежедневная сводка",
            intent="daily_briefing",
            occurred_at=now,
            timezone=settings["timezone"],
        )
        results = await self._sections(request, day, settings, trace_events, clock)
        return "\n".join(result.message for result in results)

    async def _sections(
        self,
        request: AgentRequest,
        day: date,
        settings,
        trace_events: list[BriefingSpan] | None,
        clock: Callable[[], datetime],
    ) -> list[AgentResult]:
        # Validate at the boundary, including objects constructed without validation.
        request = AgentRequest.model_validate(request.model_dump())
        used = {name: [] for name in ("schedule", "lifestyle", "knowledge")}
        recorded_errors: set[str] = set()

        async def tracked(component, operation, tool_name, work):
            started = clock()
            dispatched: list[str] = []

            def on_dispatch(name):
                dispatched.append(name)
                used[component].append(name)

            try:
                with observe_tool_dispatch(on_dispatch):
                    result = await work()
            except BaseException as exc:
                if trace_events is not None:
                    trace_events.append(
                        BriefingSpan(
                            component,
                            "tool" if dispatched else "execute",
                            started,
                            clock(),
                            "error",
                            dispatched[-1] if dispatched else None,
                            _safe_error(exc),
                        )
                    )
                    recorded_errors.add(component)
                raise
            if tool_name is not None and not dispatched:
                used[component].append(tool_name)
            if trace_events is not None:
                trace_events.append(
                    BriefingSpan(component, operation, started, clock(), "ok", tool_name)
                )
            return result

        async def schedule(request: AgentRequest) -> AgentResult:
            require_tool("briefing_schedule", SCHEDULE, self.mcp.tools)

            async def fetch():
                result = validate_result(
                    SCHEDULE, await self.mcp.call(SCHEDULE, {"date": day.isoformat()})
                )
                if result["date"] != day.isoformat():
                    raise ToolError("invalid_result")
                return result

            result = await tracked("schedule", "tool", SCHEDULE, fetch)
            lessons = result["lessons"]
            return AgentResult(
                status="success",
                message="Занятия: "
                + ("; ".join(f"{x['start']} {x['title']}" for x in lessons) or "нет"),
                confidence=1.0,
                facts=ScheduleFacts(dates=[day.isoformat()]),
                used_tools=used["schedule"],
            )

        async def weather(request: AgentRequest) -> AgentResult:
            require_tool("briefing_lifestyle", WEATHER, self.mcp.tools)

            async def fetch():
                result = validate_result(
                    WEATHER,
                    await self.mcp.call(
                        WEATHER,
                        {
                            "city": settings["city_name"],
                            "country": settings["city_country"],
                            "region": settings["city_region"],
                            "latitude": settings["latitude"],
                            "longitude": settings["longitude"],
                        },
                    ),
                )
                if result["status"] != "ok":
                    raise ToolError("invalid_result")
                return result

            result = await tracked("lifestyle", "tool", WEATHER, fetch)
            return AgentResult(
                status="success",
                message=(
                    f"Погода: {result['temperature_c']:g} °C, {result['conditions']} (Open-Meteo)."
                ),
                confidence=1.0,
                facts=WeatherFacts(temperature_c=result["temperature_c"]),
                used_tools=used["lifestyle"],
            )

        async def knowledge(request: AgentRequest) -> AgentResult:
            require_resource("briefing_knowledge", CATALOG_URI, self.mcp.resources)

            async def read():
                data = await self.mcp.read_resource(CATALOG_URI)
                return MaterialsCatalog.model_validate(data)

            catalog = await tracked("knowledge", "resource", None, read)
            require_tool("briefing_knowledge", MATERIALS, self.mcp.tools)

            async def search():
                return validate_result(
                    MATERIALS,
                    await self.mcp.call(MATERIALS, {"query": "контекстное окно", "limit": 1}),
                )

            result = await tracked("knowledge", "tool", MATERIALS, search)
            known = {entry.id for entry in catalog.documents}
            docs = [doc for doc in result["documents"] if doc["id"] in known]
            if not docs:
                return AgentResult(
                    status="not_found",
                    message="Учебный фокус: подходящий материал не найден.",
                    confidence=1.0,
                    facts=EmptyFacts(),
                    used_tools=used["knowledge"],
                )
            doc = docs[0]
            return AgentResult(
                status="success",
                message=f"Учебный фокус: повторите тему «{doc['title']}» [{doc['id']}].",
                confidence=1.0,
                facts=KnowledgeFacts(source_ids=[doc["id"]]),
                used_tools=used["knowledge"],
            )

        async def safe(coro, label, component):
            started = clock()
            try:
                async with asyncio.timeout(4):
                    result = await coro(request)
                    return AgentResult.model_validate(result.model_dump())
            except Exception as exc:
                if (
                    trace_events is not None
                    and isinstance(exc, ToolError)
                    and exc.code == "unauthorized"
                    and component not in recorded_errors
                ):
                    trace_events.append(
                        BriefingSpan(
                            component,
                            "execute",
                            started,
                            clock(),
                            "error",
                            error_code="unauthorized",
                        )
                    )
                return AgentResult(
                    status="failed",
                    message=f"{label}: данные недоступны.",
                    confidence=1.0,
                    facts=EmptyFacts(),
                    used_tools=used[component],
                    error_code=_safe_error(exc),
                )

        return list(
            await asyncio.gather(
                safe(schedule, "Занятия", "schedule"),
                safe(weather, "Погода", "lifestyle"),
                safe(knowledge, "Учебный фокус", "knowledge"),
            )
        )
