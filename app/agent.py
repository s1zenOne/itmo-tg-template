"""Агентный цикл, подтверждения, человекочитаемые результаты и аудит."""

import json
import logging
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from app.agent_router import (
    INVALID_ARGUMENTS,
    LIMIT_MESSAGE,
    MAX_MCP_CALLS,
    REMINDER_MESSAGES,
    UNAVAILABLE_MESSAGE,
    AgentRouter,
    RouteContext,
    RouteDecision,
    RoutingError,
)
from app.config import Settings
from app.llm import LLMError
from app.mcp_tools.client import MCPClient, MCPUnavailable, observe_tool_dispatch
from app.mcp_tools.contracts import (
    REMINDER,
    SCHEDULE,
    WEATHER,
    WEEK_URI,
    ToolError,
    WeekSchedule,
    validate_result,
)
from app.mcp_tools.weather import CONDITIONS
from app.reminder_intent import verified_reminder_text
from app.reminders import PreparedReminder, ReminderError, get_user_zone, reminder_display_time
from app.store import AuditDecision, PostgresStore
from app.supervisor import (
    AgentRequest,
    AgentResult,
    EmptyFacts,
    KnowledgeFacts,
    ScheduleFacts,
    SpecialistRouter,
    WeatherFacts,
    allowed_tools,
    require_resource,
    require_tool,
)
from app.tracing import ERRORS, TraceRecorder

logger = logging.getLogger("app.agent")


def _safe_error_code(error: BaseException) -> str:
    if isinstance(error, MCPUnavailable):
        return "unavailable"
    if isinstance(error, ToolError):
        if error.code in ERRORS:
            return error.code
        if error.code.endswith("_timeout"):
            return "timeout"
        if error.code.endswith("_unavailable"):
            return "unavailable"
    return "internal_error"


TOOL_REASONS = {
    WEATHER: "Для ответа нужны данные наблюдения за погодой.",
    SCHEDULE: "Для ответа нужны занятия на указанную дату.",
    REMINDER: "Пользователь запросил напоминание.",
}
TOOL_FAILURES = {
    "city_not_found": "Город не найден. Проверьте название и укажите страну.",
    "weather_timeout": "Open-Meteo не ответил вовремя. Попробуйте позже.",
    "weather_unavailable": "Open-Meteo сейчас недоступен. Попробуйте позже.",
    "weather_invalid_response": "Open-Meteo вернул некорректные данные. Попробуйте позже.",
    "schedule_unavailable": "Учебное расписание сейчас недоступно.",
    "invalid_arguments": INVALID_ARGUMENTS,
    "invalid_result": "Получены некорректные данные инструмента. Попробуйте позже.",
    "unknown_tool": "Инструмент сейчас недоступен.",
}
CONFIRMATION_FAILURES = {
    **REMINDER_MESSAGES,
    "unavailable": "Действие недоступно для этого пользователя.",
    "expired": "Срок подтверждения истёк. Отправьте новый запрос на напоминание.",
    "cancelled": "Создание напоминания отменено.",
    "executed": "По этому запросу уже создано напоминание.",
    "mismatched": "Параметры действия изменились. Подготовьте новое напоминание.",
    "unconfirmed": "Подтвердите действие кнопкой под напоминанием.",
    "unauthorized": "Подтверждение не удалось проверить. Отправьте новый запрос.",
}


@dataclass(frozen=True)
class AgentReply:
    text: str
    confirmation: UUID | None = None
    history_text: str | None = None
    close_buttons: bool = False
    replayed: bool = False


@dataclass(frozen=True)
class _Outcome:
    reply: AgentReply
    action: str
    reason: str
    tool: str | None = None
    arguments: str = "—"
    validation: str = "пройдена"
    execution: str = "без вызова"
    facts: Any = None
    error_code: str | None = None


@dataclass(frozen=True)
class SpecialistResponse:
    """Validated public result plus application-owned reply and audit metadata."""

    result: AgentResult
    outcome: _Outcome
    observed_tools: tuple[str, ...]


def _display_time(when: datetime, zone: str) -> str:
    return f"{reminder_display_time(when, zone).isoformat(sep=' ')} ({zone})"


def _reminder_card(action: PreparedReminder) -> AgentReply:
    if action.status in CONFIRMATION_FAILURES:
        return AgentReply(CONFIRMATION_FAILURES[action.status])
    if action.status == "confirmed":
        next_step = "Повторите подтверждение, чтобы получить результат."
    else:
        next_step = "Подтвердите создание в течение пяти минут с момента подготовки."
    text = (
        f"Напоминание «{action.text}»\n"
        f"Время {_display_time(action.remind_at, action.timezone)}.\n{next_step}"
    )
    history = "Я запросил подтверждение напоминания."
    return AgentReply(text, confirmation=action.action_id, history_text=history)


def _arguments(name: str, arguments: dict[str, str]) -> str:
    if name == WEATHER:
        return f"название города, {len(arguments['city'])} симв."
    if name == SCHEDULE:
        return f"дата {arguments['date']}"
    return f"момент {arguments['remind_at']}, текст {len(arguments['text'])} симв."


def _model_data(name: str, result: dict) -> dict:
    # Свободный текст источников не попадает в следующий запрос модели.
    if name == WEATHER:
        return {
            key: result[key]
            for key in (
                "status",
                "source",
                "temperature_c",
                "wind_speed",
                "wind_unit",
                "observed_at",
            )
        }
    return {
        "status": result["status"],
        "source": result["source"],
        "date": result["date"],
        "lessons": [
            {"start": lesson["start"], "end": lesson["end"]} for lesson in result["lessons"]
        ],
    }


def _read_text(name: str, result: dict) -> str:
    if name == WEATHER:
        if result["status"] == "ambiguous":
            options = [
                ", ".join(
                    part for part in (place["city"], place["region"], place["country"]) if part
                )
                for place in result["options"]
            ]
            return (
                "Найдено несколько городов. Уточните страну или область.\n"
                + "\n".join(f"• {option}" for option in options)
                + f"\nИсточник — {result['source']}."
            )
        temperature = f"{result['temperature_c']:g}".replace(".", ",")
        wind = f"{result['wind_speed']:g}".replace(".", ",")
        return (
            f"{result['city']}, {result['country']}. {temperature} °C, "
            f"{result['conditions']}. Ветер {wind} км/ч.\n"
            f"Наблюдение {result['observed_at']} ({result['timezone']}). "
            f"Источник — {result['source']}."
        )
    lines = [f"Расписание на {result['date']}."]
    lines += [
        f"• {lesson['start']}–{lesson['end']} — «{lesson['title']}», {lesson['location']}"
        for lesson in result["lessons"]
    ] or ["Занятий в учебном расписании нет."]
    return "\n".join([*lines, f"Источник — {result['source']}."])


def _current_week_requested(query: str) -> bool:
    text = query.casefold()
    return bool(
        re.search(r"(?:эту|этой|текущ\w*)\s+недел\w*", text)
        and re.search(r"расписан\w*|занят\w*|пар(?:ы|а|у|ам|ах)\b", text)
        and not re.search(r"(?:следующ\w*|прошл\w*)\s+недел\w*", text)
    )


def _confirmed_city_requested(query: str) -> bool:
    text = query.casefold()
    # Explicit place references are handled by the regular model/tool path.
    if re.search(r"\b(?:в|во|из|для)\s+(?!мо[её]м\b)[а-яёa-z]", text):
        return False
    if re.search(r"\bв\s+мо[её]м\s+городе\b", text):
        return True
    return bool(re.search(r"\b(?:зонт\w*|ветер|ветра|одет\w*)\b", text))


def _weather_advice(query: str, result: dict) -> str:
    text = query.casefold()
    advice = []
    if re.search(r"зонт\w*", text):
        wet = {
            "слабая морось",
            "умеренная морось",
            "сильная морось",
            "слабая ледяная морось",
            "сильная ледяная морось",
            "слабый ледяной дождь",
            "сильный ледяной дождь",
            "слабый дождь",
            "умеренный дождь",
            "сильный дождь",
            "слабый ливень",
            "умеренный ливень",
            "сильный ливень",
            "гроза с небольшим градом",
            "гроза с крупным градом",
            "гроза",
        }
        if result["conditions"].casefold() in wet:
            advice.append("По текущему наблюдению стоит взять зонт.")
        elif result["conditions"].casefold() in CONDITIONS.values():
            advice.append("По текущему наблюдению дождь не указан; погода может измениться.")
        else:
            advice.append("По этим данным не удалось определить, нужен ли зонт.")
    if re.search(r"оде(?:т|в)\w*|одежд\w*|надет\w*", text):
        temperature = result["temperature_c"]
        if temperature < 5:
            advice.append("Для текущей температуры выберите тёплую одежду.")
        elif temperature < 15:
            advice.append("Для текущей температуры пригодится куртка или тёплый слой.")
        else:
            advice.append("Для текущей температуры подойдёт лёгкая одежда.")
        if result["wind_speed"] >= 35:
            advice.append("Учтите сильный ветер и возьмите ветрозащиту.")
    return " ".join(advice)


class AgentService:
    def __init__(
        self,
        store: PostgresStore,
        router: AgentRouter,
        mcp: MCPClient,
        settings: Settings,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        specialist_router: SpecialistRouter | None = None,
        traces: TraceRecorder | None = None,
    ):
        self.store = store
        self.router = router
        self.mcp = mcp
        self.settings = settings
        self.clock = clock
        self.specialist_router = specialist_router
        self.traces = traces

    async def _record(
        self, owner_id: int, request_id: str, started: float, now: datetime, outcome: _Outcome
    ) -> AgentReply:
        duration = round((time.monotonic() - started) * 1000)
        await self.store.write_audit(
            owner_id,
            AuditDecision(
                now,
                outcome.action,
                outcome.tool,
                outcome.arguments,
                outcome.validation,
                outcome.execution,
                duration,
                outcome.reason,
            ),
        )
        logger.info(
            "Решение агента id=%s tool=%s result=%s duration_ms=%d",
            request_id,
            outcome.tool or "-",
            outcome.execution,
            duration,
        )
        return outcome.reply

    async def answer(
        self,
        owner_id: int,
        update_id: int,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        component_override: str | None = None,
    ) -> AgentReply:
        request_uuid = uuid4()
        request_id = request_uuid.hex
        now, started = self.clock(), time.monotonic()
        trace_id = None
        existing = await self.store.reminder_for_update(owner_id, update_id)
        if existing is not None:
            outcome = _Outcome(
                replace(_reminder_card(existing), replayed=True),
                "повтор запроса",
                "Действие уже подготовлено.",
                REMINDER,
                execution="использовано сохранённое действие",
            )
        else:
            await self.mcp.connect()
            zone = await self.store.get_timezone(owner_id)
            context = RouteContext(now, zone, self.settings.schedule_timezone)
            component = None
            if self.traces is not None:
                trace_id = await self.traces.start(owner_id, request_uuid, "unknown")
            route_started = self.clock()
            query = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            if self.specialist_router is not None:
                if component_override == "dialogue":
                    component = "dialogue"
                else:
                    try:
                        decision = await self.specialist_router.decide(
                            query, request_id, messages[1:-1]
                        )
                    except LLMError as exc:
                        outcome = _Outcome(
                            AgentReply(exc.user_message), "отказ", "Сервис модели недоступен."
                        )
                        if trace_id is not None:
                            await self.traces.span(
                                trace_id,
                                "router",
                                "route",
                                route_started,
                                self.clock(),
                                "error",
                                error_code=exc.code,
                            )
                            await self.traces.close(trace_id, "error")
                        return await self._record(owner_id, request_id, started, now, outcome)
                    component = decision.target
            if self.traces is not None:
                await self.traces.set_route(
                    trace_id, "general" if component == "dialogue" else (component or "unknown")
                )
                await self.traces.span(
                    trace_id,
                    "router",
                    "route",
                    route_started,
                    self.clock(),
                    "clarify" if component == "clarify" else "ok",
                )
            request = AgentRequest(
                request_id=request_uuid,
                trace_id=trace_id or uuid4(),
                actor_id=owner_id,
                text=query,
                intent="user_query",
                occurred_at=now,
                timezone=zone or self.settings.schedule_timezone,
            )
            if component == "clarify":
                outcome = _Outcome(
                    AgentReply(
                        "Уточните, пожалуйста: вам нужны расписание, учебные материалы или погода?"
                    ),
                    "уточнение",
                    "Маршрут не определён.",
                )
            elif component in {"schedule", "knowledge", "lifestyle"}:
                specialist = {
                    "schedule": self._schedule_specialist,
                    "knowledge": self._knowledge_specialist,
                    "lifestyle": self._lifestyle_specialist,
                }[component]
                try:
                    response = await specialist(
                        request, update_id, messages, temperature, context, trace_id
                    )
                    validated = AgentResult.model_validate(response.result.model_dump())
                    expected_status = {
                        "отказ": "failed",
                        "уточнение": "needs_clarification",
                        "не найдено": "not_found",
                    }.get(response.outcome.action, "success")
                    confirmation = response.outcome.reply.confirmation
                    if (
                        validated.message != response.outcome.reply.text
                        or validated.used_tools != list(response.observed_tools)
                        or validated.facts != (response.outcome.facts or EmptyFacts())
                        or validated.status != expected_status
                        or validated.error_code
                        != (
                            (response.outcome.error_code or "internal_error")
                            if validated.status == "failed"
                            else None
                        )
                        or (confirmation is not None and response.outcome.action != "подготовка")
                        or (response.outcome.action == "подготовка" and confirmation is None)
                    ):
                        raise ValueError("specialist result mismatch")
                    outcome = replace(
                        response.outcome,
                        reply=replace(response.outcome.reply, text=validated.message),
                    )
                except (AttributeError, TypeError, ValueError):
                    outcome = _Outcome(
                        AgentReply("Не удалось проверить результат специалиста."),
                        "отказ",
                        "Некорректный результат специалиста.",
                        validation="отклонена",
                    )
            else:
                outcome = await self._loop(
                    owner_id,
                    update_id,
                    messages,
                    temperature,
                    request_id,
                    context,
                    component,
                    trace_id,
                )
        if trace_id is not None:
            await self.traces.close(trace_id, "error" if outcome.action == "отказ" else "ok")
        return await self._record(owner_id, request_id, started, now, outcome)

    @staticmethod
    def _specialist_response(outcome: _Outcome, tools: list[str]) -> SpecialistResponse:
        status = {
            "отказ": "failed",
            "уточнение": "needs_clarification",
            "не найдено": "not_found",
        }.get(outcome.action, "success")
        result = AgentResult(
            status=status,
            message=outcome.reply.text,
            confidence=1.0,
            facts=outcome.facts or EmptyFacts(),
            used_tools=tools,
            error_code=(outcome.error_code or "internal_error") if status == "failed" else None,
        )
        return SpecialistResponse(result, outcome, tuple(tools))

    async def _call_observed(
        self, name: str, arguments: dict, used_tools: list[str] | None, dispatched: list[str]
    ) -> dict:
        def on_dispatch(tool_name: str) -> None:
            dispatched.append(tool_name)
            if used_tools is not None:
                used_tools.append(tool_name)

        with observe_tool_dispatch(on_dispatch):
            result = await self.mcp.call(name, arguments)
        # Test doubles cannot expose the SDK dispatch signal; a returned result proves their call.
        if not dispatched:
            on_dispatch(name)
        return result

    async def _tool_error_span(
        self,
        trace_id: UUID | None,
        component: str,
        started: datetime,
        dispatched: list[str],
        error: BaseException,
    ) -> None:
        if trace_id is None:
            return
        code = _safe_error_code(error)
        await self.traces.span(
            trace_id,
            component,
            "tool" if dispatched else "execute",
            started,
            self.clock(),
            "error",
            tool_name=dispatched[-1] if dispatched else None,
            error_code=code,
        )

    async def _schedule_specialist(
        self,
        request: AgentRequest,
        update_id: int,
        messages: list[dict[str, str]],
        temperature: float,
        context: RouteContext,
        trace_id: UUID | None,
    ) -> SpecialistResponse:
        tools: list[str] = []
        if _current_week_requested(request.text):
            outcome = await self._current_week(request.occurred_at, trace_id)
        else:
            outcome = await self._loop(
                request.actor_id,
                update_id,
                messages,
                temperature,
                request.request_id.hex,
                context,
                "schedule",
                trace_id,
                tools,
            )
        return self._specialist_response(outcome, tools)

    async def _knowledge_specialist(
        self,
        request: AgentRequest,
        update_id: int,
        messages: list[dict[str, str]],
        temperature: float,
        context: RouteContext,
        trace_id: UUID | None,
    ) -> SpecialistResponse:
        tools: list[str] = []
        outcome = await self._knowledge(request, trace_id, tools)
        return self._specialist_response(outcome, tools)

    async def _lifestyle_specialist(
        self,
        request: AgentRequest,
        update_id: int,
        messages: list[dict[str, str]],
        temperature: float,
        context: RouteContext,
        trace_id: UUID | None,
    ) -> SpecialistResponse:
        tools: list[str] = []
        if _confirmed_city_requested(request.text):
            outcome = await self._confirmed_city_weather(
                request.actor_id, request.text, trace_id, tools
            )
        else:
            outcome = await self._loop(
                request.actor_id,
                update_id,
                messages,
                temperature,
                request.request_id.hex,
                context,
                "lifestyle",
                trace_id,
                tools,
            )
        return self._specialist_response(outcome, tools)

    async def _confirmed_city_weather(
        self, owner_id: int, query: str, trace_id: UUID | None, tools: list[str]
    ) -> _Outcome:
        city = await self.store.get_confirmed_city(owner_id)
        if city is None:
            return _Outcome(
                AgentReply("Сначала выберите и подтвердите город командой /city."),
                "уточнение",
                "Подтверждённый город не задан.",
            )
        started = self.clock()
        dispatched: list[str] = []
        try:
            require_tool("lifestyle", WEATHER, self.mcp.tools)
            result = validate_result(
                WEATHER,
                await self._call_observed(
                    WEATHER,
                    {
                        "city": city["city_name"],
                        "country": city["city_country"],
                        "region": city["city_region"],
                        "latitude": city["latitude"],
                        "longitude": city["longitude"],
                    },
                    tools,
                    dispatched,
                ),
            )
            if result["status"] != "ok":
                raise ToolError("invalid_result")
            if trace_id is not None:
                await self.traces.span(
                    trace_id,
                    "lifestyle",
                    "tool",
                    started,
                    self.clock(),
                    "ok",
                    tool_name=WEATHER,
                )
            weather = {**result, "city": city["city_name"], "country": city["city_country"]}
            text = _read_text(WEATHER, weather)
            advice = _weather_advice(query, weather)
            if advice:
                text += f"\n{advice}"
            return _Outcome(
                AgentReply(text, history_text="Я показал проверенные данные о погоде."),
                "чтение",
                "Погода в подтверждённом городе.",
                WEATHER,
                execution="успешно",
                facts=WeatherFacts(temperature_c=result["temperature_c"]),
            )
        except (MCPUnavailable, ToolError, KeyError, TypeError, ValueError) as exc:
            await self._tool_error_span(trace_id, "lifestyle", started, dispatched, exc)
            return _Outcome(
                AgentReply("Погода в подтверждённом городе сейчас недоступна."),
                "отказ",
                "Не удалось проверить данные о погоде.",
                validation="отклонена",
                error_code=_safe_error_code(exc),
            )

    async def _current_week(self, now: datetime, trace_id: UUID | None) -> _Outcome:
        try:
            require_resource("schedule", WEEK_URI, self.mcp.resources)
            started = self.clock()
            data = WeekSchedule.model_validate(await self.mcp.read_resource(WEEK_URI))
            if data.timezone != self.settings.schedule_timezone:
                raise ValueError("unexpected schedule zone")
            today = now.astimezone(ZoneInfo(data.timezone)).date()
            monday = today - timedelta(days=today.weekday())
            dates = [(monday + timedelta(days=i)).isoformat() for i in range(7)]
            if [day.date for day in data.days] != dates:
                raise ValueError("invalid week")
            if trace_id is not None:
                await self.traces.span(
                    trace_id, "schedule", "resource", started, self.clock(), "ok"
                )
            text = "\n\n".join(_read_text(SCHEDULE, day.model_dump()) for day in data.days)
            return _Outcome(
                AgentReply(text, history_text="Я показал учебное расписание на неделю."),
                "чтение",
                "Чтение расписания текущей недели.",
                execution="успешно",
                facts=ScheduleFacts(dates=dates),
            )
        except (MCPUnavailable, ToolError, KeyError, TypeError, ValueError):
            return _Outcome(
                AgentReply("Расписание на текущую неделю сейчас недоступно."),
                "отказ",
                "Не удалось проверить ресурс расписания.",
                validation="отклонена",
            )

    async def _knowledge(
        self, request: AgentRequest, trace_id: UUID | None, tools: list[str]
    ) -> _Outcome:
        query = request.text
        from app.mcp_tools.contracts import CATALOG_URI, MATERIALS

        call_started = self.clock()
        dispatched: list[str] = []
        try:
            require_resource("knowledge", CATALOG_URI, self.mcp.resources)
            catalog = await self.mcp.read_resource(CATALOG_URI)
            require_tool("knowledge", MATERIALS, self.mcp.tools)
            call_started = self.clock()
            data = validate_result(
                MATERIALS,
                await self._call_observed(
                    MATERIALS, {"query": query, "limit": 5}, tools, dispatched
                ),
            )
            if trace_id is not None:
                await self.traces.span(
                    trace_id,
                    "knowledge",
                    "tool",
                    call_started,
                    self.clock(),
                    "ok",
                    tool_name=MATERIALS,
                )
            known = {item["id"] for item in catalog["documents"]}
            documents = [item for item in data["documents"] if item["id"] in known]
            if not documents:
                result = AgentResult(
                    status="not_found",
                    message="В учебных материалах ответ не найден.",
                    confidence=1.0,
                    facts=EmptyFacts(),
                    used_tools=[MATERIALS],
                )
            else:
                text = "\n\n".join(
                    f"{item['text']}\nИсточник: {item['title']} [{item['id']}]."
                    for item in documents
                )
                result = AgentResult(
                    status="success",
                    message=text,
                    confidence=1.0,
                    facts=KnowledgeFacts(source_ids=[item["id"] for item in documents]),
                    used_tools=[MATERIALS],
                )
            return _Outcome(
                AgentReply(
                    result.message, history_text="Я показал сведения из учебных материалов."
                ),
                "не найдено" if result.status == "not_found" else "чтение",
                "Поиск в учебных материалах.",
                MATERIALS,
                execution="успешно",
                facts=result.facts,
            )
        except (MCPUnavailable, ToolError, KeyError, TypeError, ValueError) as exc:
            await self._tool_error_span(trace_id, "knowledge", call_started, dispatched, exc)
            return _Outcome(
                AgentReply("Учебные материалы сейчас недоступны."),
                "отказ",
                "Ошибка чтения учебных материалов.",
                validation="отклонена",
                error_code=_safe_error_code(exc),
            )

    async def reject(self, owner_id: int, text: str) -> None:
        await self._record(
            owner_id,
            secrets.token_hex(8),
            time.monotonic(),
            self.clock(),
            _Outcome(
                AgentReply(text), "отказ", "Размер запроса превышает лимит.", validation="отклонена"
            ),
        )

    async def _loop(
        self,
        owner_id: int,
        update_id: int,
        messages: list[dict[str, str]],
        temperature: float,
        request_id: str,
        context: RouteContext,
        component: str | None = None,
        trace_id: UUID | None = None,
        used_tools: list[str] | None = None,
    ) -> _Outcome:
        transcript: list[dict[str, Any]] = list(messages)
        displayed: list[str] = []
        last_tool = None
        summary = "—"
        facts: Any = EmptyFacts()

        def failed(
            text: str,
            reason: str,
            *,
            validation: str = "отклонена",
            error_code: str = "internal_error",
        ) -> _Outcome:
            combined = "\n\n".join([*displayed, text])
            return _Outcome(
                AgentReply(combined),
                "отказ",
                reason,
                last_tool,
                summary,
                validation,
                "остановлено",
                facts,
                error_code,
            )

        while True:
            try:
                decision = await self.router.decide(
                    transcript,
                    temperature,
                    request_id,
                    tools=(
                        allowed_tools(component, self.mcp.tools)
                        if component is not None
                        else self.mcp.tools
                    ),
                    context=context,
                )
            except RoutingError as exc:
                if exc.code == "parallel_calls" and exc.result is not None:
                    count = sum(call.name in TOOL_REASONS for call in exc.result.tool_calls)
                    if len(displayed) + count > MAX_MCP_CALLS:
                        return failed(LIMIT_MESSAGE, "Достигнут лимит MCP-вызовов.")
                if exc.name in TOOL_REASONS:
                    last_tool = exc.name
                return failed(exc.user_message, "Предложение модели отклонено проверкой.")
            except LLMError as exc:
                return failed(exc.user_message, "Сервис модели не вернул решение.")
            if decision.action == "unavailable":
                return failed(UNAVAILABLE_MESSAGE, "MCP-соединение недоступно.", validation="—")
            if decision.action != "call_tool":
                if displayed:
                    text = "\n\n".join(displayed)
                    if decision.action == "clarify":
                        text += f"\n\n{decision.text}"
                    history = "Я показал проверенные данные инструмента."
                    return _Outcome(
                        AgentReply(text, history_text=history),
                        "чтение",
                        TOOL_REASONS[last_tool],
                        last_tool,
                        summary,
                        execution="успешно",
                        facts=facts,
                    )
                action = "уточнение" if decision.action == "clarify" else "ответ"
                reason = (
                    "Для действия нужны дополнительные данные."
                    if decision.action == "clarify"
                    else "На запрос можно ответить без инструмента."
                )
                return _Outcome(
                    AgentReply(decision.text, history_text=decision.text),
                    action,
                    reason,
                )

            if len(displayed) >= MAX_MCP_CALLS:
                return failed(LIMIT_MESSAGE, "Достигнут лимит MCP-вызовов.")
            last_tool = decision.name
            if component is not None:
                try:
                    require_tool(component, last_tool, self.mcp.tools)
                except ToolError:
                    return failed(
                        "Действие не разрешено для этого запроса.",
                        "Политика доступа отклонила действие.",
                        error_code="unauthorized",
                    )
                original = next(m["content"] for m in reversed(messages) if m["role"] == "user")
                if last_tool == REMINDER:
                    original_task = verified_reminder_text(
                        original,
                        decision.arguments["text"],
                        decision.arguments["remind_at"],
                        context.now,
                        decision.timezone or "UTC",
                    )
                    if original_task is None:
                        return failed(
                            "Уточните текст и точные дату и время напоминания.",
                            "Напоминание не соответствует исходному запросу.",
                        )
                    decision = replace(
                        decision, arguments={**decision.arguments, "text": original_task}
                    )
            summary = _arguments(last_tool, decision.arguments)
            if not self.mcp.available:
                return failed(
                    UNAVAILABLE_MESSAGE,
                    "MCP-соединение недоступно.",
                    validation="пройдена",
                    error_code="unavailable",
                )
            if last_tool == REMINDER:
                return await self._prepare(owner_id, update_id, decision, summary)
            dispatched: list[str] = []
            call_started = self.clock()
            try:
                result = validate_result(
                    last_tool,
                    await self._call_observed(
                        last_tool, decision.arguments, used_tools, dispatched
                    ),
                )
                if last_tool == SCHEDULE and result["date"] != decision.arguments["date"]:
                    raise ToolError("invalid_result")
                if trace_id is not None:
                    await self.traces.span(
                        trace_id,
                        component,
                        "tool",
                        call_started,
                        self.clock(),
                        "ok",
                        tool_name=last_tool,
                    )
            except MCPUnavailable as exc:
                await self._tool_error_span(
                    trace_id, component or "mcp", call_started, dispatched, exc
                )
                return failed(
                    UNAVAILABLE_MESSAGE,
                    "Связь с MCP-сервером прервана.",
                    validation="пройдена",
                    error_code="unavailable",
                )
            except ToolError as exc:
                await self._tool_error_span(
                    trace_id, component or "mcp", call_started, dispatched, exc
                )
                return failed(
                    TOOL_FAILURES.get(exc.code, "Данные инструмента получить не удалось."),
                    "Инструмент вернул ошибку.",
                    validation="пройдена",
                    error_code=_safe_error_code(exc),
                )
            text = _read_text(last_tool, result)
            if last_tool == SCHEDULE:
                previous = facts.dates if isinstance(facts, ScheduleFacts) else []
                facts = ScheduleFacts(dates=[*previous, result["date"]])
            elif last_tool == WEATHER and result["status"] == "ok":
                facts = WeatherFacts(temperature_c=result["temperature_c"])
            if component == "lifestyle" and last_tool == WEATHER and result["status"] == "ok":
                original = next(m["content"] for m in reversed(messages) if m["role"] == "user")
                advice = _weather_advice(original, result)
                if advice:
                    text += f"\n{advice}"
            if result["status"] == "ambiguous":
                return _Outcome(
                    AgentReply(
                        "\n\n".join([*displayed, text]), history_text="Я попросил уточнить город."
                    ),
                    "уточнение",
                    "Геокодер вернул несколько городов.",
                    last_tool,
                    summary,
                    execution="получены варианты",
                    facts=facts,
                )
            displayed.append(text)
            transcript.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [decision.result.tool_calls[0].as_message()],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": decision.call_id,
                        "content": json.dumps(_model_data(last_tool, result), ensure_ascii=False),
                    },
                ]
            )

    async def _prepare(
        self, owner_id: int, update_id: int, decision: RouteDecision, summary: str
    ) -> _Outcome:
        try:
            action = await self.store.prepare_reminder(
                owner_id,
                update_id,
                decision.arguments["text"],
                datetime.fromisoformat(decision.arguments["remind_at"]),
                decision.timezone,
            )
        except ReminderError as exc:
            return _Outcome(
                AgentReply(CONFIRMATION_FAILURES.get(exc.code, INVALID_ARGUMENTS)),
                "отказ",
                "Параметры напоминания не прошли проверку.",
                REMINDER,
                summary,
                "отклонена",
            )
        return _Outcome(
            _reminder_card(action),
            "подготовка",
            TOOL_REASONS[REMINDER],
            REMINDER,
            summary,
            execution="ожидается подтверждение",
        )

    async def confirm(self, owner_id: int, action_id: UUID, *, accepted: bool) -> AgentReply:
        now, started = self.clock(), time.monotonic()
        request_id = secrets.token_hex(8)
        try:
            if accepted:
                action = await self.store.confirm_reminder(owner_id, action_id)
                arguments = {"text": action.text, "remind_at": action.remind_at.isoformat()}
                if self.specialist_router is not None:
                    require_tool("schedule", REMINDER, self.mcp.tools)
                result = validate_result(
                    REMINDER,
                    await self.mcp.call(
                        REMINDER, arguments, owner_id=owner_id, action_id=action_id
                    ),
                )
                if (
                    datetime.fromisoformat(result["remind_at"]) != action.remind_at
                    or result["timezone"] != action.timezone
                ):
                    raise ToolError("invalid_result")
                text = (
                    f"Напоминание № {result['id']} сохранено. "
                    f"Время {_display_time(action.remind_at, action.timezone)}."
                )
                execution = "создано" if result["created"] else "возвращена сохранённая запись"
                outcome = _Outcome(
                    AgentReply(text, close_buttons=True),
                    "подтверждение",
                    "Получено согласие пользователя.",
                    REMINDER,
                    _arguments(REMINDER, arguments),
                    execution=execution,
                )
            else:
                await self.store.cancel_reminder(owner_id, action_id)
                outcome = _Outcome(
                    AgentReply("Создание напоминания отменено.", close_buttons=True),
                    "отмена",
                    "Пользователь отменил действие.",
                    REMINDER,
                    execution="отменено",
                )
        except MCPUnavailable:
            outcome = _Outcome(
                AgentReply(f"{UNAVAILABLE_MESSAGE} Затем нажмите кнопку подтверждения повторно."),
                "подтверждение",
                "Связь с MCP-сервером недоступна.",
                REMINDER,
                execution="сбой",
            )
        except (ReminderError, ToolError) as exc:
            terminal = exc.code in {"expired", "cancelled", "past", "executed"}
            text = CONFIRMATION_FAILURES.get(
                exc.code, "Не удалось проверить запись. Попробуйте подтвердить ещё раз."
            )
            outcome = _Outcome(
                AgentReply(text, close_buttons=terminal),
                "отказ",
                "Действие или результат не прошли проверку.",
                REMINDER,
                validation="отклонена",
                execution="остановлено",
            )
        return await self._record(owner_id, request_id, started, now, outcome)

    async def trace(self, owner_id: int, argument: str) -> str:
        if self.traces is None:
            return "Трассировка недоступна."
        return await self.traces.render(owner_id, argument)

    async def tools_text(self) -> str:
        if not await self.mcp.connect() or not self.mcp.tools:
            return UNAVAILABLE_MESSAGE
        lines = ["Доступные инструменты"]
        for tool in self.mcp.tools:
            read_only = tool.annotations is not None and tool.annotations.read_only_hint is True
            kind = "чтение" if read_only else "изменение, нужно подтверждение"
            lines.append(f"• {tool.name} — {tool.description or tool.name} ({kind})")
        return "\n".join(lines)

    async def timezone_text(self, owner_id: int, argument: str) -> str:
        zone = argument or await self.store.get_timezone(owner_id)
        if zone is None:
            return "Часовой пояс ещё не задан. Пример команды /timezone Europe/Moscow."
        try:
            get_user_zone(zone)
        except ReminderError:
            return REMINDER_MESSAGES["invalid_timezone"]
        if argument:
            await self.store.set_timezone(owner_id, zone)
        return f"Часовой пояс {zone}. Сейчас {_display_time(self.clock(), zone)}."

    async def why(self, owner_id: int) -> str:
        entry = await self.store.latest_audit(owner_id)
        if entry is None:
            return "У вас пока нет решений агента. Отправьте текстовый запрос."
        zone = await self.store.get_timezone(owner_id) or "UTC"
        return "\n".join(
            [
                f"Последний запрос — {_display_time(entry.requested_at, zone)}",
                f"Действие — {entry.action}",
                f"Инструмент — {entry.tool or '—'}",
                f"Аргументы — {entry.arguments}",
                f"Проверка — {entry.validation}",
                f"Выполнение — {entry.execution}",
                f"Длительность — {entry.duration_ms} мс",
                f"Причина — {entry.reason}",
            ]
        )
