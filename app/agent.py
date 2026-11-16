"""Агентный цикл, подтверждения, человекочитаемые результаты и аудит."""

import json
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

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
from app.mcp_tools.client import MCPClient, MCPUnavailable
from app.mcp_tools.contracts import REMINDER, SCHEDULE, WEATHER, ToolError, validate_result
from app.reminders import PreparedReminder, ReminderError, get_user_zone, reminder_display_time
from app.store import AuditDecision, PostgresStore

logger = logging.getLogger("app.agent")

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


class AgentService:
    def __init__(
        self,
        store: PostgresStore,
        router: AgentRouter,
        mcp: MCPClient,
        settings: Settings,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.store = store
        self.router = router
        self.mcp = mcp
        self.settings = settings
        self.clock = clock

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
    ) -> AgentReply:
        request_id = secrets.token_hex(8)
        now, started = self.clock(), time.monotonic()
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
            outcome = await self._loop(
                owner_id, update_id, messages, temperature, request_id, context
            )
        return await self._record(owner_id, request_id, started, now, outcome)

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
    ) -> _Outcome:
        transcript: list[dict[str, Any]] = list(messages)
        displayed: list[str] = []
        last_tool = None
        summary = "—"

        def failed(text: str, reason: str, *, validation: str = "отклонена") -> _Outcome:
            combined = "\n\n".join([*displayed, text])
            return _Outcome(
                AgentReply(combined),
                "отказ",
                reason,
                last_tool,
                summary,
                validation,
                "остановлено",
            )

        while True:
            try:
                decision = await self.router.decide(
                    transcript,
                    temperature,
                    request_id,
                    tools=self.mcp.tools,
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
            summary = _arguments(last_tool, decision.arguments)
            if not self.mcp.available:
                return failed(
                    UNAVAILABLE_MESSAGE, "MCP-соединение недоступно.", validation="пройдена"
                )
            if last_tool == REMINDER:
                return await self._prepare(owner_id, update_id, decision, summary)
            try:
                result = validate_result(
                    last_tool, await self.mcp.call(last_tool, decision.arguments)
                )
                if last_tool == SCHEDULE and result["date"] != decision.arguments["date"]:
                    raise ToolError("invalid_result")
            except MCPUnavailable:
                return failed(
                    UNAVAILABLE_MESSAGE, "Связь с MCP-сервером прервана.", validation="пройдена"
                )
            except ToolError as exc:
                return failed(
                    TOOL_FAILURES.get(exc.code, "Данные инструмента получить не удалось."),
                    "Инструмент вернул ошибку.",
                    validation="пройдена",
                )
            text = _read_text(last_tool, result)
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
