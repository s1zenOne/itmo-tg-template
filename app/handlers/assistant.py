"""Telegram-команды и доставка обычного текста."""

import asyncio
from contextlib import suppress
from uuid import UUID

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update

from app.agent_router import UNAVAILABLE_MESSAGE
from app.mcp_tools.contracts import ToolError
from app.notifications import NotificationService
from app.service import AssistantService

MAX_TELEGRAM_CHARS = 4096
START_TEXT = (
    "Я помогу с программированием, переводом и тренировкой.\n"
    "/study — объяснение вопросов (режим по умолчанию)\n"
    "/translate — перевод на указанный вами язык\n"
    "/quiz — вопросы для тренировки по теме\n"
    "/settings — режим, модель и температура; /settings 0.3 — изменить температуру\n"
    "/reset — очистить историю диалога\n"
    "/tools — узнать, какие действия доступны\n"
    "/timezone Europe/Moscow — задать часовой пояс\n"
    "/city город, страна — выбрать город для сводки\n"
    "/briefing_on и /briefing_off — включить или выключить сводку\n"
    "/notifications — настройки уведомлений\n"
    "/trace last или /trace ID — посмотреть свою трассу\n"
    "/why — посмотреть последнее решение агента\n"
    "/start — список команд"
)


async def _refresh_typing(bot: Bot, chat_id: int) -> None:
    while True:
        await asyncio.sleep(4)
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            # Сбой вторичного индикатора не отменяет формирование ответа.
            return


def split_reply(text: str) -> list[str]:
    return [
        text[index : index + MAX_TELEGRAM_CHARS]
        for index in range(0, len(text), MAX_TELEGRAM_CHARS)
    ]


def create_router(
    service: AssistantService, *, notifications: NotificationService | None = None
) -> Router:
    router = Router(name="assistant")

    @router.message(F.chat.type == "private", F.text)
    async def handle_text(message: Message, bot: Bot, event_update: Update) -> None:
        if message.from_user is None:
            return
        owner = message.from_user.id
        if notifications is not None:
            await notifications.register_private(owner, message.chat.id)
        keyboard = None
        raw = message.text or ""
        stripped = raw.lstrip()
        if stripped.startswith("/"):
            command, _, argument = stripped.partition(" ")
            command = command.split("@", 1)[0]
            argument = argument.strip()
            if command == "/start":
                reply = START_TEXT
            elif command in {"/study", "/translate", "/quiz"}:
                reply = await service.switch_mode(message.chat.id, command[1:])
            elif command == "/reset":
                reply = await service.reset(message.chat.id)
            elif command == "/settings":
                reply = (
                    await service.set_temperature(message.chat.id, argument)
                    if argument
                    else await service.settings_text(message.chat.id)
                )
            elif command == "/tools":
                reply = await service.agent.tools_text() if service.agent else UNAVAILABLE_MESSAGE
            elif command == "/timezone":
                reply = (
                    await service.agent.timezone_text(owner, argument)
                    if service.agent
                    else UNAVAILABLE_MESSAGE
                )
            elif command == "/why":
                reply = await service.agent.why(owner) if service.agent else UNAVAILABLE_MESSAGE
            elif command == "/trace":
                reply = (
                    await service.agent.trace(owner, argument)
                    if service.agent
                    else UNAVAILABLE_MESSAGE
                )
            elif command == "/city" and notifications is not None:
                try:
                    proposal = await notifications.prepare_city(
                        owner, message.chat.id, event_update.update_id, argument
                    )
                    reply = f"Выберите город: {proposal.label}. Подтвердить выбор?"
                    keyboard = _city_buttons(proposal.action_id)
                except ToolError as exc:
                    reply = {
                        "invalid_city": "Укажите город и страну: /city Казань, Россия.",
                        "city_not_found": "Город не найден. Укажите город и страну полностью.",
                        "city_ambiguous": "Найдено несколько городов. Уточните название и страну.",
                        "city_expired": "Подтверждение устарело. Повторите /city.",
                    }.get(exc.code, "Геокодирование недоступно. Повторите попытку позже.")
            elif command in {"/briefing_on", "/briefing_off"} and notifications is not None:
                reply = await notifications.set_briefing(owner, command == "/briefing_on")
            elif command == "/notifications" and notifications is not None:
                reply = await notifications.settings_text(owner)
            else:
                reply = "Неизвестная команда. Отправьте /start для списка команд."
        else:
            await bot.send_chat_action(chat_id=message.chat.id, action="typing")
            typing_task = asyncio.create_task(_refresh_typing(bot, message.chat.id))
            try:
                response = await service.answer_update(
                    message.chat.id, raw, owner_id=owner, update_id=event_update.update_id
                )
                reply = response.text
                if response.confirmation is not None:
                    keyboard = _reminder_buttons(response.confirmation)
            finally:
                typing_task.cancel()
                await asyncio.gather(typing_task, return_exceptions=True)
        parts = split_reply(reply)
        for index, part in enumerate(parts, 1):
            await message.answer(
                part, parse_mode=None, reply_markup=keyboard if index == len(parts) else None
            )

    @router.callback_query(F.data.startswith("reminder:"))
    async def handle_reminder(query: CallbackQuery) -> None:
        message = query.message
        if not isinstance(message, Message) or message.chat.type != "private" or not service.agent:
            await query.answer("Действие недоступно.")
            return
        try:
            _, verb, raw_id = (query.data or "").split(":")
            action_id = UUID(hex=raw_id)
            if verb not in {"yes", "no"} or raw_id != action_id.hex:
                raise ValueError
        except ValueError:
            await query.answer("Действие недоступно.")
            return
        # Запись может завершиться и после потери Telegram-ответа на callback.
        with suppress(TelegramBadRequest):
            await query.answer()
        reply = await service.agent.confirm(query.from_user.id, action_id, accepted=verb == "yes")
        await message.answer(reply.text, parse_mode=None)
        if reply.close_buttons:
            with suppress(TelegramBadRequest):
                await message.edit_reply_markup(reply_markup=None)

    @router.callback_query(F.data.startswith("city:"))
    async def handle_city(query: CallbackQuery) -> None:
        message = query.message
        if (
            not isinstance(message, Message)
            or message.chat.type != "private"
            or notifications is None
        ):
            await query.answer("Действие недоступно.")
            return
        try:
            _, verb, raw_id = (query.data or "").split(":")
            action_id = UUID(hex=raw_id)
            if verb not in {"yes", "no"} or raw_id != action_id.hex:
                raise ValueError
        except ValueError:
            await query.answer("Действие недоступно.")
            return
        accepted = verb == "yes"
        saved = await notifications.confirm_city(
            query.from_user.id, message.chat.id, action_id, accepted=accepted
        )
        with suppress(TelegramBadRequest):
            await query.answer()
        if not saved:
            await message.answer("Подтверждение недоступно или устарело. Повторите /city.")
            return
        await message.answer("Город сохранён." if accepted else "Выбор города отменён.")
        with suppress(TelegramBadRequest):
            await message.edit_reply_markup(reply_markup=None)

    return router


def _reminder_buttons(action_id: UUID) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Подтвердить", callback_data=f"reminder:yes:{action_id.hex}"
                ),
                InlineKeyboardButton(text="Отменить", callback_data=f"reminder:no:{action_id.hex}"),
            ]
        ]
    )


def _city_buttons(action_id: UUID) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Подтвердить", callback_data=f"city:yes:{action_id.hex}"),
                InlineKeyboardButton(text="Отменить", callback_data=f"city:no:{action_id.hex}"),
            ]
        ]
    )
