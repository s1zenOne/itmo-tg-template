"""Telegram-команды и доставка обычного текста."""

import asyncio
from contextlib import suppress
from uuid import UUID

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update

from app.agent_router import UNAVAILABLE_MESSAGE
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


def create_router(service: AssistantService) -> Router:
    router = Router(name="assistant")

    @router.message(F.chat.type == "private", F.text)
    async def handle_text(message: Message, bot: Bot, event_update: Update) -> None:
        if message.from_user is None:
            return
        owner = message.from_user.id
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
