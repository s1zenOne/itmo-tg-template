"""Telegram-команды и доставка обычного текста."""

import asyncio

from aiogram import Bot, F, Router
from aiogram.types import Message

from app.service import AssistantService

MAX_TELEGRAM_CHARS = 4096
START_TEXT = (
    "Я помогу с программированием, переводом и тренировкой.\n"
    "/study — объяснение вопросов (режим по умолчанию)\n"
    "/translate — перевод на указанный вами язык\n"
    "/quiz — вопросы для тренировки по теме\n"
    "/settings — режим, модель и температура; /settings 0.3 — изменить температуру\n"
    "/reset — очистить историю диалога\n"
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
    async def handle_text(message: Message, bot: Bot) -> None:
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
            else:
                reply = "Неизвестная команда. Отправьте /start для списка команд."
        else:
            await bot.send_chat_action(chat_id=message.chat.id, action="typing")
            typing_task = asyncio.create_task(_refresh_typing(bot, message.chat.id))
            try:
                reply = await service.answer(message.chat.id, raw)
            finally:
                typing_task.cancel()
                await asyncio.gather(typing_task, return_exceptions=True)
        for part in split_reply(reply):
            await message.answer(part, parse_mode=None)

    return router
