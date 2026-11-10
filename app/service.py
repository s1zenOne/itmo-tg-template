"""Сценарии ассистента без зависимостей от Telegram и SQL."""

import asyncio
import secrets

from app.config import Settings
from app.context import ContextLimitError, build_messages
from app.llm import LLMError, LLMInvalidResponse, OpenRouterChatClient
from app.prompts import MODE_PROMPTS
from app.store import PostgresStore

TEMPERATURES = {"0.0": 0.0, "0.3": 0.3, "0.7": 0.7, "1.0": 1.0}
CHANGED_MESSAGE = "Настройки изменились во время ответа. Повторите запрос."


class AssistantService:
    def __init__(self, store: PostgresStore, llm: OpenRouterChatClient, settings: Settings):
        self.store = store
        self.llm = llm
        self.settings = settings
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock(self, chat_id: int) -> asyncio.Lock:
        return self._locks.setdefault(chat_id, asyncio.Lock())

    async def answer(self, chat_id: int, text: str) -> str:
        if not text.strip():
            return "Напишите текстовый вопрос."
        async with self._lock(chat_id):
            state, history = await self.store.get_context(
                chat_id, self.settings.history_max_messages * 4
            )
            try:
                messages = build_messages(
                    MODE_PROMPTS[state.mode],
                    history,
                    text,
                    max_history_messages=self.settings.history_max_messages,
                    max_context_chars=self.settings.context_max_chars,
                )
            except ContextLimitError:
                return "Запрос слишком длинный. Сократите его и повторите."
            try:
                result = await self.llm.generate(messages, state.temperature, secrets.token_hex(8))
                reply = result.text
                if not isinstance(reply, str) or not reply.strip():
                    raise LLMInvalidResponse
            except LLMError as exc:
                saved = await self.store.save_result(chat_id, state.generation, text, None)
                return exc.user_message if saved else CHANGED_MESSAGE
            saved = await self.store.save_result(chat_id, state.generation, text, reply)
            return reply if saved else CHANGED_MESSAGE

    async def switch_mode(self, chat_id: int, mode: str) -> str:
        if mode not in MODE_PROMPTS:
            raise ValueError("Неизвестный режим")
        async with self._lock(chat_id):
            await self.store.set_mode(chat_id, mode)
        return f"Режим /{mode} включён. История очищена."

    async def reset(self, chat_id: int) -> str:
        async with self._lock(chat_id):
            await self.store.reset_history(chat_id)
        return "История очищена. Режим и температура сохранены."

    async def settings_text(self, chat_id: int) -> str:
        async with self._lock(chat_id):
            state, _ = await self.store.get_context(chat_id, 0)
        return (
            f"Режим: /{state.mode}\n"
            f"Основная модель: {self.settings.openrouter_models[0]}\n"
            f"Бесплатный резерв: {'есть' if len(self.settings.openrouter_models) > 1 else 'нет'}\n"
            f"Температура: {state.temperature:.1f}"
        )

    async def set_temperature(self, chat_id: int, value: str) -> str:
        if value not in TEMPERATURES:
            return "Допустимые значения: 0.0, 0.3, 0.7, 1.0. Пример: /settings 0.3."
        async with self._lock(chat_id):
            await self.store.set_temperature(chat_id, TEMPERATURES[value])
        return f"Температура установлена: {value}. Она действует со следующего запроса."
