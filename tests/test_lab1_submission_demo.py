"""Synthetic Telegram demonstration; opt in with RUN_LAB1_SUBMISSION_DEMO=1.

Starts an isolated local PostgreSQL cluster. No Telegram or model requests leave the test.
The test also runs against the historical lab-1 checkout, before AgentService existed.
"""

import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.config import Settings
from app.db import create_pool
from app.handlers.assistant import create_router
from app.llm import LLMResult, LLMTimeout
from app.prompts import MODE_PROMPTS
from app.service import AssistantService
from app.store import HistoryMessage, PostgresStore

pytestmark = pytest.mark.integration
LONG_REPLY = "А" * 4095 + "👋" + "Б" * 4096 + "\nконец"


@pytest.fixture
def demo_database():
    if os.environ.get("RUN_LAB1_SUBMISSION_DEMO") != "1":
        pytest.skip("Установите RUN_LAB1_SUBMISSION_DEMO=1 для временной PostgreSQL")
    initdb, pg_ctl = shutil.which("initdb"), shutil.which("pg_ctl")
    if not initdb or not pg_ctl or os.name != "posix" or os.getuid() == 0:
        pytest.fail("Требуются initdb, pg_ctl и запуск от обычного пользователя")
    with tempfile.TemporaryDirectory(prefix="itmo-demo-", dir="/tmp") as temporary:
        data = str(Path(temporary) / "data")
        subprocess.run(
            [initdb, "-D", data, "-A", "trust", "-U", "bot", "--no-locale", "-E", "UTF8"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        try:
            subprocess.run(
                [
                    pg_ctl,
                    "-D",
                    data,
                    "-l",
                    str(Path(temporary) / "postgres.log"),
                    "-w",
                    "-t",
                    "20",
                    "-o",
                    f"-c listen_addresses='' -k {temporary}",
                    "start",
                ],
                check=True,
                capture_output=True,
                timeout=25,
            )
            yield Settings(
                bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                postgres_password="unused-local-socket",
                postgres_host=temporary,
                postgres_db="postgres",
            )
        finally:
            if Path(data, "postmaster.pid").exists():
                subprocess.run(
                    [pg_ctl, "-D", data, "-w", "-t", "20", "-m", "immediate", "stop"],
                    check=True,
                    capture_output=True,
                    timeout=25,
                )


class SyntheticLLM:
    def __init__(self):
        self.dialogue_calls = []

    async def generate(self, messages, temperature, request_id, **kwargs):
        tools = kwargs.get("tools") or []
        if any(tool["function"]["name"] == "select_route" for tool in tools):
            from app.llm import NativeToolCall

            route = json.dumps({"target": "dialogue", "reason": "test", "confidence": 1.0})
            return LLMResult(
                "",
                "test/model",
                "test",
                None,
                request_id,
                tool_calls=(NativeToolCall("route", "select_route", route),),
            )
        self.dialogue_calls.append((messages, temperature))
        query = messages[-1]["content"]
        if query == "Проверка сбоя":
            raise LLMTimeout()
        text = LONG_REPLY if query == "Длинный ответ" else f"Ответ на запрос: {query}"
        return LLMResult(text, "test/model", "test", None, request_id)


def make_app(store, llm, settings):
    # Lab 1 has no agent; lab 2 has an agent but no SpecialistRouter yet.
    if (
        "agent" not in inspect.signature(AssistantService).parameters
        or importlib.util.find_spec("app.supervisor") is None
    ):
        service = AssistantService(store, llm, settings)
    else:
        from app.agent import AgentService
        from app.agent_router import AgentRouter
        from app.mcp_tools.contracts import tool_definitions
        from app.supervisor import SpecialistRouter

        mcp = Mock(available=True, tools=tool_definitions(), connect=AsyncMock(return_value=True))
        agent = AgentService(
            store, AgentRouter(llm), mcp, settings, specialist_router=SpecialistRouter(llm)
        )
        service = AssistantService(store, llm, settings, agent)
    bot = Bot("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk")
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router(service))
    return dispatcher, bot


async def send(dispatcher, bot, owner, update_id, text):
    bot.session.reset_mock()
    message = Message(
        message_id=update_id,
        date=datetime.now(UTC),
        chat=Chat(id=owner, type="private"),
        from_user=User(id=owner, is_bot=False, first_name="Тест"),
        text=text,
    )
    await dispatcher.feed_update(bot, Update(update_id=update_id, message=message))
    return [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]


async def test_two_users_restart_and_long_reply_through_telegram(demo_database):
    settings = demo_database
    llm = SyntheticLLM()
    pool = await create_pool(settings)
    try:
        store = PostgresStore(pool)
        await store.initialize()
        dispatcher, bot = make_app(store, llm, settings)
        await send(dispatcher, bot, 101, 1, "/quiz")
        await send(dispatcher, bot, 101, 2, "/settings 0.7")
        await send(dispatcher, bot, 101, 3, "Тема А")
        await send(dispatcher, bot, 202, 4, "Тема Б")
        first, first_history = await store.get_context(101, 12)
        second, second_history = await store.get_context(202, 12)
        assert (first.mode, first.temperature) == ("quiz", 0.7)
        assert (second.mode, second.temperature) == ("study", 0.3)
        assert first_history == [
            HistoryMessage("user", "Тема А"),
            HistoryMessage("assistant", "Ответ на запрос: Тема А"),
        ]
        assert second_history == [
            HistoryMessage("user", "Тема Б"),
            HistoryMessage("assistant", "Ответ на запрос: Тема Б"),
        ]

        # Replace the service, dispatcher, bot session and database pool, preserving only the DB.
        await pool.close()
        pool = await create_pool(settings)
        store = PostgresStore(pool)
        await store.initialize()
        dispatcher, bot = make_app(store, llm, settings)
        settings_a = await send(dispatcher, bot, 101, 5, "/settings")
        settings_b = await send(dispatcher, bot, 202, 6, "/settings")
        assert "/quiz" in settings_a[0].text and "0.7" in settings_a[0].text
        assert "/study" in settings_b[0].text and "0.3" in settings_b[0].text

        reply_a = await send(dispatcher, bot, 101, 7, "Продолжение А")
        assert [(part.chat_id, part.text) for part in reply_a] == [
            (101, "Ответ на запрос: Продолжение А")
        ]
        messages_a, temperature_a = llm.dialogue_calls[-1]
        assert messages_a[0]["content"].startswith(MODE_PROMPTS["quiz"])
        assert temperature_a == 0.7
        assert [item["content"] for item in messages_a[1:]] == [
            "Тема А",
            "Ответ на запрос: Тема А",
            "Продолжение А",
        ]
        reply_b = await send(dispatcher, bot, 202, 8, "Продолжение Б")
        assert [(part.chat_id, part.text) for part in reply_b] == [
            (202, "Ответ на запрос: Продолжение Б")
        ]
        messages_b, temperature_b = llm.dialogue_calls[-1]
        assert messages_b[0]["content"].startswith(MODE_PROMPTS["study"])
        assert temperature_b == 0.3
        assert [item["content"] for item in messages_b[1:]] == [
            "Тема Б",
            "Ответ на запрос: Тема Б",
            "Продолжение Б",
        ]
        before_reset_b = (await store.get_context(202, 12))[1]
        await send(dispatcher, bot, 101, 9, "/reset")
        state_a, history_a = await store.get_context(101, 12)
        assert (state_a.mode, state_a.temperature, history_a) == ("quiz", 0.7, [])
        assert (await store.get_context(202, 12))[1] == before_reset_b

        failure = await send(dispatcher, bot, 101, 10, "Проверка сбоя")
        assert [(part.chat_id, part.text) for part in failure] == [(101, LLMTimeout.user_message)]
        assert (await store.get_context(101, 12))[1] == [HistoryMessage("user", "Проверка сбоя")]
        assert (await store.get_context(202, 12))[1] == before_reset_b

        sent = await send(dispatcher, bot, 202, 11, "Длинный ответ")
        assert len(sent) == 3
        assert all(
            part.chat_id == 202 and 0 < len(part.text) <= 4096 and part.parse_mode is None
            for part in sent
        )
        assert "".join(part.text for part in sent) == LONG_REPLY
        assert (await store.get_context(202, 12))[1][-1] == HistoryMessage("assistant", LONG_REPLY)
    finally:
        await pool.close()
