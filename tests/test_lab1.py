"""Поведение лабораторной № 1 без Telegram API и LLM-провайдера."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.config import Settings
from app.context import build_messages
from app.handlers.assistant import START_TEXT, create_router
from app.llm import LLMError, LLMInvalidResponse, LLMResult, LLMTimeout, OpenRouterChatClient
from app.prompts import MODE_PROMPTS
from app.service import AssistantService
from app.store import HistoryMessage, UserState

PRIMARY = "google/gemma-4-26b-a4b-it:free"
RESERVE = "nvidia/nemotron-3.5-lightning:free"


class FakeStore:
    def __init__(self):
        self.states: dict[int, UserState] = {}
        self.histories: dict[int, list[HistoryMessage]] = {}

    def state(self, chat_id: int) -> UserState:
        return self.states.setdefault(chat_id, UserState())

    async def get_context(self, chat_id: int, limit: int):
        return self.state(chat_id), self.histories.get(chat_id, [])[-limit:].copy()

    async def set_mode(self, chat_id: int, mode: str) -> UserState:
        state = replace(
            self.state(chat_id), mode=mode, generation=self.state(chat_id).generation + 1
        )
        self.states[chat_id] = state
        self.histories[chat_id] = []
        return state

    async def reset_history(self, chat_id: int) -> UserState:
        state = replace(self.state(chat_id), generation=self.state(chat_id).generation + 1)
        self.states[chat_id] = state
        self.histories[chat_id] = []
        return state

    async def set_temperature(self, chat_id: int, temperature: float) -> UserState:
        state = replace(self.state(chat_id), temperature=temperature)
        self.states[chat_id] = state
        return state

    async def save_result(
        self, chat_id: int, generation: int, user_text: str, assistant_text: str | None
    ) -> bool:
        if self.state(chat_id).generation != generation:
            return False
        history = self.histories.setdefault(chat_id, [])
        history.append(HistoryMessage("user", user_text))
        if assistant_text is not None:
            history.append(HistoryMessage("assistant", assistant_text))
        return True


class FakeLLM:
    def __init__(self, *responses: str | Exception):
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, str]], float, str]] = []

    async def generate(self, messages, temperature, request_id):
        self.calls.append((messages, temperature, request_id))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResult(response, PRIMARY, "synthetic-provider", None, request_id)


@pytest.fixture
def service_fixture():
    store = FakeStore()
    llm = FakeLLM("Первый ответ.", "Второй ответ.", "Третий ответ.")
    settings = Settings(
        bot_token="unused", postgres_password="unused", openrouter_models=(PRIMARY, RESERVE)
    )
    return AssistantService(store, llm, settings), store, llm


async def test_study_answer_uses_system_role_and_chronological_history(service_fixture):
    service, store, llm = service_fixture

    assert await service.answer(101, "Первый вопрос") == "Первый ответ."
    assert llm.calls[0][0] == [
        {"role": "system", "content": MODE_PROMPTS["study"]},
        {"role": "user", "content": "Первый вопрос"},
    ]
    assert llm.calls[0][1] == 0.3
    assert llm.calls[0][2]
    assert await service.answer(101, "Второй вопрос") == "Второй ответ."
    assert llm.calls[1][0] == [
        {"role": "system", "content": MODE_PROMPTS["study"]},
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ."},
        {"role": "user", "content": "Второй вопрос"},
    ]
    assert store.histories[101][-1] == HistoryMessage("assistant", "Второй ответ.")


async def test_two_chats_have_independent_history_and_settings(service_fixture):
    service, store, llm = service_fixture

    await service.answer(101, "Секретный вопрос")
    await service.set_temperature(202, "0.7")
    assert await service.answer(202, "Другой вопрос") == "Второй ответ."

    assert llm.calls[1][0] == [
        {"role": "system", "content": MODE_PROMPTS["study"]},
        {"role": "user", "content": "Другой вопрос"},
    ]
    assert llm.calls[1][1] == 0.7
    assert store.state(101).temperature == 0.3
    assert store.histories[101] != store.histories[202]


async def test_switch_mode_clears_old_history_and_uses_new_prompt(service_fixture):
    service, store, llm = service_fixture
    await service.answer(101, "Вопрос по Python")

    reply = await service.switch_mode(101, "translate")

    assert reply == "Режим /translate включён. История очищена."
    assert store.state(101).mode == "translate"
    assert store.histories[101] == []
    assert len(llm.calls) == 1
    await service.answer(101, "Переведи на английский: привет")
    assert llm.calls[1][0] == [
        {"role": "system", "content": MODE_PROMPTS["translate"]},
        {"role": "user", "content": "Переведи на английский: привет"},
    ]


async def test_reset_clears_only_owner_history_and_preserves_settings(service_fixture):
    service, store, _llm = service_fixture
    await service.switch_mode(101, "quiz")
    await service.set_temperature(101, "1.0")
    await service.answer(101, "Словари")
    await service.answer(202, "Циклы")
    other_history = store.histories[202].copy()

    reply = await service.reset(101)

    assert reply == "История очищена. Режим и температура сохранены."
    assert store.histories[101] == []
    assert store.histories[202] == other_history
    assert store.state(101).mode == "quiz"
    assert store.state(101).temperature == 1.0


async def test_settings_accepts_only_four_values_and_next_request_uses_new_value(service_fixture):
    service, store, llm = service_fixture

    settings_text = await service.settings_text(101)
    assert "Режим: /study" in settings_text
    assert PRIMARY in settings_text
    assert "резерв" in settings_text.lower()
    assert "Температура: 0.3" in settings_text
    assert await service.set_temperature(101, "0.7") == (
        "Температура установлена: 0.7. Она действует со следующего запроса."
    )
    assert store.state(101).temperature == 0.7
    assert len(llm.calls) == 0
    await service.answer(101, "Вопрос")
    assert llm.calls[0][1] == 0.7

    for bad in ("0.2", "0,3", "nan", "1.00", "", "0.7 extra"):
        assert await service.set_temperature(101, bad) == (
            "Допустимые значения: 0.0, 0.3, 0.7, 1.0. Пример: /settings 0.3."
        )
        assert store.state(101).temperature == 0.7
    assert len(llm.calls) == 1


@pytest.mark.parametrize("error", [LLMTimeout(), LLMError(), LLMInvalidResponse()])
async def test_model_failure_is_safe_and_has_no_assistant_history(error):
    store = FakeStore()
    llm = FakeLLM(error)
    settings = Settings(
        bot_token="unused", postgres_password="unused", openrouter_models=(PRIMARY, RESERVE)
    )
    service = AssistantService(store, llm, settings)

    reply = await service.answer(101, "Проверь запрос")

    assert reply == error.user_message
    assert store.histories[101] == [HistoryMessage("user", "Проверь запрос")]
    assert "Проверь запрос" not in reply


async def test_empty_model_text_is_rejected_before_telegram_delivery():
    store = FakeStore()
    llm = FakeLLM("  ")
    settings = Settings(
        bot_token="unused", postgres_password="unused", openrouter_models=(PRIMARY, RESERVE)
    )
    service = AssistantService(store, llm, settings)

    assert await service.answer(101, "Вопрос") == LLMInvalidResponse.user_message
    assert store.histories[101] == [HistoryMessage("user", "Вопрос")]


async def test_oversized_current_request_is_rejected_without_model_call():
    store = FakeStore()
    llm = FakeLLM("Ответ не должен использоваться")
    settings = Settings(
        bot_token="unused",
        postgres_password="unused",
        openrouter_models=(PRIMARY, RESERVE),
        context_max_chars=1000,
    )
    service = AssistantService(store, llm, settings)

    assert await service.answer(101, "А" * 1000) == (
        "Запрос слишком длинный. Сократите его и повторите."
    )
    assert llm.calls == []
    assert store.histories.get(101, []) == []


async def test_reset_in_another_worker_cannot_restore_inflight_history():
    class DelayedLLM:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def generate(self, messages, temperature, request_id):
            self.started.set()
            await self.release.wait()
            return LLMResult("Поздний ответ", PRIMARY, None, None, request_id)

    store = FakeStore()
    llm = DelayedLLM()
    settings = Settings(
        bot_token="unused", postgres_password="unused", openrouter_models=(PRIMARY, RESERVE)
    )
    service = AssistantService(store, llm, settings)
    request = asyncio.create_task(service.answer(101, "Старый вопрос"))
    await asyncio.wait_for(llm.started.wait(), timeout=1)

    await store.reset_history(101)
    llm.release.set()
    await asyncio.wait_for(request, timeout=1)

    assert store.histories[101] == []


async def _feed_message(service: AssistantService, text: str, chat_type: str = "private"):
    bot = Bot("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk")
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router(service))
    message = Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=101, type=chat_type),
        from_user=User(id=101, is_bot=False, first_name="Студент"),
        text=text,
    )
    await dispatcher.feed_update(bot, Update(update_id=1, message=message))
    return [call.args[1] for call in bot.session.call_args_list]


async def test_long_reply_is_sent_in_order_without_losing_characters():
    long_reply = "А" * 4095 + "👋" + "Б" * 4096 + "\nконец"
    store = FakeStore()
    llm = FakeLLM(long_reply)
    settings = Settings(
        bot_token="unused", postgres_password="unused", openrouter_models=(PRIMARY, RESERVE)
    )
    service = AssistantService(store, llm, settings)

    calls = await _feed_message(service, "Дай длинный ответ")

    sent = [call for call in calls if isinstance(call, SendMessage)]
    assert len(sent) == 3
    assert all(0 < len(call.text) <= 4096 and call.parse_mode is None for call in sent)
    assert "".join(call.text for call in sent) == long_reply
    assert store.histories[101][-1] == HistoryMessage("assistant", long_reply)


async def test_commands_do_not_call_model_and_nonprivate_messages_are_ignored(service_fixture):
    service, _store, llm = service_fixture

    start_calls = await _feed_message(service, "/start")
    group_calls = await _feed_message(service, "Вопрос", chat_type="group")

    assert [call.text for call in start_calls if isinstance(call, SendMessage)] == [START_TEXT]
    assert group_calls == []
    assert llm.calls == []


async def test_telegram_commands_change_settings_without_reaching_model(service_fixture):
    service, store, llm = service_fixture

    mode_calls = await _feed_message(service, "/quiz")
    temperature_calls = await _feed_message(service, "/settings 1.0")
    unknown_calls = await _feed_message(service, "/unknown")

    assert [call.text for call in mode_calls if isinstance(call, SendMessage)] == [
        "Режим /quiz включён. История очищена."
    ]
    assert [call.text for call in temperature_calls if isinstance(call, SendMessage)] == [
        "Температура установлена: 1.0. Она действует со следующего запроса."
    ]
    assert [call.text for call in unknown_calls if isinstance(call, SendMessage)] == [
        "Неизвестная команда. Отправьте /start для списка команд."
    ]
    assert store.state(101).mode == "quiz"
    assert store.state(101).temperature == 1.0
    assert llm.calls == []


def test_context_keeps_recent_complete_turns_under_message_limit():
    history = [
        HistoryMessage("user", "старый вопрос"),
        HistoryMessage("assistant", "старый ответ"),
        HistoryMessage("user", "новый вопрос"),
        HistoryMessage("assistant", "новый ответ"),
    ]

    messages = build_messages(
        "Инструкция",
        history,
        "Текущий вопрос",
        max_history_messages=2,
        max_context_chars=1000,
    )

    assert messages == [
        {"role": "system", "content": "Инструкция"},
        {"role": "user", "content": "новый вопрос"},
        {"role": "assistant", "content": "новый ответ"},
        {"role": "user", "content": "Текущий вопрос"},
    ]


def test_context_keeps_recent_complete_turns_under_character_budget():
    history = [
        HistoryMessage("user", "старый"),
        HistoryMessage("assistant", "ответ"),
        HistoryMessage("user", "новый"),
        HistoryMessage("assistant", "ответ"),
    ]

    messages = build_messages(
        "Правила",
        history,
        "запрос",
        max_history_messages=10,
        max_context_chars=120,
    )

    assert messages == [
        {"role": "system", "content": "Правила"},
        {"role": "user", "content": "новый"},
        {"role": "assistant", "content": "ответ"},
        {"role": "user", "content": "запрос"},
    ]


def test_context_rejects_instruction_and_current_request_over_budget():
    with pytest.raises(ValueError):
        build_messages(
            "Правила",
            [],
            "длинный запрос",
            max_history_messages=10,
            max_context_chars=8,
        )


class FakeHTTPResponse:
    def __init__(self, status: int, payload: object):
        self.status = status
        self.payload = payload
        self.headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        pass

    async def json(self, **_kwargs):
        return self.payload


class FakeHTTPSession:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = (
            payload
            if payload is not None
            else {
                "model": PRIMARY,
                "choices": [
                    {"message": {"content": "Синтетический ответ"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            }
        )
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeHTTPResponse(self.status, self.payload)

    async def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("status", "payload", "expected_error"),
    [
        (503, {"error": "secret-from-server"}, LLMError),
        (200, {"model": PRIMARY, "choices": []}, LLMInvalidResponse),
        (200, {"model": PRIMARY, "choices": [{"message": {"content": "  "}}]}, LLMInvalidResponse),
        (
            200,
            {
                "model": PRIMARY,
                "choices": [{"message": {"content": "обрыв"}, "finish_reason": "length"}],
            },
            LLMInvalidResponse,
        ),
    ],
)
async def test_http_adapter_rejects_failure_and_empty_results(
    monkeypatch, caplog, status, payload, expected_error
):
    session = FakeHTTPSession(status=status, payload=payload)
    monkeypatch.setattr("app.llm.aiohttp.ClientSession", lambda **_kwargs: session)
    settings = Settings(
        bot_token="unused",
        postgres_password="unused",
        openrouter_api_key="synthetic-key",
        openrouter_models=(PRIMARY,),
    )
    async with OpenRouterChatClient(settings) as client:
        with pytest.raises(expected_error):
            await client.generate(
                [{"role": "user", "content": "synthetic-dialog-secret"}],
                0.3,
                "synthetic-id",
            )
    assert "synthetic-key" not in caplog.text
    assert "synthetic-dialog-secret" not in caplog.text
    assert "secret-from-server" not in caplog.text


async def test_http_adapter_maps_timeout_without_logging_dialogue(monkeypatch, caplog):
    session = FakeHTTPSession()

    def fail_post(*_args, **_kwargs):
        raise TimeoutError("synthetic-dialog-secret")

    session.post = fail_post
    monkeypatch.setattr("app.llm.aiohttp.ClientSession", lambda **_kwargs: session)
    settings = Settings(
        bot_token="unused",
        postgres_password="unused",
        openrouter_api_key="synthetic-key",
        openrouter_models=(PRIMARY,),
    )

    async with OpenRouterChatClient(settings) as client:
        with pytest.raises(LLMTimeout):
            await client.generate(
                [{"role": "user", "content": "synthetic-dialog-secret"}], 0.3, "synthetic-id"
            )
    assert "synthetic-dialog-secret" not in caplog.text
    assert "synthetic-key" not in caplog.text


async def test_experiment_resets_history_for_each_real_call_path(tmp_path, monkeypatch):
    from scripts import lab1_experiment as experiment

    class ExperimentStore:
        def __init__(self):
            self.resets = 0
            self.temperatures = []

        async def initialize(self):
            pass

        async def set_mode(self, _chat_id, _mode):
            pass

        async def set_temperature(self, _chat_id, temperature):
            self.temperatures.append(temperature)

        async def reset_history(self, _chat_id):
            self.resets += 1

    class ExperimentLLM:
        def __init__(self, _settings):
            self.calls = []
            self.provider = "Google AI Studio"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            pass

        async def generate(self, messages, temperature, request_id, *, model, endpoint):
            self.calls.append((messages, temperature, model, endpoint))
            return LLMResult(
                f"Ответ {len(self.calls)}",
                model,
                self.provider,
                {"total_tokens": 7},
                request_id,
            )

    store = ExperimentStore()
    llm = ExperimentLLM(None)
    pool = AsyncMock()
    monkeypatch.setattr(experiment, "create_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(experiment, "PostgresStore", lambda _pool: store)
    monkeypatch.setattr(experiment, "OpenRouterChatClient", lambda _settings: llm)
    path = tmp_path / "experiment.json"
    settings = Settings(
        bot_token="unused", postgres_password="unused", openrouter_models=(PRIMARY, RESERVE)
    )

    await experiment.run(settings, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert store.resets == 14
    assert store.temperatures == [0.0, 0.3, 0.7, 1.0, 0.3]
    assert [call[1] for call in llm.calls] == (
        [0.0, 0.3, 0.7, 1.0] + [0.0] * 3 + [0.3] * 3 + [0.7] * 3 + [1.0] * 3 + [0.3] * 2
    )
    assert len(llm.calls) == 18
    assert all(call[2:] == (PRIMARY, "google-ai-studio") for call in llm.calls)
    assert [item["role"] for item in llm.calls[0][0]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert len(data["preflight"]) == 4
    assert [item["temperature"] for item in data["preflight"]] == [0.0, 0.3, 0.7, 1.0]
    assert all(item["model"] == PRIMARY for item in data["preflight"])
    assert all(item["provider"] == "Google AI Studio" for item in data["preflight"])
    assert len(data["runs"]) == 12
    assert len(data["comparison"]) == 2
    assert data["runs"][0]["usage"] == {"total_tokens": 7}
    assert data["runs"][0]["model"] == PRIMARY
    assert data["runs"][0]["provider"] == "Google AI Studio"
    assert data["runs"][0]["response"] == "Ответ 5"
    assert data["comparison"][0]["model"] == PRIMARY
    assert data["comparison"][0]["provider"] == "Google AI Studio"
    assert data["constant_settings"]["provider_fallbacks"] is False


async def test_experiment_route_change_stops_before_recording_series(tmp_path, monkeypatch):
    from scripts import lab1_experiment as experiment

    class ChangedRouteLLM:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            pass

        async def generate(self, _messages, _temperature, request_id, *, model, endpoint):
            assert model == PRIMARY
            assert endpoint == "google-ai-studio"
            return LLMResult("Не использовать в серии", model, "Other Provider", None, request_id)

    create_pool = AsyncMock()
    monkeypatch.setattr(experiment, "create_pool", create_pool)
    monkeypatch.setattr(experiment, "OpenRouterChatClient", lambda _settings: ChangedRouteLLM())
    path = tmp_path / "route-changed.json"
    settings = Settings(bot_token="unused", postgres_password="unused")

    with pytest.raises(experiment.ExperimentRouteError):
        await experiment.run(settings, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["route_mismatch"]["actual_provider"] == "Other Provider"
    assert data["preflight"] == []
    assert data["runs"] == []
    create_pool.assert_not_awaited()
