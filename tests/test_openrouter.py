"""Контракт HTTP-адаптера OpenRouter без сетевых запросов."""

import asyncio
import logging
from collections.abc import Mapping

import pytest

from app.config import Settings
from app.llm import LLMError, LLMResult, LLMTimeout, OpenRouterChatClient

PRIMARY = "google/gemma-4-26b-a4b-it:free"
RESERVE = "nvidia/nemotron-3.5-lightning:free"
ROUTE = "nvidia/nvfp4"
URL = "https://openrouter.ai/api/v1/chat/completions"
DIALOGUE = [
    {"role": "system", "content": "Синтетическая инструкция"},
    {"role": "user", "content": "Первый вопрос"},
    {"role": "assistant", "content": "Первый ответ"},
    {"role": "user", "content": "Второй вопрос"},
]


def settings(*, models: tuple[str, ...] = (PRIMARY, RESERVE)) -> Settings:
    return Settings(
        bot_token="unused",
        postgres_password="unused",
        openrouter_api_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key="synthetic-key",
        openrouter_models=models,
        llm_timeout_seconds=2,
    )


def completion(
    *,
    model: str = PRIMARY,
    provider: str = "google-ai-studio",
    content: str = "Синтетический ответ",
    usage: dict[str, int] | None = None,
) -> dict:
    return {
        "id": "gen-synthetic",
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        "openrouter_metadata": {
            "endpoints": {
                "available": [
                    {"provider": "unused-provider", "selected": False},
                    {"provider": provider, "selected": True},
                ]
            }
        },
    }


class FakeResponse:
    def __init__(
        self,
        status: int,
        payload: object,
        *,
        headers: Mapping[str, str] | None = None,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        delay_seconds: float = 0,
    ):
        self.status = status
        self.payload = payload
        self.headers = dict(headers or {})
        self.entered = entered
        self.release = release
        self.delay_seconds = delay_seconds

    async def __aenter__(self):
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.entered is not None:
            self.entered.set()
        return self

    async def __aexit__(self, *_exc):
        pass

    async def json(self, **_kwargs):
        if self.release is not None:
            await self.release.wait()
        return self.payload

    async def text(self):
        return "synthetic-error-body-secret"


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses.copy()
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def post(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        assert self.responses, "Неожиданная дополнительная HTTP-попытка"
        return self.responses.pop(0)

    async def close(self):
        self.closed = True


def fake_session(monkeypatch, *responses: FakeResponse) -> FakeSession:
    session = FakeSession(list(responses))
    monkeypatch.setattr("app.llm.aiohttp.ClientSession", lambda **_kwargs: session)
    return session


def assert_free_request(call: tuple[str, dict], models: tuple[str, ...]) -> dict:
    url, kwargs = call
    assert url == URL
    assert kwargs["headers"]["Authorization"] == "Bearer synthetic-key"
    assert kwargs["headers"]["X-OpenRouter-Metadata"] == "enabled"
    payload = kwargs["json"]
    assert payload["model"] in models
    assert payload["model"].endswith(":free")
    assert "models" not in payload
    assert payload["provider"]["require_parameters"] is True
    assert payload["provider"]["max_price"] == {
        "prompt": 0,
        "completion": 0,
        "request": 0,
    }
    return payload


async def test_roles_temperatures_and_per_request_metadata(monkeypatch):
    session = fake_session(monkeypatch, *(FakeResponse(200, completion()) for _ in range(4)))

    async with OpenRouterChatClient(settings()) as client:
        for temperature in (0.0, 0.3, 0.7, 1.0):
            result = await client.generate(DIALOGUE, temperature, f"req-{temperature}")
            assert isinstance(result, LLMResult)
            assert result.text == "Синтетический ответ"
            assert result.model == PRIMARY
            assert result.provider == "google-ai-studio"
            assert result.usage == {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            }
            assert result.request_id == f"req-{temperature}"

    assert session.closed
    assert len(session.calls) == 4
    for temperature, call in zip((0.0, 0.3, 0.7, 1.0), session.calls, strict=True):
        payload = assert_free_request(call, (PRIMARY, RESERVE))
        assert payload["messages"] == DIALOGUE
        assert type(payload["temperature"]) is float
        assert payload["temperature"] == temperature


@pytest.mark.parametrize(
    ("status", "headers"),
    [
        (503, {}),
        (429, {"Retry-After": "0"}),
        (429, {"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "20"}),
    ],
)
async def test_temporary_endpoint_failure_moves_to_free_reserve(monkeypatch, status, headers):
    session = fake_session(
        monkeypatch,
        FakeResponse(
            status,
            {"error": {"code": status, "metadata": {"provider_name": "google-ai-studio"}}},
            headers=headers,
        ),
        FakeResponse(200, completion(model=RESERVE, provider=ROUTE)),
    )

    async with OpenRouterChatClient(settings()) as client:
        result = await client.generate(DIALOGUE, 0.3, "req-fallback")

    assert result.model == RESERVE
    assert result.provider == ROUTE
    assert result.request_id == "req-fallback"
    assert [assert_free_request(call, (PRIMARY, RESERVE))["model"] for call in session.calls] == [
        PRIMARY,
        RESERVE,
    ]


async def test_no_allowed_provider_404_moves_to_free_reserve(monkeypatch):
    session = fake_session(
        monkeypatch,
        FakeResponse(
            404,
            {
                "error": {"code": 404},
                "openrouter_metadata": {
                    "requested": PRIMARY,
                    "attempt": 0,
                    "endpoints": {
                        "total": 1,
                        "available": [{"provider": "Google AI Studio", "selected": False}],
                    },
                },
            },
        ),
        FakeResponse(200, completion(model=RESERVE, provider=ROUTE)),
    )

    async with OpenRouterChatClient(settings()) as client:
        result = await client.generate(DIALOGUE, 0.3, "req-no-provider")

    assert result.model == RESERVE
    assert [call[1]["json"]["model"] for call in session.calls] == [PRIMARY, RESERVE]


async def test_unknown_model_404_does_not_switch(monkeypatch):
    session = fake_session(monkeypatch, FakeResponse(404, {"error": {"code": 404}}))

    async with OpenRouterChatClient(settings()) as client:
        with pytest.raises(LLMError):
            await client.generate(DIALOGUE, 0.3, "req-unknown-model")

    assert len(session.calls) == 1


async def test_exhausted_free_pool_returns_safe_error(monkeypatch):
    session = fake_session(
        monkeypatch,
        FakeResponse(503, {"error": {"message": "synthetic-error-body-secret"}}),
        FakeResponse(503, {"error": {"message": "synthetic-error-body-secret"}}),
    )

    async with OpenRouterChatClient(settings()) as client:
        with pytest.raises(LLMError) as error:
            await client.generate(DIALOGUE, 0.3, "req-exhausted")

    assert "synthetic-error-body-secret" not in error.value.user_message
    assert len(session.calls) == 2
    assert [assert_free_request(call, (PRIMARY, RESERVE))["model"] for call in session.calls] == [
        PRIMARY,
        RESERVE,
    ]


async def test_long_retry_after_does_not_block_another_free_provider(monkeypatch):
    session = fake_session(
        monkeypatch,
        FakeResponse(429, {"error": {"code": 429}}, headers={"Retry-After": "2"}),
        FakeResponse(200, completion(model=RESERVE, provider=ROUTE)),
    )
    short_timeout = Settings(
        bot_token="unused",
        postgres_password="unused",
        openrouter_api_key="synthetic-key",
        openrouter_models=(PRIMARY, RESERVE),
        llm_timeout_seconds=1,
    )

    async with OpenRouterChatClient(short_timeout) as client:
        result = await client.generate(DIALOGUE, 0.3, "req-retry-after")

    assert result.model == RESERVE
    assert len(session.calls) == 2


@pytest.mark.parametrize(
    ("error_type", "expected_calls"),
    [("provider_unavailable", 2), ("content_policy_violation", 1)],
)
async def test_error_inside_http_200_uses_typed_classification(
    monkeypatch, error_type, expected_calls
):
    session = fake_session(
        monkeypatch,
        FakeResponse(200, {"error": {"code": 503, "metadata": {"error_type": error_type}}}),
        FakeResponse(200, completion(model=RESERVE, provider=ROUTE)),
    )

    async with OpenRouterChatClient(settings()) as client:
        if expected_calls == 1:
            with pytest.raises(LLMError):
                await client.generate(DIALOGUE, 0.3, "req-embedded-error")
        else:
            result = await client.generate(DIALOGUE, 0.3, "req-embedded-error")
            assert result.model == RESERVE

    assert len(session.calls) == expected_calls


@pytest.mark.parametrize(
    ("status", "payload", "headers"),
    [
        (401, {"error": {"code": 401, "message": "invalid key"}}, {}),
        (400, {"error": {"code": 400, "message": "invalid parameter"}}, {}),
        (403, {"error": {"code": 403, "message": "moderated"}}, {}),
        (
            429,
            {"error": {"code": 429, "message": "account quota exhausted"}},
            {"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "0", "Retry-After": "0"},
        ),
    ],
)
async def test_terminal_errors_never_switch_model(monkeypatch, caplog, status, payload, headers):
    session = fake_session(monkeypatch, FakeResponse(status, payload, headers=headers))

    with caplog.at_level(logging.INFO):
        async with OpenRouterChatClient(settings()) as client:
            with pytest.raises(LLMError):
                await client.generate(DIALOGUE, 0.3, "req-terminal")

    assert len(session.calls) == 1
    assert assert_free_request(session.calls[0], (PRIMARY, RESERVE))["model"] == PRIMARY
    assert "synthetic-key" not in caplog.text
    assert "Синтетическая инструкция" not in caplog.text
    assert "invalid key" not in caplog.text
    assert "invalid parameter" not in caplog.text
    assert "moderated" not in caplog.text


async def test_response_uses_actual_model_and_selected_provider(monkeypatch):
    fake_session(monkeypatch, FakeResponse(200, completion(model=RESERVE, provider=ROUTE)))

    async with OpenRouterChatClient(settings()) as client:
        result = await client.generate(DIALOGUE, 0.7, "req-actual")

    assert result.model == RESERVE
    assert result.provider == ROUTE


async def test_pinned_experiment_model_and_endpoint_disable_fallbacks(monkeypatch):
    session = fake_session(
        monkeypatch,
        FakeResponse(200, completion(model=RESERVE, provider=ROUTE)),
    )

    async with OpenRouterChatClient(settings()) as client:
        result = await client.generate(DIALOGUE, 0.3, "req-pinned", model=RESERVE, endpoint=ROUTE)

    assert result.model == RESERVE
    assert len(session.calls) == 1
    payload = assert_free_request(session.calls[0], (PRIMARY, RESERVE))
    assert payload["model"] == RESERVE
    assert payload["provider"]["only"] == [ROUTE]
    assert payload["provider"]["order"] == [ROUTE]
    assert payload["provider"]["allow_fallbacks"] is False


async def test_pinned_experiment_failure_does_not_switch_model(monkeypatch):
    session = fake_session(monkeypatch, FakeResponse(503, {"error": {"code": 503}}))

    async with OpenRouterChatClient(settings()) as client:
        with pytest.raises(LLMError):
            await client.generate(DIALOGUE, 0.3, "req-pinned-error", model=RESERVE, endpoint=ROUTE)

    assert len(session.calls) == 1
    payload = assert_free_request(session.calls[0], (PRIMARY, RESERVE))
    assert payload["model"] == RESERVE
    assert payload["provider"]["allow_fallbacks"] is False


async def test_all_fallback_attempts_share_one_timeout(monkeypatch):
    session = fake_session(
        monkeypatch,
        FakeResponse(503, {"error": {"code": 503}}, delay_seconds=0.7),
        FakeResponse(200, completion(model=RESERVE, provider=ROUTE), delay_seconds=0.7),
    )
    short_timeout = Settings(
        bot_token="unused",
        postgres_password="unused",
        openrouter_api_key="synthetic-key",
        openrouter_models=(PRIMARY, RESERVE),
        llm_timeout_seconds=1,
    )

    async with OpenRouterChatClient(short_timeout) as client:
        with pytest.raises(LLMTimeout):
            await asyncio.wait_for(client.generate(DIALOGUE, 0.3, "req-timeout"), timeout=2)

    assert len(session.calls) == 2


async def test_parallel_results_keep_their_own_metadata(monkeypatch):
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    fake_session(
        monkeypatch,
        FakeResponse(
            200,
            completion(model=PRIMARY, provider="google-ai-studio", content="Первый"),
            entered=first_entered,
            release=release_first,
        ),
        FakeResponse(200, completion(model=RESERVE, provider=ROUTE, content="Второй")),
    )

    async with OpenRouterChatClient(settings()) as client:
        first_task = asyncio.create_task(client.generate(DIALOGUE, 0.0, "req-first"))
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        second = await client.generate(DIALOGUE, 1.0, "req-second")
        release_first.set()
        first = await asyncio.wait_for(first_task, timeout=1)

    assert (first.text, first.model, first.provider, first.request_id) == (
        "Первый",
        PRIMARY,
        "google-ai-studio",
        "req-first",
    )
    assert (second.text, second.model, second.provider, second.request_id) == (
        "Второй",
        RESERVE,
        ROUTE,
        "req-second",
    )
