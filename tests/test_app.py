import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import DEFAULT_OPENROUTER_MODELS, ConfigError, Settings
from app.health import HealthState, health_result
from app.logging_setup import SecretFilter
from app.telegram import create_bot

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
PRIMARY = "google/gemma-4-26b-a4b-it:free"
RESERVE = "nvidia/nemotron-3.5-lightning:free"


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        f"BOT_TOKEN={TOKEN}\nPOSTGRES_PASSWORD=secret-db\n"
        f"OPENROUTER_API_KEY=fake-key\nOPENROUTER_MODELS={PRIMARY},{RESERVE}\n",
        encoding="utf-8",
    )
    return Settings.load(path, environ={})


def test_settings_environment_overrides_file(tmp_path):
    # Arrange
    path = tmp_path / ".env"
    path.write_text(
        f"BOT_TOKEN={TOKEN}\nPOSTGRES_PASSWORD='p$a#ss'\nPOSTGRES_PORT=5432\n"
        f"OPENROUTER_API_KEY=fake-key\nOPENROUTER_MODELS={PRIMARY},{RESERVE}\n",
        encoding="utf-8",
    )
    # Act
    config = Settings.load(
        path,
        environ={"POSTGRES_PORT": "55432", "OPENROUTER_MODELS": f"{RESERVE},{PRIMARY}"},
    )
    # Assert
    assert config.postgres_port == 55432
    assert config.postgres_password == "p$a#ss"
    assert config.openrouter_models == (RESERVE, PRIMARY)
    assert config.openrouter_api_base_url == "https://openrouter.ai/api/v1"
    assert "p$a#ss" not in repr(config)
    assert TOKEN not in repr(config)
    assert "fake-key" not in repr(config)


def test_missing_openrouter_key_is_reported_without_using_legacy_values(tmp_path):
    values = {
        "BOT_TOKEN": TOKEN,
        "POSTGRES_PASSWORD": "db-secret",
        "LLM_API_KEY": "fake-key",
        "LLM_API_BASE_URL": "https://legacy.example/v1",
        "LLM_MODEL": "legacy-paid-model",
    }

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY") as exc:
        Settings.load(tmp_path / ".env", environ=values)

    assert "fake-key" not in str(exc.value)
    assert "db-secret" not in str(exc.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://openrouter.ai/api/v1",
        "https://user:secret@openrouter.ai/api/v1",
        "https://openrouter.ai.evil.example/api/v1",
        "https://openrouter.ai/api/v1?key=secret",
        "https://openrouter.ai/api/v2",
    ],
)
def test_openrouter_url_rejects_insecure_or_other_hosts(tmp_path, url):
    with pytest.raises(ConfigError, match="OPENROUTER_API_BASE_URL") as exc:
        Settings.load(
            tmp_path / ".env",
            environ={
                "BOT_TOKEN": TOKEN,
                "POSTGRES_PASSWORD": "db-secret",
                "OPENROUTER_API_BASE_URL": url,
                "OPENROUTER_API_KEY": "fake-key",
            },
        )
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        f"{PRIMARY},{PRIMARY}",
        "openrouter/free",
        "openrouter/auto",
        "openrouter/free:free",
        "openrouter/auto:free",
        "paid/model",
        "paid/model:free,paid/model",
        "/missing-author:free",
        "author/slug:free?paid=1",
    ],
)
def test_openrouter_model_list_rejects_empty_duplicates_and_nonfree_ids(tmp_path, raw):
    with pytest.raises(ConfigError, match="OPENROUTER_MODELS"):
        Settings.load(
            tmp_path / ".env",
            environ={
                "BOT_TOKEN": TOKEN,
                "POSTGRES_PASSWORD": "db-secret",
                "OPENROUTER_API_KEY": "fake-key",
                "OPENROUTER_MODELS": raw,
            },
        )


def test_openrouter_default_pool_keeps_catalog_priority(tmp_path):
    settings = Settings.load(
        tmp_path / ".env",
        environ={
            "BOT_TOKEN": TOKEN,
            "POSTGRES_PASSWORD": "db-secret",
            "OPENROUTER_API_KEY": "fake-key",
        },
    )

    assert settings.openrouter_models == DEFAULT_OPENROUTER_MODELS
    assert settings.openrouter_models[0] == PRIMARY


@pytest.mark.parametrize("value", ["abc", "0", "65536"])
def test_invalid_port_has_safe_error(tmp_path, value):
    # Arrange
    path = tmp_path / ".env"
    # Act / Assert
    with pytest.raises(ConfigError, match="POSTGRES_PORT"):
        Settings.load(
            path,
            environ={"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "secret", "POSTGRES_PORT": value},
            require_llm=False,
        )


@pytest.mark.parametrize("proxy", ["http://user:pass@localhost:8080", "socks5://localhost:1080"])
async def test_proxy_is_attached_to_bot_session(settings, proxy):
    # Arrange
    from dataclasses import replace

    config = replace(settings, telegram_proxy_url=proxy)
    # Act
    bot = create_bot(config)
    # Assert
    assert bot.session._proxy == proxy
    await bot.session.close()


def test_bad_proxy_does_not_leak_credentials(tmp_path):
    # Arrange
    proxy = "ftp://user:very-secret@host:123"
    # Act / Assert
    with pytest.raises(ConfigError) as exc:
        Settings.load(
            tmp_path / ".env",
            environ={"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "db", "TELEGRAM_PROXY_URL": proxy},
        )
    assert "TELEGRAM_PROXY_URL" in str(exc.value)
    assert "very-secret" not in str(exc.value)


async def test_health_tracks_pool_failure_and_recovery():
    # Arrange
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    task = asyncio.create_task(asyncio.Event().wait())
    state = HealthState(pool=pool, initialized=True, polling_task=task)
    try:
        # Act / Assert
        assert (await health_result(state))[0] == 200
        pool.fetchval.assert_awaited_with("SELECT 1", timeout=3)
        pool.fetchval.side_effect = ConnectionError("secret-password")
        status, body = await health_result(state)
        assert status == 503
        assert body == {"status": "not_ready", "database": "unavailable", "polling": "running"}
        pool.fetchval.side_effect = None
        assert (await health_result(state))[0] == 200
        state.initialized = False
        assert (await health_result(state))[0] == 503
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_health_without_polling_is_not_ready():
    # Arrange
    state = HealthState(pool=None)
    # Act
    status, body = await health_result(state)
    # Assert
    assert status == 503
    assert body["polling"] == "stopped"


def test_log_filter_redacts_secrets_and_exception():
    # Arrange
    secret_filter = SecretFilter([TOKEN, "db-secret", "proxy-secret"])
    try:
        raise ValueError("proxy-secret")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "test",
            logging.ERROR,
            "",
            1,
            "Ошибка %s https://api.telegram.org/bot%s",
            ("db-secret", TOKEN),
            sys.exc_info(),
        )
    # Act
    secret_filter.filter(record)
    rendered = logging.Formatter().format(record)
    # Assert
    assert all(secret not in rendered for secret in (TOKEN, "db-secret", "proxy-secret"))
    assert "ValueError" in rendered
