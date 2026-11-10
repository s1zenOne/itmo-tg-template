"""Единственное место чтения и проверки настроек приложения."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODELS = (
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "dots-studio/dots-3-note-preview:free",
)
_FREE_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*:free\Z")


def parse_openrouter_models(raw: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in raw.split(","))
    if not all(_FREE_MODEL_ID.fullmatch(model) for model in models):
        raise ConfigError(
            "OPENROUTER_MODELS: нужны конкретные бесплатные ID вида author/slug:free."
        )
    if any(model in {"openrouter/free:free", "openrouter/auto:free"} for model in models):
        raise ConfigError("OPENROUTER_MODELS: агрегаторы OpenRouter недопустимы.")
    if len(set(models)) != len(models):
        raise ConfigError("OPENROUTER_MODELS: повторяющиеся модели недопустимы.")
    return models


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    openrouter_api_base_url: str = OPENROUTER_API_BASE_URL
    openrouter_api_key: str = field(default="", repr=False)
    openrouter_models: tuple[str, ...] = DEFAULT_OPENROUTER_MODELS
    llm_timeout_seconds: int = 30
    history_max_messages: int = 12
    context_max_chars: int = 12000
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080

    @classmethod
    def load(
        cls,
        env_file: Path | str = ".env",
        *,
        environ: Mapping[str, str] | None = None,
        require_llm: bool = True,
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        def positive_int(key: str, default: str, minimum: int, maximum: int) -> int:
            try:
                result = int(value(key, default))
                if not minimum <= result <= maximum:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно целое число от {minimum} до {maximum}.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        proxy = value("TELEGRAM_PROXY_URL")
        if proxy:
            try:
                parsed = urlsplit(proxy)
                if (
                    parsed.scheme not in {"http", "socks5"}
                    or not parsed.hostname
                    or not parsed.port
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    "TELEGRAM_PROXY_URL: нужен http://host:port или socks5://host:port; "
                    "при необходимости добавьте user:password@."
                ) from None
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        api_url = value("OPENROUTER_API_BASE_URL", OPENROUTER_API_BASE_URL).rstrip("/")
        if api_url != OPENROUTER_API_BASE_URL:
            raise ConfigError("OPENROUTER_API_BASE_URL: нужен https://openrouter.ai/api/v1.")
        api_key = value("OPENROUTER_API_KEY")
        raw_models = (
            values.get("OPENROUTER_MODELS") or ""
            if "OPENROUTER_MODELS" in values
            else ",".join(DEFAULT_OPENROUTER_MODELS)
        )
        models = parse_openrouter_models(raw_models)
        if require_llm:
            if not api_key.strip() or any(ord(char) < 32 for char in api_key):
                raise ConfigError("OPENROUTER_API_KEY: укажите ключ OpenRouter.")
        history_max_messages = positive_int("HISTORY_MAX_MESSAGES", "12", 2, 100)
        if history_max_messages % 2:
            raise ConfigError("HISTORY_MAX_MESSAGES: нужно чётное число.")
        return cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            openrouter_api_base_url=api_url,
            openrouter_api_key=api_key,
            openrouter_models=models,
            llm_timeout_seconds=positive_int("LLM_TIMEOUT_SECONDS", "30", 1, 120),
            history_max_messages=history_max_messages,
            context_max_chars=positive_int("CONTEXT_MAX_CHARS", "12000", 1000, 100000),
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
        )
