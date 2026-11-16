import logging
import re
from urllib.parse import unquote, urlsplit

from app.config import Settings


class SecretFilter(logging.Filter):
    def __init__(self, secrets: list[str]):
        super().__init__()
        self.secrets = sorted({s for s in secrets if s}, key=len, reverse=True)

    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + logging.Formatter().formatException(record.exc_info)
        if record.stack_info:
            text += "\n" + record.stack_info
        for secret in self.secrets:
            text = text.replace(secret, "[скрыто]")
        text = re.sub(r"(https?://|socks5://)[^\s/@]+:[^\s/@]+@", r"\1[скрыто]@", text)
        if record.name.startswith("aiogram"):
            text = re.sub(r"(?i)(\b(?:update|bot|user|chat) id[= :]+)\d+", r"\1[скрыто]", text)
            if text.startswith(("Run polling for bot ", "Polling stopped for bot ")):
                text = text.split(" for bot ", 1)[0]
        record.msg, record.args = text, ()
        record.exc_info = record.exc_text = record.stack_info = None
        return True


def configure_logging(settings: Settings) -> None:
    proxy = urlsplit(settings.telegram_proxy_url)
    handler = logging.StreamHandler()
    handler.addFilter(
        SecretFilter(
            [
                settings.bot_token,
                settings.postgres_password,
                settings.openrouter_api_key,
                settings.telegram_proxy_url,
                proxy.password or "",
                unquote(proxy.password or ""),
            ]
        )
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)
    # Aiogram event records contain the Telegram update ID.
    logging.getLogger("aiogram.event").disabled = True
