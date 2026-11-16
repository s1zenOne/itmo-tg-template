"""Параметры и нормализация времени напоминаний."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from functools import lru_cache
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

_DATETIME = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}"
    r"(?::[0-9]{2}(?:\.[0-9]{1,6})?)?"
    r"(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?"
)
_OFFSET = re.compile(r"UTC([+-])([01][0-9]|2[0-3]):([0-5][0-9])")


class ReminderError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PreparedReminder:
    action_id: UUID
    text: str
    remind_at: datetime
    timezone: str
    expires_at: datetime
    status: str


@dataclass(frozen=True)
class SavedReminder:
    id: int
    remind_at: datetime
    timezone: str
    created: bool


@lru_cache(maxsize=1)
def _iana_zones() -> set[str]:
    return available_timezones() - {"localtime", "posixrules"}


def get_user_zone(name: str) -> ZoneInfo:
    try:
        if name in _iana_zones():
            return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        pass
    raise ReminderError("invalid_timezone")


def normalize_reminder_text(text: str) -> str:
    cleaned = text.strip()
    if not 1 <= len(cleaned) <= 500 or "\0" in cleaned:
        raise ReminderError("invalid_text")
    return cleaned


def _wall_time(value: datetime, zone: ZoneInfo) -> datetime:
    # Проверка обоих fold отличает уникальное время от пропуска и повторения часа.
    candidates = set()
    for fold in (0, 1):
        candidate = value.replace(tzinfo=zone, fold=fold).astimezone(UTC)
        if candidate.astimezone(zone).replace(tzinfo=None) == value:
            candidates.add(candidate)
    if not candidates:
        raise ReminderError("nonexistent_time")
    if len(candidates) != 1:
        raise ReminderError("ambiguous_time")
    return candidates.pop()


def normalize_reminder_time(value: str, zone: str | None, *, now: datetime) -> tuple[datetime, str]:
    if not _DATETIME.fullmatch(value):
        raise ReminderError("invalid_time")
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            if zone is None:
                raise ReminderError("timezone_required")
            instant = _wall_time(parsed, get_user_zone(zone))
            display_zone = zone
        else:
            instant = parsed.astimezone(UTC)
            if zone and instant.astimezone(get_user_zone(zone)).utcoffset() == parsed.utcoffset():
                display_zone = zone
            else:
                display_zone = str(parsed.tzinfo)
    except ReminderError:
        raise
    except (ValueError, OverflowError):
        raise ReminderError("invalid_time") from None
    if instant <= now:
        raise ReminderError("past")
    return instant, display_zone


def reminder_display_time(instant: datetime, zone: str) -> datetime:
    fixed = _OFFSET.fullmatch(zone)
    if fixed is None:
        return instant.astimezone(get_user_zone(zone))
    direction, hours, minutes = fixed.groups()
    delta = timedelta(hours=int(hours), minutes=int(minutes))
    return instant.astimezone(timezone(delta if direction == "+" else -delta))
