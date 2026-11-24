"""Fail-closed verification that a reminder proposal is grounded in the request."""

import re
from datetime import UTC, date, datetime, timedelta

from app.reminders import ReminderError, reminder_display_time

_WEEKDAYS = {
    "понедельник": 0,
    "вторник": 1,
    "среду": 2,
    "среда": 2,
    "четверг": 3,
    "пятницу": 4,
    "пятница": 4,
    "субботу": 5,
    "суббота": 5,
    "воскресенье": 6,
}
_MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        ),
        start=1,
    )
}
_NUMBERS = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4}
_DATE = (
    r"послезавтра|завтра|сегодня|"
    + "|".join(_WEEKDAYS)
    + r"|\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}"
    + r"|\d{1,2}\s+(?:"
    + "|".join(_MONTHS)
    + r")"
)
_RELATIVE = (
    r"через\s+(?P<count>\d{1,3}|один|одну|два|две|три|четыре)\s+"
    r"(?P<unit>час|часа|часов|минуту|минуты|минут)"
)
_PREFIX = r"(?:пожалуйста[, ]+)?(?:напомни|напомните)(?:\s+мне)?\s+"
_ABSOLUTE = re.compile(
    rf"^{_PREFIX}(?P<date>{_DATE})\s+в\s+(?P<hour>\d{{1,2}})(?::(?P<minute>\d{{2}}))?"
    r"(?:\s+(?P<offset>utc[+-]\d{2}:\d{2}))?\s+(?P<task>.+)$",
    re.IGNORECASE,
)
_RELATIVE_PREFIX = re.compile(rf"^{_RELATIVE}\s+{_PREFIX}(?P<task>.+)$", re.IGNORECASE)
_RELATIVE_AFTER = re.compile(rf"^{_PREFIX}{_RELATIVE}\s+(?P<task>.+)$", re.IGNORECASE)
_EXTRA_TIME = re.compile(
    rf"(?<!\w)(?:{_DATE}|через\s+\S+\s+(?:час\w*|минут\w*)|"
    r"\d{1,2}:\d{2}|в\s+\d{1,2}(?::\d{2})?|утром|днём|вечером|ночью)(?!\w)",
    re.IGNORECASE,
)


def _task_text(value: str) -> str:
    # Ignore final punctuation only for comparison; the original task is retained for the card.
    return value.strip().rstrip(".!").rstrip()


def matches_reminder_request(
    query: str, text: str, proposed: str, now: datetime, zone: str
) -> bool:
    original = query.strip()
    if any(char in original for char in '?«»"“”'):
        return False
    if re.search(r"\b(?:не|нет|без|кроме|not|except)\b", original, re.IGNORECASE):
        return False
    match = _ABSOLUTE.fullmatch(original)
    relative = False
    if match is None:
        match = _RELATIVE_PREFIX.fullmatch(original) or _RELATIVE_AFTER.fullmatch(original)
        relative = True
    if match is None:
        return False
    task = _task_text(match.group("task"))
    if len(task) < 3 or task != _task_text(text) or _EXTRA_TIME.search(task):
        return False
    try:
        instant = datetime.fromisoformat(proposed)
        if instant.tzinfo is None or now.tzinfo is None:
            return False
        local = reminder_display_time(instant, zone)
        current = reminder_display_time(now, zone)
        if relative:
            raw_count = match.group("count").casefold()
            count = int(raw_count) if raw_count.isdigit() else _NUMBERS[raw_count]
            if count < 1:
                return False
            delta = (
                timedelta(hours=count)
                if match.group("unit").casefold().startswith("час")
                else timedelta(minutes=count)
            )
            expected = now.astimezone(UTC) + delta
            # Minute-level proposals are accepted when the user's request has minute precision.
            return instant.astimezone(UTC) in {expected, expected.replace(second=0, microsecond=0)}
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        offset = match.group("offset")
        if offset is not None:
            specified = datetime.fromisoformat(f"2000-01-01T00:00:00{offset[3:]}").utcoffset()
            if local.utcoffset() != specified:
                return False
        if (local.hour, local.minute, local.second, local.microsecond) != (hour, minute, 0, 0):
            return False
        marker = match.group("date").casefold()
        if marker in {"сегодня", "завтра", "послезавтра"}:
            days = {"сегодня": 0, "завтра": 1, "послезавтра": 2}[marker]
            expected_date = current.date() + timedelta(days=days)
        elif marker in _WEEKDAYS:
            days = (_WEEKDAYS[marker] - current.weekday()) % 7
            if days == 0:
                return False
            expected_date = current.date() + timedelta(days=days)
        elif "-" in marker:
            expected_date = date.fromisoformat(marker)
        elif "." in marker:
            expected_date = datetime.strptime(marker, "%d.%m.%Y").date()
        else:
            day, month = marker.split()
            expected_date = date(current.year, _MONTHS[month], int(day))
            # A past date without a year does not unambiguously mean next year.
            if expected_date < current.date():
                return False
        return local.date() == expected_date
    except (ValueError, TypeError, KeyError, OverflowError, ReminderError):
        return False


def verified_reminder_text(
    query: str, text: str, proposed: str, now: datetime, zone: str
) -> str | None:
    """Return the user's exact task when the proposal passes all checks."""
    if not matches_reminder_request(query, text, proposed, now, zone):
        return None
    original = query.strip()
    match = (
        _ABSOLUTE.fullmatch(original)
        or _RELATIVE_PREFIX.fullmatch(original)
        or _RELATIVE_AFTER.fullmatch(original)
    )
    return match.group("task").strip() if match else None
