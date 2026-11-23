"""Fail-closed verification that a reminder proposal is grounded in the request."""

import re
from datetime import datetime, timedelta

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


def matches_reminder_request(
    query: str, text: str, proposed: str, now: datetime, zone: str
) -> bool:
    original = " ".join(query.casefold().split())
    task = " ".join(text.casefold().split())
    if not re.match(r"^(?:пожалуйста[, ]+)?(?:напомни|напомните|remind me)\b", original):
        return False
    if any(char in original for char in '?«»"“”') or re.match(
        r"^(?:пожалуйста[, ]+)?(?:напомни|напомните)[, ]+(?:мне[, ]+)?(?:как|что|почему|зачем)\b",
        original,
    ):
        return False
    if re.search(r"\b(?:не|без|отмени|отменить|don't|do not)\s+.{0,20}(?:напомн|remind)", original):
        return False
    # Several times or calendar cues cannot be safely bound to one action.
    if re.search(r"\b(?:не|нет|без|кроме|not|except)\b", original):
        return False
    times = re.findall(r"(?<![\d+-])\d{1,2}:\d{2}(?!\d)", original)
    date_markers = re.findall(
        r"\b(?:послезавтра|завтра|сегодня|понедельник|вторник|среду|среда|"
        r"четверг|пятницу|пятница|субботу|суббота|воскресенье)\b"
        r"|(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)|(?<!\d)\d{2}\.\d{2}\.\d{4}(?!\d)",
        original,
    )
    if len(times) != 1 or len(date_markers) != 1:
        return False
    if len(task) < 3 or task not in original:
        return False
    try:
        instant = datetime.fromisoformat(proposed)
        local = reminder_display_time(instant, zone)
        current = reminder_display_time(now, zone)
    except (ValueError, TypeError, KeyError, ReminderError):
        return False
    hour, minute = (int(part) for part in times[0].split(":"))
    if (hour, minute) != (local.hour, local.minute):
        return False
    explicit = local.strftime("%Y-%m-%d") in original or local.strftime("%d.%m.%Y") in original
    if "послезавтра" in original:
        explicit |= local.date() == current.date() + timedelta(days=2)
    elif "завтра" in original:
        explicit |= local.date() == current.date() + timedelta(days=1)
    elif "сегодня" in original:
        explicit |= local.date() == current.date()
    for word, weekday in _WEEKDAYS.items():
        if re.search(rf"\b{word}\b", original):
            days = (weekday - current.weekday()) % 7 or 7
            explicit |= local.date() == current.date() + timedelta(days=days)
    return bool(explicit)
