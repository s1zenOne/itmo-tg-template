from datetime import UTC, datetime, timedelta

import pytest

from app.reminders import (
    ReminderError,
    get_user_zone,
    normalize_reminder_text,
    normalize_reminder_time,
    reminder_display_time,
)

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def test_reminder_normalizes_local_time_in_saved_zone():
    moment, zone = normalize_reminder_time("2026-11-17T09:00", "Europe/Moscow", now=NOW)
    assert moment == datetime(2026, 11, 17, 6, tzinfo=UTC)
    assert zone == "Europe/Moscow"
    assert reminder_display_time(moment, zone).isoformat() == "2026-11-17T09:00:00+03:00"


def test_reminder_explicit_offset_is_kept_for_display_without_iana_zone():
    moment, zone = normalize_reminder_time("2026-11-17T09:30+05:30", None, now=NOW)
    assert moment == datetime(2026, 11, 17, 4, tzinfo=UTC)
    assert zone == "UTC+05:30"
    assert reminder_display_time(moment, zone).isoformat() == "2026-11-17T09:30:00+05:30"


@pytest.mark.parametrize(
    ("value", "zone", "error"),
    [
        ("2026-10-31T09:00Z", None, "past"),
        ("2026-11-01T00:00Z", None, "past"),
        ("2026-11-17T09:00", None, "timezone_required"),
        ("2026-11-31T09:00Z", None, "invalid_time"),
        ("2026-11-17T09:00+01:60", None, "invalid_time"),
        ("2026-11-17", None, "invalid_time"),
        ("2027-03-28T02:30", "Europe/Berlin", "nonexistent_time"),
        ("2027-10-31T02:30", "Europe/Berlin", "ambiguous_time"),
        ("2026-11-17T09:00", "Wrong/Zone", "invalid_timezone"),
    ],
)
def test_reminder_rejects_bad_wall_time(value, zone, error):
    with pytest.raises(ReminderError, match=error):
        normalize_reminder_time(value, zone, now=NOW)


def test_reminder_explicit_offsets_disambiguate_repeated_hour():
    first, zone = normalize_reminder_time("2027-10-31T02:30+02:00", "Europe/Berlin", now=NOW)
    second, other_zone = normalize_reminder_time("2027-10-31T02:30+01:00", "Europe/Berlin", now=NOW)
    assert (zone, other_zone) == ("Europe/Berlin", "Europe/Berlin")
    assert reminder_display_time(first, zone).fold == 0
    assert reminder_display_time(second, zone).fold == 1
    assert second - first == timedelta(hours=1)


@pytest.mark.parametrize("zone", ["Wrong/Zone", "+03:00", "localtime", "posix/Europe/Moscow"])
def test_only_iana_names_can_be_saved(zone):
    with pytest.raises(ReminderError, match="invalid_timezone"):
        get_user_zone(zone)


@pytest.mark.parametrize("text", ["", "   ", "х" * 501, "invalid\0text"])
def test_empty_long_or_invalid_reminder_text_is_rejected(text):
    with pytest.raises(ReminderError, match="invalid_text"):
        normalize_reminder_text(text)


def test_reminder_text_is_stripped_before_length_check():
    assert normalize_reminder_text(f"  {'х' * 500}  ") == "х" * 500
