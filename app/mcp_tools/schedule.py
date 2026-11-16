import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from app.mcp_tools.contracts import SCHEDULE_SOURCE, ScheduleResult, ToolError, WeekSchedule

DEFAULT_PATH = Path(__file__).with_name("data") / "schedule.json"
DEFAULT_TIMEZONE = "Europe/Moscow"
MAX_FILE_BYTES = 512 * 1024


def _now() -> datetime:
    return datetime.now(UTC)


class FileSchedule:
    def __init__(
        self,
        path: Path = DEFAULT_PATH,
        timezone: str = DEFAULT_TIMEZONE,
        *,
        clock: Callable[[], datetime] = _now,
    ):
        self.path = path
        self.timezone = ZoneInfo(timezone)
        self.clock = clock

    def _read(self) -> dict[str, ScheduleResult]:
        try:
            with self.path.open("rb") as file:
                payload = file.read(MAX_FILE_BYTES + 1)
            if len(payload) > MAX_FILE_BYTES:
                raise ToolError("schedule_unavailable")
            days = json.loads(payload)
            if not isinstance(days, dict):
                raise ToolError("schedule_unavailable")
            return {day: self._day(day, lessons) for day, lessons in days.items()}
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
            raise ToolError("schedule_unavailable") from None

    @staticmethod
    def _day(date: str, lessons: list) -> ScheduleResult:
        result = ScheduleResult(status="ok", date=date, lessons=lessons, source=SCHEDULE_SOURCE)
        result.lessons.sort(key=lambda lesson: (lesson.start, lesson.end))
        return result

    async def get_schedule(self, date: str) -> ScheduleResult:
        days = await asyncio.to_thread(self._read)
        return days[date] if date in days else self._day(date, [])

    async def current_week(self) -> WeekSchedule:
        days = await asyncio.to_thread(self._read)
        today = self.clock().astimezone(self.timezone).date()
        monday = today - timedelta(days=today.weekday())
        week = [(monday + timedelta(days=offset)).isoformat() for offset in range(7)]
        return WeekSchedule(
            timezone=self.timezone.key,
            days=[days[date] if date in days else self._day(date, []) for date in week],
            source=SCHEDULE_SOURCE,
        )
