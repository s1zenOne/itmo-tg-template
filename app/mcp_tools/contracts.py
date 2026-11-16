from datetime import date, datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.types import Tool, ToolAnnotations
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.reminders import normalize_reminder_text, reminder_display_time

WEATHER = "get_weather"
SCHEDULE = "get_schedule"
REMINDER = "add_reminder"
WEEK_URI = "schedule://current-week"
WEATHER_SOURCE = "Open-Meteo"
SCHEDULE_SOURCE = "Учебное расписание"

DATE_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
SCHEDULE_INPUT_PATTERN = r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}$"
LOCAL_TIME = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]{1,6})?)?"
OFFSET = r"(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
HHMM_PATTERN = r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$"

DateText = Annotated[str, StringConstraints(pattern=DATE_PATTERN)]
AwareTimeText = Annotated[str, StringConstraints(pattern=f"^{LOCAL_TIME}{OFFSET}$")]
LessonTime = Annotated[str, StringConstraints(pattern=HHMM_PATTERN)]
CityText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
ShortLabel = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=150)]
LessonText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=250)]
ReminderText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
]


class ToolError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class WeatherInput(Contract):
    city: CityText = Field(
        description="Город. Для уточнения добавь через запятую страну или область."
    )

    @field_validator("city")
    @classmethod
    def validate_city(cls, value: str) -> str:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) > 2 or not all(parts):
            raise ValueError("Нужен город и, при необходимости, одна страна или область.")
        return value


class CalendarDate(Contract):
    date: DateText

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


class ScheduleInput(CalendarDate):
    date: Annotated[str, StringConstraints(pattern=SCHEDULE_INPUT_PATTERN)] = Field(
        description="Точная дата YYYY-MM-DD в диапазоне 2000–2099 годов."
    )


class LocalReminderInput(Contract):
    text: ReminderText = Field(description="Текст напоминания без указания даты и времени.")
    remind_at: Annotated[str, StringConstraints(pattern=f"^{LOCAL_TIME}(?:{OFFSET})?$")]

    @field_validator("text")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return normalize_reminder_text(value)

    @field_validator("remind_at")
    @classmethod
    def valid_time(cls, value: str) -> str:
        datetime.fromisoformat(value)
        return value


class ReminderInput(LocalReminderInput):
    remind_at: AwareTimeText = Field(description="Точный момент ISO 8601 с явным смещением или Z.")


class Place(Contract):
    city: ShortLabel
    country: ShortLabel
    region: ShortLabel | None


class WeatherResult(Place):
    status: Literal["ok"]
    temperature_c: FiniteFloat
    conditions: ShortLabel
    wind_speed: Annotated[FiniteFloat, Field(ge=0)]
    wind_unit: Literal["km/h"]
    observed_at: AwareTimeText
    timezone: ShortLabel
    source: Literal[WEATHER_SOURCE]

    @field_validator("observed_at", "timezone")
    @classmethod
    def check_observation(cls, value: str, info: ValidationInfo) -> str:
        if info.field_name == "observed_at":
            datetime.fromisoformat(value)
        else:
            try:
                ZoneInfo(value)
            except ZoneInfoNotFoundError as exc:
                raise ValueError("Неизвестная зона наблюдения.") from exc
        return value


class AmbiguousWeatherResult(Contract):
    status: Literal["ambiguous"]
    options: Annotated[list[Place], Field(min_length=2, max_length=10)]
    source: Literal[WEATHER_SOURCE]


class Lesson(Contract):
    start: LessonTime
    end: LessonTime
    title: LessonText
    location: LessonText

    @model_validator(mode="after")
    def check_lesson_time(self) -> "Lesson":
        if self.start >= self.end:
            raise ValueError("Конец занятия должен быть позже начала.")
        return self


class ScheduleResult(CalendarDate):
    status: Literal["ok"]
    lessons: Annotated[list[Lesson], Field(max_length=24)]
    source: Literal[SCHEDULE_SOURCE]


class WeekSchedule(Contract):
    timezone: ShortLabel
    days: Annotated[list[ScheduleResult], Field(min_length=7, max_length=7)]
    source: Literal[SCHEDULE_SOURCE]


class ReminderResult(Contract):
    status: Literal["ok"]
    id: Annotated[int, Field(gt=0)]
    remind_at: AwareTimeText
    timezone: ShortLabel
    created: bool

    @field_validator("remind_at")
    @classmethod
    def check_reminder_time(cls, value: str) -> str:
        datetime.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def check_reminder_zone(self) -> "ReminderResult":
        reminder_display_time(datetime.fromisoformat(self.remind_at), self.timezone)
        return self


class ErrorResult(Contract):
    status: Literal["error"]
    code: Annotated[str, StringConstraints(pattern=r"^[a-z_]{2,40}$")]


INPUTS = {WEATHER: WeatherInput, SCHEDULE: ScheduleInput, REMINDER: ReminderInput}
OUTPUTS = {
    WEATHER: TypeAdapter(
        Annotated[
            WeatherResult | AmbiguousWeatherResult | ErrorResult, Field(discriminator="status")
        ]
    ),
    SCHEDULE: TypeAdapter(Annotated[ScheduleResult | ErrorResult, Field(discriminator="status")]),
    REMINDER: TypeAdapter(Annotated[ReminderResult | ErrorResult, Field(discriminator="status")]),
}
DESCRIPTIONS = {
    WEATHER: "Получает текущую погоду в указанном городе из Open-Meteo.",
    SCHEDULE: "Получает занятия из учебного расписания на одну точную дату.",
    REMINDER: "Создаёт напоминание после подтверждения пользователем.",
}


def tool_definitions() -> tuple[Tool, ...]:
    return tuple(
        Tool(
            name=name,
            description=DESCRIPTIONS[name],
            input_schema=input_model.model_json_schema(),
            output_schema={"type": "object", **OUTPUTS[name].json_schema()},
            annotations=ToolAnnotations(
                read_only_hint=name != REMINDER,
                destructive_hint=False,
                idempotent_hint=True,
                open_world_hint=name == WEATHER,
            ),
        )
        for name, input_model in INPUTS.items()
    )


def validate_arguments(name: str, arguments: object, *, allow_local_reminder: bool = False) -> dict:
    if name not in INPUTS:
        raise ToolError("unknown_tool")
    model = LocalReminderInput if name == REMINDER and allow_local_reminder else INPUTS[name]
    try:
        return model.model_validate(arguments).model_dump()
    except ValidationError:
        raise ToolError("invalid_arguments") from None


def validate_result(name: str, data: object) -> dict:
    if name not in OUTPUTS:
        raise ToolError("unknown_tool")
    try:
        result = OUTPUTS[name].validate_python(data)
    except ValidationError:
        raise ToolError("invalid_result") from None
    if isinstance(result, ErrorResult):
        raise ToolError(result.code)
    return result.model_dump()
