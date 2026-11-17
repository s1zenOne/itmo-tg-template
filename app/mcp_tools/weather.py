import asyncio
import json
import re
import unicodedata
from datetime import datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError

from app.mcp_tools.contracts import (
    WEATHER_SOURCE,
    AmbiguousWeatherResult,
    Place,
    ShortLabel,
    ToolError,
    WeatherResult,
)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=4, connect=2)
MAX_BODY_BYTES = 128 * 1024
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}

CONDITIONS = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь и туман",
    51: "слабая морось",
    53: "умеренная морось",
    55: "сильная морось",
    56: "слабая ледяная морось",
    57: "сильная ледяная морось",
    61: "слабый дождь",
    63: "умеренный дождь",
    65: "сильный дождь",
    66: "слабый ледяной дождь",
    67: "сильный ледяной дождь",
    71: "слабый снег",
    73: "умеренный снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "слабый ливень",
    81: "умеренный ливень",
    82: "сильный ливень",
    85: "слабый снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с небольшим градом",
    97: "сильная гроза",
    99: "гроза с крупным градом",
}


class ProviderData(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class GeocodedPlace(ProviderData):
    name: ShortLabel
    country: ShortLabel
    admin1: ShortLabel | None = None
    latitude: Annotated[FiniteFloat, Field(ge=-90, le=90)]
    longitude: Annotated[FiniteFloat, Field(ge=-180, le=180)]

    @property
    def place(self) -> Place:
        return Place(city=self.name, country=self.country, region=self.admin1)


class GeocodingResult(ProviderData):
    generationtime_ms: FiniteFloat
    results: Annotated[list[object], Field(max_length=100)] = Field(default_factory=list)


class CurrentWeather(ProviderData):
    time: int
    temperature_2m: FiniteFloat
    weather_code: int
    wind_speed_10m: Annotated[FiniteFloat, Field(ge=0)]


class CurrentUnits(ProviderData):
    time: Literal["unixtime"]
    temperature_2m: Literal["°C"]
    wind_speed_10m: Literal["km/h"]


class ForecastResult(ProviderData):
    timezone: ShortLabel
    current: CurrentWeather
    current_units: CurrentUnits


def _normalized_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if char.isalnum())


class OpenMeteoWeather:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def _get_json(self, url: str, params: dict) -> dict:
        for attempt in range(2):
            try:
                async with self.session.get(
                    url, params=params, timeout=HTTP_TIMEOUT, allow_redirects=False
                ) as response:
                    if response.status in RETRY_STATUSES and attempt == 0:
                        continue
                    if response.status != 200:
                        raise ToolError("weather_unavailable")
                    try:
                        await response.content.readexactly(MAX_BODY_BYTES + 1)
                    except asyncio.IncompleteReadError as exc:
                        data = json.loads(exc.partial)
                        if isinstance(data, dict) and data.get("error") is not True:
                            return data
                    raise ToolError("weather_invalid_response")
            except (UnicodeError, json.JSONDecodeError):
                raise ToolError("weather_invalid_response") from None
            except TimeoutError:
                if attempt:
                    raise ToolError("weather_timeout") from None
            except aiohttp.ClientError:
                if attempt:
                    raise ToolError("weather_unavailable") from None
        raise ToolError("weather_unavailable")

    async def get_weather(self, city: str) -> WeatherResult | AmbiguousWeatherResult:
        try:
            data = await self._get_json(
                GEOCODING_URL,
                {
                    "name": city,
                    "language": "ru" if re.search("[А-Яа-яЁё]", city) else "en",
                    "count": 100,
                },
            )
            results = GeocodingResult.model_validate(data).results
            if not results:
                raise ToolError("city_not_found")
            places = []
            for result in results:
                try:
                    places.append(GeocodedPlace.model_validate(result))
                except ValidationError:
                    continue
            if not places:
                raise ToolError("weather_invalid_response")

            requested = _normalized_name(city.split(",", 1)[0])
            exact = [place for place in places if _normalized_name(place.name) == requested]
            candidates = exact or places
            options = {(place.name, place.country, place.admin1): place for place in candidates}
            if len(options) > 1:
                return AmbiguousWeatherResult(
                    status="ambiguous",
                    options=[place.place for place in list(options.values())[:10]],
                    source=WEATHER_SOURCE,
                )

            place = next(iter(options.values()))
            raw = await self._get_json(
                FORECAST_URL,
                {
                    "latitude": place.latitude,
                    "longitude": place.longitude,
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "timeformat": "unixtime",
                    "timezone": "auto",
                },
            )
            forecast = ForecastResult.model_validate(raw)
            current = forecast.current
            instant = datetime.fromtimestamp(current.time, ZoneInfo(forecast.timezone))
            return WeatherResult(
                **place.place.model_dump(),
                status="ok",
                temperature_c=current.temperature_2m,
                conditions=CONDITIONS[current.weather_code],
                wind_speed=current.wind_speed_10m,
                wind_unit=forecast.current_units.wind_speed_10m,
                observed_at=instant.isoformat(),
                timezone=forecast.timezone,
                source=WEATHER_SOURCE,
            )
        except (
            ValidationError,
            KeyError,
            ValueError,
            OSError,
            OverflowError,
            ZoneInfoNotFoundError,
        ):
            raise ToolError("weather_invalid_response") from None
