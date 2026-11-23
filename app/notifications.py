"""Настройки уведомлений из доверенных личных Telegram update."""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import aiohttp
from pydantic import ValidationError

from app.mcp_tools.contracts import ToolError
from app.mcp_tools.weather import (
    GEOCODING_URL,
    GeocodedPlace,
    GeocodingResult,
    OpenMeteoWeather,
    _normalized_name,
)
from app.reminders import ReminderError, get_user_zone


@dataclass(frozen=True)
class CityProposal:
    action_id: UUID
    place: GeocodedPlace

    @property
    def label(self) -> str:
        return ", ".join(x for x in (self.place.name, self.place.admin1, self.place.country) if x)


async def geocode_city(query: str) -> GeocodedPlace:
    """Геокодер не получает идентификатор пользователя и адрес доставки."""
    parts = [part.strip() for part in query.split(",")]
    if (
        len(parts) not in (2, 3)
        or not all(parts)
        or len(query) > 150
        or any(ord(char) < 32 for char in query)
    ):
        raise ToolError("invalid_city")
    city, country = parts[0], parts[-1]
    region = parts[1] if len(parts) == 3 else None
    async with aiohttp.ClientSession() as session:
        weather = OpenMeteoWeather(session)
        raw = await weather._get_json(
            GEOCODING_URL,
            {
                "name": city,
                "language": "ru" if re.search("[А-Яа-яЁё]", city) else "en",
                "count": 100,
            },
        )
    try:
        results = GeocodingResult.model_validate(raw).results
        matches = []
        for item in results:
            try:
                place = GeocodedPlace.model_validate(item)
            except ValidationError:
                continue
            if (
                _normalized_name(place.name) == _normalized_name(city)
                and _normalized_name(place.country) == _normalized_name(country)
                and (
                    region is None
                    or _normalized_name(place.admin1 or "") == _normalized_name(region)
                )
            ):
                matches.append(place)
    except ValidationError:
        raise ToolError("weather_invalid_response") from None
    if not matches:
        raise ToolError("city_not_found")
    unique = {(p.name, p.country, p.admin1, p.latitude, p.longitude): p for p in matches}
    if len(unique) != 1:
        raise ToolError("city_ambiguous")
    return next(iter(unique.values()))


class NotificationService:
    def __init__(
        self,
        pool,
        *,
        geocoder: Callable[[str], Awaitable[GeocodedPlace]] = geocode_city,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.pool = pool
        self.geocoder = geocoder
        self.clock = clock

    async def register_private(self, owner_id: int, destination_chat_id: int) -> None:
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO chat_settings (chat_id) VALUES ($1) ON CONFLICT DO NOTHING", owner_id
            )
            await conn.execute(
                "INSERT INTO notification_settings (owner_id, destination_chat_id) VALUES ($1, $2) "
                "ON CONFLICT (owner_id) DO UPDATE SET "
                "destination_chat_id = EXCLUDED.destination_chat_id",
                owner_id,
                destination_chat_id,
            )

    async def prepare_city(
        self, owner_id: int, chat_id: int, update_id: int, query: str
    ) -> CityProposal:
        place = await self.geocoder(query)
        async with self.pool.acquire() as conn, conn.transaction():
            # Сериализуем конкурирующие предложения и повторы одного update.
            await conn.fetchrow(
                "SELECT chat_id FROM chat_settings WHERE chat_id = $1 FOR UPDATE", owner_id
            )
            row = await conn.fetchrow(
                "SELECT * FROM pending_cities WHERE owner_id = $1 AND update_id = $2",
                owner_id,
                update_id,
            )
            if row is None:
                await conn.execute(
                    "UPDATE pending_cities SET status = 'cancelled' "
                    "WHERE owner_id = $1 AND status = 'pending'",
                    owner_id,
                )
                row = await conn.fetchrow(
                    """INSERT INTO pending_cities
                    (action_id, owner_id, destination_chat_id, update_id, city_name, city_country,
                     city_region, latitude, longitude, expires_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING *""",
                    uuid4(),
                    owner_id,
                    chat_id,
                    update_id,
                    place.name,
                    place.country,
                    place.admin1,
                    place.latitude,
                    place.longitude,
                    self.clock() + timedelta(minutes=5),
                )
            if (
                row["destination_chat_id"] != chat_id
                or row["status"] != "pending"
                or row["expires_at"] <= self.clock()
            ):
                raise ToolError("city_expired")
            saved = GeocodedPlace(
                name=row["city_name"],
                country=row["city_country"],
                admin1=row["city_region"],
                latitude=row["latitude"],
                longitude=row["longitude"],
            )
            return CityProposal(row["action_id"], saved)

    async def confirm_city(
        self, owner_id: int, chat_id: int, action_id: UUID, *, accepted: bool
    ) -> bool:
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM pending_cities WHERE owner_id = $1 AND destination_chat_id = $2 "
                "AND action_id = $3 FOR UPDATE",
                owner_id,
                chat_id,
                action_id,
            )
            if row is None or row["status"] != "pending" or row["expires_at"] <= self.clock():
                return False
            settings = await conn.fetchrow(
                "SELECT destination_chat_id FROM notification_settings "
                "WHERE owner_id=$1 FOR UPDATE",
                owner_id,
            )
            if settings is None or settings["destination_chat_id"] != chat_id:
                return False
            await conn.execute(
                "UPDATE pending_cities SET status = $2 WHERE action_id = $1",
                action_id,
                "confirmed" if accepted else "cancelled",
            )
            if accepted:
                await conn.execute(
                    """UPDATE notification_settings SET city_name=$2, city_country=$3,
                    city_region=$4,
                    latitude=$5, longitude=$6 WHERE owner_id=$1 AND destination_chat_id=$7""",
                    owner_id,
                    row["city_name"],
                    row["city_country"],
                    row["city_region"],
                    row["latitude"],
                    row["longitude"],
                    chat_id,
                )
            return True

    async def set_briefing(self, owner_id: int, enabled: bool) -> str:
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                """SELECT n.city_name, n.latitude, n.longitude, c.timezone
                FROM notification_settings n
                JOIN chat_settings c ON c.chat_id=n.owner_id WHERE n.owner_id=$1 FOR UPDATE OF n""",
                owner_id,
            )
            if row is None:
                return "Настройки недоступны. Напишите боту в личном чате."
            if enabled:
                try:
                    get_user_zone(row["timezone"] or "")
                except ReminderError:
                    return "Сначала задайте часовой пояс командой /timezone."
                if not row["city_name"] or row["latitude"] is None or row["longitude"] is None:
                    return "Сначала выберите и подтвердите город командой /city."
            await conn.execute(
                "UPDATE notification_settings SET briefing_enabled=$2 WHERE owner_id=$1",
                owner_id,
                enabled,
            )
        return "Ежедневная сводка включена." if enabled else "Ежедневная сводка выключена."

    async def settings_text(self, owner_id: int) -> str:
        row = await self.pool.fetchrow(
            """SELECT n.city_name, n.city_country, n.city_region, n.briefing_enabled, c.timezone
            FROM notification_settings n JOIN chat_settings c ON c.chat_id=n.owner_id
            WHERE n.owner_id=$1""",
            owner_id,
        )
        if row is None:
            return "Настройки недоступны. Напишите боту в личном чате."
        city = (
            ", ".join(x for x in (row["city_name"], row["city_region"], row["city_country"]) if x)
            or "не выбран"
        )
        state = "включена" if row["briefing_enabled"] else "выключена"
        return (
            f"Город: {city}\nЧасовой пояс: {row['timezone'] or 'не задан'}\n"
            f"Ежедневная сводка: {state}"
        )
