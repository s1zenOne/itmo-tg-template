"""Проверка MCP через независимый клиент из официального SDK."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from jsonschema import Draft202012Validator
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]
WEEK_URI = "schedule://current-week"


async def run(weather_city: str | None) -> int:
    async with Client(
        StdioServerParameters(
            command=sys.executable, args=["-m", "app.mcp_tools.server"], cwd=ROOT
        ),
        read_timeout_seconds=20,
    ) as client:
        tools = (await client.list_tools()).tools
        resources = (await client.list_resources()).resources
        assert {tool.name: tool.annotations.read_only_hint for tool in tools} == {
            "get_weather": True,
            "get_schedule": True,
            "add_reminder": False,
            "search_materials": True,
        }
        assert {str(resource.uri) for resource in resources} == {WEEK_URI, "materials://catalog"}
        for tool in tools:
            Draft202012Validator.check_schema(tool.input_schema)
            Draft202012Validator.check_schema(tool.output_schema)
            assert tool.input_schema.get("additionalProperties") is False
        schemas = {tool.name: tool.output_schema for tool in tools}

        result = await client.call_tool("get_schedule", {"date": "2026-10-12"})
        assert not result.is_error
        Draft202012Validator(schemas["get_schedule"]).validate(result.structured_content)
        assert result.structured_content["date"] == "2026-10-12"
        assert result.structured_content["lessons"][0] == {
            "start": "10:00",
            "end": "11:30",
            "title": "Алгоритмы и структуры данных",
            "location": "Аудитория 301",
        }

        week = await client.read_resource(WEEK_URI)
        days = json.loads(week.contents[0].text)["days"]
        monday = date.fromisoformat(days[0]["date"])
        assert monday.weekday() == 0
        assert [day["date"] for day in days] == [
            (monday + timedelta(days=i)).isoformat() for i in range(7)
        ]

        catalog = await client.read_resource("materials://catalog")
        assert len(json.loads(catalog.contents[0].text)["documents"]) >= 3
        found = await client.call_tool("search_materials", {"query": "MCP", "limit": 2})
        assert not found.is_error and found.structured_content["documents"]
        Draft202012Validator(schemas["search_materials"]).validate(found.structured_content)

        denied = await client.call_tool(
            "add_reminder",
            {
                "text": "Синтетическая проверка",
                "remind_at": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            },
        )
        assert denied.is_error and denied.structured_content == {
            "status": "error",
            "code": "unauthorized",
        }
        Draft202012Validator(schemas["add_reminder"]).validate(denied.structured_content)
        report = {
            "tools": [tool.name for tool in tools],
            "input_and_output_schemas": "verified",
            "schedule_lessons": len(result.structured_content["lessons"]),
            "resource": WEEK_URI,
            "resource_days": len(days),
            "unsigned_reminder": denied.structured_content["code"],
        }
        weather_failed = False
        if weather_city is not None:
            weather = await client.call_tool("get_weather", {"city": weather_city})
            Draft202012Validator(schemas["get_weather"]).validate(weather.structured_content)
            report["weather"] = weather.structured_content
            weather_failed = bool(weather.is_error)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(weather_failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-city", help="Город для обращения к живому Open-Meteo")
    raise SystemExit(asyncio.run(run(parser.parse_args().weather_city)))
