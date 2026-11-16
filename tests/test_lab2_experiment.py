import pytest

from scripts.lab2_experiment import _case_result, _metrics, arguments_match


@pytest.mark.parametrize(
    ("expected", "actual", "equivalent"),
    [
        ("Санкт-Петербург", "St. Petersburg, Russian Federation", True),
        ("Санкт-Петербург", "St. Petersburg, USA", False),
        ("Казань", "Kazan", True),
        ("Тбилиси, Грузия", "Tbilisi, Georgia", True),
        ("Тбилиси, Грузия", "Тбилиси", False),
    ],
)
def test_experiment_city_equivalence(expected: str, actual: str, equivalent: bool) -> None:
    assert arguments_match("get_weather", {"city": expected}, {"city": actual}) is equivalent


def test_experiment_normalized_reminder() -> None:
    expected = {"remind_at": "2026-10-13T18:30:00+03:00", "text": "отправить отчёт"}
    actual = {"remind_at": "2026-10-13T15:30:00Z", "text": " Отправить отчет. "}
    assert arguments_match("add_reminder", expected, actual)
    actual["remind_at"] = "2026-10-13T18:30:00Z"
    assert not arguments_match("add_reminder", expected, actual)


def test_experiment_counts_missing_tool_as_argument_error() -> None:
    schedule = {
        "id": "schedule",
        "query": "Сегодня?",
        "expected_action": "get_schedule",
        "expected_arguments": {"date": "2026-10-12"},
    }
    respond = {
        "id": "respond",
        "query": "Как дела?",
        "expected_action": "respond",
        "expected_arguments": {},
    }
    rows = [
        _case_result(case, action="respond", arguments={}, status="validated")
        for case in (schedule, respond)
    ]
    assert _metrics(rows) == {
        "routing_accuracy": {"correct": 1, "total": 2},
        "argument_accuracy": {"correct": 0, "total": 1},
    }


def test_experiment_settings_only_require_openrouter_key(tmp_path):
    from app.config import Settings

    settings = Settings.load(
        tmp_path / ".env", environ={"OPENROUTER_API_KEY": "synthetic-key"}, require_runtime=False
    )
    assert settings.openrouter_api_key == "synthetic-key"
    assert settings.bot_token == settings.postgres_password == ""
