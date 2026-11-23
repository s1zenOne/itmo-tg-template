import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from app.agent import AgentService
from app.agent_router import AgentRouter
from app.config import Settings
from app.llm import LLMResult, NativeToolCall
from app.mcp_tools.contracts import SCHEDULE, tool_definitions
from app.prompts import MODE_PROMPTS
from app.service import AssistantService
from app.store import UserState
from app.supervisor import SpecialistRouter


def response(text="", name=None, arguments=None):
    calls = () if name is None else (NativeToolCall("one", name, json.dumps(arguments)),)
    return LLMResult(text, "test/model", "test", None, "request", tool_calls=calls)


def service_for(mode, responses):
    store = Mock(
        get_context=AsyncMock(return_value=(UserState(mode=mode), [])),
        save_result=AsyncMock(return_value=True),
        reminder_for_update=AsyncMock(return_value=None),
        get_timezone=AsyncMock(return_value="Europe/Moscow"),
        write_audit=AsyncMock(),
    )
    mcp = Mock(
        available=True,
        tools=tool_definitions(),
        connect=AsyncMock(return_value=True),
        call=AsyncMock(
            return_value={
                "status": "ok",
                "date": "2026-10-13",
                "lessons": [],
                "source": "Учебное расписание",
            }
        ),
        read_resource=AsyncMock(),
    )
    llm = Mock(generate=AsyncMock(side_effect=responses))
    settings = Settings(bot_token="unused", postgres_password="unused")
    agent = AgentService(
        store,
        AgentRouter(llm),
        mcp,
        settings,
        clock=lambda: datetime(2026, 10, 12, 6, tzinfo=UTC),
        specialist_router=SpecialistRouter(llm),
    )
    return AssistantService(store, llm, settings, agent), store, llm, mcp


async def test_study_mode_routes_to_schedule_and_preserves_its_system_instruction():
    service, store, llm, mcp = service_for(
        "study",
        [
            response(
                name="select_route",
                arguments={"target": "schedule", "reason": "x", "confidence": 0.9},
            ),
            response(name=SCHEDULE, arguments={"date": "2026-10-13"}),
            response(text="Готово."),
        ],
    )
    result = await service.answer_update(101, "Какие пары завтра?", owner_id=101, update_id=1)
    assert "2026-10-13" in result.text
    assert MODE_PROMPTS["study"] in llm.generate.await_args_list[1].args[0][0]["content"]
    assert llm.generate.await_args_list[0].kwargs["tools"][0]["function"]["name"] == "select_route"
    mcp.call.assert_awaited_once_with(SCHEDULE, {"date": "2026-10-13"})
    store.save_result.assert_awaited_once()


@pytest.mark.parametrize(
    "mode,query", [("translate", "Переведи на английский: привет"), ("quiz", "Вопрос о Python")]
)
async def test_previous_modes_keep_system_instruction_without_mcp_tools(mode, query):
    service, store, llm, mcp = service_for(mode, [response(text="Ответ в выбранном режиме.")])
    result = await service.answer_update(101, query, owner_id=101, update_id=1)
    assert result.text == "Ответ в выбранном режиме."
    llm.generate.assert_awaited_once()
    model_call = llm.generate.await_args
    assert MODE_PROMPTS[mode] in model_call.args[0][0]["content"]
    exposed = {tool["function"]["name"] for tool in model_call.kwargs["tools"]}
    assert not exposed.intersection(tool.name for tool in mcp.tools)
    mcp.call.assert_not_awaited()
    mcp.read_resource.assert_not_awaited()
    store.save_result.assert_awaited_once()
