import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import AssistantModelMessage, ModelResponse, ModelUsage
from myclaw.schedule.records import ScheduledWork
from myclaw.schedule.scheduled_work_execution import (
    ScheduledWorkModelSettings,
    ScheduledWorkRunner,
)
from myclaw.session.session import Session
from myclaw.tools.models import ModelToolCall
from myclaw.tools.tool_gateway import ToolGateway
from tests.fixtures import FakeTool, ScriptedFakeProvider

NOW = datetime(2026, 7, 12, 23, 0, 0, 123000, tzinfo=timezone(timedelta(hours=8)))
TASK_SESSION_ID = "20260712-220000-123000_0f8fad5b-d9cb-469f-a165-70867728950e"
TASK_ID = "550e8400-e29b-41d4-a716-446655440000"


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    return state


def _task() -> ScheduledWork:
    return ScheduledWork(
        id=TASK_ID,
        title="Weekly project review",
        cron="0 9 * * 1",
        prompt="Review the current project and summarize open risks.",
        created_at=NOW - timedelta(hours=1),
        enabled=True,
        session_id=TASK_SESSION_ID,
    )


def _runner(
    state: WorkspaceState,
    provider: ScriptedFakeProvider,
    gateway: ToolGateway | None = None,
) -> ScheduledWorkRunner:
    return ScheduledWorkRunner(
        provider=provider,
        workspace_state=state,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session: ToolGateway() if gateway is None else gateway,
    )


@pytest.mark.asyncio
async def test_scheduled_work_persists_a_complete_session_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    response = ModelResponse(
        message=AssistantModelMessage(content="Scheduled result."),
        usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
        finish_reason="stop",
    )
    runner = _runner(state, ScriptedFakeProvider(completions=(response,)))

    result = await runner.run(_task())

    assert result.status == "completed"
    session = Session.load(state, TASK_SESSION_ID)
    assert [(message["role"], message["content"]) for message in session.messages] == [
        ("user", "Review the current project and summarize open risks."),
        ("assistant", "Scheduled result."),
    ]
    assert session.metadata["title"] == "Weekly project review"


@pytest.mark.asyncio
async def test_scheduled_work_preserves_tool_call_relationships(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    tool_call = ModelToolCall(
        id="call_read",
        name="read_file",
        arguments='{"path":"README.md"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="", tool_calls=(tool_call,)),
                usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Read complete."),
                usage=ModelUsage(input_tokens=10, output_tokens=3, total_tokens=13),
                finish_reason="stop",
            ),
        )
    )
    gateway = ToolGateway()
    gateway.register_tools(
        (
            FakeTool(
                name="read_file",
                description="Read a file.",
                required=("path",),
                outcomes=("README contents",),
            ),
        )
    )
    runner = _runner(state, provider, gateway)

    result = await runner.run(_task())

    assert result.status == "completed"
    messages = Session.load(state, TASK_SESSION_ID).messages
    assert messages[1]["tool_calls"] == [
        {
            "id": "call_read",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        }
    ]
    assert messages[2]["tool_call_id"] == "call_read"
    assert messages[2]["name"] == "read_file"
    assert messages[2]["content"] == "README contents"


@pytest.mark.asyncio
async def test_scheduled_model_failure_is_recorded_as_an_assistant_error(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    provider = ScriptedFakeProvider(
        completions=(ModelCallError(ErrorInfo(code="model_failed", message="Model unavailable.")),)
    )
    runner = _runner(state, provider)

    result = await runner.run(_task())

    assert result.status == "failed"
    assert result.error == ErrorInfo(code="model_failed", message="Model unavailable.")
    assistant = Session.load(state, TASK_SESSION_ID).messages[-1]
    assert assistant["role"] == "assistant"
    assert assistant["status"] == "error"
    assert assistant["error"] == {
        "code": "model_failed",
        "message": "Model unavailable.",
    }


@pytest.mark.asyncio
async def test_scheduled_work_returns_a_safe_error_for_malformed_session_field_types(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    path = state.sessions_directory / f"{TASK_SESSION_ID}.jsonl"
    path.write_text(
        json.dumps(
            {
                "session_id": TASK_SESSION_ID,
                "created_at": "2026-07-12T22:00:00.123+08:00",
                "updated_at": "2026-07-12T22:00:00.123+08:00",
                "last_consolidated": 0,
                "metadata": {
                    "title": "Weekly project review",
                    "token_usage": "not-an-object",
                },
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    runner = _runner(state, ScriptedFakeProvider())

    result = await runner.run(_task())

    assert result.status == "failed"
    assert result.content == ""
    assert result.error == ErrorInfo(
        code="persistence_error",
        message="Scheduled Work Session could not be updated.",
    )
