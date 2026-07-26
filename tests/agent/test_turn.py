import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.turn import AgentTurn, AgentTurnLane
from myclaw.agent.workspace import Workspace
from myclaw.config.agent_home import AgentHome
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from myclaw.session.records import (
    AssistantSessionMessage,
    ConversationSession,
    SessionMessage,
    ToolSessionMessage,
    UserSessionMessage,
)
from myclaw.session.session_store import JsonlSessionStore
from myclaw.tools.models import ModelToolCall, ToolDefinition, ToolExecutionContext, ToolResult
from myclaw.tools.tool_artifacts import externalize_tool_result
from myclaw.tools.tool_gateway import ToolGateway
from tests.fixtures import FakeTool, ScriptedFakeProvider, StreamScript

LOCAL_TIMEZONE = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 18, 18, 30, 12, 123456, tzinfo=LOCAL_TIMEZONE)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
USER_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
REQUEST_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
ASSISTANT_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
TOOL_MESSAGE_UUID = UUID("a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34")
FOLLOW_UP_REQUEST_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
FINAL_ASSISTANT_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")


@dataclass(frozen=True, slots=True)
class _ModelSettings:
    model: str = "test-model"
    max_output: int = 1024
    temperature: float = 0.2
    reasoning_effort: None = None
    timeout_seconds: int = 30


class _MissingToolAppendSessionStore(JsonlSessionStore):
    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        if isinstance(message, ToolSessionMessage):
            raise OSError("injected pre-write Tool append failure")
        await super().append_message(session_id, message)


class _IndeterminateToolAppendSessionStore(JsonlSessionStore):
    _fail_reconciliation_load = False

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        if isinstance(message, ToolSessionMessage):
            self._fail_reconciliation_load = True
            raise OSError("injected indeterminate Tool append failure")
        await super().append_message(session_id, message)

    async def load(self, session_id: str) -> ConversationSession:
        if self._fail_reconciliation_load:
            self._fail_reconciliation_load = False
            raise OSError("injected reconciliation load failure")
        return await super().load(session_id)


def _usage() -> ModelUsage:
    return ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15)


def _io_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    return Path(f"\\\\?\\{path.absolute()}")


def _provider(
    lane: AgentTurnLane,
    responses: tuple[ModelResponse, ...],
) -> ScriptedFakeProvider:
    if lane == "foreground":
        return ScriptedFakeProvider(
            streams=(
                StreamScript(events=(ModelCompleted(response=response),)) for response in responses
            )
        )
    return ScriptedFakeProvider(completions=responses)


async def _collect(turn: AsyncIterator[object]) -> tuple[object, ...]:
    return tuple([item async for item in turn])


async def _run_artifact_turn(
    *,
    lane: AgentTurnLane,
    sessions: JsonlSessionStore,
    session_id: str,
    agent_home: Path,
    workspace: Path,
    completes: bool,
) -> tuple[tuple[object, ...], Path]:
    tool_call = ModelToolCall(
        id="call_read",
        name="read_file",
        arguments='{"path":"README.md"}',
    )
    first_response = ModelResponse(
        message=AssistantModelMessage(content="Reading.", tool_calls=(tool_call,)),
        usage=_usage(),
        finish_reason="tool_calls",
    )
    responses: tuple[ModelResponse, ...] = (first_response,)
    if completes:
        responses += (
            ModelResponse(
                message=AssistantModelMessage(content="Finished."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    provider = _provider(lane, responses)
    tool = FakeTool(
        definition=ToolDefinition(
            name="read_file",
            description="Read a test file.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        outcomes=("oversized artifact contents",),
    )
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane=lane,
            workspace=workspace,
            agent_home=agent_home,
            session_id=session_id,
        ),
        tools=(tool,),
    )
    externalized: list[ToolResult] = []

    def externalize(result: ToolResult) -> ToolResult:
        projected = externalize_tool_result(
            result,
            agent_home=agent_home,
            workspace=Workspace.from_path(workspace),
            session_id=session_id,
            max_tool_result_chars=4,
        )
        externalized.append(projected)
        return projected

    turn = AgentTurn(
        lane=lane,
        provider=provider,
        sessions=sessions,
        session_id=session_id,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_MESSAGE_UUID,
                FOLLOW_UP_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        system_prompt="Test system prompt.",
        tool_gateway=gateway,
        externalize_result=externalize,
    )

    payloads = await _collect(turn.run("Inspect the project."))

    result = externalized[0]
    assert result.artifact is not None
    artifact_path = _io_path(
        agent_home / "sessions" / Workspace.from_path(workspace).slug / Path(result.artifact.path)
    )
    return payloads, artifact_path


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["foreground", "scheduled_work"])
async def test_agent_turn_lanes_share_one_persisted_tool_loop(
    lane: AgentTurnLane,
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=lambda: SESSION_UUID,
    )
    session = sessions.prepare()
    tool_call = ModelToolCall(
        id="call_read",
        name="read_file",
        arguments='{"path":"README.md"}',
    )
    provider = _provider(
        lane,
        (
            ModelResponse(
                message=AssistantModelMessage(content="Reading.", tool_calls=(tool_call,)),
                usage=_usage(),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Finished."),
                usage=_usage(),
                finish_reason="stop",
            ),
        ),
    )
    tool = FakeTool(
        definition=ToolDefinition(
            name="read_file",
            description="Read a test file.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        outcomes=("file contents",),
    )
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane=lane,
            workspace=workspace,
            agent_home=agent_home,
            session_id=session.id,
        ),
        tools=(tool,),
    )
    turn = AgentTurn(
        lane=lane,
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=iter(
            (
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_MESSAGE_UUID,
                FOLLOW_UP_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        system_prompt="Test system prompt.",
        tool_gateway=gateway,
    )

    payloads = await _collect(turn.run("Inspect the project."))

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "ToolStartedPayload",
        "ToolCompletedPayload",
        "TurnCompletedPayload",
    ]
    reloaded = await sessions.load(session.id)
    assert [type(message) for message in reloaded.messages] == [
        UserSessionMessage,
        AssistantSessionMessage,
        ToolSessionMessage,
        AssistantSessionMessage,
    ]
    tool_message = reloaded.messages[2]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.content == "file contents"
    requests = provider.stream_requests if lane == "foreground" else provider.complete_requests
    assert len(requests) == 2
    assert all(isinstance(request, ModelRequest) for request in requests)
    assert [request.route for request in requests if isinstance(request, ModelRequest)] == [
        "chat" if lane == "foreground" else "cron",
        "chat" if lane == "foreground" else "cron",
    ]
    assert [request.stream for request in requests if isinstance(request, ModelRequest)] == [
        lane == "foreground",
        lane == "foreground",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["foreground", "scheduled_work"])
async def test_agent_turn_lanes_persist_a_published_tool_artifact(
    lane: AgentTurnLane,
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=lambda: SESSION_UUID,
    )
    session = sessions.prepare()

    payloads, artifact_path = await _run_artifact_turn(
        lane=lane,
        sessions=sessions,
        session_id=session.id,
        agent_home=agent_home,
        workspace=workspace,
        completes=True,
    )

    assert type(payloads[-1]).__name__ == "TurnCompletedPayload"
    assert artifact_path.read_text(encoding="utf-8") == "oversized artifact contents"


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["foreground", "scheduled_work"])
async def test_agent_turn_lanes_accept_an_orphan_after_known_session_failure(
    lane: AgentTurnLane,
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = _MissingToolAppendSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=lambda: SESSION_UUID,
    )
    session = sessions.prepare()

    payloads, artifact_path = await _run_artifact_turn(
        lane=lane,
        sessions=sessions,
        session_id=session.id,
        agent_home=agent_home,
        workspace=workspace,
        completes=False,
    )

    assert type(payloads[-1]).__name__ == "TurnFailedPayload"
    assert artifact_path.read_text(encoding="utf-8") == "oversized artifact contents"


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["foreground", "scheduled_work"])
async def test_agent_turn_lanes_preserve_an_indeterminate_tool_artifact(
    lane: AgentTurnLane,
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = _IndeterminateToolAppendSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=lambda: SESSION_UUID,
    )
    session = sessions.prepare()

    payloads, artifact_path = await _run_artifact_turn(
        lane=lane,
        sessions=sessions,
        session_id=session.id,
        agent_home=agent_home,
        workspace=workspace,
        completes=False,
    )

    assert type(payloads[-1]).__name__ == "TurnFailedPayload"
    assert artifact_path.read_text(encoding="utf-8") == "oversized artifact contents"
