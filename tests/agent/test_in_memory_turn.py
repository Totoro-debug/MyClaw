import asyncio
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.turn import AgentTurn
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextDelta,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session import Session
from myclaw.tools.artifacts import ArtifactReference
from myclaw.tools.models import ModelToolCall, ToolResult, ToolResultStatus
from myclaw.tools.tool_gateway import ToolGateway
from tests.fixtures import FakeTool, ScriptedFakeProvider, StreamScript

LOCAL_TIMEZONE = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 18, 18, 30, 12, 123456, tzinfo=LOCAL_TIMEZONE)
REQUEST_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
ASSISTANT_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
TOOL_MESSAGE_UUID = UUID("a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34")
FOLLOW_UP_REQUEST_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
FINAL_ASSISTANT_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")


class _ResultGateway(ToolGateway):
    def __init__(self, *outcomes: ToolResult | BaseException) -> None:
        super().__init__()
        self._outcomes = list(outcomes)
        self.calls: list[ModelToolCall] = []

    async def call(self, tool_call: ModelToolCall) -> ToolResult:
        self.calls.append(tool_call)
        if not self._outcomes:
            raise AssertionError("No scripted Tool result remains")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_agent_turn_uses_one_active_session_for_the_complete_tool_round(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    tool_call = ModelToolCall(
        id="call_read",
        name="read_file",
        arguments='{"path":"README.md"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Reading.", tool_calls=(tool_call,)
                            ),
                            usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Finished."),
                            usage=ModelUsage(input_tokens=16, output_tokens=5, total_tokens=21),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = ToolGateway()
    gateway.register_tools(
        (
            FakeTool(
                name="read_file",
                description="Read a test file.",
                required=("path",),
                outcomes=("file contents",),
            ),
        )
    )
    persist_calls: list[list[dict[str, object]]] = []
    persist = session.persist

    def record_persist() -> None:
        persist_calls.append(copy.deepcopy(session.messages))
        persist()

    monkeypatch.setattr(session, "persist", record_persist)
    artifact = ArtifactReference(
        path=f"artifacts/{session.session_id}/call_read.txt",
        total_chars=13,
        preview_chars=7,
    )

    def externalize_result(result: ToolResult) -> ToolResult:
        assert result.content == "file contents"
        return ToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            content="preview",
            artifact=artifact,
        )

    turn = AgentTurn(
        lane="foreground",
        provider=provider,
        session=session,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=iter(
            (
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_MESSAGE_UUID,
                FOLLOW_UP_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        system_prompt="Test system prompt.",
        tool_gateway=gateway,
        externalize_result=externalize_result,
    )

    payloads = [payload async for payload in turn.run("Inspect the project.")]

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "ToolStartedPayload",
        "ToolCompletedPayload",
        "TurnCompletedPayload",
    ]
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.messages[1]["content"] == "Reading."
    assert session.messages[1]["tool_calls"] == [tool_call.to_dict()]
    assert session.messages[1]["status"] == "completed"
    assert session.messages[1]["error"] is None
    assert session.messages[1]["token_usage"] == {
        "model_calls": 1,
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }
    assert session.messages[2] | {"timestamp": "ignored"} == {
        "role": "tool",
        "content": "preview",
        "timestamp": "ignored",
        "tool_call_id": "call_read",
        "name": "read_file",
        "status": "success",
        "artifact": artifact.to_dict(),
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 28,
        "output_tokens": 8,
        "total_tokens": 36,
    }
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    assert [message.role for message in second_request.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert len(persist_calls) == 1
    await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("status", "content"),
    (("error", "read_file could not complete."), ("refused", "Permission denied.")),
)
@pytest.mark.asyncio
async def test_agent_turn_preserves_non_success_tool_results_in_the_active_session(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: ToolResultStatus,
    content: str,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    tool_call = ModelToolCall(
        id="call_read",
        name="read_file",
        arguments='{"path":"README.md"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Checking.", tool_calls=(tool_call,)
                            ),
                            usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Handled."),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = _ResultGateway(
        ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            status=status,
            content=content,
            artifact=None,
        )
    )
    persisted_snapshots: list[list[dict[str, object]]] = []

    def record_persist() -> None:
        persisted_snapshots.append(copy.deepcopy(session.messages))

    monkeypatch.setattr(session, "persist", record_persist)
    turn = AgentTurn(
        lane="foreground",
        provider=provider,
        session=session,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="Test system prompt.",
        tool_gateway=gateway,
    )

    payloads = [payload async for payload in turn.run("Check the file.")]

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "ToolStartedPayload",
        "ToolCompletedPayload",
        "TurnCompletedPayload",
    ]
    assert len(gateway.calls) == 1
    assert session.messages[2] | {"timestamp": "ignored"} == {
        "role": "tool",
        "content": content,
        "timestamp": "ignored",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "status": status,
        "artifact": None,
    }
    follow_up = provider.stream_requests[1]
    assert isinstance(follow_up, ModelRequest)
    assert follow_up.messages[2].to_dict() == {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": content,
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 13,
        "output_tokens": 4,
        "total_tokens": 17,
    }
    assert len(persisted_snapshots) == 1


@pytest.mark.asyncio
async def test_agent_turn_records_partial_model_failure_and_one_zero_usage_call(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    failure = ModelCallError(ErrorInfo(code="provider_unavailable", message="Try later."))
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(TextDelta(delta="Partial response"),),
                error=failure,
            ),
        )
    )
    persisted_snapshots: list[list[dict[str, object]]] = []
    monkeypatch.setattr(
        session,
        "persist",
        lambda: persisted_snapshots.append(copy.deepcopy(session.messages)),
    )
    turn = AgentTurn(
        lane="foreground",
        provider=provider,
        session=session,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="Test system prompt.",
        tool_gateway=None,
    )

    payloads = [payload async for payload in turn.run("Start a response.")]

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "TextDeltaPayload",
        "TurnFailedPayload",
    ]
    assert session.messages[1] | {"timestamp": "ignored"} == {
        "role": "assistant",
        "content": "Partial response",
        "timestamp": "ignored",
        "tool_calls": [],
        "status": "error",
        "error": {"code": "provider_unavailable", "message": "Try later."},
        "token_usage": {
            "model_calls": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert len(persisted_snapshots) == 1


@pytest.mark.asyncio
async def test_agent_turn_repairs_every_unfinished_tool_call_before_cancel_terminal(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    tool_calls = (
        ModelToolCall(id="call_first", name="read_file", arguments='{"path":"one"}'),
        ModelToolCall(id="call_second", name="read_file", arguments='{"path":"two"}'),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="I will inspect both.", tool_calls=tool_calls
                            ),
                            usage=ModelUsage(input_tokens=9, output_tokens=4, total_tokens=13),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    gateway = _ResultGateway()
    persisted_snapshots: list[list[dict[str, object]]] = []
    monkeypatch.setattr(
        session,
        "persist",
        lambda: persisted_snapshots.append(copy.deepcopy(session.messages)),
    )
    turn = AgentTurn(
        lane="foreground",
        provider=provider,
        session=session,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="Test system prompt.",
        tool_gateway=gateway,
        cancel_requested=iter((False, True)).__next__,
    )

    payloads = [payload async for payload in turn.run("Inspect both files.")]

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "ToolStartedPayload",
        "TurnCancelledPayload",
    ]
    assert gateway.calls == []
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert session.messages[1]["content"] == "I will inspect both."
    assert session.messages[1]["tool_calls"] == [call.to_dict() for call in tool_calls]
    assert [message["tool_call_id"] for message in session.messages[2:]] == [
        "call_first",
        "call_second",
    ]
    assert all(message["status"] == "error" for message in session.messages[2:])
    assert all(message["artifact"] is None for message in session.messages[2:])
    assert len(persisted_snapshots) == 1
    assert persisted_snapshots[0] == session.messages


@pytest.mark.asyncio
async def test_agent_turn_repairs_partial_session_state_before_one_persist(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="Partial answer"),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Complete answer"),
                            usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    persist_calls: list[list[dict[str, object]]] = []
    persist = session.persist

    def record_persist() -> None:
        persist_calls.append(copy.deepcopy(session.messages))
        persist()

    monkeypatch.setattr(
        session,
        "persist",
        record_persist,
    )
    turn = AgentTurn(
        lane="foreground",
        provider=provider,
        session=session,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID,)).__next__,
        system_prompt="Test system prompt.",
        tool_gateway=None,
        cancel_requested=iter((False, True)).__next__,
    )

    payloads = [payload async for payload in turn.run("Start an answer.")]

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "TextDeltaPayload",
        "TurnCancelledPayload",
    ]
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.messages[1]["status"] == "interrupted"
    assert session.messages[1]["error"] == {
        "code": "turn_cancelled",
        "message": "Turn interrupted by user.",
    }
    assert len(persist_calls) == 1


@pytest.mark.asyncio
async def test_persist_scheduling_failure_does_not_replace_or_add_terminal_events(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Completed."),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    persist_calls = 0

    def fail_persist() -> None:
        nonlocal persist_calls
        persist_calls += 1
        raise OSError("disk unavailable")

    monkeypatch.setattr(session, "persist", fail_persist)
    conversation = StreamingConversationPort(
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
    )

    events = [event async for event in conversation.submit("Complete this turn.")]

    assert [event.type for event in events] == ["turn_started", "turn_completed"]
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_streaming_conversation_port_accepts_the_active_session(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="In memory."),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    persist_calls: list[list[dict[str, object]]] = []
    persist = session.persist

    def record_persist() -> None:
        persist_calls.append(copy.deepcopy(session.messages))
        persist()

    monkeypatch.setattr(session, "persist", record_persist)
    conversation = StreamingConversationPort(
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID, ASSISTANT_UUID, FINAL_ASSISTANT_UUID)).__next__,
    )

    events = conversation.submit("Use the active session.")
    started = await anext(events)
    terminal = await anext(events)

    assert [started.type, terminal.type] == ["turn_started", "turn_completed"]
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert len(persist_calls) == 1
    session_path = state.sessions_directory / f"{session.session_id}.jsonl"
    assert not session_path.exists()
    await events.aclose()
    await asyncio.sleep(0)
    assert session_path.exists()


@dataclass(frozen=True, slots=True)
class _ModelSettings:
    model: str = "test-model"
    max_output: int = 1024
    temperature: float = 0.2
    reasoning_effort: None = None
    timeout_seconds: int = 30
