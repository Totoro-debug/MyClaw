import asyncio
import copy
from collections.abc import AsyncIterator
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
    ModelStreamEvent,
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
FOLLOW_UP_REQUEST_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
TURN_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
THIRD_REQUEST_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")


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


class _CloseTrackingProvider:
    def __init__(self) -> None:
        self.stream_closed = False
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        try:
            yield TextDelta(delta="Partial response")
            await asyncio.Event().wait()
        finally:
            self.stream_closed = True

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected completion request: {request!r}")

    async def close(self) -> None:
        return None


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
        new_uuid=iter((REQUEST_UUID, FOLLOW_UP_REQUEST_UUID)).__next__,
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
        new_uuid=iter((TURN_UUID, REQUEST_UUID)).__next__,
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


@pytest.mark.asyncio
async def test_tool_gateway_faults_remain_safe_linked_results_before_model_continues(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    private_argument = "private-tool-argument"
    private_failure = "private-tool-failure"
    calls = (
        ModelToolCall(
            id="call_unknown",
            name="unknown_capability",
            arguments=f'{{"secret":"{private_argument}"}}',
        ),
        ModelToolCall(
            id="call_invalid",
            name="send_notice",
            arguments=f'{{"message":42,"secret":"{private_argument}"}}',
        ),
        ModelToolCall(
            id="call_failed",
            name="send_notice",
            arguments='{"message":"hello"}',
        ),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Trying tools.", tool_calls=calls
                            ),
                            usage=ModelUsage(input_tokens=8, output_tokens=3, total_tokens=11),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Handled safely."),
                            usage=ModelUsage(input_tokens=12, output_tokens=4, total_tokens=16),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    tool = FakeTool(
        name="send_notice",
        description="Send one notice.",
        required=("message",),
        outcomes=(RuntimeError(private_failure),),
    )
    gateway = ToolGateway()
    gateway.register_tools((tool,))
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
        new_uuid=iter((TURN_UUID, REQUEST_UUID, FOLLOW_UP_REQUEST_UUID)).__next__,
        tool_gateway=gateway,
    )

    events = [event async for event in conversation.submit("Use the tools.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    assert len(tool.calls) == 1
    assert [message["tool_call_id"] for message in session.messages[2:5]] == [
        "call_unknown",
        "call_invalid",
        "call_failed",
    ]
    assert [message["content"] for message in session.messages[2:5]] == [
        "The requested tool is not available.",
        "Invalid arguments for send_notice.",
        "send_notice could not complete the request.",
    ]
    assert all(message["status"] == "error" for message in session.messages[2:5])
    assert all("id" not in message for message in session.messages)
    assert private_argument not in repr(events)
    assert private_failure not in repr(events)
    assert private_argument not in repr(session.messages[2:])
    assert private_failure not in repr(session.messages)
    follow_up = provider.stream_requests[1]
    assert isinstance(follow_up, ModelRequest)
    assert [message.to_dict()["tool_call_id"] for message in follow_up.messages[2:5]] == [
        "call_unknown",
        "call_invalid",
        "call_failed",
    ]
    await conversation.close()


@pytest.mark.asyncio
async def test_multiple_tool_calls_keep_mixed_results_and_relationships_in_order(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    calls = tuple(
        ModelToolCall(id=f"call_{index}", name="read_file", arguments="{}") for index in range(3)
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=calls),
                            usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Mixed results handled."),
                            usage=ModelUsage(input_tokens=9, output_tokens=3, total_tokens=12),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = _ResultGateway(
        ToolResult("call_0", "read_file", "success", "first", None),
        ToolResult("call_1", "read_file", "error", "second", None),
        ToolResult("call_2", "read_file", "refused", "third", None),
    )
    turn = AgentTurn(
        lane="foreground",
        provider=provider,
        session=session,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID, FOLLOW_UP_REQUEST_UUID)).__next__,
        system_prompt="Test system prompt.",
        tool_gateway=gateway,
    )

    payloads = [payload async for payload in turn.run("Read three files.")]

    assert type(payloads[-1]).__name__ == "TurnCompletedPayload"
    assert [message["tool_call_id"] for message in session.messages[2:5]] == [
        "call_0",
        "call_1",
        "call_2",
    ]
    assert [message["status"] for message in session.messages[2:5]] == [
        "success",
        "error",
        "refused",
    ]
    follow_up = provider.stream_requests[1]
    assert isinstance(follow_up, ModelRequest)
    assert [message.to_dict()["content"] for message in follow_up.messages[2:5]] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_multiple_tool_rounds_remain_ordered_in_session_and_provider_history(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    first_call = ModelToolCall(id="call_first", name="read_file", arguments="{}")
    second_call = ModelToolCall(id="call_second", name="read_file", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="First round.", tool_calls=(first_call,)
                            ),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Second round.", tool_calls=(second_call,)
                            ),
                            usage=ModelUsage(input_tokens=7, output_tokens=2, total_tokens=9),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="All done."),
                            usage=ModelUsage(input_tokens=10, output_tokens=3, total_tokens=13),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = _ResultGateway(
        ToolResult("call_first", "read_file", "success", "first result", None),
        ToolResult("call_second", "read_file", "success", "second result", None),
    )
    turn = AgentTurn(
        lane="foreground",
        provider=provider,
        session=session,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID, FOLLOW_UP_REQUEST_UUID, THIRD_REQUEST_UUID)).__next__,
        system_prompt="Test system prompt.",
        tool_gateway=gateway,
    )

    payloads = [payload async for payload in turn.run("Run both rounds.")]

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "ToolStartedPayload",
        "ToolCompletedPayload",
        "ToolStartedPayload",
        "ToolCompletedPayload",
        "TurnCompletedPayload",
    ]
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.messages[2]["tool_call_id"] == first_call.id
    assert session.messages[4]["tool_call_id"] == second_call.id
    third_request = provider.stream_requests[2]
    assert isinstance(third_request, ModelRequest)
    assert [message.role for message in third_request.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


@pytest.mark.asyncio
async def test_tool_execution_cancellation_repairs_every_unfinished_relationship(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    calls = (
        ModelToolCall(id="call_first", name="read_file", arguments="{}"),
        ModelToolCall(id="call_second", name="read_file", arguments="{}"),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Starting.", tool_calls=calls),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    turn = AgentTurn(
        lane="foreground",
        provider=provider,
        session=session,
        settings=_ModelSettings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="Test system prompt.",
        tool_gateway=_ResultGateway(asyncio.CancelledError()),
    )

    payloads = [payload async for payload in turn.run("Cancel during execution.")]

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "ToolStartedPayload",
        "TurnCancelledPayload",
    ]
    assert [message["tool_call_id"] for message in session.messages[2:]] == [
        "call_first",
        "call_second",
    ]
    assert all(message["status"] == "error" for message in session.messages[2:])


@pytest.mark.asyncio
async def test_cancellation_after_tool_completion_stops_before_model_continuation(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    call = ModelToolCall(id="call_done", name="read_file", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Reading.", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    gateway = _ResultGateway(
        ToolResult("call_done", "read_file", "success", "complete result", None)
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
        cancel_requested=iter((False, False, True)).__next__,
    )

    payloads = [payload async for payload in turn.run("Cancel after completion.")]

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "ToolStartedPayload",
        "ToolCompletedPayload",
        "TurnCancelledPayload",
    ]
    assert len(provider.stream_requests) == 1
    assert session.messages[2]["tool_call_id"] == call.id
    assert session.messages[2]["status"] == "success"


@pytest.mark.asyncio
async def test_unexpected_provider_failure_is_safe_in_events_and_session_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    private_failure = "private-provider-response"
    provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=RuntimeError(private_failure)),)
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

    payloads = [payload async for payload in turn.run("Fail safely.")]

    assert [type(payload).__name__ for payload in payloads] == [
        "TurnStartedPayload",
        "TurnFailedPayload",
    ]
    assert session.messages[-1]["error"] == {
        "code": "model_failed",
        "message": "The model request failed.",
    }
    assert private_failure not in repr(payloads)
    assert private_failure not in repr(session.messages)


@pytest.mark.asyncio
async def test_closing_stream_repairs_partial_content_and_closes_provider_iterator(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state)
    provider = _CloseTrackingProvider()
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
        new_uuid=iter((TURN_UUID, REQUEST_UUID)).__next__,
    )
    events = conversation.submit("Start a streamed response.")

    assert (await anext(events)).type == "turn_started"
    assert (await anext(events)).type == "text_delta"
    await events.aclose()

    assert provider.stream_closed
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.messages[-1]["content"] == "Partial response"
    assert session.messages[-1]["status"] == "interrupted"
    assert session.messages[-1]["error"] == {
        "code": "turn_cancelled",
        "message": "Turn interrupted by user.",
    }


@dataclass(frozen=True, slots=True)
class _ModelSettings:
    model: str = "test-model"
    max_output: int = 1024
    temperature: float = 0.2
    reasoning_effort: None = None
    timeout_seconds: int = 30
