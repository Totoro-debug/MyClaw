import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from myclaw.agent.events import (
    AgentEvent,
    TurnCancelledPayload,
    TurnFailedPayload,
)
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
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
from myclaw.session.records import (
    AssistantSessionMessage,
    ConversationSession,
    SessionError,
    SessionMessage,
    ToolSessionMessage,
)
from myclaw.session.session_store import JsonlSessionStore
from myclaw.tools.base import BaseTool
from myclaw.tools.models import ModelToolCall, ToolResult
from myclaw.tools.tool_artifacts import externalize_tool_result
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures import (
    FakeClock,
    FakeTool,
    ScriptedFakeProvider,
    StreamScript,
    validate_agent_event_sequence,
)
from tests.fixtures.diagnostic_capture import capture_diagnostics

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
USER_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
REQUEST_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
ASSISTANT_UUID = UUID("a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34")
TOOL_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
REPAIR_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")
RETRY_REPAIR_UUID = UUID("84f40c92-f82d-4ce8-a57e-7d6f476893ed")
ERROR_UUID = UUID("3d813cbb-47fb-45df-91b5-0f4b6c7f7648")
REQUEST_TWO_UUID = UUID("886313e1-3b8a-4a2d-9f7f-77611a4b6f4e")
ASSISTANT_TWO_UUID = UUID("b3f37212-6f3a-4a1b-8d2e-78ab3f9c4567")


def _gateway(*tools: BaseTool) -> ToolGateway:
    gateway = ToolGateway()
    gateway.register_tools(tuple(tools))
    return gateway


def _io_path(path: Path) -> Path:
    return HOST_FILESYSTEM.path_for_io(path)


def _externalizer(*, workspace: Path, session_id: str) -> Callable[[ToolResult], ToolResult]:
    workspace_state = WorkspaceState(Workspace.from_path(workspace))
    workspace_state.initialize(agent_home_root=Path.home() / ".myclaw")

    def externalize(result: ToolResult) -> ToolResult:
        return externalize_tool_result(
            result,
            workspace_state=workspace_state,
            session_id=session_id,
            max_tool_result_chars=1,
        )

    return externalize


def _artifact_path(*, workspace: Path, session_id: str, filename: str) -> Path:
    state = WorkspaceState(Workspace.from_path(workspace))
    return state.sessions_directory / "artifacts" / session_id / filename


class InitialAppendFailingStore(JsonlSessionStore):
    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        del session_id, message
        raise OSError("private-api-key-and-traceback-detail")


class BlockingInitialAppendStore(JsonlSessionStore):
    append_started: asyncio.Event

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self.append_started.set()
        await asyncio.Event().wait()
        await super().append_message(session_id, message)


class AssistantAppendFailingStore(JsonlSessionStore):
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count == 2:
            raise OSError("private-assistant-publication-detail")
        await super().append_message(session_id, message)


class AssistantAppendThenFailingStore(JsonlSessionStore):
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        await super().append_message(session_id, message)
        if self._append_count == 2:
            raise OSError("private-post-assistant-publication-detail")


class AssistantAppendThenBlockingStore(JsonlSessionStore):
    assistant_append_durable: asyncio.Event
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        await super().append_message(session_id, message)
        if self._append_count == 2:
            self.assistant_append_durable.set()
            await asyncio.Event().wait()


class BlockingModelErrorAppendStore(JsonlSessionStore):
    error_append_started: asyncio.Event
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count == 2:
            self.error_append_started.set()
            await asyncio.Event().wait()
        await super().append_message(session_id, message)


class ModelErrorAppendThenBlockingStore(JsonlSessionStore):
    error_append_durable: asyncio.Event
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        await super().append_message(session_id, message)
        if self._append_count == 2:
            self.error_append_durable.set()
            await asyncio.Event().wait()


class LoadFailingStore(JsonlSessionStore):
    async def load(self, session_id: str) -> ConversationSession:
        del session_id
        raise ValueError("private-corrupt-jsonl-content")


class ToolAppendFailingStore(JsonlSessionStore):
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count == 3:
            assert isinstance(message, ToolSessionMessage)
            assert message.artifact is not None
            raise OSError("private-tool-result-publication-detail")
        await super().append_message(session_id, message)


class ToolAppendAndRepairFailingStore(JsonlSessionStore):
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count >= 3:
            raise OSError("private-persistent-tool-publication-detail")
        await super().append_message(session_id, message)


class ToolAppendThenFailingStore(JsonlSessionStore):
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        await super().append_message(session_id, message)
        if self._append_count == 3:
            raise OSError("private-post-publication-detail")


class ToolAppendAndReconcileLoadFailingStore(JsonlSessionStore):
    _append_count = 0
    _fail_next_load = False

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count == 3:
            self._fail_next_load = True
            raise OSError("private-tool-publication-detail")
        await super().append_message(session_id, message)

    async def load(self, session_id: str) -> ConversationSession:
        if self._fail_next_load:
            self._fail_next_load = False
            raise OSError("private-reconciliation-load-detail")
        return await super().load(session_id)


class SecondRepairAppendThenFailingStore(JsonlSessionStore):
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count == 3:
            raise OSError("private-original-tool-publication-detail")
        await super().append_message(session_id, message)
        if self._append_count == 5:
            raise OSError("private-post-repair-publication-detail")


class SecondRepairAppendFailingOnceStore(JsonlSessionStore):
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count in {3, 5}:
            raise OSError("private-transient-repair-publication-detail")
        await super().append_message(session_id, message)


class ToolAppendWithBlockingReconcileStore(JsonlSessionStore):
    reconcile_started: asyncio.Event
    _append_count = 0
    _block_next_load = False

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count == 3:
            self._block_next_load = True
            raise OSError("private-tool-publication-detail")
        await super().append_message(session_id, message)

    async def load(self, session_id: str) -> ConversationSession:
        if self._block_next_load:
            self._block_next_load = False
            self.reconcile_started.set()
            await asyncio.Event().wait()
        return await super().load(session_id)


class ToolAppendThenBlockingStore(JsonlSessionStore):
    tool_append_durable: asyncio.Event
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        await super().append_message(session_id, message)
        if self._append_count == 3:
            self.tool_append_durable.set()
            await asyncio.Event().wait()


class RepairAppendThenBlockingStore(JsonlSessionStore):
    repair_append_durable: asyncio.Event
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count == 3:
            raise OSError("private-original-tool-publication-detail")
        await super().append_message(session_id, message)
        if self._append_count == 4:
            self.repair_append_durable.set()
            await asyncio.Event().wait()


class SecondCancellationRepairAppendFailingOnceStore(JsonlSessionStore):
    _append_count = 0

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        self._append_count += 1
        if self._append_count == 4:
            raise OSError("private-transient-cancellation-repair-detail")
        await super().append_message(session_id, message)


class CloseTrackingProvider:
    def __init__(self) -> None:
        self.stream_closed = False

    async def stream(self, request: object):  # type: ignore[no-untyped-def]
        del request
        try:
            yield TextDelta(delta="partial")
            await asyncio.Event().wait()
        finally:
            self.stream_closed = True

    async def complete(self, request: object) -> ModelResponse:
        raise AssertionError(f"Unexpected completion request: {request!r}")

    async def close(self) -> None:
        return None


class CloseTrackedCompletionStream:
    def __init__(self, event: ModelStreamEvent, closed: list[bool]) -> None:
        self._event = event
        self._closed = closed
        self._emitted = False

    def __aiter__(self) -> "CloseTrackedCompletionStream":
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._emitted:
            raise StopAsyncIteration
        self._emitted = True
        return self._event

    async def aclose(self) -> None:
        self._closed[0] = True


class FailingCloseCompletionStream(CloseTrackedCompletionStream):
    async def aclose(self) -> None:
        try:
            raise OSError("private-stream-close-cause")
        except OSError as error:
            raise RuntimeError("private-stream-close-detail") from error


class CleanupFailingProvider:
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        return FailingCloseCompletionStream(
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Completed safely."),
                    usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                    finish_reason="stop",
                )
            ),
            [False],
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected completion request: {request!r}")

    async def close(self) -> None:
        return None


class TitleCloseTrackingProvider:
    def __init__(self) -> None:
        self._title_closed = [False]
        self._streams: list[AsyncIterator[ModelStreamEvent]] = []

    @property
    def title_stream_closed(self) -> bool:
        return self._title_closed[0]

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        is_title = request.system_prompt == "Generate one title."
        event = ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(
                    content="Generated title" if is_title else "Main response."
                ),
                usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                finish_reason="stop",
            )
        )
        stream = CloseTrackedCompletionStream(
            event,
            self._title_closed if is_title else [False],
        )
        self._streams.append(stream)
        return stream

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected completion request: {request!r}")

    async def close(self) -> None:
        return None


class BlockingCloseCompletionStream(CloseTrackedCompletionStream):
    def __init__(self, event: ModelStreamEvent, close_started: asyncio.Event) -> None:
        super().__init__(event, [False])
        self._close_started = close_started
        self._close_attempts = 0

    async def aclose(self) -> None:
        self._close_attempts += 1
        if self._close_attempts == 1:
            self._close_started.set()
            await asyncio.Event().wait()


class BlockingRoundCloseProvider:
    def __init__(self, tool_call: ModelToolCall) -> None:
        self.close_started = asyncio.Event()
        self.stream_requests: list[ModelRequest] = []
        self._tool_call = tool_call

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.stream_requests.append(request)
        if len(self.stream_requests) == 1:
            return BlockingCloseCompletionStream(
                ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(
                            content="First round.",
                            tool_calls=(self._tool_call,),
                        ),
                        usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                        finish_reason="tool_calls",
                    )
                ),
                self.close_started,
            )
        return CloseTrackedCompletionStream(
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Second round must not run."),
                    usage=ModelUsage(input_tokens=4, output_tokens=3, total_tokens=7),
                    finish_reason="stop",
                )
            ),
            [False],
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected completion request: {request!r}")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_initial_session_publication_failure_is_one_safe_terminal_event(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = InitialAppendFailingStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    provider = ScriptedFakeProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID)).__next__,
    )

    capture = capture_diagnostics()

    with capture.session(session.id):
        events = [event async for event in conversation.submit("Do not leak disk details.")]
    capture.close()

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == ErrorInfo(
        code="persistence_error",
        message="Conversation Session could not be updated.",
    )
    assert "private-api-key-and-traceback-detail" not in str(failed.to_dict())
    assert provider.stream_requests == []
    assert not sessions.path_for(session.id).exists()
    content = capture.text
    event_text = capture.event_text
    records = [line for line in content.splitlines() if "myclaw.agent.turn:" in line]
    assert len(records) == 1
    assert (
        "Agent Turn failed code=persistence_error operation=session_append type=OSError"
        in records[0]
    )
    assert "Traceback (most recent call last)" in content
    assert "OSError" in content
    assert "private-api-key-and-traceback-detail" not in event_text
    assert "private-api-key-and-traceback-detail" in content
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_closing_conversation_stream_closes_provider_iterator_immediately(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    provider = CloseTrackingProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )
    capture = capture_diagnostics()

    with capture.session(session.id):
        stream = conversation.submit("Start a streamed response.")
        assert (await anext(stream)).type == "turn_started"
        assert (await anext(stream)).type == "text_delta"
        await stream.aclose()
    capture.close()

    assert provider.stream_closed is True
    assert capture.text == ""


@pytest.mark.asyncio
async def test_foreground_stream_cleanup_failure_is_logged_as_an_independent_error(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    conversation = StreamingConversationPort(
        provider=CleanupFailingProvider(),
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )
    capture = capture_diagnostics()

    with capture.session(session.id):
        events = [event async for event in conversation.submit("Complete the turn.")]
    capture.close()

    assert [event.type for event in events] == ["turn_started", "turn_completed"]
    content = capture.text
    event_text = capture.event_text
    records = [line for line in content.splitlines() if "myclaw.agent.turn:" in line]
    assert len(records) == 1
    assert (
        "Agent Turn cleanup failed code=model_failed operation=provider_stream_close "
        "type=RuntimeError" in records[0]
    )
    assert "Traceback (most recent call last)" in content
    assert "OSError" in content
    assert "RuntimeError" in content
    assert "private-stream-close-cause" not in event_text
    assert "private-stream-close-detail" not in event_text
    assert "private-stream-close-cause" in content
    assert "private-stream-close-detail" in content


@pytest.mark.asyncio
async def test_title_generation_closes_provider_iterator_after_completion(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    provider = TitleCloseTrackingProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
        title_prompt="Generate one title.",
    )

    capture = capture_diagnostics()

    with capture.session(session.id):
        events = [event async for event in conversation.submit("Name this session.")]
        await conversation.close()
    capture.close()

    assert [event.type for event in events] == ["turn_started", "turn_completed"]
    assert provider.title_stream_closed is True
    assert capture.text == ""


@pytest.mark.asyncio
async def test_closing_conversation_stream_persists_streamed_text_as_interrupted(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    conversation = StreamingConversationPort(
        provider=CloseTrackingProvider(),
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )
    stream = conversation.submit("Start a streamed response.")

    assert (await anext(stream)).type == "turn_started"
    assert (await anext(stream)).type == "text_delta"
    await stream.aclose()

    persisted = await sessions.load(session.id)
    assert [message.role for message in persisted.messages] == ["user", "assistant"]
    interrupted = persisted.messages[-1]
    assert isinstance(interrupted, AssistantSessionMessage)
    assert (interrupted.content, interrupted.status, interrupted.error) == (
        "partial",
        "interrupted",
        SessionError(code="turn_cancelled", message="Turn interrupted by user."),
    )


@pytest.mark.asyncio
async def test_same_task_cancellation_after_turn_start_stops_before_provider(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Too late."),
                            usage=ModelUsage(input_tokens=2, output_tokens=2, total_tokens=4),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )
    stream = conversation.submit("Cancel this turn.")

    assert (await anext(stream)).type == "turn_started"
    await conversation.cancel_active_turn()
    remaining = [event async for event in stream]

    assert [event.type for event in remaining] == ["turn_cancelled"]
    assert provider.stream_requests == []
    assert [message.role for message in (await sessions.load(session.id)).messages] == ["user"]


@pytest.mark.asyncio
async def test_external_cancellation_during_initial_publication_is_one_terminal_event(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = BlockingInitialAppendStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    sessions.append_started = asyncio.Event()
    session = sessions.prepare()
    provider = ScriptedFakeProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID)).__next__,
    )

    async def collect() -> list[AgentEvent]:
        return [event async for event in conversation.submit("Cancel while saving.")]

    turn = asyncio.create_task(collect())
    await sessions.append_started.wait()
    await conversation.cancel_active_turn()
    events = await turn

    assert [event.type for event in events] == ["turn_started", "turn_cancelled"]
    assert provider.stream_requests == []
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("termination", "store_type", "stream_text", "expected_roles"),
    (
        (
            "cancel",
            ToolAppendAndRepairFailingStore,
            False,
            ["user", "assistant"],
        ),
        (
            "close",
            ToolAppendAndRepairFailingStore,
            False,
            ["user", "assistant"],
        ),
        (
            "cancel",
            JsonlSessionStore,
            True,
            ["user", "assistant", "tool"],
        ),
    ),
    ids=("persistent-repair-failure", "close-repair-failure", "streamed-tool-call"),
)
async def test_tool_stage_cancellation_survives_persistent_repair_failure(
    agent_home: Path,
    workspace: Path,
    termination: str,
    store_type: type[JsonlSessionStore],
    stream_text: bool,
    expected_roles: list[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = store_type(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    tool_call = ModelToolCall(
        id="call_cancelled",
        name="inspect",
        arguments='{"query":"private-cancelled-tool-argument"}',
    )
    stream_events = (
        *((TextDelta(delta="About to inspect."),) if stream_text else ()),
        ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(
                    content="About to inspect.",
                    tool_calls=(tool_call,),
                ),
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                finish_reason="tool_calls",
            )
        ),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=stream_events,
            ),
        )
    )
    tool = FakeTool(
        name="inspect",
        description="Inspect one value.",
        required=("query",),
        outcomes=("must not execute",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID, TOOL_UUID, REPAIR_UUID)
        ).__next__,
        tool_gateway=_gateway(tool),
    )
    capture = capture_diagnostics()

    with capture.session(session.id):
        stream = conversation.submit("Cancel before the sensitive tool executes.")
        events: list[AgentEvent] = []
        while not events or events[-1].type != "tool_started":
            events.append(await anext(stream))
        if termination == "cancel":
            await conversation.cancel_active_turn()
            events.append(await anext(stream))
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
            expected_events = ["turn_started"]
            if stream_text:
                expected_events.append("text_delta")
            expected_events.extend(("tool_started", "turn_cancelled"))
            assert [event.type for event in events] == expected_events
            cancelled = events[-1]
            assert isinstance(cancelled.payload, TurnCancelledPayload)
            assert cancelled.payload.partial_content == ""
            validate_agent_event_sequence(events)
        else:
            await stream.aclose()
            assert [event.type for event in events] == ["turn_started", "tool_started"]
    capture.close()
    rendered = str([event.to_dict() for event in events])
    assert "private-cancelled-tool-argument" not in rendered
    assert "private-persistent-tool-publication-detail" not in rendered
    assert tool.calls == []
    persisted = await sessions.load(session.id)
    assert [message.role for message in persisted.messages] == expected_roles
    content = capture.text
    event_text = capture.event_text
    records = [line for line in content.splitlines() if "myclaw.agent.turn:" in line]
    if store_type is ToolAppendAndRepairFailingStore:
        assert len(records) == 1
        assert (
            "Agent Turn failed code=persistence_error "
            "operation=interrupted_state_repair type=OSError" in records[0]
        )
        assert "Traceback (most recent call last)" in content
    else:
        assert records == []
    assert "private-cancelled-tool-argument" not in event_text
    assert "private-persistent-tool-publication-detail" not in event_text


@pytest.mark.asyncio
async def test_corrupt_session_before_model_call_is_a_safe_persistence_terminal(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = LoadFailingStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    provider = ScriptedFakeProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID)).__next__,
    )

    capture = capture_diagnostics()

    with capture.session(session.id):
        events = [event async for event in conversation.submit("Stop before provider use.")]
    capture.close()

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == ErrorInfo(
        code="persistence_error",
        message="Conversation Session could not be read.",
    )
    assert "private-corrupt-jsonl-content" not in str(failed.to_dict())
    assert provider.stream_requests == []
    content = capture.text
    event_text = capture.event_text
    records = [line for line in content.splitlines() if "myclaw.agent.turn:" in line]
    assert len(records) == 1
    assert (
        "Agent Turn failed code=persistence_error operation=session_read type=ValueError"
        in records[0]
    )
    assert "Traceback (most recent call last)" in content
    assert "private-corrupt-jsonl-content" not in event_text
    assert "private-corrupt-jsonl-content" in content
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_model_error_publication_failure_reports_only_persistence_error(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = AssistantAppendFailingStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    failure = ModelCallError(
        ErrorInfo(code="provider_unavailable", message="The provider is unavailable.")
    )
    failure.__cause__ = RuntimeError("private-provider-response-body")
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(), error=failure),))
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )

    capture = capture_diagnostics()

    with capture.session(session.id):
        events = [event async for event in conversation.submit("Fail without leaking.")]
    capture.close()

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == ErrorInfo(
        code="persistence_error",
        message="Conversation Session could not be updated.",
    )
    rendered = str(failed.to_dict())
    assert "private-provider-response-body" not in rendered
    assert "private-assistant-publication-detail" not in rendered
    assert [message.role for message in (await sessions.load(session.id)).messages] == ["user"]
    content = capture.text
    event_text = capture.event_text
    records = [line for line in content.splitlines() if "myclaw.agent.turn:" in line]
    assert len(records) == 2
    assert "Agent Turn failed code=provider_unavailable type=ModelCallError" in records[0]
    assert (
        "Agent Turn failed code=persistence_error operation=terminal_state_append type=OSError"
        in records[1]
    )
    assert content.count("Traceback (most recent call last)") >= 2
    assert "ModelCallError" in content
    assert "OSError" in content
    assert "private-provider-response-body" not in event_text
    assert "private-assistant-publication-detail" not in event_text
    assert "private-provider-response-body" in content
    assert "private-assistant-publication-detail" in content
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_cancellation_during_model_error_publication_is_one_terminal_event(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = BlockingModelErrorAppendStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    sessions.error_append_started = asyncio.Event()
    session = sessions.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(),
                error=ModelCallError(
                    ErrorInfo(code="provider_unavailable", message="Provider unavailable.")
                ),
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )

    async def collect() -> list[AgentEvent]:
        return [event async for event in conversation.submit("Fail, then cancel.")]

    turn = asyncio.create_task(collect())
    await sessions.error_append_started.wait()
    await conversation.cancel_active_turn()
    events = await turn

    assert [event.type for event in events] == ["turn_started", "turn_cancelled"]
    assert [message.role for message in (await sessions.load(session.id)).messages] == ["user"]
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_cancellation_after_durable_model_error_does_not_duplicate_partial_text(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = ModelErrorAppendThenBlockingStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    sessions.error_append_durable = asyncio.Event()
    session = sessions.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(TextDelta(delta="Partial provider text."),),
                error=ModelCallError(
                    ErrorInfo(code="provider_unavailable", message="Provider unavailable.")
                ),
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID, ERROR_UUID)).__next__,
    )

    async def collect() -> list[AgentEvent]:
        return [event async for event in conversation.submit("Fail after partial text.")]

    turn = asyncio.create_task(collect())
    await sessions.error_append_durable.wait()
    await conversation.cancel_active_turn()
    events = await turn

    assert [event.type for event in events] == [
        "turn_started",
        "text_delta",
        "turn_cancelled",
    ]
    persisted = await sessions.load(session.id)
    assistants = [
        message for message in persisted.messages if isinstance(message, AssistantSessionMessage)
    ]
    assert [(message.content, message.status, message.error) for message in assistants] == [
        (
            "Partial provider text.",
            "error",
            SessionError(code="provider_unavailable", message="Provider unavailable."),
        )
    ]
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_foreground_model_failure_logs_safe_metadata_without_redacting_exception(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    private_values = (
        "private-user-message",
        "private-system-prompt",
        "private-memory-content",
        "private-file-content",
        "private-web-content",
        "private-provider-response-body",
        "private-api-key",
    )
    failure = ModelCallError(
        ErrorInfo(code="provider_unavailable", message="The provider is unavailable.")
    )
    failure.__cause__ = RuntimeError(
        f"{private_values[5]} Authorization: Bearer {private_values[6]}"
    )
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(), error=failure),))
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
        system_prompt=" ".join(private_values[1:5]),
    )
    capture = capture_diagnostics()
    with capture.session(session.id):
        events = [event async for event in conversation.submit(private_values[0])]
    capture.close()

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    content = capture.text
    event_text = capture.event_text
    records = [line for line in content.splitlines() if "myclaw.agent.turn:" in line]
    assert len(records) == 1
    assert " ERROR " in records[0]
    assert "Agent Turn failed code=provider_unavailable type=ModelCallError" in records[0]
    assert "Traceback (most recent call last)" in content
    assert "ModelCallError" in content
    assert "RuntimeError" in content
    assert all(value not in event_text for value in private_values)
    assert private_values[5] in content
    assert private_values[6] in content


@pytest.mark.asyncio
async def test_unexpected_provider_stream_failure_is_normalized_without_secret_text(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(),
                error=RuntimeError("api_key=sk-private\nTraceback: SDK internals"),
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )

    events = [event async for event in conversation.submit("Trigger provider failure.")]

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == ErrorInfo(
        code="model_failed",
        message="The model request failed.",
    )
    persisted = await sessions.load(session.id)
    assert "sk-private" not in str([message.to_dict() for message in persisted.messages])
    assert "Traceback" not in str([event.to_dict() for event in events])
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_unsupported_provider_stream_event_is_a_safe_model_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(cast(ModelStreamEvent, "api_key=sk-invalid-event"),),
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )

    events = [event async for event in conversation.submit("Return an invalid event.")]

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == ErrorInfo(
        code="model_failed",
        message="The model request failed.",
    )
    assert "sk-invalid-event" not in str([event.to_dict() for event in events])
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_empty_completed_provider_response_is_a_model_failure_not_disk_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content=""),
                            usage=ModelUsage(input_tokens=2, output_tokens=0, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID, ERROR_UUID)).__next__,
    )

    events = [event async for event in conversation.submit("Return no response.")]

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == ErrorInfo(
        code="model_failed",
        message="The model request failed.",
    )
    persisted = await sessions.load(session.id)
    assert [message.role for message in persisted.messages] == ["user", "assistant"]
    assistant = persisted.messages[-1]
    assert isinstance(assistant, AssistantSessionMessage)
    assert assistant.error == SessionError(
        code="model_failed",
        message="The model request failed.",
    )
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_assistant_publication_failure_never_reports_turn_completed(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = AssistantAppendFailingStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Generated but not durable."),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )
    capture = capture_diagnostics()

    with capture.session(session.id):
        events = [event async for event in conversation.submit("Persist before completing.")]
    capture.close()

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == ErrorInfo(
        code="persistence_error",
        message="Conversation Session could not be updated.",
    )
    assert "private-assistant-publication-detail" not in str(failed.to_dict())
    assert [message.role for message in (await sessions.load(session.id)).messages] == ["user"]
    assert len(provider.stream_requests) == 1
    content = capture.text
    event_text = capture.event_text
    records = [line for line in content.splitlines() if "myclaw.agent.turn:" in line]
    assert len(records) == 1
    assert (
        "Agent Turn failed code=persistence_error operation=session_append type=OSError"
        in records[0]
    )
    assert "private-assistant-publication-detail" not in event_text
    assert "private-assistant-publication-detail" in content
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_durable_assistant_publication_error_repairs_its_tool_call(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = AssistantAppendThenFailingStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    tool_call = ModelToolCall(id="call_durable", name="inspect", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Inspecting.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    tool = FakeTool(
        name="inspect",
        description="Inspect.",
        outcomes=("must-not-run",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID, REPAIR_UUID)).__next__,
        tool_gateway=_gateway(tool),
    )

    events = [event async for event in conversation.submit("Run one durable tool call.")]

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error.code == "persistence_error"
    persisted = await sessions.load(session.id)
    assert [message.role for message in persisted.messages] == ["user", "assistant", "tool"]
    repair = persisted.messages[-1]
    assert isinstance(repair, ToolSessionMessage)
    assert (repair.tool_call_id, repair.status, repair.content) == (
        tool_call.id,
        "error",
        "Assistant response could not be persisted.",
    )
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_cancellation_after_durable_assistant_does_not_duplicate_streamed_text(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = AssistantAppendThenBlockingStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    sessions.assistant_append_durable = asyncio.Event()
    session = sessions.prepare()
    tool_call = ModelToolCall(id="call_after_assistant", name="inspect", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="Streamed response."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Streamed response.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    tool = FakeTool(
        name="inspect",
        description="Inspect.",
        outcomes=("must-not-run",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                ERROR_UUID,
                REPAIR_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(tool),
    )

    async def collect() -> list[AgentEvent]:
        return [event async for event in conversation.submit("Cancel after assistant write.")]

    turn = asyncio.create_task(collect())
    await sessions.assistant_append_durable.wait()
    await conversation.cancel_active_turn()
    events = await turn

    assert [event.type for event in events] == [
        "turn_started",
        "text_delta",
        "turn_cancelled",
    ]
    persisted = await sessions.load(session.id)
    assert [message.role for message in persisted.messages] == ["user", "assistant", "tool"]
    assistants = [
        message for message in persisted.messages if isinstance(message, AssistantSessionMessage)
    ]
    assert [(message.content, message.status) for message in assistants] == [
        ("Streamed response.", "completed")
    ]
    repair = persisted.messages[-1]
    assert isinstance(repair, ToolSessionMessage)
    assert repair.tool_call_id == tool_call.id
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_type", "expected_roles", "artifact_persisted", "expected_tool_status"),
    (
        (ToolAppendFailingStore, ["user", "assistant", "tool"], True, "error"),
        (ToolAppendAndRepairFailingStore, ["user", "assistant"], True, None),
        (ToolAppendThenFailingStore, ["user", "assistant", "tool"], True, "success"),
        (
            ToolAppendAndReconcileLoadFailingStore,
            ["user", "assistant", "tool"],
            True,
            "error",
        ),
    ),
    ids=(
        "repair-publishes-correlated-error",
        "persistent-disk-failure",
        "message-durable-before-error",
        "publication-unknown-repairs-without-discard",
    ),
)
async def test_tool_result_publication_failure_stops_with_correlated_safe_history(
    agent_home: Path,
    workspace: Path,
    store_type: type[JsonlSessionStore],
    expected_roles: list[str],
    artifact_persisted: bool,
    expected_tool_status: str | None,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = store_type(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    tool_call = ModelToolCall(
        id="call_sensitive",
        name="inspect",
        arguments='{"query":"private-raw-tool-argument"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Inspecting.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    tool = FakeTool(
        name="inspect",
        description="Inspect one value.",
        required=("query",),
        outcomes=("private-raw-tool-result",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_UUID,
                REPAIR_UUID,
                RETRY_REPAIR_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(tool),
        externalize_result=_externalizer(
            workspace=workspace,
            session_id=session.id,
        ),
    )
    capture = capture_diagnostics()

    with capture.session(session.id):
        events = [event async for event in conversation.submit("Run one sensitive tool.")]
    capture.close()

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "turn_failed",
    ]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == ErrorInfo(
        code="persistence_error",
        message="Conversation Session could not be updated.",
    )
    rendered = str([event.to_dict() for event in events])
    assert "private-raw-tool-argument" not in rendered
    assert "private-raw-tool-result" not in rendered
    assert "private-tool-result-publication-detail" not in rendered
    assert "private-persistent-tool-publication-detail" not in rendered
    artifact_path = _artifact_path(
        workspace=workspace,
        session_id=session.id,
        filename="call_sensitive.txt",
    )
    assert _io_path(artifact_path).exists() is artifact_persisted
    persisted = await sessions.load(session.id)
    assert [message.role for message in persisted.messages] == expected_roles
    if expected_tool_status is not None:
        tool_message = persisted.messages[-1]
        assert isinstance(tool_message, ToolSessionMessage)
        assert (tool_message.tool_call_id, tool_message.status) == (
            tool_call.id,
            expected_tool_status,
        )
        if expected_tool_status == "error":
            assert tool_message.content == "Tool result could not be persisted."
    assert len(provider.stream_requests) == 1
    content = capture.text
    event_text = capture.event_text
    primary_records = [
        line
        for line in content.splitlines()
        if "myclaw.agent.turn:" in line and "operation=session_append" in line
    ]
    assert len(primary_records) == 1
    assert "Agent Turn failed code=persistence_error" in primary_records[0]
    assert "type=OSError" in primary_records[0]
    assert "private-raw-tool-argument" not in event_text
    assert "private-raw-tool-result" not in event_text
    assert "private-tool-result-publication-detail" not in event_text
    assert "private-persistent-tool-publication-detail" not in event_text
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_type",
    (SecondRepairAppendThenFailingStore, SecondRepairAppendFailingOnceStore),
    ids=("durable-before-error", "transient-before-write"),
)
async def test_repair_publication_fault_does_not_duplicate_or_drop_tool_message(
    agent_home: Path,
    workspace: Path,
    store_type: type[JsonlSessionStore],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = store_type(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    tool_calls = (
        ModelToolCall(id="call_one", name="inspect", arguments="{}"),
        ModelToolCall(id="call_two", name="inspect", arguments="{}"),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Inspecting twice.",
                                tool_calls=tool_calls,
                            ),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    tool = FakeTool(
        name="inspect",
        description="Inspect.",
        outcomes=("first-result",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_UUID,
                REPAIR_UUID,
                RETRY_REPAIR_UUID,
                ERROR_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(tool),
    )

    events = [event async for event in conversation.submit("Run two calls.")]

    assert [event.type for event in events] == ["turn_started", "tool_started", "turn_failed"]
    persisted = await sessions.load(session.id)
    tool_messages = [
        message for message in persisted.messages if isinstance(message, ToolSessionMessage)
    ]
    assert [message.tool_call_id for message in tool_messages] == ["call_one", "call_two"]
    assert [message.content for message in tool_messages] == [
        "Tool result could not be persisted.",
        "Tool result could not be persisted.",
    ]
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_cancellation_during_tool_reconciliation_is_safely_terminal(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = ToolAppendWithBlockingReconcileStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    sessions.reconcile_started = asyncio.Event()
    session = sessions.prepare()
    tool_call = ModelToolCall(id="call_cancel_reconcile", name="inspect", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Inspecting.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    tool = FakeTool(
        name="inspect",
        description="Inspect.",
        outcomes=("private-oversized-result",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID, TOOL_UUID, REPAIR_UUID)
        ).__next__,
        tool_gateway=_gateway(tool),
        externalize_result=_externalizer(
            workspace=workspace,
            session_id=session.id,
        ),
    )

    async def collect() -> list[AgentEvent]:
        return [event async for event in conversation.submit("Cancel reconciliation.")]

    turn = asyncio.create_task(collect())
    await sessions.reconcile_started.wait()
    await conversation.cancel_active_turn()
    events = await turn

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "turn_cancelled",
    ]
    persisted = await sessions.load(session.id)
    repair = persisted.messages[-1]
    assert isinstance(repair, ToolSessionMessage)
    assert (repair.tool_call_id, repair.content) == (
        tool_call.id,
        "Tool call interrupted because the turn was cancelled.",
    )
    artifact_path = _artifact_path(
        workspace=workspace,
        session_id=session.id,
        filename="call_cancel_reconcile.txt",
    )
    assert _io_path(artifact_path).exists()
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_cancellation_while_closing_tool_round_stops_before_next_provider_call(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    tool_call = ModelToolCall(id="call_before_close", name="inspect", arguments="{}")
    provider = BlockingRoundCloseProvider(tool_call)
    tool = FakeTool(
        name="inspect",
        description="Inspect.",
        outcomes=("first-result",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(tool),
    )

    async def collect() -> list[AgentEvent]:
        return [event async for event in conversation.submit("Cancel between rounds.")]

    turn = asyncio.create_task(collect())
    await provider.close_started.wait()
    await conversation.cancel_active_turn()
    events = await turn

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_cancelled",
    ]
    assert len(provider.stream_requests) == 1
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_cancellation_after_durable_tool_result_keeps_one_message_and_artifact(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = ToolAppendThenBlockingStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    sessions.tool_append_durable = asyncio.Event()
    session = sessions.prepare()
    tool_call = ModelToolCall(id="call_durable_cancel", name="inspect", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Inspecting.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    tool = FakeTool(
        name="inspect",
        description="Inspect.",
        outcomes=("private-oversized-result",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID, TOOL_UUID, REPAIR_UUID)
        ).__next__,
        tool_gateway=_gateway(tool),
        externalize_result=_externalizer(
            workspace=workspace,
            session_id=session.id,
        ),
    )

    async def collect() -> list[AgentEvent]:
        return [event async for event in conversation.submit("Cancel after Tool write.")]

    turn = asyncio.create_task(collect())
    await sessions.tool_append_durable.wait()
    await conversation.cancel_active_turn()
    events = await turn

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "turn_cancelled",
    ]
    persisted = await sessions.load(session.id)
    tool_messages = [
        message for message in persisted.messages if isinstance(message, ToolSessionMessage)
    ]
    assert [(message.tool_call_id, message.status) for message in tool_messages] == [
        (tool_call.id, "success")
    ]
    artifact_path = _artifact_path(
        workspace=workspace,
        session_id=session.id,
        filename="call_durable_cancel.txt",
    )
    assert _io_path(artifact_path).exists()
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_cancellation_after_durable_repair_does_not_duplicate_tool_message(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = RepairAppendThenBlockingStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    sessions.repair_append_durable = asyncio.Event()
    session = sessions.prepare()
    tool_call = ModelToolCall(id="call_repair_cancel", name="inspect", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Inspecting.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    tool = FakeTool(
        name="inspect",
        description="Inspect.",
        outcomes=("first-result",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_UUID,
                REPAIR_UUID,
                RETRY_REPAIR_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(tool),
    )

    async def collect() -> list[AgentEvent]:
        return [event async for event in conversation.submit("Cancel after repair write.")]

    turn = asyncio.create_task(collect())
    await sessions.repair_append_durable.wait()
    await conversation.cancel_active_turn()
    events = await turn

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "turn_cancelled",
    ]
    persisted = await sessions.load(session.id)
    tool_messages = [
        message for message in persisted.messages if isinstance(message, ToolSessionMessage)
    ]
    assert [(message.tool_call_id, message.content) for message in tool_messages] == [
        (
            tool_call.id,
            "Tool result could not be persisted.",
        )
    ]
    validate_agent_event_sequence(events)


@pytest.mark.asyncio
async def test_cancellation_retries_a_transient_second_tool_repair_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = SecondCancellationRepairAppendFailingOnceStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = sessions.prepare()
    tool_calls = (
        ModelToolCall(id="call_cancel_one", name="inspect", arguments="{}"),
        ModelToolCall(id="call_cancel_two", name="inspect", arguments="{}"),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Inspecting twice.",
                                tool_calls=tool_calls,
                            ),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    tool = FakeTool(
        name="inspect",
        description="Inspect.",
        outcomes=("must-not-run",),
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                REPAIR_UUID,
                RETRY_REPAIR_UUID,
                ERROR_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(tool),
    )
    stream = conversation.submit("Cancel two calls.")

    assert (await anext(stream)).type == "turn_started"
    assert (await anext(stream)).type == "tool_started"
    await conversation.cancel_active_turn()
    remaining = [event async for event in stream]

    assert [event.type for event in remaining] == ["turn_cancelled"]
    persisted = await sessions.load(session.id)
    tool_messages = [
        message for message in persisted.messages if isinstance(message, ToolSessionMessage)
    ]
    assert [message.tool_call_id for message in tool_messages] == [
        "call_cancel_one",
        "call_cancel_two",
    ]
    assert all(
        message.content == "Tool call interrupted because the turn was cancelled."
        for message in tool_messages
    )
