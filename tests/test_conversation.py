import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent_home import AgentHome
from myclaw.contracts import (
    AgentEvent,
    AssistantModelMessage,
    AssistantSessionMessage,
    ConversationPort,
    ErrorInfo,
    ModelCallError,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    SessionError,
    TextDelta,
    TextDeltaPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    UserSessionMessage,
    validate_agent_event_sequence,
)
from myclaw.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.repl import run_repl
from myclaw.session_store import JsonlSessionStore
from myclaw.workspace import Workspace
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
USER_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
REQUEST_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
ASSISTANT_UUID = UUID("a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34")
TURN_TWO_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
USER_TWO_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")
REQUEST_TWO_UUID = UUID("886313e1-3b8a-4a2d-9f7f-77611a4b6f4e")
ASSISTANT_TWO_UUID = UUID("b3f37212-6f3a-4a1b-8d2e-78ab3f9c4567")


class CancellableThenSuccessfulProvider:
    def __init__(self, second_response: ModelResponse) -> None:
        self._second_response = second_response
        self._stream_count = 0
        self.first_stream_waiting = asyncio.Event()
        self.stream_requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.stream_requests.append(request)
        self._stream_count += 1
        if self._stream_count == 1:
            yield TextDelta(delta="Partial answer")
            self.first_stream_waiting.set()
            await asyncio.Event().wait()
            return
        yield ModelCompleted(response=self._second_response)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_nonblank_turn_streams_ordered_deltas_then_persists_one_completed_assistant(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    usage = ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144)
    response = ModelResponse(
        message=AssistantModelMessage(content="I will inspect the files."),
        usage=usage,
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        streams=[
            StreamScript(
                events=(
                    TextDelta(delta="I will "),
                    TextDelta(delta="inspect the files."),
                    ModelCompleted(response=response),
                )
            )
        ]
    )
    turn_uuids = iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID))
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=turn_uuids.__next__,
    )
    events = conversation.submit("Help me inspect this project.")

    started = await anext(events)
    first_delta = await anext(events)
    after_first_delta = await store.load(session.id)
    second_delta = await anext(events)
    after_second_delta = await store.load(session.id)
    completed = await anext(events)
    with pytest.raises(StopAsyncIteration):
        await anext(events)

    observed = (started, first_delta, second_delta, completed)
    assert [event.type for event in observed] == [
        "turn_started",
        "text_delta",
        "text_delta",
        "turn_completed",
    ]
    assert [event.event_id for event in observed] == [0, 1, 2, 3]
    assert [event.turn_id for event in observed] == [TURN_UUID] * 4
    assert isinstance(first_delta.payload, TextDeltaPayload)
    assert isinstance(second_delta.payload, TextDeltaPayload)
    assert [first_delta.payload.delta, second_delta.payload.delta] == [
        "I will ",
        "inspect the files.",
    ]
    assert isinstance(completed.payload, TurnCompletedPayload)
    assert completed.payload.content == "I will inspect the files."
    assert completed.payload.usage == usage
    validate_agent_event_sequence(observed)

    assert [message.role for message in after_first_delta.messages] == ["user"]
    assert [message.role for message in after_second_delta.messages] == ["user"]
    reloaded = await store.load(session.id)
    assert reloaded.messages == (
        UserSessionMessage(
            id=str(USER_UUID),
            created_at=NOW,
            content="Help me inspect this project.",
        ),
        AssistantSessionMessage(
            id=str(ASSISTANT_UUID),
            created_at=NOW,
            content="I will inspect the files.",
            tool_calls=(),
            status="completed",
            error=None,
            usage=usage,
        ),
    )
    assert len([message for message in reloaded.messages if message.role == "assistant"]) == 1
    assert len(provider.stream_requests) == 1
    request = provider.stream_requests[0]
    assert isinstance(request, ModelRequest)
    assert request.route == "chat"
    assert request.stream is True
    assert request.model == "test-model"
    assert [message.to_dict() for message in request.messages] == [
        {
            "role": "user",
            "content": (
                "<runtime_context>\n"
                "current_time: 2026-07-11T15:30:12.123+08:00\n"
                f"session_id: {session.id}\n"
                "</runtime_context>\n\n"
                "<user_input>\n"
                "Help me inspect this project.\n"
                "</user_input>"
            ),
        }
    ]
    assert provider.complete_requests == []


@pytest.mark.asyncio
async def test_consecutive_turns_send_raw_short_term_memory_and_wrap_only_current_input(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    first_usage = ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15)
    second_usage = ModelUsage(input_tokens=19, output_tokens=4, total_tokens=23)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="First answer."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="First answer."),
                            usage=first_usage,
                            finish_reason="stop",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    TextDelta(delta="Second answer."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Second answer."),
                            usage=second_usage,
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
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
                TURN_TWO_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
    )

    first_events = [event async for event in conversation.submit("First raw input.")]
    clock.advance(65.333)
    second_events = [event async for event in conversation.submit("Second raw input.")]

    assert [event.type for event in first_events] == [
        "turn_started",
        "text_delta",
        "turn_completed",
    ]
    assert [event.type for event in second_events] == [
        "turn_started",
        "text_delta",
        "turn_completed",
    ]
    assert [
        [message.to_dict() for message in request.messages]
        for request in provider.stream_requests
        if isinstance(request, ModelRequest)
    ] == [
        [
            {
                "role": "user",
                "content": (
                    "<runtime_context>\n"
                    "current_time: 2026-07-11T15:30:12.123+08:00\n"
                    f"session_id: {session.id}\n"
                    "</runtime_context>\n\n"
                    "<user_input>\n"
                    "First raw input.\n"
                    "</user_input>"
                ),
            }
        ],
        [
            {"role": "user", "content": "First raw input."},
            {"role": "assistant", "content": "First answer.", "tool_calls": []},
            {
                "role": "user",
                "content": (
                    "<runtime_context>\n"
                    "current_time: 2026-07-11T15:31:17.456+08:00\n"
                    f"session_id: {session.id}\n"
                    "</runtime_context>\n\n"
                    "<user_input>\n"
                    "Second raw input.\n"
                    "</user_input>"
                ),
            },
        ],
    ]
    reloaded = await store.load(session.id)
    assert reloaded.metadata.id == session.id
    assert reloaded.messages == (
        UserSessionMessage(
            id=str(USER_UUID),
            created_at=NOW,
            content="First raw input.",
        ),
        AssistantSessionMessage(
            id=str(ASSISTANT_UUID),
            created_at=NOW,
            content="First answer.",
            tool_calls=(),
            status="completed",
            error=None,
            usage=first_usage,
        ),
        UserSessionMessage(
            id=str(USER_TWO_UUID),
            created_at=datetime(2026, 7, 11, 15, 31, 17, 456000, tzinfo=LOCAL_OFFSET),
            content="Second raw input.",
        ),
        AssistantSessionMessage(
            id=str(ASSISTANT_TWO_UUID),
            created_at=datetime(2026, 7, 11, 15, 31, 17, 456000, tzinfo=LOCAL_OFFSET),
            content="Second answer.",
            tool_calls=(),
            status="completed",
            error=None,
            usage=second_usage,
        ),
    )


@pytest.mark.asyncio
async def test_final_model_failure_emits_one_failed_terminal_and_persists_safe_error(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    safe_error = ErrorInfo(
        code="provider_timeout",
        message="The model timed out after all attempts.",
    )
    failure = ModelCallError(safe_error)
    failure.__cause__ = RuntimeError("unsafe SDK response body")
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(), error=failure),))
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
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

    events = [event async for event in conversation.submit("Keep this raw input.")]

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    assert [event.event_id for event in events] == [0, 1]
    assert [event.turn_id for event in events] == [TURN_UUID, TURN_UUID]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == safe_error
    validate_agent_event_sequence(events)
    reloaded = await store.load(session.id)
    assert reloaded.messages == (
        UserSessionMessage(
            id=str(USER_UUID),
            created_at=NOW,
            content="Keep this raw input.",
        ),
        AssistantSessionMessage(
            id=str(ASSISTANT_UUID),
            created_at=NOW,
            content="",
            tool_calls=(),
            status="error",
            error=SessionError(
                code="provider_timeout",
                message="The model timed out after all attempts.",
            ),
            usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        ),
    )


@pytest.mark.asyncio
async def test_model_failure_after_streamed_text_persists_the_observed_partial_content(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(TextDelta(delta="Observed partial"),),
                error=ModelCallError(
                    ErrorInfo(code="provider_unavailable", message="The model became unavailable.")
                ),
            ),
        )
    )
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
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

    events = [event async for event in conversation.submit("Keep visible partial output.")]

    assert [event.type for event in events] == [
        "turn_started",
        "text_delta",
        "turn_failed",
    ]
    validate_agent_event_sequence(events)
    reloaded = await store.load(session.id)
    assistant = reloaded.messages[-1]
    assert isinstance(assistant, AssistantSessionMessage)
    assert (
        assistant.content,
        assistant.status,
        assistant.error,
    ) == (
        "Observed partial",
        "error",
        SessionError(
            code="provider_unavailable",
            message="The model became unavailable.",
        ),
    )


@pytest.mark.asyncio
async def test_stream_without_completion_still_emits_one_safe_failed_terminal(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = ScriptedFakeProvider(streams=(StreamScript(events=()),))
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
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

    events = [event async for event in conversation.submit("Require a terminal event.")]

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1]
    assert isinstance(failed.payload, TurnFailedPayload)
    assert failed.payload.error == ErrorInfo(
        code="model_failed",
        message="The model stream ended without a complete response.",
    )
    validate_agent_event_sequence(events)
    reloaded = await store.load(session.id)
    assistant = reloaded.messages[-1]
    assert isinstance(assistant, AssistantSessionMessage)
    assert assistant.status == "error"
    assert assistant.error == SessionError(
        code="model_failed",
        message="The model stream ended without a complete response.",
    )


@pytest.mark.asyncio
async def test_turn_after_failure_keeps_raw_user_history_but_omits_pure_error_assistant(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    second_response = ModelResponse(
        message=AssistantModelMessage(content="Recovered answer."),
        usage=ModelUsage(input_tokens=11, output_tokens=2, total_tokens=13),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(),
                error=ModelCallError(
                    ErrorInfo(code="model_failed", message="The model call failed.")
                ),
            ),
            StreamScript(events=(ModelCompleted(response=second_response),)),
        )
    )
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
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
                TURN_TWO_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
    )

    first_events = [event async for event in conversation.submit("First raw input.")]
    second_events = [event async for event in conversation.submit("Try again.")]

    assert [event.type for event in first_events] == ["turn_started", "turn_failed"]
    assert [event.type for event in second_events] == ["turn_started", "turn_completed"]
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    assert [message.to_dict() for message in second_request.messages] == [
        {"role": "user", "content": "First raw input."},
        {
            "role": "user",
            "content": (
                "<runtime_context>\n"
                "current_time: 2026-07-11T15:30:12.123+08:00\n"
                f"session_id: {session.id}\n"
                "</runtime_context>\n\n"
                "<user_input>\n"
                "Try again.\n"
                "</user_input>"
            ),
        },
    ]
    assert [message.role for message in (await store.load(session.id)).messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_cancel_active_turn_persists_partial_then_releases_next_foreground_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    second_response = ModelResponse(
        message=AssistantModelMessage(content="A complete next answer."),
        usage=ModelUsage(input_tokens=16, output_tokens=5, total_tokens=21),
        finish_reason="stop",
    )
    provider = CancellableThenSuccessfulProvider(second_response)
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
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
                TURN_TWO_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
    )

    first_turn = asyncio.create_task(
        _collect_events(conversation.submit("Please start this answer."))
    )
    await provider.first_stream_waiting.wait()
    await conversation.cancel_active_turn()
    overlapping_turn = conversation.submit("Too early.")
    with pytest.raises(RuntimeError, match="foreground turn is already active"):
        await anext(overlapping_turn)
    first_events = await first_turn
    second_events = [event async for event in conversation.submit("Continue cleanly.")]

    assert [event.type for event in first_events] == [
        "turn_started",
        "text_delta",
        "turn_cancelled",
    ]
    assert [event.event_id for event in first_events] == [0, 1, 2]
    assert [event.turn_id for event in first_events] == [TURN_UUID] * 3
    cancelled = first_events[-1]
    assert isinstance(cancelled.payload, TurnCancelledPayload)
    assert cancelled.payload.partial_content == "Partial answer"
    validate_agent_event_sequence(first_events)
    assert [event.type for event in second_events] == ["turn_started", "turn_completed"]
    validate_agent_event_sequence(second_events)

    reloaded = await store.load(session.id)
    interrupted = reloaded.messages[1]
    assert isinstance(interrupted, AssistantSessionMessage)
    assert (
        interrupted.content,
        interrupted.status,
        interrupted.error,
        interrupted.usage,
    ) == (
        "Partial answer",
        "interrupted",
        SessionError(code="turn_cancelled", message="Turn interrupted by user."),
        ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
    )
    second_request = provider.stream_requests[1]
    assert [message.to_dict() for message in second_request.messages] == [
        {"role": "user", "content": "Please start this answer."},
        {
            "role": "assistant",
            "content": "Partial answer\n\n[Turn interrupted by user.]",
            "tool_calls": [],
        },
        {
            "role": "user",
            "content": (
                "<runtime_context>\n"
                "current_time: 2026-07-11T15:30:12.123+08:00\n"
                f"session_id: {session.id}\n"
                "</runtime_context>\n\n"
                "<user_input>\n"
                "Continue cleanly.\n"
                "</user_input>"
            ),
        },
    ]


@pytest.mark.asyncio
async def test_close_waits_for_the_active_foreground_turn_to_finish_cancelling(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = CancellableThenSuccessfulProvider(
        ModelResponse(
            message=AssistantModelMessage(content="unused"),
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="stop",
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
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
    turn = asyncio.create_task(_collect_events(conversation.submit("Stop and persist this turn.")))
    await provider.first_stream_waiting.wait()

    await conversation.close()

    assert turn.done()
    assert [event.type for event in await turn] == [
        "turn_started",
        "text_delta",
        "turn_cancelled",
    ]
    interrupted = (await store.load(session.id)).messages[-1]
    assert isinstance(interrupted, AssistantSessionMessage)
    assert interrupted.status == "interrupted"


@pytest.mark.asyncio
async def test_close_waits_for_turn_cleanup_without_waiting_for_unrelated_caller_work(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = CancellableThenSuccessfulProvider(
        ModelResponse(
            message=AssistantModelMessage(content="unused"),
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="stop",
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
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
    unrelated_work = asyncio.Event()
    turn_finished = asyncio.Event()

    async def caller() -> None:
        await _collect_events(conversation.submit("Close only this turn."))
        turn_finished.set()
        await unrelated_work.wait()

    caller_task = asyncio.create_task(caller())
    await provider.first_stream_waiting.wait()
    try:
        await asyncio.wait_for(conversation.close(), timeout=0.5)

        assert turn_finished.is_set()
        assert not caller_task.done()
    finally:
        unrelated_work.set()
        await caller_task


@pytest.mark.asyncio
async def test_submit_is_rejected_after_conversation_close(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = ScriptedFakeProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=lambda: TURN_UUID,
    )
    await conversation.close()

    events = conversation.submit("This turn must not start.")
    with pytest.raises(RuntimeError, match="Conversation Port is closed"):
        await anext(events)

    assert provider.stream_requests == []


@pytest.mark.asyncio
async def test_repl_writer_failure_closes_the_active_turn_iterator(
    agent_home: Path,
    workspace: Path,
) -> None:
    class OneInput:
        async def read(self) -> str | None:
            return "Start the failing render."

    class FailingWriter:
        async def write_delta(self, delta: str) -> None:
            del delta
            raise OSError("writer failed")

        async def finish_turn(self) -> None:
            return None

        async def write_line(self, content: str) -> None:
            del content

    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = CancellableThenSuccessfulProvider(
        ModelResponse(
            message=AssistantModelMessage(content="A clean second answer."),
            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            finish_reason="stop",
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
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
                TURN_TWO_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
    )

    with pytest.raises(OSError, match="writer failed"):
        await run_repl(
            conversation=conversation,
            input_reader=OneInput(),
            writer=FailingWriter(),
        )

    second = [event async for event in conversation.submit("Start a clean next turn.")]
    assert [event.type for event in second] == ["turn_started", "turn_completed"]


@pytest.mark.asyncio
async def test_typed_cancellation_without_partial_emits_cancelled_but_no_empty_assistant(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(),
                error=ModelCallError(
                    ErrorInfo(code="turn_cancelled", message="The model request was cancelled.")
                ),
            ),
        )
    )
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID)).__next__,
    )

    events = [event async for event in conversation.submit("Cancel without output.")]

    assert [event.type for event in events] == ["turn_started", "turn_cancelled"]
    assert [event.event_id for event in events] == [0, 1]
    cancelled = events[-1]
    assert isinstance(cancelled.payload, TurnCancelledPayload)
    assert cancelled.payload.partial_content == ""
    validate_agent_event_sequence(events)
    reloaded = await store.load(session.id)
    assert reloaded.messages == (
        UserSessionMessage(
            id=str(USER_UUID),
            created_at=NOW,
            content="Cancel without output.",
        ),
    )


@pytest.mark.asyncio
async def test_conversation_port_rejects_an_overlapping_foreground_submit(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="First answer."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="First answer."),
                            usage=ModelUsage(input_tokens=8, output_tokens=3, total_tokens=11),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
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
    first_turn = conversation.submit("First foreground input.")
    overlapping_turn = conversation.submit("Overlapping input.")

    assert (await anext(first_turn)).type == "turn_started"
    with pytest.raises(RuntimeError, match="foreground turn is already active"):
        await anext(overlapping_turn)
    remaining_events = [event async for event in first_turn]

    assert [event.type for event in remaining_events] == ["text_delta", "turn_completed"]
    assert len(provider.stream_requests) == 1
    assert [message.role for message in (await store.load(session.id)).messages] == [
        "user",
        "assistant",
    ]


async def _collect_events(events: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in events]
