from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent_home import AgentHome
from myclaw.contracts import (
    AssistantModelMessage,
    AssistantSessionMessage,
    ConversationPort,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextDelta,
    TextDeltaPayload,
    TurnCompletedPayload,
    UserSessionMessage,
    validate_agent_event_sequence,
)
from myclaw.conversation import ChatModelSettings, StreamingConversationPort
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
