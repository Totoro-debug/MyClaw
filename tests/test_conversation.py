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
        {"role": "user", "content": "Help me inspect this project."}
    ]
    assert provider.complete_requests == []
