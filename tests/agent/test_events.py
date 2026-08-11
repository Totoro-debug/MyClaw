from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from myclaw.agent.events import (
    AgentEvent,
    ConfirmationRequestedPayload,
    TextDeltaPayload,
    ToolStartedPayload,
)

LOCAL_OFFSET = timezone(timedelta(hours=8))


def test_agent_events_reject_invalid_envelopes_payload_pairs_and_summaries() -> None:
    valid = AgentEvent(
        type="text_delta",
        event_id=3,
        turn_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        created_at=datetime(2026, 7, 11, 15, 30, 13, 123000, tzinfo=LOCAL_OFFSET),
        payload=TextDeltaPayload(delta="Hello"),
    )

    with pytest.raises(ValueError, match="event_id"):
        replace(valid, event_id=-1)
    with pytest.raises(ValueError, match="UUID4"):
        replace(valid, turn_id=UUID("123e4567-e89b-12d3-a456-426614174000"))
    with pytest.raises(ValueError, match="does not match"):
        replace(valid, type="turn_started")
    with pytest.raises(ValueError, match="delta"):
        TextDeltaPayload(delta="")
    with pytest.raises(ValueError, match="240"):
        ToolStartedPayload(
            tool_call_id="call_123",
            tool_name="read_file",
            summary="x" * 241,
        )

    confirmation = ConfirmationRequestedPayload(
        confirmation_id=UUID("16fd2706-8baf-4334-8c7f-ada847da0314"),
        turn_id=valid.turn_id,
        tool_call_id="call_confirm",
        tool_name="schedule",
        reason="Confirm Schedule Job: Remember this",
        summary="Add a Schedule Job",
        details={"message": "Remember this"},
        warnings=("This changes Workspace State.",),
    )
    assert confirmation.reason == "Confirm Schedule Job: Remember this"
    AgentEvent(
        type="confirmation_requested",
        event_id=4,
        turn_id=valid.turn_id,
        created_at=valid.created_at,
        payload=confirmation,
    )
    with pytest.raises(ValueError, match="turn_id does not match"):
        AgentEvent(
            type="confirmation_requested",
            event_id=4,
            turn_id=UUID("0f8fad5b-d9cb-469f-a165-70867728950e"),
            created_at=valid.created_at,
            payload=confirmation,
        )
