from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from myclaw.agent.events import (
    AgentEvent,
    BackgroundCompletedPayload,
    ProgressPayload,
    TextDeltaPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnStartedPayload,
)
from myclaw.errors import ErrorInfo
from myclaw.provider.models import ModelUsage

LOCAL_OFFSET = timezone(timedelta(hours=8))


def test_all_agent_event_payloads_and_the_envelope_match_the_frozen_shapes() -> None:
    usage = ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144)
    error = ErrorInfo(code="model_failed", message="Model call failed.")
    payloads = (
        TurnStartedPayload(),
        TextDeltaPayload(delta="Hello"),
        ProgressPayload(status="running", summary="Preparing context"),
        ToolStartedPayload(
            tool_call_id="call_123",
            tool_name="read_file",
            summary="Reading a file",
        ),
        ToolCompletedPayload(
            tool_call_id="call_123",
            tool_name="read_file",
            status="success",
            summary="File read",
        ),
        TurnCompletedPayload(content="Done.", usage=usage),
        TurnFailedPayload(error=error),
        TurnCancelledPayload(partial_content="Partial answer"),
        BackgroundCompletedPayload(
            kind="scheduled_work",
            title="Weekly project review",
            session_id="20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000",
            status="completed",
            summary="Review complete",
        ),
    )

    assert [payload.to_dict() for payload in payloads] == [
        {},
        {"delta": "Hello"},
        {"status": "running", "summary": "Preparing context"},
        {
            "tool_call_id": "call_123",
            "tool_name": "read_file",
            "summary": "Reading a file",
        },
        {
            "tool_call_id": "call_123",
            "tool_name": "read_file",
            "status": "success",
            "summary": "File read",
        },
        {
            "content": "Done.",
            "usage": {"input_tokens": 120, "output_tokens": 24, "total_tokens": 144},
        },
        {
            "error": {
                "code": "model_failed",
                "message": "Model call failed.",
                "retryable": False,
                "retry_after_seconds": None,
            }
        },
        {"partial_content": "Partial answer"},
        {
            "kind": "scheduled_work",
            "title": "Weekly project review",
            "session_id": "20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000",
            "status": "completed",
            "summary": "Review complete",
        },
    ]

    event = AgentEvent(
        type="text_delta",
        event_id=3,
        turn_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        created_at=datetime(2026, 7, 11, 15, 30, 13, 123000, tzinfo=LOCAL_OFFSET),
        payload=TextDeltaPayload(delta="Hello"),
    )
    assert event.to_dict() == {
        "type": "text_delta",
        "event_id": 3,
        "turn_id": "550e8400-e29b-41d4-a716-446655440000",
        "created_at": "2026-07-11T15:30:13.123+08:00",
        "payload": {"delta": "Hello"},
    }


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
        ProgressPayload(status="running", summary="x" * 241)
