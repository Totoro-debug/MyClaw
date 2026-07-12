from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from myclaw.contracts import (
    AgentEvent,
    BackgroundCompletedPayload,
    ErrorInfo,
    ModelUsage,
    PermissionRequestedPayload,
    ProgressPayload,
    TextDeltaPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnStartedPayload,
    validate_agent_event_sequence,
)

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
        PermissionRequestedPayload(
            request_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            tool_call_id="call_124",
            tool_name="write_file",
            action="write",
            resource="README.md",
            risk_summary="This changes a Workspace file.",
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
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
            "tool_call_id": "call_124",
            "tool_name": "write_file",
            "action": "write",
            "resource": "README.md",
            "risk_summary": "This changes a Workspace file.",
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


def test_agent_event_sequence_accepts_started_ordered_deltas_and_one_terminal() -> None:
    turn_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    created_at = datetime(2026, 7, 11, 15, 30, 13, tzinfo=LOCAL_OFFSET)
    events = (
        AgentEvent(
            type="turn_started",
            event_id=10,
            turn_id=turn_id,
            created_at=created_at,
            payload=TurnStartedPayload(),
        ),
        AgentEvent(
            type="text_delta",
            event_id=11,
            turn_id=turn_id,
            created_at=created_at,
            payload=TextDeltaPayload(delta="Hel"),
        ),
        AgentEvent(
            type="text_delta",
            event_id=13,
            turn_id=turn_id,
            created_at=created_at,
            payload=TextDeltaPayload(delta="lo"),
        ),
        AgentEvent(
            type="turn_completed",
            event_id=14,
            turn_id=turn_id,
            created_at=created_at,
            payload=TurnCompletedPayload(
                content="Hello",
                usage=ModelUsage(input_tokens=10, output_tokens=1, total_tokens=11),
            ),
        ),
    )

    validate_agent_event_sequence(events)


def test_agent_event_sequence_rejects_unstable_ids_turns_and_terminal_events() -> None:
    turn_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    other_turn_id = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
    created_at = datetime(2026, 7, 11, 15, 30, 13, tzinfo=LOCAL_OFFSET)
    started = AgentEvent(
        type="turn_started",
        event_id=10,
        turn_id=turn_id,
        created_at=created_at,
        payload=TurnStartedPayload(),
    )
    delta = AgentEvent(
        type="text_delta",
        event_id=11,
        turn_id=turn_id,
        created_at=created_at,
        payload=TextDeltaPayload(delta="Hello"),
    )
    completed = AgentEvent(
        type="turn_completed",
        event_id=12,
        turn_id=turn_id,
        created_at=created_at,
        payload=TurnCompletedPayload(
            content="Hello",
            usage=ModelUsage(input_tokens=10, output_tokens=1, total_tokens=11),
        ),
    )
    cancelled = AgentEvent(
        type="turn_cancelled",
        event_id=13,
        turn_id=turn_id,
        created_at=created_at,
        payload=TurnCancelledPayload(partial_content="Hello"),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_agent_event_sequence((started, replace(completed, event_id=10)))
    with pytest.raises(ValueError, match="same turn_id"):
        validate_agent_event_sequence((started, replace(completed, turn_id=other_turn_id)))
    with pytest.raises(ValueError, match="turn_started"):
        validate_agent_event_sequence((delta, completed))
    with pytest.raises(ValueError, match="exactly one terminal"):
        validate_agent_event_sequence((started, delta))
    with pytest.raises(ValueError, match="exactly one terminal"):
        validate_agent_event_sequence((started, completed, cancelled))
    with pytest.raises(ValueError, match="terminal event must be last"):
        validate_agent_event_sequence(
            (started, replace(completed, event_id=11), replace(delta, event_id=12))
        )


def test_agent_event_sequence_rejects_background_completion_during_foreground_turn() -> None:
    foreground_turn_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    background_run_id = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
    created_at = datetime(2026, 7, 11, 15, 30, 13, tzinfo=LOCAL_OFFSET)
    started = AgentEvent(
        type="turn_started",
        event_id=20,
        turn_id=foreground_turn_id,
        created_at=created_at,
        payload=TurnStartedPayload(),
    )
    background = AgentEvent(
        type="background_completed",
        event_id=21,
        turn_id=background_run_id,
        created_at=created_at,
        payload=BackgroundCompletedPayload(
            kind="scheduled_work",
            title="Weekly project review",
            session_id=("20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000"),
            status="completed",
            summary="Review complete",
        ),
    )
    completed = AgentEvent(
        type="turn_completed",
        event_id=22,
        turn_id=foreground_turn_id,
        created_at=created_at,
        payload=TurnCompletedPayload(
            content="Done.",
            usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
        ),
    )

    with pytest.raises(ValueError, match="background_completed"):
        validate_agent_event_sequence((started, background, completed))


def test_agent_event_sequence_accepts_background_only_completion() -> None:
    validate_agent_event_sequence(
        (
            AgentEvent(
                type="background_completed",
                event_id=21,
                turn_id=UUID("0f8fad5b-d9cb-469f-a165-70867728950e"),
                created_at=datetime(2026, 7, 11, 15, 30, 13, tzinfo=LOCAL_OFFSET),
                payload=BackgroundCompletedPayload(
                    kind="scheduled_work",
                    title="Weekly project review",
                    session_id=("20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000"),
                    status="completed",
                    summary="Review complete",
                ),
            ),
        )
    )
