from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from myclaw.management.service import RuntimeStatus, SessionListingEntry

LOCAL_OFFSET = timezone(timedelta(hours=8))


def test_runtime_status_exposes_the_documented_management_fields() -> None:
    status = RuntimeStatus(
        version="0.1.0",
        chat_model="anthropic-default/model-id",
        chat_reasoning_effort="medium",
        uptime_seconds=123,
        estimated_input_tokens=4200,
        context_window=200000,
        context_used_percent=2.1,
        session_message_count=12,
        last_consolidated=4,
        cumulative_usage={
            "model_calls": 5,
            "input_tokens": 6100,
            "output_tokens": 900,
            "total_tokens": 7000,
        },
    )

    assert status.to_dict() == {
        "version": "0.1.0",
        "chat_model": "anthropic-default/model-id",
        "chat_reasoning_effort": "medium",
        "uptime_seconds": 123,
        "estimated_input_tokens": 4200,
        "context_window": 200000,
        "context_used_percent": 2.1,
        "session_message_count": 12,
        "last_consolidated": 4,
        "cumulative_usage": {
            "model_calls": 5,
            "input_tokens": 6100,
            "output_tokens": 900,
            "total_tokens": 7000,
        },
    }


def test_runtime_status_rejects_negative_or_boolean_counters() -> None:
    status = RuntimeStatus(
        version="0.1.0",
        chat_model="anthropic-default/model-id",
        chat_reasoning_effort="medium",
        uptime_seconds=123,
        estimated_input_tokens=4200,
        context_window=200000,
        context_used_percent=2.1,
        session_message_count=12,
        last_consolidated=4,
        cumulative_usage={
            "model_calls": 5,
            "input_tokens": 6100,
            "output_tokens": 900,
            "total_tokens": 7000,
        },
    )

    with pytest.raises(ValueError, match="uptime_seconds"):
        replace(status, uptime_seconds=-1)
    with pytest.raises(ValueError, match="session_message_count"):
        replace(status, session_message_count=True)


def test_session_summary_exposes_only_picker_fields_and_validates_message_count() -> None:
    summary = SessionListingEntry(
        id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
        title="MyClaw implementation",
        created_at=datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET),
        updated_at=datetime(2026, 7, 11, 15, 31, 2, 456000, tzinfo=LOCAL_OFFSET),
        message_count=12,
    )

    assert summary.to_dict() == {
        "id": "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
        "title": "MyClaw implementation",
        "created_at": "2026-07-11T15:30:12.123+08:00",
        "updated_at": "2026-07-11T15:31:02.456+08:00",
        "message_count": 12,
    }
    with pytest.raises(ValueError, match="message_count"):
        replace(summary, message_count=-1)
