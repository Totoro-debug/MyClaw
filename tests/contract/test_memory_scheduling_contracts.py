from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from myclaw.contracts import ScheduledWork, SummaryEntry, serialize_scheduled_work

LOCAL_OFFSET = timezone(timedelta(hours=8))


def test_summary_entry_serializes_with_exactly_three_keys() -> None:
    entry = SummaryEntry(
        index=1,
        timestamp=datetime(2026, 7, 11, 16, tzinfo=LOCAL_OFFSET),
        content="The user is implementing MyClaw and prefers a file-first architecture.",
    )

    assert entry.to_json_line() == (
        '{"index":1,"timestamp":"2026-07-11T16:00:00.000+08:00",'
        '"content":"The user is implementing MyClaw and prefers a file-first architecture."}\n'
    )
    assert set(entry.to_dict()) == {"index", "timestamp", "content"}


def test_summary_entry_rejects_invalid_index_time_and_content() -> None:
    with pytest.raises(ValueError, match="index must start at 1"):
        SummaryEntry(index=0, timestamp=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET), content="x")
    with pytest.raises(ValueError, match="timezone-aware"):
        SummaryEntry(index=1, timestamp=datetime(2026, 7, 11), content="x")
    with pytest.raises(ValueError, match="content"):
        SummaryEntry(index=1, timestamp=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET), content="")


def test_scheduled_work_serializes_as_the_exact_seven_key_array_record() -> None:
    work = ScheduledWork(
        id="550e8400-e29b-41d4-a716-446655440000",
        title="Weekly project review",
        cron="0 9 * * 1",
        prompt="Review the current project and summarize open risks.",
        created_at=datetime(2026, 7, 11, 16, tzinfo=LOCAL_OFFSET),
        enabled=True,
        session_id="20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000",
    )

    assert work.to_dict() == {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "title": "Weekly project review",
        "cron": "0 9 * * 1",
        "prompt": "Review the current project and summarize open risks.",
        "created_at": "2026-07-11T16:00:00.000+08:00",
        "enabled": True,
        "session_id": "20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000",
    }
    assert serialize_scheduled_work((work,)) == (
        '[{"id":"550e8400-e29b-41d4-a716-446655440000",'
        '"title":"Weekly project review","cron":"0 9 * * 1",'
        '"prompt":"Review the current project and summarize open risks.",'
        '"created_at":"2026-07-11T16:00:00.000+08:00","enabled":true,'
        '"session_id":"20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000"}]'
    )


def test_scheduled_work_rejects_values_outside_the_record_contract() -> None:
    valid = ScheduledWork(
        id="550e8400-e29b-41d4-a716-446655440000",
        title="Weekly project review",
        cron="0 9 * * 1",
        prompt="Review the current project.",
        created_at=datetime(2026, 7, 11, 16, tzinfo=LOCAL_OFFSET),
        enabled=True,
        session_id="20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000",
    )

    with pytest.raises(ValueError, match="UUID4"):
        replace(valid, id="123e4567-e89b-12d3-a456-426614174000")
    with pytest.raises(ValueError, match="title"):
        replace(valid, title="")
    with pytest.raises(ValueError, match="120"):
        replace(valid, title="x" * 121)
    with pytest.raises(ValueError, match="20000"):
        replace(valid, prompt="x" * 20001)
    with pytest.raises(ValueError, match="5-field"):
        replace(valid, cron="0 0 9 * * 1")
    with pytest.raises(ValueError, match="enabled"):
        replace(valid, enabled=False)
    with pytest.raises(ValueError, match="Session ID"):
        replace(
            valid,
            session_id="20260711-160000-000000_123e4567-e89b-12d3-a456-426614174000",
        )


def test_scheduled_work_rejects_range_invalid_five_field_cron() -> None:
    with pytest.raises(ValueError, match="valid 5-field"):
        ScheduledWork(
            id="550e8400-e29b-41d4-a716-446655440000",
            title="Invalid schedule",
            cron="99 99 99 99 99",
            prompt="This must never be persisted.",
            created_at=datetime(2026, 7, 11, 16, tzinfo=LOCAL_OFFSET),
            enabled=True,
            session_id=("20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000"),
        )
