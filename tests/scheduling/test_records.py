from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from myclaw.schedule.records import ScheduledWork

LOCAL_OFFSET = timezone(timedelta(hours=8))


def test_scheduled_work_exports_the_exact_seven_key_record() -> None:
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
    assert replace(valid, enabled=False).enabled is False
    for invalid_enabled in (cast(bool, 0), cast(bool, "false")):
        with pytest.raises(ValueError, match="enabled"):
            replace(valid, enabled=invalid_enabled)
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
            session_id="20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000",
        )
