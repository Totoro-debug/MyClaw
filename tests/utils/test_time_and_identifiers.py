from datetime import datetime, timedelta, timezone
from uuid import UUID

from myclaw.session.identifiers import make_session_id
from myclaw.utils.time import format_rfc3339_milliseconds


def test_time_and_session_id_use_the_frozen_persisted_formats() -> None:
    local_time = datetime(
        2026,
        7,
        11,
        15,
        30,
        12,
        123456,
        tzinfo=timezone(timedelta(hours=8)),
    )
    session_uuid = UUID("550e8400-e29b-41d4-a716-446655440000")

    assert format_rfc3339_milliseconds(local_time) == "2026-07-11T15:30:12.123+08:00"
    assert (
        make_session_id(local_time, session_uuid)
        == "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000"
    )
