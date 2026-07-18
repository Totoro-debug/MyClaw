"""Conversation Session ID generation and validation."""

import re
from datetime import datetime
from uuid import UUID

from myclaw.utils.validation import (
    require_aware_datetime,
    require_uuid4,
    require_uuid4_string,
)

_SESSION_ID_PATTERN = re.compile(
    r"(?P<timestamp>\d{8}-\d{6}-\d{6})_"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def require_session_id(value: str, *, field: str = "session_id") -> None:
    """Validate the complete timestamp-plus-UUID4 Conversation Session ID."""
    match = _SESSION_ID_PATTERN.fullmatch(value)
    if match is None:
        msg = f"{field} must be a valid Session ID"
        raise ValueError(msg)
    try:
        datetime.strptime(match.group("timestamp"), "%Y%m%d-%H%M%S-%f")
        require_uuid4_string(match.group("uuid"), field=field)
    except ValueError as exc:
        msg = f"{field} must be a valid Session ID"
        raise ValueError(msg) from exc


def make_session_id(created_at: datetime, session_uuid: UUID) -> str:
    """Build the frozen local timestamp plus UUID4 Conversation Session ID."""
    require_aware_datetime(created_at, field="created_at")
    require_uuid4(session_uuid, field="session_uuid")
    return f"{created_at:%Y%m%d-%H%M%S-%f}_{session_uuid}"
