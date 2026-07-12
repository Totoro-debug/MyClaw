"""Shared validation and formatting for runtime contract records."""

import re
from datetime import datetime
from math import isfinite
from uuid import UUID

_SESSION_ID_PATTERN = re.compile(
    r"(?P<timestamp>\d{8}-\d{6}-\d{6})_"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def require_nonnegative_int(value: int, *, field: str) -> None:
    """Validate persisted counters without accepting bool as an integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{field} must be a nonnegative integer"
        raise ValueError(msg)


def require_nonnegative_number(value: float, *, field: str) -> None:
    """Validate a finite nonnegative numeric measurement without accepting bool."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        or not isfinite(value)
    ):
        msg = f"{field} must be a finite nonnegative number"
        raise ValueError(msg)


def require_aware_datetime(value: datetime, *, field: str) -> None:
    """Reject datetimes that cannot carry the required persisted UTC offset."""
    if value.utcoffset() is None:
        msg = f"{field} must be timezone-aware"
        raise ValueError(msg)


def require_uuid4(value: UUID, *, field: str) -> None:
    """Reject UUIDs outside the canonical lowercase RFC 4122 UUID4 form."""
    if value.version != 4 or str(value) != str(value).lower():
        msg = f"{field} must be a canonical UUID4"
        raise ValueError(msg)


def require_uuid4_string(value: str, *, field: str) -> None:
    """Validate a persisted lowercase, hyphenated UUID4 string."""
    try:
        parsed = UUID(value)
        require_uuid4(parsed, field=field)
    except (ValueError, AttributeError) as exc:
        msg = f"{field} must be a canonical UUID4"
        raise ValueError(msg) from exc
    if str(parsed) != value:
        msg = f"{field} must be a canonical UUID4"
        raise ValueError(msg)


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


def format_rfc3339_milliseconds(value: datetime) -> str:
    """Format an aware datetime using the frozen persisted time representation."""
    require_aware_datetime(value, field="timestamp")
    return value.isoformat(timespec="milliseconds")


def make_session_id(created_at: datetime, session_uuid: UUID) -> str:
    """Build the frozen local timestamp plus UUID4 Conversation Session ID."""
    require_aware_datetime(created_at, field="created_at")
    require_uuid4(session_uuid, field="session_uuid")
    return f"{created_at:%Y%m%d-%H%M%S-%f}_{session_uuid}"
