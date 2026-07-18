"""Shared validation for runtime values and persisted records."""

from datetime import datetime
from math import isfinite
from uuid import UUID


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
