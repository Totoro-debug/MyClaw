"""Frozen time formatting used by persisted records and runtime events."""

from datetime import datetime

from myclaw.utils.validation import require_aware_datetime


def format_rfc3339_milliseconds(value: datetime) -> str:
    """Format an aware datetime using the frozen persisted time representation."""
    require_aware_datetime(value, field="timestamp")
    return value.isoformat(timespec="milliseconds")
