"""Shared wall-clock seam for cron schedulers."""

import asyncio
import time as _time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, tzinfo
from time import monotonic
from typing import Protocol


class SchedulerClock(Protocol):
    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


class AsyncioSchedulerClock:
    """Use an injected wall clock in the host's current local timezone."""

    def __init__(self, *, now: Callable[[], datetime]) -> None:
        self._now = now
        self._timezone = _system_timezone()

    def now(self) -> datetime:
        return self._now().astimezone(self._timezone)

    def monotonic(self) -> float:
        return monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


def _system_timezone() -> tzinfo:
    """Return a rule-bearing adapter for the host's system-local timezone."""
    return _SystemLocalTimezone()


class _SystemLocalTimezone(tzinfo):
    """Resolve host-local rules for each datetime through the standard library."""

    def fromutc(self, value: datetime) -> datetime:
        if value.tzinfo is not self:
            raise ValueError("fromutc() requires a datetime using this timezone")
        local = _host_local(value.replace(tzinfo=UTC))
        return local.replace(tzinfo=self)

    def utcoffset(self, value: datetime | None) -> timedelta | None:
        local = self._local_wall_time(value)
        return None if local is None else local.utcoffset()

    def dst(self, value: datetime | None) -> timedelta | None:
        local = self._local_wall_time(value)
        if local is None:
            return None
        daylight = local.dst()
        if daylight is not None:
            return daylight
        offset = local.utcoffset()
        if offset is None:
            return None
        return offset - timedelta(seconds=-_time.timezone)

    def tzname(self, value: datetime | None) -> str | None:
        local = self._local_wall_time(value)
        return None if local is None else local.tzname()

    @staticmethod
    def _local_wall_time(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _host_local(value.replace(tzinfo=None))


def _host_local(value: datetime) -> datetime:
    """Apply the host's timezone rules to an aware instant or naive local wall time."""
    return value.astimezone()
