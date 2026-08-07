"""Shared wall-clock seam for cron schedulers."""

import asyncio
from collections.abc import Callable
from datetime import datetime
from time import monotonic
from typing import Protocol

from tzlocal import get_localzone


class SchedulerClock(Protocol):
    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


class AsyncioSchedulerClock:
    """Use an injected wall clock in the system local timezone."""

    def __init__(self, *, now: Callable[[], datetime]) -> None:
        self._now = now
        self._timezone = get_localzone()

    def now(self) -> datetime:
        return self._now().astimezone(self._timezone)

    def monotonic(self) -> float:
        return monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
