"""Deterministic clock for tests that must not use wall-clock delays."""

import asyncio
from datetime import datetime, timedelta


class FakeClock:
    """Control timezone-aware wall time and monotonic elapsed time together."""

    def __init__(self, start: datetime) -> None:
        if start.utcoffset() is None:
            msg = "FakeClock requires a timezone-aware start time"
            raise ValueError(msg)
        self._now = start
        self._monotonic = 0.0
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            msg = "FakeClock cannot move backwards"
            raise ValueError(msg)
        self._now += timedelta(seconds=seconds)
        self._monotonic += seconds

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)
        await asyncio.sleep(0)
