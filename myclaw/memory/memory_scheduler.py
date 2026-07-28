"""Cron scheduling for silent periodic Memory Tasks."""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, tzinfo
from typing import Protocol

from croniter import croniter  # type: ignore[import-untyped]
from tzlocal import get_localzone

from myclaw.memory.memory_task import MemoryManager
from myclaw.runtime_log import log_sanitized_exception

logger = logging.getLogger(__name__)


class MemorySchedulerClock(Protocol):
    """Timezone-aware wall clock boundary used by Memory Task scheduling."""

    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


class AsyncioMemorySchedulerClock:
    """Use an injected local wall clock and asyncio for production waits."""

    def __init__(self, *, now: Callable[[], datetime]) -> None:
        self._now = now
        self._timezone = get_localzone()

    def now(self) -> datetime:
        return self._now().astimezone(self._timezone)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class MemoryTaskScheduler:
    """Run one Memory Manager on configured local-time cron boundaries."""

    def __init__(
        self,
        *,
        manager: MemoryManager,
        schedule: str,
        clock: MemorySchedulerClock,
    ) -> None:
        self._manager = manager
        self._schedule = schedule
        self._clock = clock
        self._loop_task: asyncio.Task[None] | None = None
        self._run_tasks: set[asyncio.Task[None]] = set()
        self._timezone: tzinfo | None = None
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Memory Task scheduler is closed")
        if self._loop_task is None:
            startup = self._clock.now()
            if startup.utcoffset() is None or startup.tzinfo is None:
                raise ValueError("Memory Task scheduler clock must be timezone-aware")
            self._timezone = startup.tzinfo
            self._loop_task = asyncio.create_task(self._run())

    async def close(self) -> None:
        self._closed = True
        task = self._loop_task
        if task is None:
            return
        task.cancel()
        running = tuple(self._run_tasks)
        for run_task in running:
            run_task.cancel()
        results = await asyncio.gather(task, *running, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                log_sanitized_exception(
                    logger,
                    logging.ERROR,
                    "Memory Task scheduler cleanup failed",
                    result,
                )

    async def _run(self) -> None:
        timezone = self._timezone
        if timezone is None:
            raise RuntimeError("Memory Task scheduler was not started")
        while True:
            current = self._clock.now().astimezone(timezone)
            next_run = croniter(self._schedule, current).get_next(datetime)
            await self._clock.sleep(max(0.0, next_run.timestamp() - current.timestamp()))
            run_task = asyncio.create_task(self._trigger())
            self._run_tasks.add(run_task)
            run_task.add_done_callback(self._run_tasks.discard)

    async def _trigger(self) -> None:
        try:
            await self._manager.run_periodic()
        except Exception as error:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            log_sanitized_exception(
                logger,
                logging.ERROR,
                "Memory Task trigger crashed",
                error,
            )
            return
