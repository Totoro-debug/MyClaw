"""Cron scheduling for silent periodic Memory Tasks."""

import asyncio
from datetime import datetime, tzinfo

from croniter import croniter  # type: ignore[import-untyped]
from loguru import logger

from myclaw.logging.session import without_session_log
from myclaw.memory.memory_task import MemoryManager
from myclaw.utils.scheduler import SchedulerClock


class MemoryTaskScheduler:
    """Run one Memory Manager on configured local-time cron boundaries."""

    def __init__(
        self,
        *,
        manager: MemoryManager,
        schedule: str,
        clock: SchedulerClock,
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
            with without_session_log():
                self._loop_task = asyncio.create_task(self._run())

    async def close(self) -> None:
        with without_session_log():
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
                    logger.opt(exception=result).error("Memory Task scheduler cleanup failed")

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
            logger.opt(exception=error).error("Memory Task trigger crashed")
            return
