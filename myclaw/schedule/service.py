"""Runtime-local dispatcher and execution lifecycle for Schedule Jobs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from loguru import logger

from myclaw.agent.run import (
    AgentRunCancelledPayload,
    AgentRunCompletedPayload,
    AgentRunFailedPayload,
    AgentRunInterface,
)
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.logging.session import session_log
from myclaw.schedule.model import ScheduleJob
from myclaw.schedule.store import ScheduleStore
from myclaw.session.session import Session, SessionStoragePartition

ScheduleHealth = Literal["available", "faulted"]


class ScheduleClock(Protocol):
    """Aware wall-clock and cancellable wait seam used by Schedule Service."""

    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True, slots=True)
class ScheduleServiceStatus:
    """The intentionally small Management status surface for Schedule Service."""

    status: ScheduleHealth
    active_job_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "active_job_count": self.active_job_count,
        }


class ScheduleService:
    """Dispatch due at Jobs through the shared Agent Run boundary."""

    def __init__(
        self,
        *,
        store: ScheduleStore,
        agent_run: AgentRunInterface,
        workspace_state: WorkspaceState,
        clock: ScheduleClock,
    ) -> None:
        self._store = store
        self._agent_run = agent_run
        self._workspace_state = workspace_state
        self._clock = clock
        self._loop_task: asyncio.Task[None] | None = None
        self._run_tasks: set[asyncio.Task[None]] = set()
        self._active_job_ids: set[str] = set()
        self._consumed_at_jobs: set[str] = set()
        self._closing = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None
        self._faulted = False

    def start(self) -> None:
        """Start the single dispatcher; repeated starts are idempotent."""
        if self._close_task is not None:
            raise RuntimeError("Schedule Service is closed")
        if self._loop_task is not None:
            return
        current = self._clock.now()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("Schedule Service clock must be timezone-aware")
        self._loop_task = asyncio.create_task(self._dispatch())

    async def close(self) -> None:
        """Cancel and await the dispatcher and every reserved Job run."""
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_owned_tasks())
            self._close_task = task
        await _await_shared(task)

    def status_snapshot(self) -> ScheduleServiceStatus:
        """Return health and active reservations without exposing Job details."""
        health: ScheduleHealth = (
            "faulted" if self._faulted or self._store.health == "faulted" else "available"
        )
        return ScheduleServiceStatus(status=health, active_job_count=len(self._active_job_ids))

    async def _close_owned_tasks(self) -> None:
        self._closing.set()
        loop_task = self._loop_task
        if loop_task is not None:
            loop_task.cancel()
        running = tuple(self._run_tasks)
        for task in running:
            task.cancel()
        owned = (() if loop_task is None else (loop_task,)) + running
        results = await asyncio.gather(*owned, return_exceptions=True)
        self._active_job_ids.clear()
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result,
                asyncio.CancelledError,
            ):
                self._faulted = True
                logger.opt(exception=result).error("Schedule Service shutdown failed")

    async def _dispatch(self) -> None:
        revision = self._store.revision
        try:
            while not self._closing.is_set():
                if self._store.health == "faulted":
                    self._faulted = True
                    return
                jobs = await self._store.snapshot()
                current = self._clock.now()
                due = sorted(
                    (
                        job
                        for job in jobs
                        if job.schedule.kind == "at"
                        and job.job_id not in self._consumed_at_jobs
                        and _is_due(job, current)
                    ),
                    key=lambda job: job.job_id,
                )
                if due:
                    for job in due:
                        self._reserve(job)
                    revision = self._store.revision
                    await asyncio.sleep(0)
                    continue

                delay = _next_delay(jobs, current)
                revision = await self._wait_for_wake(delay, revision)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._faulted = True
            logger.opt(exception=error).error(
                "Schedule Service dispatcher failed type={}",
                type(error).__name__,
            )

    def _reserve(self, job: ScheduleJob) -> None:
        if job.job_id in self._active_job_ids or job.job_id in self._consumed_at_jobs:
            return
        self._active_job_ids.add(job.job_id)
        self._consumed_at_jobs.add(job.job_id)
        task = asyncio.create_task(self._run_job(job))
        self._run_tasks.add(task)
        task.add_done_callback(self._run_finished)

    async def _wait_for_wake(self, delay: float, revision: int) -> int:
        sleep_task = asyncio.create_task(self._clock.sleep(min(60.0, max(0.0, delay))))
        change_task = asyncio.create_task(self._store.wait_for_change(revision))
        close_task = asyncio.create_task(self._closing.wait())
        tasks = (sleep_task, change_task, close_task)
        try:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if close_task in done:
                return revision
            if change_task in done:
                return change_task.result()
            return self._store.revision
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_job(self, job: ScheduleJob) -> None:
        session: Session | None = None
        terminal: Literal["ok", "error"] | None = None
        terminal_error: str | None = None
        payloads: AsyncIterator[object] | None = None
        correlation: AbstractContextManager[None] = nullcontext()
        try:
            try:
                session = Session.load(
                    self._workspace_state,
                    job.session_id,
                    partition=SessionStoragePartition.SCHEDULE,
                    now=self._clock.now,
                )
            except FileNotFoundError:
                session = Session.create_schedule(
                    self._workspace_state,
                    job.job_id,
                    now=self._clock.now,
                )
            correlation = session_log(session)
            with correlation:
                payloads = self._agent_run.run_agent(
                    session,
                    job.message,
                    route="schedule",
                    stream=False,
                )
                async for payload in payloads:
                    if isinstance(payload, AgentRunCompletedPayload):
                        terminal = "ok"
                        terminal_error = None
                    elif isinstance(payload, AgentRunFailedPayload):
                        terminal = "error"
                        terminal_error = payload.error.message
                    elif isinstance(payload, AgentRunCancelledPayload):
                        return
                if terminal is None:
                    terminal = "error"
                    terminal_error = "Schedule Job execution failed."
        except asyncio.CancelledError:
            raise
        except Exception as error:
            terminal = "error"
            terminal_error = "Schedule Job execution failed."
            logger.opt(exception=error).error(
                "Schedule Job execution failed job_id={} type={}",
                job.job_id,
                type(error).__name__,
            )
        finally:
            await _close_payloads(payloads)
            if session is not None:
                try:
                    session.close()
                except Exception as error:
                    logger.opt(exception=error).error(
                        "Schedule Session close failed job_id={} type={}",
                        job.job_id,
                        type(error).__name__,
                    )
            if terminal is not None:
                await self._commit_terminal(job, terminal, terminal_error)
            self._active_job_ids.discard(job.job_id)

    async def _commit_terminal(
        self,
        job: ScheduleJob,
        terminal: Literal["ok", "error"],
        error: str | None,
    ) -> None:
        operation = asyncio.create_task(self._remove_at_job(job))
        cancellation: asyncio.CancelledError | None = None
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError as caught:
                cancellation = caught
        try:
            await operation
        except Exception as failure:
            self._faulted = True
            logger.opt(exception=failure).error(
                "Schedule terminal update failed job_id={} kind={} outcome={}",
                job.job_id,
                job.schedule.kind,
                terminal,
            )
        else:
            if terminal == "error":
                logger.warning(
                    "Schedule Job failed job_id={} kind={} error={}",
                    job.job_id,
                    job.schedule.kind,
                    error or "Schedule Job execution failed.",
                )
        if cancellation is not None:
            raise cancellation

    async def _remove_at_job(self, job: ScheduleJob) -> None:
        remove_job = getattr(self._store, "remove_job", None)
        if job.source == "system" and callable(remove_job):
            await remove_job(job.job_id)
            return
        await self._store.remove_user_job(job.job_id)

    def _run_finished(self, task: asyncio.Task[None]) -> None:
        self._run_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as error:
            self._faulted = True
            logger.opt(exception=error).error(
                "Schedule Job task failed type={}",
                type(error).__name__,
            )


def _is_due(job: ScheduleJob, current: datetime) -> bool:
    at_time = job.schedule.at_datetime
    if at_time is None:
        return False
    return current >= at_time


def _next_delay(jobs: tuple[ScheduleJob, ...], current: datetime) -> float:
    future = [
        (job.schedule.at_datetime - current).total_seconds()
        for job in jobs
        if job.schedule.kind == "at"
        and job.schedule.at_datetime is not None
        and job.schedule.at_datetime > current
    ]
    return min(future, default=60.0)


async def _close_payloads(payloads: object | None) -> None:
    if payloads is None:
        return
    close = getattr(payloads, "aclose", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except BaseException:
        pass


async def _await_shared(task: asyncio.Task[None]) -> None:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as caught:
            cancellation = caught
    try:
        task.result()
    except BaseException as error:
        if cancellation is not None:
            raise cancellation from error
        raise
    if cancellation is not None:
        raise cancellation


__all__ = ["ScheduleClock", "ScheduleService", "ScheduleServiceStatus"]
