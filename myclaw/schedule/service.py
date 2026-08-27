"""Runtime-local dispatcher and execution lifecycle for Schedule Jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

from croniter import croniter  # type: ignore[import-untyped]
from loguru import logger

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log
from myclaw.schedule.model import DREAM_JOB_ID, JobSchedule, ScheduleJob
from myclaw.schedule.store import (
    ScheduleStaleRemovalError,
    ScheduleStateError,
    ScheduleStoreFaultedError,
    WorkspaceScheduleStore,
)
from myclaw.utils.async_tasks import await_task_preserving_cancellation

ScheduleHealth = Literal["available", "faulted"]


class ScheduleClock(Protocol):
    """Aware wall-clock and cancellable wait seam used by Schedule Service."""

    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


type ScheduleJobExecutor = Callable[[ScheduleJob], Awaitable[None]]
type DreamExecutor = Callable[[], Awaitable[object]]
type _ExecutionLane = Literal["user", "dream"]


class ScheduleJobExecutionError(Exception):
    """A safe, structured failure returned by one scheduled Job execution."""

    def __init__(self, error: ErrorInfo) -> None:
        if not isinstance(error, ErrorInfo):
            raise TypeError("Schedule Job execution errors require ErrorInfo")
        self.error = error
        super().__init__(error.message)


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
    """Own Schedule persistence and dispatch Jobs through one execution callback."""

    def __init__(
        self,
        *,
        workspace_state: WorkspaceState,
        clock: ScheduleClock,
        execute_user_job: ScheduleJobExecutor,
        execute_dream: DreamExecutor,
    ) -> None:
        self._store = WorkspaceScheduleStore(workspace_state)
        self._clock = clock
        self._execute_user_job = execute_user_job
        self._execute_dream = execute_dream
        self._loop_task: asyncio.Task[None] | None = None
        self._run_tasks: set[asyncio.Task[None]] = set()
        self._terminal_commit_tasks: set[asyncio.Task[ScheduleJob | None]] = set()
        self._active_job_ids: set[str] = set()
        self._consumed_at_jobs: set[str] = set()
        self._every_deadlines: dict[str, _EveryDeadline] = {}
        self._cron_cursors: dict[str, _CronCursor] = {}
        self._last_wall_timestamp: float | None = None
        self._last_monotonic: float | None = None
        self._closing = asyncio.Event()
        self._faulted_event = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None
        self._pause_task: asyncio.Task[None] | None = None
        self._abort_task: asyncio.Task[None] | None = None
        self._faulted = False
        self._aborted = False
        self._paused = False
        self._dispatcher_error_logged = False
        self._terminal_store_error_logged = False

    def start(self) -> None:
        """Start the single dispatcher; repeated starts are idempotent."""
        if self._paused:
            self.resume()
            return
        self._prepare_start()
        self._activate_prepared()

    def _prepare_start(self) -> None:
        """Validate dispatcher activation without creating owned tasks."""
        if self._close_task is not None or self._aborted:
            raise RuntimeError("Schedule Service is closed")
        if self._loop_task is not None:
            return
        if self._faulted or self._store.health == "faulted":
            self._faulted = True
            self._faulted_event.set()
            return
        current = self._clock.now()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("Schedule Service clock must be timezone-aware")
        self._clock.monotonic()

    def _activate_prepared(self) -> None:
        """Activate a preflighted dispatcher using only task creation."""
        if self._loop_task is not None or self._faulted:
            return
        self._loop_task = asyncio.create_task(self._dispatch())

    async def close(self) -> None:
        """Cancel and await the dispatcher and every reserved Job run."""
        if self._aborted:
            await self.abort_and_wait()
            return
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_owned_tasks())
            self._close_task = task
        await await_task_preserving_cancellation(task)

    async def pause_and_drain(self) -> None:
        """Stop new occurrences and await every in-flight dispatcher operation."""
        if self._aborted:
            await self.abort_and_wait()
            return
        if self._close_task is not None:
            await await_task_preserving_cancellation(self._close_task)
            return
        self._paused = True
        task = self._pause_task
        if task is None:
            task = asyncio.create_task(self._pause_owned_tasks())
            self._pause_task = task
        await await_task_preserving_cancellation(task)

    def resume(self) -> None:
        """Resume dispatch after a completed pause barrier."""
        if self._aborted or self._close_task is not None:
            raise RuntimeError("Schedule Service is closed")
        if not self._paused:
            return
        if self._pause_task is not None and not self._pause_task.done():
            raise RuntimeError("Schedule Service pause is still draining")
        self._paused = False
        self._pause_task = None
        self._prepare_start()
        self._activate_prepared()

    def status_snapshot(self) -> ScheduleServiceStatus:
        """Return health and active reservations without exposing Job details."""
        health: ScheduleHealth = (
            "faulted" if self._faulted or self._store.health == "faulted" else "available"
        )
        return ScheduleServiceStatus(status=health, active_job_count=len(self._active_job_ids))

    def cancellation_requested(self) -> bool:
        """Return whether Runtime shutdown has requested Schedule execution cancellation."""
        return self._closing.is_set()

    def abort(self) -> None:
        """Synchronously detach Schedule work for an abandoned generation."""
        if self._aborted:
            return
        self._aborted = True
        self._closing.set()
        self._paused = True
        loop_task = self._loop_task
        if loop_task is not None and not loop_task.done():
            loop_task.cancel()
        for run_task in tuple(self._run_tasks):
            if not run_task.done():
                run_task.cancel()
        for commit_task in tuple(self._terminal_commit_tasks):
            if not commit_task.done():
                commit_task.cancel()

    async def abort_and_wait(self) -> None:
        """Cancel and drain all dispatcher, Job, and terminal persistence tasks."""
        self.abort()
        task = self._abort_task
        if task is None:
            task = asyncio.create_task(self._drain_cancelled_tasks())
            self._abort_task = task
        await await_task_preserving_cancellation(task)

    async def add_user_job(self, job: ScheduleJob) -> ScheduleJob:
        """Add one user-owned Job through the Schedule persistence boundary."""
        if self._aborted:
            raise RuntimeError("Schedule Service is no longer active")
        return await self._store.add_user_job(job)

    async def public_snapshot(self) -> tuple[ScheduleJob, ...]:
        """Return the public user-owned Job snapshot."""
        if self._aborted:
            raise RuntimeError("Schedule Service is no longer active")
        return await self._store.public_snapshot()

    async def remove_user_job(
        self,
        job_id: str,
        *,
        expected: ScheduleJob | None = None,
    ) -> bool:
        """Remove one user-owned Job with the Store's optimistic expectation."""
        if self._aborted:
            raise RuntimeError("Schedule Service is no longer active")
        return await self._store.remove_user_job(job_id, expected=expected)

    async def register_dream_job(self, *, schedule: JobSchedule) -> ScheduleJob:
        if self._aborted:
            raise RuntimeError("Schedule Service is no longer active")
        job = _new_dream_job(schedule, now_ms=_epoch_milliseconds(self._clock.now()))
        return await self._store._register_system_job(job)

    def _register_dream_job_sync(self, *, schedule: JobSchedule) -> ScheduleJob:
        if self._aborted:
            raise RuntimeError("Schedule Service is no longer active")
        job = _new_dream_job(schedule, now_ms=_epoch_milliseconds(self._clock.now()))
        return self._store._register_system_job_sync(job)

    async def _pause_owned_tasks(self) -> None:
        loop_task = self._loop_task
        if loop_task is not None and not loop_task.done():
            loop_task.cancel()
        if loop_task is not None:
            await asyncio.gather(loop_task, return_exceptions=True)
        self._loop_task = None
        await self._cancel_and_drain_job_tasks()
        self._active_job_ids.clear()

    async def _drain_cancelled_tasks(self) -> None:
        loop_task = self._loop_task
        if loop_task is not None and loop_task is not asyncio.current_task():
            await asyncio.gather(loop_task, return_exceptions=True)
        await self._cancel_and_drain_job_tasks()
        self._loop_task = None
        self._active_job_ids.clear()
        self._every_deadlines.clear()
        self._cron_cursors.clear()

    async def _cancel_and_drain_job_tasks(self) -> None:
        while self._run_tasks or self._terminal_commit_tasks:
            run_tasks = tuple(self._run_tasks)
            terminal_tasks = tuple(self._terminal_commit_tasks)
            tasks = run_tasks + terminal_tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._run_tasks.difference_update(run_tasks)
            self._terminal_commit_tasks.difference_update(terminal_tasks)

    async def _close_owned_tasks(self) -> None:
        self._closing.set()
        self._paused = True
        loop_task = self._loop_task
        if loop_task is not None:
            loop_task.cancel()
        running = tuple(self._run_tasks)
        for task in running:
            task.cancel()
        owned = (() if loop_task is None else (loop_task,)) + running
        results = await asyncio.gather(*owned, return_exceptions=True)
        self._loop_task = None
        self._run_tasks.difference_update(running)
        done_terminal_tasks = tuple(
            task for task in tuple(self._terminal_commit_tasks) if task.done()
        )
        self._terminal_commit_tasks.difference_update(done_terminal_tasks)
        while self._terminal_commit_tasks:
            terminal_tasks = tuple(self._terminal_commit_tasks)
            await asyncio.gather(*terminal_tasks, return_exceptions=True)
            self._terminal_commit_tasks.difference_update(
                task for task in self._terminal_commit_tasks if task.done()
            )
        self._active_job_ids.clear()
        self._every_deadlines.clear()
        self._cron_cursors.clear()
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result,
                asyncio.CancelledError,
            ):
                self._latch_fault()
                logger.error(
                    "Schedule Service shutdown failed type={}",
                    type(result).__name__,
                )

    async def _dispatch(self) -> None:
        revision = self._store.revision
        try:
            while not self._closing.is_set() and not self._paused:
                if self._faulted or self._store.health == "faulted":
                    self._latch_fault()
                    return
                jobs = await self._store.snapshot()
                current = self._clock.now()
                current_monotonic = self._clock.monotonic()
                forward_jump = self._record_clock_sample(current, current_monotonic)
                self._sync_every_deadlines(jobs, current, current_monotonic)
                self._sync_cron_cursors(jobs, current)
                if forward_jump:
                    self._skip_missed_cron_occurrences(jobs, current)
                due = sorted(
                    (
                        job
                        for job in jobs
                        if job.job_id not in self._consumed_at_jobs
                        and _is_due(
                            job,
                            current,
                            current_monotonic,
                            self._every_deadlines,
                            self._cron_cursors,
                        )
                    ),
                    key=lambda job: job.job_id,
                )
                if due:
                    reserved = await self._store.reserve_due(tuple(due))
                    for job in reserved:
                        self._reserve(job, current_monotonic=current_monotonic)
                    revision = self._store.revision
                    await asyncio.sleep(0)
                    continue

                delay = _next_delay(
                    jobs,
                    current,
                    current_monotonic,
                    self._every_deadlines,
                    self._cron_cursors,
                )
                revision = await self._wait_for_wake(delay, revision)
        except asyncio.CancelledError:
            raise
        except ScheduleStoreFaultedError:
            self._latch_fault()
        except Exception as error:
            self._latch_fault()
            if not self._dispatcher_error_logged:
                self._dispatcher_error_logged = True
                logger.error(
                    "Schedule Service dispatcher failed type={}",
                    type(error).__name__,
                )

    def _reserve(self, job: ScheduleJob, *, current_monotonic: float) -> None:
        if self._faulted or self._paused:
            return
        lane = self._execution_lane(job)
        if job.job_id in self._active_job_ids:
            if self._consume_recurring_occurrence(job, current_monotonic=current_monotonic):
                with logger.contextualize(session_id=job.session_id):
                    logger.warning(
                        "Schedule Job occurrence skipped while active job_id={} kind={}",
                        job.job_id,
                        job.schedule.kind,
                    )
            return
        if job.job_id in self._consumed_at_jobs:
            return
        self._consume_recurring_occurrence(job, current_monotonic=current_monotonic)
        if job.schedule.kind == "at":
            self._consumed_at_jobs.add(job.job_id)
        task = asyncio.create_task(self._run_job(job, lane=lane))
        self._active_job_ids.add(job.job_id)
        self._run_tasks.add(task)
        task.add_done_callback(self._run_finished)

    def _execution_lane(self, job: ScheduleJob) -> _ExecutionLane:
        if job.source == "user":
            return "user"
        if job.source == "system" and job.job_id == DREAM_JOB_ID:
            return "dream"
        raise ScheduleStateError(self._store.path)

    def _sync_every_deadlines(
        self,
        jobs: tuple[ScheduleJob, ...],
        current: datetime,
        current_monotonic: float,
    ) -> None:
        active_ids: set[str] = set()
        for job in jobs:
            if job.schedule.kind != "every":
                continue
            active_ids.add(job.job_id)
            anchor_ms = _every_anchor_ms(job)
            every_seconds = job.schedule.every_seconds
            if every_seconds is None:
                raise ValueError("every Schedule Job must define every_seconds")
            existing = self._every_deadlines.get(job.job_id)
            if existing is not None:
                if existing.anchor_ms == anchor_ms and existing.every_seconds == every_seconds:
                    continue
                # A finishing run fixes its monotonic deadline before the Store publishes
                # the matching wall-clock anchor. Keep that mapping while the run is active.
                if job.job_id in self._active_job_ids:
                    continue
            due_at = _every_due_at(job, anchor_ms)
            delay = max(0.0, (due_at - current).total_seconds())
            self._every_deadlines[job.job_id] = _EveryDeadline(
                anchor_ms=anchor_ms,
                deadline=current_monotonic + delay,
                every_seconds=every_seconds,
            )
        for job_id in tuple(self._every_deadlines):
            if job_id not in active_ids:
                del self._every_deadlines[job_id]

    def _consume_every_occurrence(self, job: ScheduleJob, *, current_monotonic: float) -> None:
        deadline = self._every_deadlines.get(job.job_id)
        every_seconds = job.schedule.every_seconds
        if deadline is None or every_seconds is None:
            return
        next_deadline = deadline.deadline + every_seconds
        if next_deadline <= current_monotonic:
            next_deadline = current_monotonic + every_seconds
        self._every_deadlines[job.job_id] = _EveryDeadline(
            anchor_ms=deadline.anchor_ms,
            deadline=next_deadline,
            every_seconds=every_seconds,
        )

    def _consume_recurring_occurrence(
        self,
        job: ScheduleJob,
        *,
        current_monotonic: float,
    ) -> bool:
        if job.schedule.kind == "every":
            self._consume_every_occurrence(job, current_monotonic=current_monotonic)
            return True
        if job.schedule.kind == "cron":
            self._consume_cron_occurrence(job)
            return True
        return False

    def _record_clock_sample(self, current: datetime, monotonic: float) -> bool:
        wall_timestamp = _instant_timestamp(current)
        last_wall_timestamp = self._last_wall_timestamp
        last_monotonic = self._last_monotonic
        self._last_wall_timestamp = wall_timestamp
        self._last_monotonic = monotonic
        if last_wall_timestamp is None or last_monotonic is None:
            return False
        return wall_timestamp - last_wall_timestamp > monotonic - last_monotonic + 1.0

    def _sync_cron_cursors(
        self,
        jobs: tuple[ScheduleJob, ...],
        current: datetime,
    ) -> None:
        active_ids: set[str] = set()
        for job in jobs:
            if job.schedule.kind != "cron":
                continue
            active_ids.add(job.job_id)
            cron_expr = job.schedule.cron_expr
            timezone = job.schedule.timezone
            if cron_expr is None or timezone is None:
                raise ValueError("cron Schedule must define cron_expr and timezone")
            existing = self._cron_cursors.get(job.job_id)
            if existing is not None and (
                existing.created_at_ms == job.created_at_ms
                and existing.cron_expr == cron_expr
                and existing.timezone == timezone
            ):
                continue
            self._cron_cursors[job.job_id] = _new_cron_cursor(
                cron_expr,
                timezone,
                current,
                created_at_ms=job.created_at_ms,
            )
        for job_id in tuple(self._cron_cursors):
            if job_id not in active_ids:
                del self._cron_cursors[job_id]

    def _skip_missed_cron_occurrences(
        self,
        jobs: tuple[ScheduleJob, ...],
        current: datetime,
    ) -> None:
        current_timestamp = _instant_timestamp(current)
        for job in jobs:
            if job.schedule.kind != "cron":
                continue
            cursor = self._cron_cursors.get(job.job_id)
            if cursor is None:
                continue
            while _instant_timestamp(cursor.next_occurrence) < current_timestamp:
                self._consume_cron_occurrence(job)
                logger.warning(
                    "Schedule Job Cron occurrence skipped after wall-clock jump job_id={}",
                    job.job_id,
                )

    def _consume_cron_occurrence(self, job: ScheduleJob) -> None:
        cursor = self._cron_cursors.get(job.job_id)
        if cursor is None:
            raise ValueError("Cron Schedule cursor is not initialized")
        cursor.next_occurrence = _next_cron_occurrence(cursor)

    async def _wait_for_wake(self, delay: float, revision: int) -> int:
        sleep_task = asyncio.create_task(self._clock.sleep(min(60.0, max(0.0, delay))))
        change_task = asyncio.create_task(self._store.wait_for_change(revision))
        close_task = asyncio.create_task(self._closing.wait())
        fault_task = asyncio.create_task(self._faulted_event.wait())
        tasks = (sleep_task, change_task, close_task, fault_task)
        try:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if close_task in done or fault_task in done:
                return revision
            if change_task in done:
                return change_task.result()
            return self._store.revision
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_job(self, job: ScheduleJob, *, lane: _ExecutionLane) -> None:
        terminal: Literal["ok", "error"] | None = None
        terminal_error: str | None = None
        terminal_ready = False
        try:
            if lane == "dream":
                result = await self._execute_dream()
                if _is_memory_task_running(result):
                    return
                result_error = getattr(result, "error", None)
                if isinstance(result_error, ErrorInfo):
                    terminal = "error"
                    terminal_error = result_error.message
                else:
                    terminal = "ok"
                terminal_ready = True
            else:
                await self._execute_user_job(job)
                terminal = "ok"
                terminal_ready = True
        except asyncio.CancelledError:
            raise
        except ScheduleJobExecutionError as failure:
            terminal = "error"
            terminal_error = failure.error.message
            terminal_ready = True
        except Exception as error:
            terminal = "error"
            terminal_error = "Schedule Job execution failed."
            terminal_ready = True
            logger.error(
                "Schedule Job execution failed job_id={} type={}",
                job.job_id,
                type(error).__name__,
            )
        finally:
            try:
                if terminal is not None and terminal_ready:
                    await self._commit_terminal(job, terminal, terminal_error)
            finally:
                self._active_job_ids.discard(job.job_id)

    async def _commit_terminal(
        self,
        job: ScheduleJob,
        terminal: Literal["ok", "error"],
        error: str | None,
    ) -> None:
        finished_at_ms = _epoch_milliseconds(self._clock.now())
        if job.schedule.kind == "every":
            every_seconds = job.schedule.every_seconds
            if every_seconds is None:
                raise ValueError("every Schedule Job must define every_seconds")
            self._every_deadlines[job.job_id] = _EveryDeadline(
                anchor_ms=finished_at_ms,
                deadline=self._clock.monotonic() + every_seconds,
                every_seconds=every_seconds,
            )
        operation = asyncio.create_task(
            self._remove_at_job(job)
            if job.schedule.kind == "at"
            else (
                self._store._commit_system_terminal
                if job.source == "system"
                else self._store.commit_terminal
            )(
                job.job_id,
                expected=job,
                finished_at_ms=finished_at_ms,
                status=terminal,
                error=error,
                now_ms=finished_at_ms,
            )
        )
        self._terminal_commit_tasks.add(operation)
        operation.add_done_callback(self._terminal_commit_tasks.discard)
        cancellation: asyncio.CancelledError | None = None
        failure: BaseException | None = None
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError as caught:
                cancellation = caught
            except BaseException as caught:
                failure = caught
                break
        if failure is None:
            try:
                await operation
            except asyncio.CancelledError as caught:
                if cancellation is None:
                    cancellation = caught
            except BaseException as caught:
                failure = caught
        if failure is not None and not isinstance(failure, asyncio.CancelledError):
            self._latch_fault()
            if not self._terminal_store_error_logged:
                self._terminal_store_error_logged = True
                if job.source == "user":
                    with session_log(self._store.workspace_state, job.session_id):
                        logger.error(
                            "Schedule terminal update failed job_id={} kind={} outcome={} type={}",
                            job.job_id,
                            job.schedule.kind,
                            terminal,
                            type(failure).__name__,
                        )
                else:
                    logger.error(
                        "Schedule terminal update failed job_id={} kind={} outcome={} type={}",
                        job.job_id,
                        job.schedule.kind,
                        terminal,
                        type(failure).__name__,
                    )
        if cancellation is not None:
            raise cancellation

    async def _remove_at_job(self, job: ScheduleJob) -> None:
        await self._store._remove_terminal_job(job.job_id, expected=job)

    def _run_finished(self, task: asyncio.Task[None]) -> None:
        self._run_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as error:
            self._latch_fault()
            logger.error(
                "Schedule Job task failed type={}",
                type(error).__name__,
            )

    def _latch_fault(self) -> None:
        self._faulted = True
        self._faulted_event.set()


@dataclass(frozen=True, slots=True)
class _EveryDeadline:
    anchor_ms: int
    deadline: float
    every_seconds: int


def _is_due(
    job: ScheduleJob,
    current: datetime,
    current_monotonic: float,
    every_deadlines: dict[str, _EveryDeadline],
    cron_cursors: dict[str, _CronCursor],
) -> bool:
    if job.schedule.kind == "at":
        at_time = job.schedule.at_datetime
        return at_time is not None and current >= at_time
    if job.schedule.kind == "every":
        deadline = every_deadlines.get(job.job_id)
        return deadline is not None and current_monotonic >= deadline.deadline
    if job.schedule.kind == "cron":
        cursor = cron_cursors.get(job.job_id)
        return cursor is not None and _instant_timestamp(
            cursor.next_occurrence
        ) <= _instant_timestamp(current)
    return False


def _next_delay(
    jobs: tuple[ScheduleJob, ...],
    current: datetime,
    current_monotonic: float,
    every_deadlines: dict[str, _EveryDeadline],
    cron_cursors: dict[str, _CronCursor],
) -> float:
    future: list[float] = []
    for job in jobs:
        if job.schedule.kind == "at":
            at_time = job.schedule.at_datetime
            if at_time is not None and at_time > current:
                future.append((at_time - current).total_seconds())
        elif job.schedule.kind == "every":
            deadline = every_deadlines.get(job.job_id)
            if deadline is not None and deadline.deadline > current_monotonic:
                future.append(deadline.deadline - current_monotonic)
        elif job.schedule.kind == "cron":
            cursor = cron_cursors.get(job.job_id)
            if cursor is not None:
                delay = _instant_timestamp(cursor.next_occurrence) - _instant_timestamp(current)
                if delay > 0:
                    future.append(delay)
    return min(future, default=60.0)


def _every_anchor_ms(job: ScheduleJob) -> int:
    return (
        job.state.last_finished_at_ms
        if job.state.last_finished_at_ms is not None
        else job.created_at_ms
    )


def _every_due_at(job: ScheduleJob, anchor_ms: int) -> datetime:
    every_seconds = job.schedule.every_seconds
    if every_seconds is None:
        raise ValueError("every Schedule Job must define every_seconds")
    return _datetime_from_epoch_milliseconds(anchor_ms) + timedelta(seconds=every_seconds)


def _datetime_from_epoch_milliseconds(value: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)


def _epoch_milliseconds(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Schedule Service clock must be timezone-aware")
    return int((value - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 1000)


@dataclass(slots=True)
class _CronCursor:
    created_at_ms: int
    cron_expr: str
    timezone: str
    zone: ZoneInfo
    iterator: croniter
    next_occurrence: datetime


def _new_cron_cursor(
    cron_expr: str,
    timezone: str,
    current: datetime,
    *,
    created_at_ms: int,
) -> _CronCursor:
    zone = ZoneInfo(timezone)
    iterator = croniter(cron_expr, current.astimezone(zone))
    cursor = _CronCursor(
        created_at_ms=created_at_ms,
        cron_expr=cron_expr,
        timezone=timezone,
        zone=zone,
        iterator=iterator,
        next_occurrence=current,
    )
    cursor.next_occurrence = _next_cron_occurrence(cursor)
    return cursor


def _next_cron_occurrence(cursor: _CronCursor) -> datetime:
    while True:
        candidate = cursor.iterator.get_next(datetime)
        if not isinstance(candidate, datetime) or candidate.tzinfo is None:
            raise ValueError("Cron library returned a naive occurrence")
        local_candidate = candidate.astimezone(cursor.zone)
        matches_expression = croniter.match(
            cursor.cron_expr,
            local_candidate.replace(tzinfo=None),
        )
        if matches_expression and _local_time_exists(local_candidate, cursor.zone):
            return candidate


def _instant_timestamp(value: datetime) -> float:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Schedule Service clock must be timezone-aware")
    return value.timestamp()


def _new_dream_job(schedule: JobSchedule, *, now_ms: int) -> ScheduleJob:
    if schedule.kind == "at":
        raise ValueError("Dream Schedule Job must be recurring")
    return ScheduleJob(
        job_id=DREAM_JOB_ID,
        source="system",
        message="Internal Dream schedule.",
        schedule=schedule,
        created_at_ms=now_ms,
        updated_at_ms=now_ms,
    )


def _is_memory_task_running(result: object) -> bool:
    error = getattr(result, "error", None)
    return isinstance(error, ErrorInfo) and error.code == "memory_task_running"


def _local_time_exists(value: datetime, zone: ZoneInfo) -> bool:
    naive = value.replace(tzinfo=None)
    for fold in (0, 1):
        candidate = naive.replace(tzinfo=zone, fold=fold)
        round_trip = candidate.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
        if round_trip == naive:
            return True
    return False


__all__ = [
    "ScheduleClock",
    "ScheduleJobExecutionError",
    "ScheduleJobExecutor",
    "ScheduleService",
    "ScheduleServiceStatus",
    "ScheduleStaleRemovalError",
]
