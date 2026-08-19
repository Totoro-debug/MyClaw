"""Runtime-local dispatcher and execution lifecycle for Schedule Jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

from croniter import croniter  # type: ignore[import-untyped]
from loguru import logger

from myclaw.agent.run import (
    AgentRunCancelledPayload,
    AgentRunCompletedPayload,
    AgentRunEmitter,
    AgentRunFailedPayload,
    AgentRunPayload,
    AgentRunRoute,
    ToolResultExternalizer,
    build_assistant_repair_message,
)
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log
from myclaw.provider.errors import ModelCallError
from myclaw.schedule.model import ScheduleJob
from myclaw.schedule.store import ScheduleStoreFaultedError, WorkspaceScheduleStore
from myclaw.session.session import Session, SessionStoragePartition

ScheduleHealth = Literal["available", "faulted"]


class ScheduleClock(Protocol):
    """Aware wall-clock and cancellable wait seam used by Schedule Service."""

    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


type ScheduleContextPreparer = Callable[
    [Session, dict[str, Any]],
    Awaitable[list[dict[str, Any]]],
]


class _AwaitableAgentRun(Protocol):
    async def run(
        self,
        messages: Sequence[dict[str, Any]],
        current_user: dict[str, Any],
        *,
        route: AgentRunRoute,
        emitter: AgentRunEmitter,
        externalize_result: ToolResultExternalizer | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]: ...


class _ScheduleRunEmitter:
    """Consume background Agent Run progress while retaining its terminal outcome."""

    def __init__(self) -> None:
        self.terminal: (
            AgentRunCompletedPayload | AgentRunFailedPayload | AgentRunCancelledPayload | None
        ) = None

    async def emit(self, payload: AgentRunPayload) -> None:
        if isinstance(
            payload,
            (AgentRunCompletedPayload, AgentRunFailedPayload, AgentRunCancelledPayload),
        ):
            self.terminal = payload


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
    """Dispatch due at, every, and cron Jobs through the shared Agent Run boundary."""

    def __init__(
        self,
        *,
        store: WorkspaceScheduleStore,
        agent_run: _AwaitableAgentRun,
        workspace_state: WorkspaceState,
        clock: ScheduleClock,
        context_preparer: ScheduleContextPreparer | None = None,
        externalize_result_for: Callable[[Session], ToolResultExternalizer] | None = None,
    ) -> None:
        self._store = store
        self._agent_run = agent_run
        self._workspace_state = workspace_state
        self._clock = clock
        self._context_preparer = context_preparer
        self._externalize_result_for = externalize_result_for
        self._loop_task: asyncio.Task[None] | None = None
        self._run_tasks: set[asyncio.Task[None]] = set()
        self._active_job_ids: set[str] = set()
        self._consumed_at_jobs: set[str] = set()
        self._every_deadlines: dict[str, _EveryDeadline] = {}
        self._cron_cursors: dict[str, _CronCursor] = {}
        self._last_wall_timestamp: float | None = None
        self._last_monotonic: float | None = None
        self._closing = asyncio.Event()
        self._faulted_event = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None
        self._faulted = False
        self._dispatcher_error_logged = False
        self._terminal_store_error_logged = False

    def start(self) -> None:
        """Start the single dispatcher; repeated starts are idempotent."""
        if self._close_task is not None:
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
            while not self._closing.is_set():
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
                        self._reserve(job)
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

    def _reserve(self, job: ScheduleJob) -> None:
        if self._faulted:
            return
        if job.job_id in self._active_job_ids:
            if self._consume_recurring_occurrence(job):
                with logger.contextualize(session_id=job.session_id):
                    logger.warning(
                        "Schedule Job occurrence skipped while active job_id={} kind={}",
                        job.job_id,
                        job.schedule.kind,
                    )
            return
        if job.job_id in self._consumed_at_jobs:
            return
        self._consume_recurring_occurrence(job)
        if job.schedule.kind == "at":
            self._consumed_at_jobs.add(job.job_id)
        task = asyncio.create_task(self._run_job(job))
        self._active_job_ids.add(job.job_id)
        self._run_tasks.add(task)
        task.add_done_callback(self._run_finished)

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
            existing = self._every_deadlines.get(job.job_id)
            if existing is not None:
                if existing.anchor_ms == anchor_ms:
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
            )
        for job_id in tuple(self._every_deadlines):
            if job_id not in active_ids:
                del self._every_deadlines[job_id]

    def _consume_every_occurrence(self, job: ScheduleJob) -> None:
        deadline = self._every_deadlines.get(job.job_id)
        every_seconds = job.schedule.every_seconds
        if deadline is None or every_seconds is None:
            return
        self._every_deadlines[job.job_id] = _EveryDeadline(
            anchor_ms=deadline.anchor_ms,
            deadline=deadline.deadline + every_seconds,
        )

    def _consume_recurring_occurrence(self, job: ScheduleJob) -> bool:
        if job.schedule.kind == "every":
            self._consume_every_occurrence(job)
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

    async def _run_job(self, job: ScheduleJob) -> None:
        session: Session | None = None
        terminal: Literal["ok", "error"] | None = None
        terminal_error: str | None = None
        terminal_code: str | None = None
        terminal_ready = False
        with session_log(self._workspace_state, job.session_id):
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
                run = self._agent_run
                (
                    terminal,
                    terminal_error,
                    terminal_code,
                    terminal_ready,
                ) = await self._run_awaitable_job(
                    session,
                    job,
                    run,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                terminal = "error"
                terminal_error = "Schedule Job execution failed."
                logger.error(
                    "Schedule Job execution failed job_id={} type={}",
                    job.job_id,
                    type(error).__name__,
                )
            finally:
                try:
                    if terminal is not None and terminal_ready:
                        if terminal_code is not None:
                            logger.warning(
                                "Schedule Job failed job_id={} kind={} code={}",
                                job.job_id,
                                job.schedule.kind,
                                terminal_code,
                            )
                        await self._commit_terminal(job, terminal, terminal_error)
                finally:
                    if session is not None:
                        try:
                            session.close()
                        except Exception as error:
                            logger.error(
                                "Schedule Session close failed job_id={} type={}",
                                job.job_id,
                                type(error).__name__,
                            )
                    self._active_job_ids.discard(job.job_id)

    async def _run_awaitable_job(
        self,
        session: Session,
        job: ScheduleJob,
        run: _AwaitableAgentRun,
    ) -> tuple[Literal["ok", "error"] | None, str | None, str | None, bool]:
        current_user = {"role": "user", "content": job.message}
        context_preparer = self._context_preparer
        try:
            if context_preparer is None:
                raise RuntimeError("Schedule Agent Run requires a context preparer")
            messages = await context_preparer(session, deepcopy(current_user))
        except asyncio.CancelledError:
            try:
                session.append_messages([current_user])
                session.persist()
            except Exception as error:
                logger.error(
                    "Schedule cancellation persistence failed job_id={} type={}",
                    job.job_id,
                    type(error).__name__,
                )
            raise
        except ModelCallError as failure:
            return _persist_schedule_failure(session, current_user, failure)
        except Exception:
            return _persist_schedule_failure(session, current_user, _schedule_model_failure())

        emitter = _ScheduleRunEmitter()

        try:
            if self._externalize_result_for is None:
                increment = await run.run(
                    messages,
                    current_user,
                    route="schedule",
                    emitter=emitter,
                    cancel_requested=self._closing.is_set,
                )
            else:
                increment = await run.run(
                    messages,
                    current_user,
                    route="schedule",
                    emitter=emitter,
                    externalize_result=self._externalize_result_for(session),
                    cancel_requested=self._closing.is_set,
                )
        except ModelCallError as failure:
            return _persist_schedule_failure(session, current_user, failure)
        except Exception:
            return _persist_schedule_failure(session, current_user, _schedule_model_failure())
        if not isinstance(increment, list):
            return _persist_schedule_failure(session, current_user, _schedule_model_failure())
        terminal = emitter.terminal
        if terminal is None:
            return _persist_schedule_failure(session, current_user, _schedule_model_failure())

        session.append_messages(increment)
        session.persist()
        if isinstance(terminal, AgentRunCancelledPayload):
            return None, None, None, False
        if isinstance(terminal, AgentRunFailedPayload):
            return "error", terminal.error.message, terminal.error.code, True
        return "ok", None, None, True

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
            )
        operation = asyncio.create_task(
            self._remove_at_job(job)
            if job.schedule.kind == "at"
            else self._store.commit_terminal(
                job.job_id,
                expected=job,
                finished_at_ms=finished_at_ms,
                status=terminal,
                error=error,
                now_ms=finished_at_ms,
            )
        )
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
        await self._store.remove_job(job.job_id, expected=job)

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


def _local_time_exists(value: datetime, zone: ZoneInfo) -> bool:
    naive = value.replace(tzinfo=None)
    for fold in (0, 1):
        candidate = naive.replace(tzinfo=zone, fold=fold)
        round_trip = candidate.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
        if round_trip == naive:
            return True
    return False


def _schedule_failure_increment(
    current_user: dict[str, Any],
    failure: ModelCallError,
) -> list[dict[str, Any]]:
    return [
        deepcopy(current_user),
        build_assistant_repair_message(
            content="",
            status="error",
            error=failure.error,
        ),
    ]


def _schedule_model_failure() -> ModelCallError:
    return ModelCallError(ErrorInfo(code="model_failed", message="The model request failed."))


def _persist_schedule_failure(
    session: Session,
    current_user: dict[str, Any],
    failure: ModelCallError,
) -> tuple[Literal["error"], str, str, bool]:
    session.append_messages(_schedule_failure_increment(current_user, failure))
    session.persist()
    return "error", failure.error.message, failure.error.code, True


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
