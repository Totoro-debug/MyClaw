"""Runtime-local Scheduled Work coordination and background Agent Events."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, tzinfo
from typing import Literal, Protocol
from uuid import UUID

from croniter import croniter  # type: ignore[import-untyped]
from loguru import logger
from tzlocal import get_localzone

from myclaw.agent.events import AgentEvent, BackgroundCompletedPayload
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log, without_session_log
from myclaw.schedule.records import ScheduledWork
from myclaw.schedule.scheduled_work import ScheduledWorkPersistenceError, ScheduledWorkStore
from myclaw.schedule.scheduled_work_execution import ScheduledWorkRunner


@dataclass(frozen=True, slots=True)
class _BackgroundEventDraft:
    turn_id: UUID
    created_at: datetime
    payload: BackgroundCompletedPayload


@dataclass(frozen=True, slots=True)
class _BrokerClosed:
    pass


_BROKER_CLOSED = _BrokerClosed()


@dataclass(frozen=True, slots=True)
class ScheduledWorkTriggerResult:
    """Stable runtime-local outcome for one Scheduled Work trigger."""

    status: Literal["completed", "failed", "skipped"]
    content: str
    error: ErrorInfo | None


class RuntimeEventBroker:
    """Sequence runtime events and queue background completions for one UI consumer."""

    def __init__(self) -> None:
        self._background: asyncio.Queue[_BackgroundEventDraft | _BrokerClosed] = asyncio.Queue()
        self._next_event_id = 0
        self._closed = False

    def sequence_foreground(self, event: AgentEvent) -> AgentEvent:
        sequenced = replace(event, event_id=self._next_event_id)
        self._next_event_id += 1
        return sequenced

    async def publish_background(
        self,
        *,
        turn_id: UUID,
        created_at: datetime,
        payload: BackgroundCompletedPayload,
    ) -> None:
        self._raise_if_closed()
        await self._background.put(
            _BackgroundEventDraft(
                turn_id=turn_id,
                created_at=created_at,
                payload=payload,
            )
        )

    async def next_background_event(self) -> AgentEvent:
        draft = self._require_draft(await self._background.get())
        return self._materialize(draft)

    def next_background_event_nowait(self) -> AgentEvent | None:
        try:
            item = self._background.get_nowait()
        except asyncio.QueueEmpty:
            return None
        draft = self._require_draft(item)
        return self._materialize(draft)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._background.put_nowait(_BROKER_CLOSED)

    def _require_draft(
        self,
        item: _BackgroundEventDraft | _BrokerClosed,
    ) -> _BackgroundEventDraft:
        if item is _BROKER_CLOSED:
            self._background.put_nowait(_BROKER_CLOSED)
            self._raise_if_closed()
        if not isinstance(item, _BackgroundEventDraft):
            raise AssertionError("unknown background event queue item")
        return item

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("Runtime event broker is closed")

    def _materialize(self, draft: _BackgroundEventDraft) -> AgentEvent:
        event = AgentEvent(
            type="background_completed",
            event_id=self._next_event_id,
            turn_id=draft.turn_id,
            created_at=draft.created_at,
            payload=draft.payload,
        )
        self._next_event_id += 1
        return event


class ScheduledWorkCoordinator:
    """Run Scheduled Work and publish its safe terminal background event."""

    def __init__(
        self,
        *,
        runner: ScheduledWorkRunner,
        events: RuntimeEventBroker,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        workspace_state: WorkspaceState | None = None,
    ) -> None:
        self._runner = runner
        self._events = events
        self._now = now
        self._new_uuid = new_uuid
        self._active_task_ids: set[str] = set()
        self._workspace_state = workspace_state

    async def trigger(self, task: ScheduledWork) -> ScheduledWorkTriggerResult:
        if task.id in self._active_task_ids:
            return ScheduledWorkTriggerResult(
                status="skipped",
                content="",
                error=None,
            )
        self._active_task_ids.add(task.id)
        run_id = self._new_uuid()
        correlation = (
            session_log(self._workspace_state, task.session_id)
            if self._workspace_state is not None
            else logger.contextualize(session_id=task.session_id)
        )
        try:
            with correlation:
                result = await self._runner.run(task)
                if result.status == "completed":
                    summary = result.content or "Scheduled Work completed."
                else:
                    summary = (
                        result.error.message
                        if result.error is not None
                        else "Scheduled Work failed."
                    )
                try:
                    await self._events.publish_background(
                        turn_id=run_id,
                        created_at=self._now(),
                        payload=BackgroundCompletedPayload(
                            kind="scheduled_work",
                            title=task.title,
                            session_id=task.session_id,
                            status=result.status,
                            summary=summary[:240],
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                    logger.opt(exception=error).error("Scheduled Work event publication failed")
                    raise
                return ScheduledWorkTriggerResult(
                    status=result.status,
                    content=result.content,
                    error=result.error,
                )
        finally:
            self._active_task_ids.discard(task.id)


class ScheduledWorkSchedulerClock(Protocol):
    """Timezone-aware wall clock boundary used by Scheduled Work."""

    def now(self) -> datetime: ...

    async def sleep(self, seconds: float) -> None: ...


class AsyncioScheduledWorkSchedulerClock:
    """Use an injected wall clock in the system local timezone."""

    def __init__(self, *, now: Callable[[], datetime]) -> None:
        self._now = now
        self._timezone = get_localzone()

    def now(self) -> datetime:
        return self._now().astimezone(self._timezone)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class ScheduledWorkScheduler:
    """Reload and trigger enabled Scheduled Work on local cron boundaries."""

    def __init__(
        self,
        *,
        store: ScheduledWorkStore,
        coordinator: ScheduledWorkCoordinator,
        clock: ScheduledWorkSchedulerClock,
    ) -> None:
        self._store = store
        self._coordinator = coordinator
        self._clock = clock
        self._loop_task: asyncio.Task[None] | None = None
        self._run_tasks: set[asyncio.Task[ScheduledWorkTriggerResult]] = set()
        self._timezone: tzinfo | None = None
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Scheduled Work scheduler is closed")
        if self._loop_task is not None:
            return
        startup = self._clock.now()
        if startup.tzinfo is None or startup.utcoffset() is None:
            raise ValueError("Scheduled Work scheduler clock must be timezone-aware")
        self._timezone = startup.tzinfo
        self._loop_task = asyncio.create_task(self._run(startup.astimezone(self._timezone)))

    async def close(self) -> None:
        self._closed = True
        loop_task = self._loop_task
        running = tuple(self._run_tasks)
        if loop_task is not None:
            loop_task.cancel()
        for run_task in running:
            run_task.cancel()
        loop_owned = () if loop_task is None else (loop_task,)
        owned = (*loop_owned, *running)
        results = await asyncio.gather(
            *owned,
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                with without_session_log():
                    logger.opt(exception=result).error("Scheduled Work scheduler cleanup failed")

    async def _run(self, previous: datetime) -> None:
        with without_session_log():
            timezone = self._timezone
            if timezone is None:
                raise RuntimeError("Scheduled Work scheduler was not started")
            while True:
                try:
                    records = tuple(record for record in self._store.load() if record.enabled)
                except ScheduledWorkPersistenceError as error:
                    logger.opt(exception=error).error(
                        "Scheduled Work definitions could not be loaded"
                    )
                    await self._clock.sleep(60.0)
                    previous = self._clock.now().astimezone(timezone)
                    continue
                waits = (
                    croniter(record.cron, previous).get_next(datetime).timestamp()
                    - previous.timestamp()
                    for record in records
                )
                await self._clock.sleep(max(0.0, min(60.0, min(waits, default=60.0))))
                current = self._clock.now().astimezone(timezone)
                try:
                    records = tuple(record for record in self._store.load() if record.enabled)
                except ScheduledWorkPersistenceError as error:
                    logger.opt(exception=error).error(
                        "Scheduled Work definitions could not be loaded"
                    )
                    previous = current
                    continue
                for record in records:
                    next_run = croniter(record.cron, previous).get_next(datetime)
                    if next_run.timestamp() <= current.timestamp():
                        run_task = asyncio.create_task(self._coordinator.trigger(record))
                        self._run_tasks.add(run_task)
                        run_task.add_done_callback(self._run_finished)
                previous = current

    def _run_finished(self, task: asyncio.Task[ScheduledWorkTriggerResult]) -> None:
        self._run_tasks.discard(task)
        if not task.cancelled():
            task.exception()
