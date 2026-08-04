import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from myclaw.agent.events import (
    AgentEvent,
    BackgroundCompletedPayload,
    TurnCompletedPayload,
    TurnStartedPayload,
)
from myclaw.errors import ErrorInfo
from myclaw.provider.models import ModelUsage
from myclaw.schedule.background_coordination import (
    RuntimeEventBroker,
    ScheduledWorkCoordinator,
    ScheduledWorkScheduler,
)
from myclaw.schedule.records import ScheduledWork
from myclaw.schedule.scheduled_work_execution import (
    ScheduledWorkRunner,
    ScheduledWorkRunResult,
)
from myclaw.terminal.repl import run_repl

LOCAL_TIMEZONE = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 4, 11, 0, 0, tzinfo=LOCAL_TIMEZONE)
TASK_ID = "550e8400-e29b-41d4-a716-446655440000"
TASK_SESSION_ID = "20260804-110000-123000_0f8fad5b-d9cb-469f-a165-70867728950e"
RUN_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
SECOND_RUN_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")


def _task(
    *,
    task_id: str = TASK_ID,
    title: str = "Weekly project review",
    enabled: bool = True,
) -> ScheduledWork:
    return ScheduledWork(
        id=task_id,
        title=title,
        cron="* * * * *",
        prompt=f"Run {title}.",
        created_at=NOW,
        enabled=enabled,
        session_id=TASK_SESSION_ID,
    )


class _CancellableThenSuccessfulRunner(ScheduledWorkRunner):
    def __init__(self) -> None:
        self.calls: list[ScheduledWork] = []
        self.first_started = asyncio.Event()
        self.first_cancelled = asyncio.Event()

    async def run(self, task: ScheduledWork) -> ScheduledWorkRunResult:
        self.calls.append(task)
        if len(self.calls) == 1:
            self.first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.first_cancelled.set()
                raise
        return ScheduledWorkRunResult(
            status="completed",
            content="Retry completed.",
            error=None,
        )


class _ResultRunner(ScheduledWorkRunner):
    def __init__(self, result: ScheduledWorkRunResult) -> None:
        self.result = result
        self.calls: list[ScheduledWork] = []

    async def run(self, task: ScheduledWork) -> ScheduledWorkRunResult:
        self.calls.append(task)
        return self.result


class _BlockingRunner(ScheduledWorkRunner):
    def __init__(self) -> None:
        self.calls: list[ScheduledWork] = []
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run(self, task: ScheduledWork) -> ScheduledWorkRunResult:
        self.calls.append(task)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("Blocking runner was released without cancellation")


class _MutableStore:
    def __init__(self, records: tuple[ScheduledWork, ...] = ()) -> None:
        self.records = records
        self.loads = 0

    def load(self) -> tuple[ScheduledWork, ...]:
        self.loads += 1
        return self.records


class _ControlledClock:
    def __init__(self) -> None:
        self.current = NOW
        self.sleeps: list[float] = []
        self.sleeping = asyncio.Event()
        self.release = asyncio.Event()

    def now(self) -> datetime:
        return self.current

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.sleeping.set()
        await self.release.wait()
        self.release.clear()
        self.sleeping.clear()

    async def advance(self, seconds: float) -> None:
        await asyncio.wait_for(self.sleeping.wait(), timeout=1)
        self.current += timedelta(seconds=seconds)
        self.release.set()


class _ScriptedInput:
    def __init__(self, values: tuple[str | None, ...]) -> None:
        self.values = iter(values)

    async def read(self) -> str | None:
        return next(self.values)


class _RecordingWriter:
    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []

    async def write_delta(self, delta: str) -> None:
        self.operations.append(("delta", delta))

    async def finish_turn(self) -> None:
        self.operations.append(("finish", ""))

    async def write_line(self, content: str) -> None:
        self.operations.append(("line", content))


class _ForegroundConversation:
    def __init__(self, broker: RuntimeEventBroker) -> None:
        self.broker = broker

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        assert text == "Run foreground"
        yield AgentEvent(
            type="turn_started",
            event_id=0,
            turn_id=RUN_UUID,
            created_at=NOW,
            payload=TurnStartedPayload(),
        )
        await self.broker.publish_background(
            turn_id=SECOND_RUN_UUID,
            created_at=NOW,
            payload=BackgroundCompletedPayload(
                kind="scheduled_work",
                title="Daily status",
                session_id=TASK_SESSION_ID,
                status="completed",
                summary="Completed during the foreground turn.",
            ),
        )
        yield AgentEvent(
            type="turn_completed",
            event_id=1,
            turn_id=RUN_UUID,
            created_at=NOW,
            payload=TurnCompletedPayload(
                content="Foreground done.",
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            ),
        )

    async def cancel_active_turn(self) -> None:
        return None


@pytest.mark.asyncio
async def test_same_scheduled_work_skips_overlap_and_can_retrigger_after_cancellation() -> None:
    runner = _CancellableThenSuccessfulRunner()
    broker = RuntimeEventBroker()
    run_ids = iter((RUN_UUID, SECOND_RUN_UUID))
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=broker,
        now=lambda: NOW,
        new_uuid=run_ids.__next__,
    )
    task = _task()

    first = asyncio.create_task(coordinator.trigger(task))
    await asyncio.wait_for(runner.first_started.wait(), timeout=1)
    overlapping = await coordinator.trigger(task)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    retried = await coordinator.trigger(task)
    event = await broker.next_background_event()

    assert overlapping.status == "skipped"
    assert runner.first_cancelled.is_set()
    assert retried.status == "completed"
    assert [called.id for called in runner.calls] == [TASK_ID, TASK_ID]
    assert event.type == "background_completed"
    assert event.turn_id == SECOND_RUN_UUID
    assert isinstance(event.payload, BackgroundCompletedPayload)
    assert event.payload.summary == "Retry completed."


@pytest.mark.asyncio
async def test_failed_scheduled_work_publishes_one_safe_failure_event() -> None:
    failure = ErrorInfo(code="model_failed", message="Scheduled model failed safely.")
    runner = _ResultRunner(ScheduledWorkRunResult(status="failed", content="", error=failure))
    broker = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=broker,
        now=lambda: NOW,
        new_uuid=lambda: RUN_UUID,
    )

    result = await coordinator.trigger(_task())
    event = await broker.next_background_event()

    assert result.status == "failed"
    assert result.error == failure
    assert isinstance(event.payload, BackgroundCompletedPayload)
    assert event.payload.status == "failed"
    assert event.payload.summary == failure.message


@pytest.mark.asyncio
async def test_runtime_event_broker_sequences_both_lanes_and_close_wakes_waiters() -> None:
    broker = RuntimeEventBroker()
    foreground = broker.sequence_foreground(
        AgentEvent(
            type="turn_started",
            event_id=99,
            turn_id=RUN_UUID,
            created_at=NOW,
            payload=TurnStartedPayload(),
        )
    )
    await broker.publish_background(
        turn_id=SECOND_RUN_UUID,
        created_at=NOW,
        payload=BackgroundCompletedPayload(
            kind="scheduled_work",
            title="Daily status",
            session_id=TASK_SESSION_ID,
            status="completed",
            summary="Done.",
        ),
    )
    background = await broker.next_background_event()
    waiting = asyncio.create_task(broker.next_background_event())
    await asyncio.sleep(0)
    broker.close()

    with pytest.raises(RuntimeError, match="closed"):
        await waiting
    with pytest.raises(RuntimeError, match="closed"):
        assert isinstance(background.payload, BackgroundCompletedPayload)
        await broker.publish_background(
            turn_id=RUN_UUID,
            created_at=NOW,
            payload=background.payload,
        )

    assert foreground.event_id == 0
    assert background.event_id == 1


@pytest.mark.asyncio
async def test_repl_renders_queued_background_completion_after_foreground_terminal() -> None:
    broker = RuntimeEventBroker()
    writer = _RecordingWriter()

    await run_repl(
        conversation=_ForegroundConversation(broker),
        input_reader=_ScriptedInput(("Run foreground", "exit")),
        writer=writer,
        background_events=broker,
    )

    assert writer.operations == [
        ("finish", ""),
        (
            "line",
            "[Scheduled Work] Daily status (completed): Completed during the foreground turn.",
        ),
    ]


@pytest.mark.asyncio
async def test_scheduler_reloads_enabled_work_and_ignores_disabled_records() -> None:
    enabled = _task()
    disabled = _task(
        task_id="6fa459ea-ee8a-4ca4-894e-db77e160355e",
        title="Disabled task",
        enabled=False,
    )
    store = _MutableStore()
    runner = _ResultRunner(
        ScheduledWorkRunResult(status="completed", content="Scheduled result.", error=None)
    )
    broker = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=broker,
        now=lambda: NOW,
        new_uuid=lambda: RUN_UUID,
    )
    clock = _ControlledClock()
    scheduler = ScheduledWorkScheduler(store=store, coordinator=coordinator, clock=clock)
    scheduler.start()
    await asyncio.wait_for(clock.sleeping.wait(), timeout=1)
    store.records = (enabled, disabled)

    await clock.advance(60)
    event = await asyncio.wait_for(broker.next_background_event(), timeout=1)
    await scheduler.close()

    assert store.loads >= 2
    assert [task.id for task in runner.calls] == [enabled.id]
    assert isinstance(event.payload, BackgroundCompletedPayload)
    assert event.payload.title == enabled.title


@pytest.mark.asyncio
async def test_scheduler_close_cancels_and_awaits_active_scheduled_work() -> None:
    store = _MutableStore((_task(),))
    runner = _BlockingRunner()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=RuntimeEventBroker(),
        now=lambda: NOW,
        new_uuid=lambda: RUN_UUID,
    )
    clock = _ControlledClock()
    scheduler = ScheduledWorkScheduler(store=store, coordinator=coordinator, clock=clock)
    scheduler.start()

    await clock.advance(60)
    await asyncio.wait_for(runner.started.wait(), timeout=1)
    await scheduler.close()

    assert runner.cancelled.is_set()
    assert [task.id for task in runner.calls] == [TASK_ID]
