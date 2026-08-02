import asyncio
import gc
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest

from myclaw.agent.events import (
    AgentEvent,
    BackgroundCompletedPayload,
    ConversationPort,
    TurnCancelledPayload,
    TurnFailedPayload,
    TurnStartedPayload,
)
from myclaw.agent.runtime import PreparedReplRuntime, prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.memory.conversation_summary import WorkspaceJsonlSummaryStore
from myclaw.memory.memory_task import WorkspaceFileMemoryStore
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.schedule.background_coordination import (
    RuntimeEventBroker,
    ScheduledWorkCoordinator,
    ScheduledWorkScheduler,
)
from myclaw.schedule.records import ScheduledWork
from myclaw.schedule.scheduled_work import (
    ScheduledWorkPersistenceError,
    WorkspaceJsonScheduledWorkStore,
)
from myclaw.schedule.scheduled_work_execution import (
    ScheduledWorkModelSettings,
    ScheduledWorkRunner,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session_resume import SwitchableConversationPort
from myclaw.session.session_store import JsonlSessionStore
from myclaw.terminal.repl import run_repl
from myclaw.tools.tool_gateway import ToolGateway
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript, persist_scheduled_work
from tests.fixtures.log_capture import install_log_capture

LOCAL_TIMEZONE = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 13, 0, 30, 0, 123456, tzinfo=LOCAL_TIMEZONE)
TASK_ID = "550e8400-e29b-41d4-a716-446655440000"
TASK_SESSION_ID = "20260713-003000-123000_0f8fad5b-d9cb-469f-a165-70867728950e"
USER_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
REQUEST_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")
ASSISTANT_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
RUN_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
RUN_TWO_UUID = UUID("a8098c1a-f86e-4f33-8a28-25f602f8e603")


def _task() -> ScheduledWork:
    return ScheduledWork(
        id=TASK_ID,
        title="Weekly project review",
        cron="0 9 * * 1",
        prompt="Review the current project and summarize open risks.",
        created_at=NOW,
        enabled=True,
        session_id=TASK_SESSION_ID,
    )


def _usage() -> ModelUsage:
    return ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15)


class BlockingCompletionProvider(ScriptedFakeProvider):
    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request: object) -> ModelResponse:
        self.complete_requests.append(request)
        self.started.set()
        await self.release.wait()
        return ModelResponse(
            message=AssistantModelMessage(content=self._content),
            usage=_usage(),
            finish_reason="stop",
        )


class CancellableThenSuccessfulCompletionProvider(ScriptedFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = asyncio.Event()
        self.first_cancelled = asyncio.Event()
        self._calls = 0

    async def complete(self, request: object) -> ModelResponse:
        self.complete_requests.append(request)
        self._calls += 1
        if self._calls == 1:
            self.first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.first_cancelled.set()
                raise
        return ModelResponse(
            message=AssistantModelMessage(content="Retry completed."),
            usage=_usage(),
            finish_reason="stop",
        )


class ConcurrentCompletionProvider(ScriptedFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started: set[str] = set()
        self.both_started = asyncio.Event()
        self.release = {"first": asyncio.Event(), "second": asyncio.Event()}

    async def complete(self, request: object) -> ModelResponse:
        assert isinstance(request, ModelRequest)
        self.complete_requests.append(request)
        current = request.messages[-1].content
        key = "first" if "Run first task." in current else "second"
        self.started.add(key)
        if self.started == {"first", "second"}:
            self.both_started.set()
        await self.release[key].wait()
        content = "First finished second." if key == "first" else "Second finished first."
        return ModelResponse(
            message=AssistantModelMessage(content=content),
            usage=_usage(),
            finish_reason="stop",
        )


class RuntimeCoordinationProvider(ScriptedFakeProvider):
    def __init__(self, *, memory_failure: bool) -> None:
        super().__init__()
        self._memory_failure = memory_failure
        self.memory_finished = asyncio.Event()

    async def complete(self, request: object) -> ModelResponse:
        assert isinstance(request, ModelRequest)
        self.complete_requests.append(request)
        if request.route == "memory":
            self.memory_finished.set()
            if self._memory_failure:
                raise ModelCallError(
                    ErrorInfo(code="model_failed", message="Memory periodic failed safely.")
                )
            return ModelResponse(
                message=AssistantModelMessage(content="No memory update needed."),
                usage=_usage(),
                finish_reason="stop",
            )
        assert request.route == "cron"
        return ModelResponse(
            message=AssistantModelMessage(content="Scheduled work remains visible."),
            usage=_usage(),
            finish_reason="stop",
        )

    async def stream(self, request: object) -> AsyncIterator[ModelStreamEvent]:
        assert isinstance(request, ModelRequest)
        self.stream_requests.append(request)
        if "<long_term_memory>" in request.system_prompt:
            yield TextDelta(delta="Foreground continues.")
            yield ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Foreground continues."),
                    usage=_usage(),
                    finish_reason="stop",
                )
            )
            return
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content='"Coordination"'),
                usage=_usage(),
                finish_reason="stop",
            )
        )


class WaitingExitInput:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.read_calls = 0

    async def read(self) -> str:
        self.read_calls += 1
        self.started.set()
        await self.release.wait()
        return "exit"


class SimultaneousEofInput:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def read(self) -> None:
        self.started.set()
        await self.release.wait()
        return None


class ReleasedConversationInput:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._calls = 0

    async def read(self) -> str:
        self._calls += 1
        if self._calls == 1:
            self.started.set()
            await self.release.wait()
            return "Continue foreground."
        return "exit"


class RecordingWriter:
    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []
        self.written = asyncio.Event()

    async def write_delta(self, delta: str) -> None:
        self.operations.append(("delta", delta))
        self.written.set()

    async def finish_turn(self) -> None:
        self.operations.append(("finish", ""))
        self.written.set()

    async def write_line(self, content: str) -> None:
        self.operations.append(("line", content))
        self.written.set()


class ScriptedInput:
    def __init__(self, values: tuple[str | None, ...]) -> None:
        self._values = iter(values)

    async def read(self) -> str | None:
        return next(self._values)


class TerminalQueueInput:
    def __init__(self, order: list[str]) -> None:
        self._order = order
        self._calls = 0
        self.second_started = asyncio.Event()
        self.release = asyncio.Event()

    async def read(self) -> str:
        self._calls += 1
        if self._calls == 1:
            return "Run foreground."
        self._order.append("input:second")
        self.second_started.set()
        await self.release.wait()
        return "exit"


class OrderingWriter:
    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.background_written = asyncio.Event()

    async def write_delta(self, delta: str) -> None:
        self._order.append(f"delta:{delta}")

    async def finish_turn(self) -> None:
        self._order.append("finish")

    async def write_line(self, content: str) -> None:
        self._order.append(f"background:{content}")
        self.background_written.set()


class ForegroundTerminalConversation:
    def __init__(
        self,
        *,
        broker: RuntimeEventBroker,
        terminal: Literal["failed", "cancelled"],
    ) -> None:
        self._broker = broker
        self._terminal = terminal

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        assert text == "Run foreground."
        yield AgentEvent(
            type="turn_started",
            event_id=0,
            turn_id=RUN_UUID,
            created_at=NOW,
            payload=TurnStartedPayload(),
        )
        await self._broker.publish_background(
            turn_id=RUN_TWO_UUID,
            created_at=NOW,
            payload=BackgroundCompletedPayload(
                kind="scheduled_work",
                title="Daily status",
                session_id=TASK_SESSION_ID,
                status="completed",
                summary="Completed before foreground terminal.",
            ),
        )
        if self._terminal == "failed":
            yield AgentEvent(
                type="turn_failed",
                event_id=1,
                turn_id=RUN_UUID,
                created_at=NOW,
                payload=TurnFailedPayload(
                    error=ErrorInfo(
                        code="model_failed",
                        message="Foreground failed safely.",
                    )
                ),
            )
        else:
            yield AgentEvent(
                type="turn_cancelled",
                event_id=1,
                turn_id=RUN_UUID,
                created_at=NOW,
                payload=TurnCancelledPayload(partial_content=""),
            )

    async def cancel_active_turn(self) -> None:
        return None


class ControlledSchedulerClock:
    def __init__(self, current: datetime) -> None:
        self._current = current
        self.sleeps: list[float] = []
        self._sleeping = asyncio.Event()
        self._release = asyncio.Event()

    def now(self) -> datetime:
        return self._current

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._sleeping.set()
        await self._release.wait()
        self._release.clear()
        self._sleeping.clear()

    async def advance(self, seconds: float) -> None:
        await asyncio.wait_for(self._sleeping.wait(), timeout=1)
        self._current += timedelta(seconds=seconds)
        self._release.set()


class LoadFailingOnceScheduledWorkStore(WorkspaceJsonScheduledWorkStore):
    def __init__(self, workspace_state: WorkspaceState) -> None:
        super().__init__(workspace_state)
        self._failed = False

    def load(self) -> tuple[ScheduledWork, ...]:
        if not self._failed:
            self._failed = True
            raise ScheduledWorkPersistenceError("PRIVATE SCHEDULED WORK DEFINITION PAYLOAD")
        return super().load()


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


def _state(workspace: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


@pytest.mark.asyncio
async def test_scheduler_refreshes_persisted_work_and_ignores_disabled_records(
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_now = NOW.replace(microsecond=0)
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace)
    store = WorkspaceJsonScheduledWorkStore(state)
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: clock_now,
        new_uuid=uuid4,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Scheduled result."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: clock_now,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: clock_now,
        new_uuid=lambda: RUN_UUID,
    )
    clock = ControlledSchedulerClock(clock_now)
    scheduler = ScheduledWorkScheduler(
        store=store,
        coordinator=coordinator,
        clock=clock,
    )

    scheduler.start()
    await _wait_until(lambda: clock.sleeps == [60.0])
    persist_scheduled_work(
        state.path,
        (
            replace(_task(), cron="31 0 * * *"),
            replace(
                _task(),
                id="11111111-1111-4111-8111-111111111111",
                title="Disabled status",
                cron="31 0 * * *",
                enabled=False,
                session_id=("20260713-003000-123000_22222222-2222-4222-8222-222222222222"),
            ),
        ),
    )
    await clock.advance(60)
    event = await asyncio.wait_for(events.next_background_event(), timeout=1)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(events.next_background_event(), timeout=0.01)
    await scheduler.close()

    assert event.turn_id == RUN_UUID
    assert isinstance(event.payload, BackgroundCompletedPayload)
    assert event.payload.summary == "Scheduled result."
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_scheduler_retries_after_a_store_load_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_now = NOW.replace(microsecond=0)
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace)
    store = LoadFailingOnceScheduledWorkStore(state)
    persist_scheduled_work(state.path, (replace(_task(), cron="32 0 * * *"),))
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: clock_now,
        new_uuid=uuid4,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Recovered schedule."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: clock_now,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: clock_now,
        new_uuid=lambda: RUN_UUID,
    )
    clock = ControlledSchedulerClock(clock_now)
    scheduler = ScheduledWorkScheduler(
        store=store,
        coordinator=coordinator,
        clock=clock,
    )
    lifetime = install_log_capture(home)

    scheduler.start()
    try:
        await _wait_until(lambda: clock.sleeps == [60.0])
        await clock.advance(60)
        await _wait_until(lambda: clock.sleeps == [60.0, 60.0])
        await clock.advance(60)
        event = await asyncio.wait_for(events.next_background_event(), timeout=1)
    finally:
        await scheduler.close()
        lifetime.close()

    assert isinstance(event.payload, BackgroundCompletedPayload)
    assert event.payload.summary == "Recovered schedule."
    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert content.count(" ERROR ") == 1
    assert (
        "session=- myclaw.schedule.background_coordination: "
        "Scheduled Work definitions could not be loaded"
    ) in content
    assert "ScheduledWorkPersistenceError" in content
    assert "PRIVATE SCHEDULED WORK DEFINITION PAYLOAD" not in content
    assert _task().prompt not in content


@pytest.mark.asyncio
async def test_scheduler_continues_after_one_scheduled_run_fails(
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_now = NOW.replace(microsecond=0)
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace)
    store = WorkspaceJsonScheduledWorkStore(state)
    persist_scheduled_work(state.path, (replace(_task(), cron="* * * * *"),))
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: clock_now,
        new_uuid=uuid4,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo(code="model_failed", message="First scheduled run failed.")),
            ModelResponse(
                message=AssistantModelMessage(content="Next scheduled run completed."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: clock_now,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: clock_now,
        new_uuid=iter((RUN_UUID, RUN_TWO_UUID)).__next__,
    )
    clock = ControlledSchedulerClock(clock_now)
    scheduler = ScheduledWorkScheduler(
        store=store,
        coordinator=coordinator,
        clock=clock,
    )

    scheduler.start()
    try:
        await _wait_until(lambda: clock.sleeps == [60.0])
        await clock.advance(60)
        failed_event = await asyncio.wait_for(events.next_background_event(), timeout=1)
        await _wait_until(lambda: clock.sleeps == [60.0, 60.0])
        await clock.advance(60)
        completed_event = await asyncio.wait_for(events.next_background_event(), timeout=1)
    finally:
        await scheduler.close()

    assert isinstance(failed_event.payload, BackgroundCompletedPayload)
    assert failed_event.payload.status == "failed"
    assert failed_event.payload.summary == "First scheduled run failed."
    assert isinstance(completed_event.payload, BackgroundCompletedPayload)
    assert completed_event.payload.status == "completed"
    assert completed_event.payload.summary == "Next scheduled run completed."
    assert (failed_event.turn_id, completed_event.turn_id) == (RUN_UUID, RUN_TWO_UUID)


@pytest.mark.asyncio
async def test_scheduler_consumes_an_unhandled_scheduled_run_after_it_is_logged(
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_now = NOW.replace(microsecond=0)
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace)
    store = WorkspaceJsonScheduledWorkStore(state)
    persist_scheduled_work(state.path, (replace(_task(), cron="31 0 * * *"),))
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: clock_now,
        new_uuid=uuid4,
    )
    attempted = asyncio.Event()

    class UnexpectedFailureProvider(ScriptedFakeProvider):
        async def complete(self, request: object) -> ModelResponse:
            self.complete_requests.append(request)
            attempted.set()
            raise RuntimeError("PRIVATE TASK PROMPT")

    runner = ScheduledWorkRunner(
        provider=UnexpectedFailureProvider(),
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: clock_now,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    clock = ControlledSchedulerClock(clock_now)
    scheduler = ScheduledWorkScheduler(
        store=store,
        coordinator=ScheduledWorkCoordinator(
            runner=runner,
            events=RuntimeEventBroker(),
            now=lambda: clock_now,
            new_uuid=lambda: RUN_UUID,
        ),
        clock=clock,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    lifetime = install_log_capture(home)

    try:
        scheduler.start()
        await _wait_until(lambda: clock.sleeps == [60.0])
        await clock.advance(60)
        await attempted.wait()
        await _wait_until(lambda: clock.sleeps == [60.0, 60.0])
        gc.collect()
        await asyncio.sleep(0)
    finally:
        await scheduler.close()
        del scheduler
        gc.collect()
        await asyncio.sleep(0)
        lifetime.close()
        loop.set_exception_handler(previous_handler)

    assert unhandled == []
    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert content.count(" ERROR ") == 1
    assert (
        f"session={TASK_SESSION_ID} myclaw.schedule.scheduled_work_execution: "
        "Scheduled Work crashed"
    ) in content
    assert "RuntimeError: [REDACTED]" in content
    assert "PRIVATE TASK PROMPT" not in content


@pytest.mark.asyncio
async def test_scheduler_close_cancels_and_awaits_running_scheduled_work(
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_now = NOW.replace(microsecond=0)
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace)
    store = WorkspaceJsonScheduledWorkStore(state)
    persist_scheduled_work(state.path, (replace(_task(), cron="31 0 * * *"),))
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: clock_now,
        new_uuid=uuid4,
    )
    provider = CancellableThenSuccessfulCompletionProvider()
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: clock_now,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    clock = ControlledSchedulerClock(clock_now)
    scheduler = ScheduledWorkScheduler(
        store=store,
        coordinator=ScheduledWorkCoordinator(
            runner=runner,
            events=events,
            now=lambda: clock_now,
            new_uuid=lambda: RUN_UUID,
        ),
        clock=clock,
    )
    lifetime = install_log_capture(home)
    existing_tasks = asyncio.all_tasks()

    scheduler.start()
    await _wait_until(lambda: clock.sleeps == [60.0])
    await clock.advance(60)
    await asyncio.wait_for(provider.first_started.wait(), timeout=1)
    await scheduler.close()
    await asyncio.sleep(0)
    lifetime.close()

    assert provider.first_cancelled.is_set()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(events.next_background_event(), timeout=0.01)
    assert asyncio.all_tasks() == existing_tasks
    assert not (agent_home / "logs").exists()
    with pytest.raises(RuntimeError, match="scheduler is closed"):
        scheduler.start()


@pytest.mark.asyncio
async def test_scheduled_work_scheduler_records_a_distinct_shutdown_cleanup_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_now = NOW.replace(microsecond=0)
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace)
    store = WorkspaceJsonScheduledWorkStore(state)
    persist_scheduled_work(state.path, (replace(_task(), cron="31 0 * * *"),))
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: clock_now,
        new_uuid=uuid4,
    )
    started = asyncio.Event()

    class CleanupFailingProvider(ScriptedFakeProvider):
        async def complete(self, request: object) -> ModelResponse:
            self.complete_requests.append(request)
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise RuntimeError("technical Scheduled Work cleanup failure") from cancellation
            raise AssertionError("cleanup test wait returned unexpectedly")

    runner = ScheduledWorkRunner(
        provider=CleanupFailingProvider(),
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: clock_now,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    clock = ControlledSchedulerClock(clock_now)
    scheduler = ScheduledWorkScheduler(
        store=store,
        coordinator=ScheduledWorkCoordinator(
            runner=runner,
            events=RuntimeEventBroker(),
            now=lambda: clock_now,
            new_uuid=lambda: RUN_UUID,
        ),
        clock=clock,
    )
    lifetime = install_log_capture(home)

    scheduler.start()
    await _wait_until(lambda: clock.sleeps == [60.0])
    await clock.advance(60)
    await started.wait()
    await scheduler.close()
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert content.count(" ERROR ") == 1
    assert (
        f"session={TASK_SESSION_ID} myclaw.schedule.background_coordination: "
        "Scheduled Work scheduler cleanup failed"
    ) in content
    assert "CancelledError" in content
    assert "RuntimeError: [REDACTED]" in content
    assert "technical Scheduled Work cleanup failure" not in content
    assert _task().prompt not in content


class CoordinatedStreamingProvider(ScriptedFakeProvider):
    def __init__(self, events: RuntimeEventBroker) -> None:
        super().__init__()
        self._events = events
        self.background_published = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: object) -> AsyncIterator[ModelStreamEvent]:
        self.stream_requests.append(request)
        yield TextDelta(delta="alpha")
        await self._events.publish_background(
            turn_id=RUN_UUID,
            created_at=NOW,
            payload=BackgroundCompletedPayload(
                kind="scheduled_work",
                title="Weekly project review",
                session_id=TASK_SESSION_ID,
                status="completed",
                summary="Background finished during streaming.",
            ),
        )
        self.background_published.set()
        await self.release.wait()
        yield TextDelta(delta="omega")
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content="alphaomega"),
                usage=_usage(),
                finish_reason="stop",
            )
        )


def _unused_conversation(session_id: str) -> ConversationPort:
    raise AssertionError(f"Unexpected foreground conversation for {session_id}")


@pytest.mark.asyncio
async def test_completed_scheduled_work_publishes_one_background_event(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="No open risks were found."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: NOW,
        new_uuid=lambda: RUN_UUID,
    )
    lifetime = install_log_capture(home)

    result = await coordinator.trigger(_task())
    event = await events.next_background_event()
    lifetime.close()

    assert result.status == "completed"
    assert result.content == "No open risks were found."
    assert result.error is None
    assert event.type == "background_completed"
    assert event.event_id == 0
    assert event.turn_id == RUN_UUID
    assert event.created_at == NOW
    payload = event.payload
    assert isinstance(payload, BackgroundCompletedPayload)
    assert payload.kind == "scheduled_work"
    assert payload.title == "Weekly project review"
    assert payload.session_id == TASK_SESSION_ID
    assert payload.status == "completed"
    assert payload.summary == "No open risks were found."
    session = await sessions.load(TASK_SESSION_ID)
    assert [message.role for message in session.messages] == ["user", "assistant"]
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_background_event_publication_failure_is_recorded_once(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=lambda: USER_UUID,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="PRIVATE BACKGROUND RESULT"),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=iter((USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    events.close()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: NOW,
        new_uuid=lambda: RUN_UUID,
    )
    lifetime = install_log_capture(home)

    with pytest.raises(RuntimeError, match="Runtime event broker is closed"):
        await coordinator.trigger(_task())
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert content.count(" ERROR ") == 1
    assert (
        f"session={TASK_SESSION_ID} myclaw.schedule.background_coordination: "
        "Scheduled Work event publication failed"
    ) in content
    assert "RuntimeError: [REDACTED]" in content
    assert "Runtime event broker is closed" not in content
    for private_content in (
        _task().title,
        _task().prompt,
        "PRIVATE BACKGROUND RESULT",
    ):
        assert private_content not in content


@pytest.mark.asyncio
async def test_failed_scheduled_work_publishes_safe_event_and_does_not_stop_another(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(
                ErrorInfo(
                    code="model_failed",
                    message="Scheduled model call failed safely.",
                )
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Second task recovered."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: NOW,
        new_uuid=iter((RUN_UUID, RUN_TWO_UUID)).__next__,
    )
    failed_task = _task()
    succeeding_task = replace(
        failed_task,
        id="11111111-1111-4111-8111-111111111111",
        title="Daily status",
        session_id="20260713-003000-123000_22222222-2222-4222-8222-222222222222",
    )

    failed = await coordinator.trigger(failed_task)
    completed = await coordinator.trigger(succeeding_task)
    failed_event = await events.next_background_event()
    completed_event = await events.next_background_event()

    assert failed.status == "failed"
    assert failed.error == ErrorInfo(
        code="model_failed",
        message="Scheduled model call failed safely.",
    )
    assert completed.status == "completed"
    assert completed.content == "Second task recovered."
    assert isinstance(failed_event.payload, BackgroundCompletedPayload)
    assert failed_event.payload.status == "failed"
    assert failed_event.payload.summary == "Scheduled model call failed safely."
    assert isinstance(completed_event.payload, BackgroundCompletedPayload)
    assert completed_event.payload.status == "completed"
    assert completed_event.payload.summary == "Second task recovered."


@pytest.mark.asyncio
async def test_overlapping_trigger_of_the_same_scheduled_work_is_skipped(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    provider = BlockingCompletionProvider("Only one run completed.")
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: NOW,
        new_uuid=iter((RUN_UUID, RUN_TWO_UUID)).__next__,
    )
    lifetime = install_log_capture(home)
    task = _task()
    first_execution = asyncio.create_task(coordinator.trigger(task))
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    skipped = None
    overlap_timed_out = False
    try:
        skipped = await asyncio.wait_for(coordinator.trigger(task), timeout=0.05)
    except TimeoutError:
        overlap_timed_out = True
    finally:
        provider.release.set()
        first = await first_execution
    lifetime.close()

    assert overlap_timed_out is False
    assert skipped is not None
    assert skipped.status == "skipped"
    assert skipped.content == ""
    assert skipped.error is None
    assert first.status == "completed"
    event = await events.next_background_event()
    assert event.turn_id == RUN_UUID
    assert len(provider.complete_requests) == 1
    session = await sessions.load(task.session_id)
    assert [message.role for message in session.messages] == ["user", "assistant"]
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(events.next_background_event(), timeout=0.01)
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_cancelled_scheduled_work_emits_no_event_and_can_be_retriggered(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    provider = CancellableThenSuccessfulCompletionProvider()
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: NOW,
        new_uuid=iter((RUN_UUID, RUN_TWO_UUID)).__next__,
    )
    lifetime = install_log_capture(home)
    task = _task()
    cancelled = asyncio.create_task(coordinator.trigger(task))
    await asyncio.wait_for(provider.first_started.wait(), timeout=1)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(events.next_background_event(), timeout=0.01)

    retried = await coordinator.trigger(task)
    event = await events.next_background_event()
    lifetime.close()

    assert retried.status == "completed"
    assert event.turn_id == RUN_TWO_UUID
    assert isinstance(event.payload, BackgroundCompletedPayload)
    assert event.payload.status == "completed"
    assert event.payload.summary == "Retry completed."
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_different_scheduled_work_runs_concurrently_and_events_follow_completion_order(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    provider = ConcurrentCompletionProvider()
    runner = ScheduledWorkRunner(
        provider=provider,
        sessions=sessions,
        workspace=workspace,
        long_term_memory="# Long-term Memory\n",
        settings=ScheduledWorkModelSettings(
            model="cron-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway_for=lambda _session_id: ToolGateway(),
    )
    events = RuntimeEventBroker()
    coordinator = ScheduledWorkCoordinator(
        runner=runner,
        events=events,
        now=lambda: NOW,
        new_uuid=iter((RUN_UUID, RUN_TWO_UUID)).__next__,
    )
    first_task = replace(_task(), prompt="Run first task.")
    second_task = replace(
        _task(),
        id="11111111-1111-4111-8111-111111111111",
        title="Daily status",
        prompt="Run second task.",
        session_id="20260713-003000-123000_22222222-2222-4222-8222-222222222222",
    )

    first_execution = asyncio.create_task(coordinator.trigger(first_task))
    second_execution = asyncio.create_task(coordinator.trigger(second_task))
    await asyncio.wait_for(provider.both_started.wait(), timeout=1)
    provider.release["second"].set()
    second_result = await second_execution
    provider.release["first"].set()
    first_result = await first_execution

    assert first_result.status == "completed"
    assert second_result.status == "completed"
    first_event = await events.next_background_event()
    second_event = await events.next_background_event()
    first_payload = first_event.payload
    second_payload = second_event.payload
    assert isinstance(first_payload, BackgroundCompletedPayload)
    assert isinstance(second_payload, BackgroundCompletedPayload)
    assert (first_event.event_id, first_event.turn_id, first_payload.title) == (
        0,
        RUN_TWO_UUID,
        "Daily status",
    )
    assert (second_event.event_id, second_event.turn_id, second_payload.title) == (
        1,
        RUN_UUID,
        "Weekly project review",
    )
    first_session = await sessions.load(first_task.session_id)
    second_session = await sessions.load(second_task.session_id)
    assert first_session.messages[-1].content == "First finished second."
    assert second_session.messages[-1].content == "Second finished first."


@pytest.mark.asyncio
async def test_idle_repl_displays_background_completion_without_restarting_input() -> None:
    events = RuntimeEventBroker()
    conversation = SwitchableConversationPort(
        session_id=TASK_SESSION_ID,
        build_conversation=_unused_conversation,
    )
    input_reader = WaitingExitInput()
    writer = RecordingWriter()
    repl = asyncio.create_task(
        run_repl(
            conversation=conversation,
            input_reader=input_reader,
            writer=writer,
            background_events=events,
        )
    )
    await asyncio.wait_for(input_reader.started.wait(), timeout=1)

    await events.publish_background(
        turn_id=RUN_UUID,
        created_at=NOW,
        payload=BackgroundCompletedPayload(
            kind="scheduled_work",
            title="Weekly project review",
            session_id=TASK_SESSION_ID,
            status="completed",
            summary="No open risks were found.",
        ),
    )
    await asyncio.wait_for(writer.written.wait(), timeout=1)
    input_reader.release.set()
    await repl

    assert writer.operations == [
        (
            "line",
            "[Scheduled Work] Weekly project review (completed): No open risks were found.",
        )
    ]
    assert input_reader.read_calls == 1


@pytest.mark.asyncio
async def test_background_completion_wins_when_it_arrives_with_eof() -> None:
    events = RuntimeEventBroker()
    conversation = SwitchableConversationPort(
        session_id=TASK_SESSION_ID,
        build_conversation=_unused_conversation,
    )
    input_reader = SimultaneousEofInput()
    writer = RecordingWriter()
    repl = asyncio.create_task(
        run_repl(
            conversation=conversation,
            input_reader=input_reader,
            writer=writer,
            background_events=events,
        )
    )
    await asyncio.wait_for(input_reader.started.wait(), timeout=1)

    input_reader.release.set()
    await events.publish_background(
        turn_id=RUN_UUID,
        created_at=NOW,
        payload=BackgroundCompletedPayload(
            kind="scheduled_work",
            title="Weekly project review",
            session_id=TASK_SESSION_ID,
            status="completed",
            summary="Completed with EOF.",
        ),
    )
    await repl

    assert writer.operations == [
        (
            "line",
            "[Scheduled Work] Weekly project review (completed): Completed with EOF.",
        )
    ]


@pytest.mark.asyncio
async def test_explicit_exit_wins_when_background_completion_arrives_in_the_same_tick() -> None:
    events = RuntimeEventBroker()
    conversation = SwitchableConversationPort(
        session_id=TASK_SESSION_ID,
        build_conversation=_unused_conversation,
    )
    input_reader = WaitingExitInput()
    writer = RecordingWriter()
    repl = asyncio.create_task(
        run_repl(
            conversation=conversation,
            input_reader=input_reader,
            writer=writer,
            background_events=events,
        )
    )
    await asyncio.wait_for(input_reader.started.wait(), timeout=1)

    input_reader.release.set()
    await events.publish_background(
        turn_id=RUN_UUID,
        created_at=NOW,
        payload=BackgroundCompletedPayload(
            kind="scheduled_work",
            title="Weekly project review",
            session_id=TASK_SESSION_ID,
            status="completed",
            summary="Must be cancelled by explicit exit.",
        ),
    )
    await repl

    assert writer.operations == []


@pytest.mark.asyncio
async def test_repl_retrieves_a_done_input_failure_when_background_rendering_fails() -> None:
    class FailingInput:
        async def read(self) -> str | None:
            await asyncio.sleep(0)
            raise LookupError("input failed concurrently")

    class ImmediateBackground:
        async def next_background_event(self) -> AgentEvent:
            await asyncio.sleep(0)
            return AgentEvent(
                type="background_completed",
                event_id=0,
                turn_id=RUN_UUID,
                created_at=NOW,
                payload=BackgroundCompletedPayload(
                    kind="scheduled_work",
                    title="Concurrent failure",
                    session_id=TASK_SESSION_ID,
                    status="failed",
                    summary="Rendering will fail.",
                ),
            )

        def next_background_event_nowait(self) -> AgentEvent | None:
            return None

    class FailingWriter(RecordingWriter):
        async def write_line(self, content: str) -> None:
            del content
            raise ValueError("background rendering failed")

    conversation = SwitchableConversationPort(
        session_id=TASK_SESSION_ID,
        build_conversation=_unused_conversation,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    diagnostics: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
    try:
        with pytest.raises(ValueError, match="background rendering failed"):
            await run_repl(
                conversation=conversation,
                input_reader=FailingInput(),
                writer=FailingWriter(),
                background_events=ImmediateBackground(),
            )
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert diagnostics == []


@pytest.mark.asyncio
async def test_closing_event_broker_wakes_waiter_and_rejects_new_events() -> None:
    events = RuntimeEventBroker()
    waiter = asyncio.create_task(events.next_background_event())
    await asyncio.sleep(0)

    events.close()
    events.close()

    with pytest.raises(RuntimeError, match="event broker is closed"):
        await waiter
    with pytest.raises(RuntimeError, match="event broker is closed"):
        await events.publish_background(
            turn_id=RUN_UUID,
            created_at=NOW,
            payload=BackgroundCompletedPayload(
                kind="scheduled_work",
                title="Weekly project review",
                session_id=TASK_SESSION_ID,
                status="completed",
                summary="Must not be queued.",
            ),
        )


@pytest.mark.asyncio
async def test_background_completion_waits_for_foreground_stream_terminal(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    session_id = sessions.prepare().id
    events = RuntimeEventBroker()
    provider = CoordinatedStreamingProvider(events)
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session_id,
        settings=ChatModelSettings(
            model="chat-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    writer = RecordingWriter()
    repl = asyncio.create_task(
        run_repl(
            conversation=conversation,
            input_reader=ScriptedInput(("Run foreground.", "exit")),
            writer=writer,
            background_events=events,
        )
    )

    await asyncio.wait_for(provider.background_published.wait(), timeout=1)
    assert writer.operations == [("delta", "alpha")]
    provider.release.set()
    await repl

    assert writer.operations == [
        ("delta", "alpha"),
        ("delta", "omega"),
        ("finish", ""),
        (
            "line",
            "[Scheduled Work] Weekly project review (completed): "
            "Background finished during streaming.",
        ),
    ]


@pytest.mark.asyncio
async def test_queued_background_is_rendered_before_starting_the_next_input_read(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    events = RuntimeEventBroker()
    provider = CoordinatedStreamingProvider(events)
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=sessions.prepare().id,
        settings=ChatModelSettings(
            model="chat-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    order: list[str] = []
    input_reader = TerminalQueueInput(order)
    writer = OrderingWriter(order)
    repl = asyncio.create_task(
        run_repl(
            conversation=conversation,
            input_reader=input_reader,
            writer=writer,
            background_events=events,
        )
    )
    await asyncio.wait_for(provider.background_published.wait(), timeout=1)
    provider.release.set()
    await asyncio.wait_for(writer.background_written.wait(), timeout=1)
    await asyncio.wait_for(input_reader.second_started.wait(), timeout=1)
    input_reader.release.set()
    await repl

    assert order == [
        "delta:alpha",
        "delta:omega",
        "finish",
        "background:[Scheduled Work] Weekly project review (completed): "
        "Background finished during streaming.",
        "input:second",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal", "terminal_operation"),
    (("failed", ("line", "Foreground failed safely.")), ("cancelled", ("finish", ""))),
)
async def test_background_completion_waits_for_failed_or_cancelled_terminal(
    terminal: Literal["failed", "cancelled"],
    terminal_operation: tuple[str, str],
) -> None:
    broker = RuntimeEventBroker()
    delegate = ForegroundTerminalConversation(broker=broker, terminal=terminal)
    conversation = SwitchableConversationPort(
        session_id=TASK_SESSION_ID,
        build_conversation=lambda _session_id: delegate,
        event_sequencer=broker,
    )
    writer = RecordingWriter()

    await run_repl(
        conversation=conversation,
        input_reader=ScriptedInput(("Run foreground.", "exit")),
        writer=writer,
        background_events=broker,
    )

    assert writer.operations == [
        terminal_operation,
        (
            "line",
            "[Scheduled Work] Daily status (completed): Completed before foreground terminal.",
        ),
    ]


@pytest.mark.asyncio
async def test_runtime_broker_sequences_foreground_and_background_events_globally(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    session_id = sessions.prepare().id
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="first"),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="first"),
                            usage=_usage(),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    TextDelta(delta="second"),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="second"),
                            usage=_usage(),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    delegate = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session_id,
        settings=ChatModelSettings(
            model="chat-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=45,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    broker = RuntimeEventBroker()
    conversation = SwitchableConversationPort(
        session_id=session_id,
        build_conversation=lambda _session_id: delegate,
        event_sequencer=broker,
    )

    first = [event async for event in conversation.submit("First turn.")]
    await broker.publish_background(
        turn_id=RUN_UUID,
        created_at=NOW,
        payload=BackgroundCompletedPayload(
            kind="scheduled_work",
            title="Weekly project review",
            session_id=TASK_SESSION_ID,
            status="completed",
            summary="Completed between turns.",
        ),
    )
    background = await broker.next_background_event()
    second = [event async for event in conversation.submit("Second turn.")]

    observed = (*first, background, *second)
    assert [event.event_id for event in observed] == list(range(7))
    assert [event.type for event in observed] == [
        "turn_started",
        "text_delta",
        "turn_completed",
        "background_completed",
        "turn_started",
        "text_delta",
        "turn_completed",
    ]


@pytest.mark.asyncio
async def test_runtime_event_ids_remain_global_after_switching_sessions(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    first_session_id = sessions.prepare().id
    second_session_id = sessions.prepare().id
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="first"),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="first"),
                            usage=_usage(),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    TextDelta(delta="resumed"),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="resumed"),
                            usage=_usage(),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )

    def conversation_for(session_id: str) -> ConversationPort:
        return StreamingConversationPort(
            provider=provider,
            sessions=sessions,
            session_id=session_id,
            settings=ChatModelSettings(
                model="chat-model",
                max_output=1024,
                temperature=0.1,
                reasoning_effort=None,
                timeout_seconds=45,
            ),
            now=lambda: NOW,
            new_uuid=uuid4,
        )

    broker = RuntimeEventBroker()
    conversation = SwitchableConversationPort(
        session_id=first_session_id,
        build_conversation=conversation_for,
        event_sequencer=broker,
    )

    first = [event async for event in conversation.submit("First session turn.")]
    await broker.publish_background(
        turn_id=RUN_UUID,
        created_at=NOW,
        payload=BackgroundCompletedPayload(
            kind="scheduled_work",
            title="Weekly project review",
            session_id=TASK_SESSION_ID,
            status="completed",
            summary="Completed before resume.",
        ),
    )
    background = await broker.next_background_event()
    conversation.switch_session(second_session_id)
    resumed = [event async for event in conversation.submit("Resumed session turn.")]

    observed = (*first, background, *resumed)
    assert [event.event_id for event in observed] == list(range(7))
    assert resumed[0].turn_id != first[0].turn_id
    assert conversation.session_id == second_session_id


@pytest.mark.asyncio
async def test_fresh_prepared_runtimes_each_run_scheduled_work(
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_now = NOW.replace(microsecond=0)
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    legacy_path = agent_home / "scheduled-work.json"
    legacy_bytes = b"legacy Agent Home Scheduled Work must not be parsed\xff"
    legacy_path.write_bytes(legacy_bytes)
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    scheduled_clock = ControlledSchedulerClock(clock_now)
    memory_clock = ControlledSchedulerClock(clock_now)
    contents = iter(("First runtime schedule.", "Second runtime schedule."))
    providers: list[ScriptedFakeProvider] = []

    def prepare_runtime() -> PreparedReplRuntime:
        provider = ScriptedFakeProvider(
            completions=(
                ModelResponse(
                    message=AssistantModelMessage(content=next(contents)),
                    usage=_usage(),
                    finish_reason="stop",
                ),
            )
        )
        providers.append(provider)
        return prepare_repl_runtime(
            agent_home=home,
            workspace=workspace,
            configuration=ConfigLoader(home).load(),
            provider_factory=lambda _configuration: provider,
            now=scheduled_clock.now,
            new_uuid=uuid4,
            memory_scheduler_clock=memory_clock,
            scheduled_work_scheduler_clock=scheduled_clock,
        )

    first_runtime = prepare_runtime()
    persist_scheduled_work(state.path, (replace(_task(), cron="* * * * *"),))

    async def run_once(runtime: PreparedReplRuntime) -> list[tuple[str, str]]:
        sleep_count = len(scheduled_clock.sleeps)
        input_reader = WaitingExitInput()
        writer = RecordingWriter()
        execution = asyncio.create_task(runtime.run(input_reader=input_reader, writer=writer))
        await asyncio.wait_for(input_reader.started.wait(), timeout=1)
        await _wait_until(lambda: len(scheduled_clock.sleeps) > sleep_count)
        await scheduled_clock.advance(60)
        await asyncio.wait_for(writer.written.wait(), timeout=1)
        input_reader.release.set()
        await execution
        return writer.operations

    first = await run_once(first_runtime)
    second = await run_once(prepare_runtime())

    assert first == [
        (
            "line",
            "[Scheduled Work] Weekly project review (completed): First runtime schedule.",
        )
    ]
    assert second == [
        (
            "line",
            "[Scheduled Work] Weekly project review (completed): Second runtime schedule.",
        )
    ]
    assert [len(provider.complete_requests) for provider in providers] == [1, 1]
    assert legacy_path.read_bytes() == legacy_bytes


@pytest.mark.asyncio
async def test_scheduled_work_does_not_cross_trigger_between_workspaces(
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_now = NOW.replace(microsecond=0)
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    legacy_path = agent_home / "scheduled-work.json"
    legacy_bytes = b"legacy Agent Home definitions must remain unread\xff"
    legacy_path.write_bytes(legacy_bytes)

    first_workspace = workspace / "first"
    second_workspace = workspace / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first_state = WorkspaceState(Workspace.from_path(first_workspace))
    second_state = WorkspaceState(Workspace.from_path(second_workspace))
    first_state.initialize(agent_home_root=Path.home() / ".myclaw")
    second_state.initialize(agent_home_root=Path.home() / ".myclaw")
    first_memory = "# First Workspace Memory\n"
    second_memory = "# Second Workspace Memory\n"
    first_state.long_term_memory_path.write_text(first_memory, encoding="utf-8")
    second_state.long_term_memory_path.write_text(second_memory, encoding="utf-8")
    first_task = replace(
        _task(),
        id="11111111-1111-4111-8111-111111111111",
        title="First Workspace review",
        cron="* * * * *",
        prompt="Run only for the first Workspace.",
        session_id="20260713-003000-123000_22222222-2222-4222-8222-222222222222",
    )
    second_task = replace(
        _task(),
        id="33333333-3333-4333-8333-333333333333",
        title="Second Workspace review",
        cron="* * * * *",
        prompt="Run only for the second Workspace.",
        session_id="20260713-003000-123000_44444444-4444-4444-8444-444444444444",
    )
    persist_scheduled_work(first_state.path, (first_task,))
    persist_scheduled_work(second_state.path, (second_task,))

    first_clock = ControlledSchedulerClock(clock_now)
    second_clock = ControlledSchedulerClock(clock_now)
    first_provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="First Workspace result."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )
    second_provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Second Workspace result."),
                usage=_usage(),
                finish_reason="stop",
            ),
        )
    )

    def prepare(
        *,
        state: WorkspaceState,
        provider: ScriptedFakeProvider,
        clock: ControlledSchedulerClock,
    ) -> PreparedReplRuntime:
        return prepare_repl_runtime(
            agent_home=home,
            workspace=Path(state.workspace.path),
            configuration=ConfigLoader(home).load(),
            provider_factory=lambda _configuration: provider,
            now=clock.now,
            new_uuid=uuid4,
            memory_scheduler_clock=ControlledSchedulerClock(clock_now),
            scheduled_work_scheduler_clock=clock,
        )

    first_runtime = prepare(state=first_state, provider=first_provider, clock=first_clock)
    second_runtime = prepare(state=second_state, provider=second_provider, clock=second_clock)
    first_input = WaitingExitInput()
    second_input = WaitingExitInput()
    first_writer = RecordingWriter()
    second_writer = RecordingWriter()
    first_execution = asyncio.create_task(
        first_runtime.run(input_reader=first_input, writer=first_writer)
    )
    second_execution = asyncio.create_task(
        second_runtime.run(input_reader=second_input, writer=second_writer)
    )

    await asyncio.wait_for(first_input.started.wait(), timeout=1)
    await asyncio.wait_for(second_input.started.wait(), timeout=1)
    await _wait_until(lambda: first_clock.sleeps == [60.0])
    await _wait_until(lambda: second_clock.sleeps == [60.0])
    await first_clock.advance(60)
    await asyncio.wait_for(first_writer.written.wait(), timeout=1)

    assert len(first_provider.complete_requests) == 1
    assert second_provider.complete_requests == []
    assert second_writer.operations == []

    await second_clock.advance(60)
    await asyncio.wait_for(second_writer.written.wait(), timeout=1)
    first_input.release.set()
    second_input.release.set()
    await asyncio.gather(first_execution, second_execution)

    assert first_writer.operations == [
        (
            "line",
            "[Scheduled Work] First Workspace review (completed): First Workspace result.",
        )
    ]
    assert second_writer.operations == [
        (
            "line",
            "[Scheduled Work] Second Workspace review (completed): Second Workspace result.",
        )
    ]
    first_request = first_provider.complete_requests[0]
    second_request = second_provider.complete_requests[0]
    assert isinstance(first_request, ModelRequest)
    assert isinstance(second_request, ModelRequest)
    assert first_memory in first_request.system_prompt
    assert second_memory not in first_request.system_prompt
    assert second_memory in second_request.system_prompt
    assert first_memory not in second_request.system_prompt
    assert first_runtime.sessions.path_for(first_task.session_id).parent == (
        first_state.sessions_directory
    )
    assert second_runtime.sessions.path_for(second_task.session_id).parent == (
        second_state.sessions_directory
    )
    assert legacy_path.read_bytes() == legacy_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_failure", "expected_cursor"),
    ((False, 1), (True, 0)),
    ids=("success", "failure"),
)
async def test_periodic_memory_stays_silent_while_scheduled_work_remains_visible(
    memory_failure: bool,
    expected_cursor: int,
    agent_home: Path,
    workspace: Path,
) -> None:
    clock_now = NOW.replace(microsecond=0)
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            'schedule = "15 * * * *"',
            'schedule = "* * * * *"',
        ).replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("A pending memory summary.", clock_now)
    scheduled_clock = ControlledSchedulerClock(clock_now)
    memory_clock = ControlledSchedulerClock(clock_now)
    provider = RuntimeCoordinationProvider(memory_failure=memory_failure)
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=scheduled_clock.now,
        new_uuid=uuid4,
        memory_scheduler_clock=memory_clock,
        scheduled_work_scheduler_clock=scheduled_clock,
    )
    persist_scheduled_work(state.path, (replace(_task(), cron="* * * * *"),))
    input_reader = ReleasedConversationInput()
    writer = RecordingWriter()
    execution = asyncio.create_task(runtime.run(input_reader=input_reader, writer=writer))
    await asyncio.wait_for(input_reader.started.wait(), timeout=1)
    await _wait_until(lambda: scheduled_clock.sleeps == [60.0] and memory_clock.sleeps == [60.0])

    await memory_clock.advance(60)
    await scheduled_clock.advance(60)
    await asyncio.wait_for(provider.memory_finished.wait(), timeout=1)
    await _wait_until(
        lambda: any(
            operation
            == (
                "line",
                "[Scheduled Work] Weekly project review (completed): "
                "Scheduled work remains visible.",
            )
            for operation in writer.operations
        )
    )
    for _ in range(100):
        cursor = await WorkspaceFileMemoryStore(state).read_summary_cursor()
        if cursor == expected_cursor:
            break
        await asyncio.sleep(0)

    assert cursor == expected_cursor
    assert writer.operations == [
        (
            "line",
            "[Scheduled Work] Weekly project review (completed): Scheduled work remains visible.",
        )
    ]

    input_reader.release.set()
    await execution

    assert writer.operations == [
        (
            "line",
            "[Scheduled Work] Weekly project review (completed): Scheduled work remains visible.",
        ),
        ("delta", "Foreground continues."),
        ("finish", ""),
    ]
    routes = []
    for request in provider.complete_requests:
        assert isinstance(request, ModelRequest)
        routes.append(request.route)
    assert sorted(routes) == ["cron", "memory"]
