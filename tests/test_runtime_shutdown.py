import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import _DeferredConversationPort, prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.logging.session import session_log
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.session.conversation import ChatModelSettings
from myclaw.session.session import Session
from myclaw.tools.tool_gateway import ToolGateway
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript
from tests.fixtures.diagnostic_capture import capture_diagnostics

NOW = datetime(2026, 7, 13, 0, 30, tzinfo=timezone(timedelta(hours=8)))


def _session_id() -> str:
    return f"20260713-003000-000000_{uuid4()}"


def _fixed_gateway(workspace: Path, agent_home: Path) -> ToolGateway:
    identity = Workspace.from_path(workspace)
    state = WorkspaceState(identity)
    state.initialize(agent_home_root=agent_home)
    return ToolGateway(
        workspace=identity,
        schedule_store=WorkspaceScheduleStore(state),
    )


class BlockingSchedulerClock:
    def __init__(self) -> None:
        self.sleep_started = asyncio.Event()
        self.sleep_stopped = asyncio.Event()

    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds
        self.sleep_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.sleep_stopped.set()


class FailingSchedulerClock:
    def now(self) -> datetime:
        raise RuntimeError("Schedule Service clock failed to start")

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


class UnusedInput:
    async def read(self) -> str | None:
        raise AssertionError("REPL input must not start after scheduler startup failure")


class FailingInput:
    async def read(self) -> str | None:
        raise LookupError("input failed")


class BlockingInput:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    async def read(self) -> str | None:
        self.started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("blocking input unexpectedly resumed")
        finally:
            self.stopped.set()


class SilentWriter:
    async def write_delta(self, delta: str) -> None:
        del delta

    async def finish_turn(self) -> None:
        return None

    async def write_line(self, content: str) -> None:
        del content


class FailingDeltaWriter(SilentWriter):
    async def write_delta(self, delta: str) -> None:
        del delta
        raise OSError("writer failed")


class ScriptedInput:
    def __init__(self, values: Iterable[str | None]) -> None:
        self._values = deque(values)

    async def read(self) -> str | None:
        return self._values.popleft()


@pytest.mark.asyncio
async def test_partial_scheduler_start_failure_closes_the_started_memory_loop(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    memory_clock = BlockingSchedulerClock()
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
        memory_scheduler_clock=memory_clock,
        schedule_scheduler_clock=FailingSchedulerClock(),
    )
    baseline = asyncio.all_tasks()

    try:
        with pytest.raises(RuntimeError, match="Schedule Service clock failed to start"):
            await runtime.run(input_reader=UnusedInput(), writer=SilentWriter())
        await asyncio.sleep(0)

        assert asyncio.all_tasks() == baseline
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_async_start_rolls_back_a_partial_scheduler_failure_before_raising(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
        memory_scheduler_clock=BlockingSchedulerClock(),
        schedule_scheduler_clock=FailingSchedulerClock(),
    )
    baseline = asyncio.all_tasks()
    log_capture = capture_diagnostics()
    state = WorkspaceState(Workspace.from_path(workspace))
    ambient_session_id = _session_id()

    try:
        with session_log(state, ambient_session_id):
            with pytest.raises(RuntimeError, match="Schedule Service clock failed to start"):
                await runtime.start()
        await asyncio.sleep(0)
    finally:
        log_capture.close()

    assert asyncio.all_tasks() == baseline
    assert not (state.logs_directory / f"{ambient_session_id}.log").exists()
    content = log_capture.text
    assert content.count(" ERROR ") == 1
    marker = "Runtime startup failed type=RuntimeError"
    assert content.count(marker) == 1


@pytest.mark.asyncio
async def test_normal_repl_exit_closes_the_runtime_model_provider(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Finished."),
                            usage=ModelUsage(input_tokens=3, output_tokens=1, total_tokens=4),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    await runtime.run(
        input_reader=ScriptedInput(("Complete one turn.", "exit")),
        writer=SilentWriter(),
    )

    assert provider.closed


@pytest.mark.parametrize("terminal_input", [None, "exit", "  QuIt  "])
@pytest.mark.asyncio
async def test_normal_eof_and_exit_shutdown_do_not_create_diagnostic_log(
    agent_home: Path,
    workspace: Path,
    terminal_input: str | None,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    log_capture = capture_diagnostics()

    try:
        await runtime.run(
            input_reader=ScriptedInput((terminal_input,)),
            writer=SilentWriter(),
        )
    finally:
        log_capture.close()

    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_prepared_runtime_rejects_a_second_repl_invocation(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    await runtime.run(
        input_reader=ScriptedInput(("exit",)),
        writer=SilentWriter(),
    )

    with pytest.raises(RuntimeError, match="Prepared Runtime is closed"):
        await runtime.run(input_reader=UnusedInput(), writer=SilentWriter())


@pytest.mark.asyncio
async def test_runtime_run_preserves_the_primary_error_when_cleanup_also_fails(
    agent_home: Path,
    workspace: Path,
) -> None:
    class FailingCloseProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            self.closed = True
            raise RuntimeError("provider cleanup failed")

    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = FailingCloseProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Prepared."),
                            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    _ = [event async for event in runtime.conversation.submit("Cache the provider.")]

    with pytest.raises(LookupError, match="input failed") as raised:
        await runtime.run(input_reader=FailingInput(), writer=SilentWriter())

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "provider cleanup failed"
    assert provider.closed


@pytest.mark.asyncio
async def test_external_runtime_close_waits_for_the_repl_and_input_to_stop(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    input_reader = BlockingInput()
    running = asyncio.create_task(runtime.run(input_reader=input_reader, writer=SilentWriter()))
    await input_reader.started.wait()

    await runtime.close()

    assert running.done()
    assert input_reader.stopped.is_set()
    await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_writer_failure_finishes_runtime_shutdown_without_task_leaks(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="Partial output."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Complete output."),
                            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    baseline = asyncio.all_tasks()

    with pytest.raises(OSError, match="writer failed"):
        await asyncio.wait_for(
            runtime.run(
                input_reader=ScriptedInput(("Render this turn.",)),
                writer=FailingDeltaWriter(),
            ),
            timeout=1,
        )
    await asyncio.sleep(0)

    assert provider.closed
    assert asyncio.all_tasks() == baseline


@pytest.mark.asyncio
async def test_deferred_conversation_does_not_construct_after_close_during_pre_submit(
    agent_home: Path,
    workspace: Path,
) -> None:
    before_started = asyncio.Event()
    release_before = asyncio.Event()

    async def before_submit() -> None:
        before_started.set()
        await release_before.wait()

    async def preserve_history(session: Session) -> Session:
        return session

    async def submit() -> list[str]:
        return [event.type async for event in conversation.submit("Do not construct after close.")]

    home = AgentHome(agent_home)
    home.initialize()
    session = Session.create(WorkspaceState(Workspace.from_path(workspace)), now=lambda: NOW)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Must not run."),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = _DeferredConversationPort(
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        system_prompt="system",
        title_prompt=session_title_prompt(),
        tool_gateway=_fixed_gateway(workspace, agent_home),
        history_preparer=preserve_history,
        before_submit=before_submit,
        on_foreground_terminal=lambda: None,
    )
    submitting = asyncio.create_task(submit())
    await before_started.wait()

    try:
        await conversation.close()
        assert submitting.done()
    finally:
        release_before.set()
        if not submitting.done():
            submitting.cancel()
        outcomes = await asyncio.gather(submitting, return_exceptions=True)

    assert isinstance(outcomes[0], asyncio.CancelledError)
    assert provider.stream_requests == []


@pytest.mark.asyncio
async def test_deferred_conversation_interrupts_pre_submit_without_closing_the_port(
    agent_home: Path,
    workspace: Path,
) -> None:
    before_started = asyncio.Event()
    release_first_before = asyncio.Event()
    before_calls = 0

    async def before_submit() -> None:
        nonlocal before_calls
        before_calls += 1
        if before_calls == 1:
            before_started.set()
            await release_first_before.wait()

    async def preserve_history(session: Session) -> Session:
        return session

    async def submit(text: str) -> list[str]:
        return [event.type async for event in conversation.submit(text)]

    home = AgentHome(agent_home)
    home.initialize()
    session = Session.create(WorkspaceState(Workspace.from_path(workspace)), now=lambda: NOW)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Second turn completed."),
                            usage=ModelUsage(input_tokens=2, output_tokens=2, total_tokens=4),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = _DeferredConversationPort(
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        system_prompt="system",
        title_prompt=session_title_prompt(),
        tool_gateway=_fixed_gateway(workspace, agent_home),
        history_preparer=preserve_history,
        before_submit=before_submit,
        on_foreground_terminal=lambda: None,
    )
    first_submit = asyncio.create_task(submit("Interrupt before materialization."))
    await before_started.wait()

    try:
        await conversation.cancel_active_turn()
        await conversation.cancel_active_turn()
        await asyncio.sleep(0)

        assert first_submit.done()
        assert first_submit.cancelling() == 1
    finally:
        release_first_before.set()
        if not first_submit.done():
            first_submit.cancel()
        first_outcome = (await asyncio.gather(first_submit, return_exceptions=True))[0]

    assert isinstance(first_outcome, asyncio.CancelledError)
    assert provider.stream_requests == []
    assert await submit("Continue after interrupt.") == ["turn_started", "turn_completed"]
    await conversation.close()


@pytest.mark.asyncio
async def test_deferred_conversation_interrupts_later_pre_submit_with_an_existing_delegate(
    agent_home: Path,
    workspace: Path,
) -> None:
    before_started = asyncio.Event()
    release_before = asyncio.Event()
    before_calls = 0

    async def before_submit() -> None:
        nonlocal before_calls
        before_calls += 1
        if before_calls == 2:
            before_started.set()
            await release_before.wait()

    async def preserve_history(session: Session) -> Session:
        return session

    async def submit(text: str) -> list[str]:
        return [event.type async for event in conversation.submit(text)]

    home = AgentHome(agent_home)
    home.initialize()
    session = Session.create(WorkspaceState(Workspace.from_path(workspace)), now=lambda: NOW)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Delegate materialized."),
                            usage=ModelUsage(input_tokens=2, output_tokens=2, total_tokens=4),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Continued after interrupt."),
                            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = _DeferredConversationPort(
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.1,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        system_prompt="system",
        title_prompt=session_title_prompt(),
        tool_gateway=_fixed_gateway(workspace, agent_home),
        history_preparer=preserve_history,
        before_submit=before_submit,
        on_foreground_terminal=lambda: None,
    )
    assert await submit("Create the delegate.") == ["turn_started", "turn_completed"]
    blocked_submit = asyncio.create_task(submit("Interrupt before the second delegate call."))
    await before_started.wait()

    try:
        await conversation.cancel_active_turn()
        await conversation.cancel_active_turn()
        await asyncio.sleep(0)

        assert blocked_submit.done()
        assert blocked_submit.cancelling() == 1
    finally:
        release_before.set()
        if not blocked_submit.done():
            blocked_submit.cancel()
        blocked_outcome = (await asyncio.gather(blocked_submit, return_exceptions=True))[0]

    assert isinstance(blocked_outcome, asyncio.CancelledError)
    assert len(provider.stream_requests) == 1
    assert await submit("Continue with the delegate.") == ["turn_started", "turn_completed"]
    await conversation.close()


@pytest.mark.asyncio
async def test_interrupt_controller_restores_handler_and_drains_foreground_cancels() -> None:
    from types import FrameType

    from myclaw.terminal.interrupts import (
        ForegroundInterruptController,
        SignalDisposition,
    )

    previous_calls = 0
    cancel_calls = 0
    cancel_started = asyncio.Event()
    release_cancel = asyncio.Event()

    def previous_handler(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        nonlocal previous_calls
        previous_calls += 1

    class SignalSetter:
        def __init__(self) -> None:
            self.current: SignalDisposition = previous_handler

        def __call__(self, signum: int, handler: SignalDisposition) -> SignalDisposition:
            assert signum > 0
            prior = self.current
            self.current = handler
            return prior

    async def cancel_foreground() -> None:
        nonlocal cancel_calls
        cancel_calls += 1
        cancel_started.set()
        await release_cancel.wait()

    signals = SignalSetter()
    controller = ForegroundInterruptController(
        loop=asyncio.get_running_loop(),
        cancel_foreground=cancel_foreground,
        set_signal=signals,
    )
    controller.install()
    assert callable(signals.current)

    signals.current(2, None)
    await cancel_started.wait()
    closing = asyncio.create_task(controller.close())
    await asyncio.sleep(0)

    try:
        assert not closing.done()
        during_close = signals.current
        assert callable(during_close)
        during_close(2, None)
        await asyncio.sleep(0)
        assert cancel_calls == 1
    finally:
        release_cancel.set()
        await closing
        controller.restore()

    assert cancel_calls == 1
    assert signals.current is previous_handler
    assert previous_calls == 0


@pytest.mark.asyncio
async def test_repeated_and_idle_interrupts_cancel_only_foreground_until_exit(
    agent_home: Path,
    workspace: Path,
) -> None:
    from myclaw.terminal.interrupts import (
        ForegroundInterruptController,
        SignalDisposition,
    )

    class InterruptibleProvider:
        def __init__(self) -> None:
            self.started = (asyncio.Event(), asyncio.Event())
            self.stopped = (asyncio.Event(), asyncio.Event())
            self.chat_calls = 0
            self.closed = False

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            if request.system_prompt == session_title_prompt():
                yield ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(content="Interrupt test"),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="stop",
                    )
                )
                return
            index = self.chat_calls
            self.chat_calls += 1
            self.started[index].set()
            try:
                yield TextDelta(delta=f"Partial {index + 1}")
                await asyncio.Event().wait()
            finally:
                self.stopped[index].set()

        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise AssertionError(f"Unexpected completion: {request!r}")

        async def close(self) -> None:
            self.closed = True

    class ControlledInput:
        def __init__(self) -> None:
            self._calls = 0
            self.idle = asyncio.Event()
            self.release_second = asyncio.Event()
            self.waiting_for_exit = asyncio.Event()
            self.release_exit = asyncio.Event()

        async def read(self) -> str:
            self._calls += 1
            if self._calls == 1:
                return "First interruptible turn."
            if self._calls == 2:
                self.idle.set()
                await self.release_second.wait()
                return "Second interruptible turn."
            self.waiting_for_exit.set()
            await self.release_exit.wait()
            return "exit"

    def previous_handler(signum: int, frame: object) -> None:
        del signum, frame

    class SignalSetter:
        def __init__(self) -> None:
            self.current: SignalDisposition = previous_handler

        def __call__(self, signum: int, handler: SignalDisposition) -> SignalDisposition:
            assert signum > 0
            previous = self.current
            self.current = handler
            return previous

    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = InterruptibleProvider()
    memory_clock = BlockingSchedulerClock()
    scheduled_clock = BlockingSchedulerClock()
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        memory_scheduler_clock=memory_clock,
        schedule_scheduler_clock=scheduled_clock,
    )
    input_reader = ControlledInput()
    signals = SignalSetter()
    interrupts = ForegroundInterruptController(
        loop=asyncio.get_running_loop(),
        cancel_foreground=runtime.conversation.cancel_active_turn,
        set_signal=signals,
    )
    interrupts.install()
    log_capture = capture_diagnostics()
    running = asyncio.create_task(runtime.run(input_reader=input_reader, writer=SilentWriter()))
    await provider.started[0].wait()
    await memory_clock.sleep_started.wait()
    await scheduled_clock.sleep_started.wait()
    try:
        first_handler = signals.current
        assert callable(first_handler)
        first_handler(2, None)
        await input_reader.idle.wait()
        assert not memory_clock.sleep_stopped.is_set()
        assert not scheduled_clock.sleep_stopped.is_set()

        idle_handler = signals.current
        assert callable(idle_handler)
        idle_handler(2, None)
        await asyncio.sleep(0)
        assert not running.done()
        input_reader.release_second.set()

        await provider.started[1].wait()
        second_handler = signals.current
        assert callable(second_handler)
        second_handler(2, None)
        await input_reader.waiting_for_exit.wait()
        assert running.cancelling() == 0
        assert not memory_clock.sleep_stopped.is_set()
        assert not scheduled_clock.sleep_stopped.is_set()

        input_reader.release_exit.set()
        await running
    finally:
        input_reader.release_second.set()
        input_reader.release_exit.set()
        await asyncio.gather(running, return_exceptions=True)
        await interrupts.close()
        interrupts.restore()
        log_capture.close()

    assert provider.stopped[0].is_set()
    assert provider.stopped[1].is_set()
    assert memory_clock.sleep_stopped.is_set()
    assert scheduled_clock.sleep_stopped.is_set()
    assert provider.closed
    assert signals.current is previous_handler
    content = log_capture.text
    records = [line for line in content.splitlines() if "myclaw.provider.model_router:" in line]
    assert len(records) == 3
    assert all(" WARNING " in record for record in records)
    assert all(
        "Default Model Route selected code=route_unavailable" in record for record in records
    )
    assert " ERROR " not in content
    assert "cancel" not in content.lower()
    assert "shutdown failed" not in content.lower()
