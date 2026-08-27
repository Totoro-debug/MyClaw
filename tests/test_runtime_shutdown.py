import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.message_bus import InboundMessage
from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import prepare_runtime
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.logging.session import session_log
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    TextDelta,
)
from myclaw.tools.base import OpenAIToolSchema
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    BlockingTaskFramingEvaluator,
    DeterministicTaskFramingEvaluator,
    ScriptedFakeProvider,
    StreamScript,
)
from tests.fixtures.diagnostic_capture import capture_diagnostics

NOW = datetime(2026, 7, 13, 0, 30, tzinfo=timezone(timedelta(hours=8)))


def _session_id() -> str:
    return f"20260713-003000-000000_{uuid4()}"


class _FixedScheduleClock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


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
        return NOW

    def monotonic(self) -> float:
        raise RuntimeError("Schedule Service clock failed to start")

    async def sleep(self, seconds: float) -> None:
        del seconds


class UnusedInput:
    async def read(self) -> str | None:
        raise AssertionError("REPL input must not start after scheduler startup failure")


class FailingInput:
    async def read(self) -> str | None:
        raise LookupError("input failed")


class CacheThenFailingInput:
    def __init__(self) -> None:
        self._read_count = 0

    async def read(self) -> str | None:
        if self._read_count == 0:
            self._read_count += 1
            return "Cache the provider."
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
async def test_runtime_shutdown_cancels_blocked_framing_and_reclaims_title_work(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = ScriptedFakeProvider()
    framer = BlockingTaskFramingEvaluator()
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    runtime.agent_loop._task_framer = framer

    await runtime.start()
    await runtime.bus.put_inbound(InboundMessage("Blocked framing during shutdown"))
    await framer.started.wait()
    execution_task = runtime.agent_loop._execution_task
    assert execution_task is not None
    title_work = tuple(runtime.agent_loop._title_work.values())
    assert len(title_work) == 1
    title_task = title_work[0].task
    execution_done = asyncio.Event()
    title_done = asyncio.Event()
    execution_task.add_done_callback(lambda _task: execution_done.set())
    title_task.add_done_callback(lambda _task: title_done.set())

    await runtime.close()

    await asyncio.wait_for(framer.cancelled.wait(), timeout=3)
    await asyncio.wait_for(execution_done.wait(), timeout=3)
    await asyncio.wait_for(title_done.wait(), timeout=3)
    terminal = await asyncio.wait_for(runtime.bus.get_outbound(), timeout=3)
    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert runtime.session.messages == []
    assert runtime.session.metadata["title"] == "Untitled session"
    assert runtime.session.metadata["token_usage"] == {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert runtime.agent_loop._execution_task is None
    assert runtime.agent_loop._title_work == {}
    assert runtime.agent_loop._aborted_tasks == set()
    assert provider.closed


@pytest.mark.asyncio
async def test_scheduler_preflight_failure_starts_no_runtime_tasks(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
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
async def test_start_preflight_failure_requires_no_async_cleanup(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
        schedule_scheduler_clock=FailingSchedulerClock(),
    )
    baseline = asyncio.all_tasks()
    log_capture = capture_diagnostics()
    state = WorkspaceState(workspace)
    ambient_session_id = _session_id()

    try:
        with session_log(state, ambient_session_id):
            with pytest.raises(RuntimeError, match="Schedule Service clock failed to start"):
                await runtime.start()
    finally:
        await runtime.close()
        log_capture.close()

    assert asyncio.all_tasks() == baseline
    assert not (state.logs_directory / f"{ambient_session_id}.log").exists()
    content = log_capture.text
    assert content.count(" ERROR ") == 1
    marker = "Runtime validation failed type=RuntimeError"
    assert content.count(marker) == 1


@pytest.mark.asyncio
async def test_runtime_start_activation_failure_aborts_all_owned_tasks(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    baseline = asyncio.all_tasks()

    def fail_schedule_activation() -> None:
        raise RuntimeError("Schedule Service activation failed")

    monkeypatch.setattr(runtime.schedule_service, "_activate_prepared", fail_schedule_activation)

    with pytest.raises(RuntimeError, match="Schedule Service activation failed"):
        await runtime.start()
    await asyncio.sleep(0)

    assert runtime.agent_loop._consumer_task is None
    assert runtime.schedule_service._loop_task is None
    assert runtime.schedule_service._run_tasks == set()
    assert runtime.schedule_service._terminal_commit_tasks == set()
    assert runtime._dream._aborted is True
    assert runtime._router._aborted is True
    assert asyncio.all_tasks() == baseline


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
    runtime = prepare_runtime(
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
    runtime = prepare_runtime(
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
    runtime = prepare_runtime(
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
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    with pytest.raises(LookupError, match="input failed") as raised:
        await runtime.run(input_reader=CacheThenFailingInput(), writer=SilentWriter())

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
    runtime = prepare_runtime(
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
    runtime = prepare_runtime(
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
async def test_repeated_and_idle_cancellations_cancel_only_foreground_until_exit(
    agent_home: Path,
    workspace: Path,
) -> None:
    class InterruptibleProvider:
        def __init__(self) -> None:
            self.started = (asyncio.Event(), asyncio.Event())
            self.stopped = (asyncio.Event(), asyncio.Event())
            self.chat_calls = 0
            self.closed = False

        async def stream(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str,
            max_output: int,
            temperature: float,
            reasoning_effort: ReasoningEffort | None,
            timeout: int,
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            if messages and messages[0] == {
                "role": "system",
                "content": session_title_prompt(),
            }:
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

        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str,
            max_output: int,
            temperature: float,
            reasoning_effort: ReasoningEffort | None,
            timeout: int,
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            raise AssertionError(f"Unexpected completion: {messages!r}")

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

    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = InterruptibleProvider()
    scheduled_clock = BlockingSchedulerClock()
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        schedule_scheduler_clock=scheduled_clock,
    )
    runtime.agent_loop._task_framer = DeterministicTaskFramingEvaluator()
    input_reader = ControlledInput()
    log_capture = capture_diagnostics()
    running = asyncio.create_task(runtime.run(input_reader=input_reader, writer=SilentWriter()))
    await provider.started[0].wait()
    await scheduled_clock.sleep_started.wait()
    try:
        await runtime.control.cancel_active_run()
        await input_reader.idle.wait()
        assert not scheduled_clock.sleep_stopped.is_set()

        await runtime.control.cancel_active_run()
        await asyncio.sleep(0)
        assert not running.done()
        input_reader.release_second.set()

        await provider.started[1].wait()
        await runtime.control.cancel_active_run()
        await input_reader.waiting_for_exit.wait()
        assert running.cancelling() == 0
        assert not scheduled_clock.sleep_stopped.is_set()

        input_reader.release_exit.set()
        await running
    finally:
        input_reader.release_second.set()
        input_reader.release_exit.set()
        await asyncio.gather(running, return_exceptions=True)
        log_capture.close()

    assert provider.stopped[0].is_set()
    assert provider.stopped[1].is_set()
    assert scheduled_clock.sleep_stopped.is_set()
    assert provider.closed
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
