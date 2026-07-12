import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from uuid import uuid4

import pytest

import myclaw.memory_scheduler as memory_scheduler_module
from myclaw.agent_home import AgentHome
from myclaw.config import ConfigLoader
from myclaw.contracts import (
    AssistantModelMessage,
    ErrorInfo,
    MemoryTaskResult,
    ModelCallError,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
)
from myclaw.conversation_summary import JsonlSummaryStore
from myclaw.memory_scheduler import AsyncioMemorySchedulerClock, MemoryTaskScheduler
from myclaw.memory_task import FileMemoryStore, MemoryManager, MemoryTaskModelSettings
from myclaw.runtime import prepare_repl_runtime
from tests.fixtures import ScriptedFakeProvider, StreamScript
from tests.test_config import VALID_CONFIG

LOCAL = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 16, 10, tzinfo=LOCAL)


class SpringForwardTimezone(tzinfo):
    """Minimal rule-bearing timezone with a 2026 spring-forward transition."""

    _standard = timedelta(hours=-5)
    _daylight = timedelta(hours=-4)
    _wall_transition = datetime(2026, 3, 8, 3)
    _utc_transition = datetime(2026, 3, 8, 7)

    def utcoffset(self, value: datetime | None) -> timedelta | None:
        if value is None:
            return None
        wall_time = value.replace(tzinfo=None)
        return self._daylight if wall_time >= self._wall_transition else self._standard

    def dst(self, value: datetime | None) -> timedelta | None:
        offset = self.utcoffset(value)
        return None if offset is None else offset - self._standard

    def tzname(self, value: datetime | None) -> str | None:
        return "FDT" if self.dst(value) else "FST"

    def fromutc(self, value: datetime) -> datetime:
        utc_time = value.replace(tzinfo=None)
        offset = self._daylight if utc_time >= self._utc_transition else self._standard
        return (utc_time + offset).replace(tzinfo=self)


class ControlledClock:
    """Release scheduler sleeps only when the test advances wall time."""

    def __init__(self, start: datetime) -> None:
        self._now = start
        self.sleeps: list[float] = []
        self._waiters: list[tuple[datetime, asyncio.Future[None]]] = []

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        future = asyncio.get_running_loop().create_future()
        deadline = self._now + timedelta(seconds=seconds)
        self._waiters.append((deadline, future))
        await future

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        for deadline, future in tuple(self._waiters):
            if deadline <= self._now and not future.done():
                future.set_result(None)
                self._waiters.remove((deadline, future))

    def report_in_timezone(self, zone: tzinfo) -> None:
        self._now = self._now.astimezone(zone)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def _response(
    content: str,
    *,
    tool_calls: tuple[ModelToolCall, ...] = (),
) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content, tool_calls=tool_calls),
        usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
        finish_reason="tool_calls" if tool_calls else "stop",
    )


def _manager(
    *,
    home: AgentHome,
    provider: ScriptedFakeProvider,
    summaries: JsonlSummaryStore,
) -> MemoryManager:
    return MemoryManager(
        provider=provider,
        summaries=summaries,
        memory=FileMemoryStore(home),
        long_term_path=home.path / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )


@pytest.mark.asyncio
async def test_periodic_memory_task_runs_when_the_manager_is_idle(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No update needed."),))
    manager = _manager(home=home, provider=provider, summaries=summaries)

    result = await manager.run_periodic()

    assert result == MemoryTaskResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )


@pytest.mark.asyncio
async def test_hourly_memory_schedule_triggers_only_at_next_local_cron_boundary(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No update needed."),))
    clock = ControlledClock(NOW)
    scheduler = MemoryTaskScheduler(
        manager=_manager(home=home, provider=provider, summaries=summaries),
        schedule="0 * * * *",
        clock=clock,
    )

    scheduler.start()
    await _wait_until(lambda: clock.sleeps == [50 * 60])
    clock.advance(50 * 60 - 1)
    await asyncio.sleep(0)
    assert provider.complete_requests == []

    clock.advance(1)
    await _wait_until(lambda: len(provider.complete_requests) == 1)
    await scheduler.close()

    assert await FileMemoryStore(home).read_summary_cursor() == 1


@pytest.mark.asyncio
async def test_periodic_trigger_is_skipped_while_the_previous_run_is_still_active(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("A pending summary.", NOW)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider(ScriptedFakeProvider):
        async def complete(self, request: object) -> ModelResponse:
            self.complete_requests.append(request)
            started.set()
            await release.wait()
            return _response("No update needed.")

    provider = BlockingProvider()
    clock = ControlledClock(NOW)
    scheduler = MemoryTaskScheduler(
        manager=_manager(home=home, provider=provider, summaries=summaries),
        schedule="0 * * * *",
        clock=clock,
    )

    scheduler.start()
    await _wait_until(lambda: clock.sleeps == [50 * 60])
    clock.advance(50 * 60)
    await started.wait()
    try:
        await _wait_until(lambda: clock.sleeps == [50 * 60, 60 * 60])
        clock.advance(60 * 60)
        await _wait_until(lambda: clock.sleeps == [50 * 60, 60 * 60, 60 * 60])
        assert len(provider.complete_requests) == 1
    finally:
        release.set()
        await scheduler.close()


@pytest.mark.asyncio
async def test_manual_memory_task_reports_running_while_a_periodic_run_is_active(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("A pending summary.", NOW)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider(ScriptedFakeProvider):
        async def complete(self, request: object) -> ModelResponse:
            self.complete_requests.append(request)
            started.set()
            await release.wait()
            return _response("No update needed.")

    provider = BlockingProvider()
    manager = _manager(home=home, provider=provider, summaries=summaries)
    periodic = asyncio.create_task(manager.run_periodic())
    await started.wait()
    (agent_home / "memory" / ".cursor").write_bytes(b"corrupt\n")

    try:
        manual = await manager.run_manual()
    finally:
        release.set()
        await periodic

    assert manual.error == ErrorInfo(
        code="memory_task_running",
        message="A Memory Task is already running.",
    )
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_runtime_starts_the_configured_memory_schedule_with_the_injected_clock(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    summaries = JsonlSummaryStore(home)
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No update needed."),))
    clock = ControlledClock(NOW)
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=clock.now,
        new_uuid=uuid4,
        memory_scheduler_clock=clock,
    )

    runtime.start()
    await _wait_until(lambda: clock.sleeps == [5 * 60])
    clock.advance(5 * 60)
    await _wait_until(lambda: len(provider.complete_requests) == 1)
    await runtime.close()

    assert await FileMemoryStore(home).read_summary_cursor() == 1


@pytest.mark.asyncio
async def test_prepared_runtime_can_run_again_with_a_fresh_memory_scheduler(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    clock = ControlledClock(NOW)
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=clock.now,
        new_uuid=uuid4,
        memory_scheduler_clock=clock,
    )

    class ExitInput:
        async def read(self) -> str:
            await asyncio.sleep(0)
            return "exit"

    class SilentWriter:
        async def write_delta(self, delta: str) -> None:
            del delta

        async def finish_turn(self) -> None:
            return None

        async def write_line(self, content: str) -> None:
            del content

    await runtime.run(input_reader=ExitInput(), writer=SilentWriter())
    await runtime.run(input_reader=ExitInput(), writer=SilentWriter())

    assert clock.sleeps == [5 * 60, 5 * 60]


@pytest.mark.asyncio
async def test_custom_schedule_keeps_the_runtime_startup_local_timezone(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No update needed."),))
    clock = ControlledClock(datetime(2026, 7, 11, 8, 50, tzinfo=LOCAL))
    scheduler = MemoryTaskScheduler(
        manager=_manager(home=home, provider=provider, summaries=summaries),
        schedule="0 9 * * *",
        clock=clock,
    )

    scheduler.start()
    await _wait_until(lambda: clock.sleeps == [10 * 60])
    clock.advance(10 * 60)
    clock.report_in_timezone(UTC)
    await _wait_until(lambda: len(clock.sleeps) == 2)
    await scheduler.close()

    assert clock.sleeps == [10 * 60, 24 * 60 * 60]


@pytest.mark.asyncio
async def test_local_schedule_wait_uses_elapsed_time_across_daylight_saving_change(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = ControlledClock(datetime(2026, 3, 8, 1, 50, tzinfo=SpringForwardTimezone()))
    scheduler = MemoryTaskScheduler(
        manager=_manager(
            home=home,
            provider=ScriptedFakeProvider(),
            summaries=JsonlSummaryStore(home),
        ),
        schedule="30 3 * * *",
        clock=clock,
    )

    scheduler.start()
    await _wait_until(lambda: len(clock.sleeps) == 1)
    await scheduler.close()

    assert clock.sleeps == [40 * 60]


def test_production_scheduler_clock_uses_the_rule_bearing_system_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zone = SpringForwardTimezone()
    monkeypatch.setattr(memory_scheduler_module, "get_localzone", lambda: zone)
    instants = iter(
        (
            datetime(2026, 3, 8, 6, 50, tzinfo=UTC),
            datetime(2026, 3, 8, 7, 30, tzinfo=UTC),
        )
    )
    clock = AsyncioMemorySchedulerClock(now=lambda: next(instants))

    before_transition = clock.now()
    after_transition = clock.now()

    assert before_transition.tzinfo is zone
    assert (before_transition.hour, before_transition.minute, before_transition.utcoffset()) == (
        1,
        50,
        timedelta(hours=-5),
    )
    assert after_transition.tzinfo is zone
    assert (after_transition.hour, after_transition.minute, after_transition.utcoffset()) == (
        3,
        30,
        timedelta(hours=-4),
    )


@pytest.mark.asyncio
async def test_periodic_memory_edit_is_visible_on_disk_but_chat_uses_the_startup_snapshot(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    startup_memory = (agent_home / "memory" / "memory.md").read_text(encoding="utf-8")
    updated_memory = startup_memory.replace(
        "## User Preference\n",
        "## User Preference\n\nPrefers concise status reports.\n",
    )
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    summaries = JsonlSummaryStore(home)
    await summaries.append("The user prefers concise status reports.", NOW)
    clock = ControlledClock(NOW)
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="edit-memory",
                        name="edit_file",
                        arguments={
                            "path": str(agent_home / "memory" / "memory.md"),
                            "old_text": startup_memory,
                            "new_text": updated_memory,
                        },
                    ),
                ),
            ),
            _response("Memory updated."),
        ),
        streams=(StreamScript(events=(ModelCompleted(response=_response("Old snapshot.")),)),),
    )
    configuration = ConfigLoader(home).load()
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _configuration: provider,
        now=clock.now,
        new_uuid=uuid4,
        memory_scheduler_clock=clock,
    )

    runtime.start()
    await _wait_until(lambda: clock.sleeps == [5 * 60])
    clock.advance(5 * 60)
    await _wait_until(lambda: len(provider.complete_requests) == 2)
    memory_view = await runtime.management_dispatcher.dispatch("/memory")
    _ = [event async for event in runtime.conversation.submit("What do you remember?")]
    await runtime.close()

    first_chat = provider.stream_requests[0]
    assert isinstance(first_chat, ModelRequest)
    assert memory_view.output == updated_memory
    assert startup_memory in first_chat.system_prompt
    assert updated_memory not in first_chat.system_prompt

    restarted_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(ModelCompleted(response=_response("New snapshot.")),)),)
    )
    restarted = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _configuration: restarted_provider,
        now=clock.now,
        new_uuid=uuid4,
        memory_scheduler_clock=clock,
    )
    _ = [event async for event in restarted.conversation.submit("What do you remember now?")]
    await restarted.close()

    restarted_chat = restarted_provider.stream_requests[0]
    assert isinstance(restarted_chat, ModelRequest)
    assert updated_memory in restarted_chat.system_prompt


@pytest.mark.asyncio
async def test_periodic_failure_is_isolated_and_all_periodic_results_stay_silent(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    summaries = JsonlSummaryStore(home)
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo("model_failed", "Memory model failed.")),
            _response("No update needed."),
        )
    )
    clock = ControlledClock(NOW)
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=clock.now,
        new_uuid=uuid4,
        memory_scheduler_clock=clock,
    )
    exit_repl = asyncio.Event()

    class WaitingInput:
        async def read(self) -> str:
            await exit_repl.wait()
            return "exit"

    class RecordingWriter:
        def __init__(self) -> None:
            self.operations: list[tuple[str, str]] = []

        async def write_delta(self, delta: str) -> None:
            self.operations.append(("delta", delta))

        async def finish_turn(self) -> None:
            self.operations.append(("finish", ""))

        async def write_line(self, content: str) -> None:
            self.operations.append(("line", content))

    writer = RecordingWriter()
    repl = asyncio.create_task(runtime.run(input_reader=WaitingInput(), writer=writer))

    await _wait_until(lambda: clock.sleeps == [5 * 60])
    clock.advance(5 * 60)
    await _wait_until(lambda: len(provider.complete_requests) == 1)
    assert await FileMemoryStore(home).read_summary_cursor() == 0

    await _wait_until(lambda: clock.sleeps == [5 * 60, 60 * 60])
    clock.advance(60 * 60)
    await _wait_until(lambda: len(provider.complete_requests) == 2)
    exit_repl.set()
    await repl

    assert await FileMemoryStore(home).read_summary_cursor() == 1
    assert writer.operations == []


@pytest.mark.asyncio
async def test_scheduler_close_cancels_and_awaits_an_active_memory_task(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("A pending summary.", NOW)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class CancellationAwareProvider(ScriptedFakeProvider):
        async def complete(self, request: object) -> ModelResponse:
            self.complete_requests.append(request)
            if len(self.complete_requests) == 1:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return _response("No update needed.")

    provider = CancellationAwareProvider()
    manager = _manager(home=home, provider=provider, summaries=summaries)
    clock = ControlledClock(NOW)
    scheduler = MemoryTaskScheduler(
        manager=manager,
        schedule="0 * * * *",
        clock=clock,
    )
    existing_tasks = asyncio.all_tasks()

    scheduler.start()
    await _wait_until(lambda: clock.sleeps == [50 * 60])
    clock.advance(50 * 60)
    await started.wait()
    await scheduler.close()
    await asyncio.sleep(0)

    assert cancelled.is_set()
    assert asyncio.all_tasks() == existing_tasks
    with pytest.raises(RuntimeError, match="scheduler is closed"):
        scheduler.start()

    recovered = await manager.run_manual()
    assert recovered.cursor == 1
