import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

import myclaw.utils.scheduler as scheduler_module
from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import PreparedRuntime, prepare_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log
from myclaw.memory.conversation_summary import WorkspaceJsonlSummaryStore
from myclaw.memory.memory_scheduler import MemoryTaskScheduler
from myclaw.memory.memory_task import (
    MemoryManager,
    MemoryTaskModelRouter,
    MemoryTaskResult,
    WorkspaceFileMemoryStore,
)
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouter
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall
from myclaw.utils.scheduler import AsyncioSchedulerClock
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    FakeClock,
    ProviderCall,
    ScriptedFakeProvider,
    ScriptedFakeRouter,
    StreamScript,
)
from tests.fixtures.diagnostic_capture import capture_diagnostics, configured_process_logging
from tests.runtime_bus import collect_foreground_outbound

LOCAL = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 16, 10, tzinfo=LOCAL)
SESSION_ID = "20260711-161000-000000_550e8400-e29b-41d4-a716-446655440000"


def _state(home: AgentHome) -> WorkspaceState:
    workspace = home.path.parent / "workspace-state"
    workspace.mkdir(exist_ok=True)
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


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
    router: MemoryTaskModelRouter,
    summaries: WorkspaceJsonlSummaryStore,
) -> MemoryManager:
    return MemoryManager(
        router=router,
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(_state(home)),
        long_term_path=home.path / "memory" / "memory.md",
        batch_size=10,
    )


def _record_complete(
    provider: ScriptedFakeProvider,
    *,
    messages: Sequence[dict[str, object]],
    tools: Sequence[OpenAIToolSchema],
    model: str,
    max_output: int,
    temperature: float,
    reasoning_effort: ReasoningEffort | None,
    timeout: int,
    continuation: ModelContinuation | None = None,
) -> None:
    provider.complete_requests.append(
        ProviderCall(
            messages=list(messages),
            tools=tuple(tools),
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            continuation=continuation,
        )
    )


def _router(provider: ScriptedFakeProvider) -> ScriptedFakeRouter:
    return ScriptedFakeRouter(provider)


@pytest.mark.asyncio
async def test_periodic_memory_task_runs_when_the_manager_is_idle(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No update needed."),))
    manager = _manager(home=home, router=_router(provider), summaries=summaries)

    result = await manager.run_periodic()

    assert result == MemoryTaskResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )


@pytest.mark.asyncio
async def test_periodic_memory_task_does_not_borrow_a_foreground_session_log(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("PRIVATE PERIODIC SUMMARY", NOW)
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo(code="model_failed", message="PRIVATE PERIODIC OUTPUT")),
        )
    )
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    router = ModelRouter(
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
    )
    with configured_process_logging(), session_log(state, SESSION_ID):
        result = await _manager(
            home=home,
            router=router,
            summaries=summaries,
        ).run_periodic()

    assert result is not None
    assert result.error == ErrorInfo(
        code="model_failed",
        message="PRIVATE PERIODIC OUTPUT",
    )
    assert result.cursor == 1
    assert capsys.readouterr().err == "Memory Task failed code=model_failed\n"
    assert not (state.logs_directory / f"{SESSION_ID}.log").exists()
    await router.close()


@pytest.mark.asyncio
async def test_memory_scheduler_trigger_does_not_borrow_a_foreground_session_log(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("PRIVATE SCHEDULER SUMMARY", NOW)
    attempted = asyncio.Event()

    class UnexpectedFailureProvider(ScriptedFakeProvider):
        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str = "test-model",
            max_output: int = 1024,
            temperature: float = 0.2,
            reasoning_effort: ReasoningEffort | None = None,
            timeout: int = 30,
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            _record_complete(
                self,
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )
            attempted.set()
            try:
                raise OSError("technical storage cause")
            except OSError as cause:
                raise RuntimeError("technical trigger failure") from cause

    clock = ControlledClock(NOW)
    scheduler = MemoryTaskScheduler(
        manager=_manager(
            home=home,
            router=_router(UnexpectedFailureProvider()),
            summaries=summaries,
        ),
        schedule="0 * * * *",
        clock=clock,
    )
    with configured_process_logging(), session_log(state, SESSION_ID):
        scheduler.start()
        await _wait_until(lambda: clock.sleeps == [50 * 60])
        clock.advance(50 * 60)
        await attempted.wait()
        await asyncio.sleep(0)
        await scheduler.close()

    assert capsys.readouterr().err == "Memory Task trigger crashed\n"
    assert not (state.logs_directory / f"{SESSION_ID}.log").exists()


@pytest.mark.asyncio
async def test_hourly_memory_schedule_triggers_only_at_next_local_cron_boundary(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No update needed."),))
    clock = ControlledClock(NOW)
    scheduler = MemoryTaskScheduler(
        manager=_manager(home=home, router=_router(provider), summaries=summaries),
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

    assert await WorkspaceFileMemoryStore(_state(home)).read_summary_cursor() == 1


@pytest.mark.asyncio
async def test_periodic_trigger_is_skipped_while_the_previous_run_is_still_active(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("A pending summary.", NOW)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider(ScriptedFakeProvider):
        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str = "test-model",
            max_output: int = 1024,
            temperature: float = 0.2,
            reasoning_effort: ReasoningEffort | None = None,
            timeout: int = 30,
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            _record_complete(
                self,
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )
            started.set()
            await release.wait()
            return _response("No update needed.")

    provider = BlockingProvider()
    clock = ControlledClock(NOW)
    scheduler = MemoryTaskScheduler(
        manager=_manager(home=home, router=_router(provider), summaries=summaries),
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
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("A pending summary.", NOW)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider(ScriptedFakeProvider):
        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str = "test-model",
            max_output: int = 1024,
            temperature: float = 0.2,
            reasoning_effort: ReasoningEffort | None = None,
            timeout: int = 30,
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            _record_complete(
                self,
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )
            started.set()
            await release.wait()
            return _response("No update needed.")

    provider = BlockingProvider()
    manager = _manager(home=home, router=_router(provider), summaries=summaries)
    periodic = asyncio.create_task(manager.run_periodic())
    await started.wait()
    (_state(home).memory_directory / ".cursor").write_bytes(b"corrupt\n")

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
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No update needed."),))
    clock = ControlledClock(NOW)
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=clock.now,
        new_uuid=uuid4,
        memory_scheduler_clock=clock,
    )

    await runtime.start()
    await _wait_until(lambda: clock.sleeps == [5 * 60])
    clock.advance(5 * 60)
    await _wait_until(lambda: len(provider.complete_requests) == 1)
    await runtime.close()

    assert await WorkspaceFileMemoryStore(state).read_summary_cursor() == 1


@pytest.mark.asyncio
async def test_each_prepared_runtime_starts_a_fresh_memory_scheduler(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    clock = ControlledClock(NOW)

    def _make_runtime() -> PreparedRuntime:
        return prepare_runtime(
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

    await _make_runtime().run(input_reader=ExitInput(), writer=SilentWriter())
    await _make_runtime().run(input_reader=ExitInput(), writer=SilentWriter())

    assert clock.sleeps == [5 * 60, 5 * 60]


@pytest.mark.asyncio
async def test_custom_schedule_keeps_the_runtime_startup_local_timezone(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No update needed."),))
    clock = ControlledClock(datetime(2026, 7, 11, 8, 50, tzinfo=LOCAL))
    scheduler = MemoryTaskScheduler(
        manager=_manager(home=home, router=_router(provider), summaries=summaries),
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
            router=_router(ScriptedFakeProvider()),
            summaries=WorkspaceJsonlSummaryStore(_state(home)),
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
    monkeypatch.setattr(scheduler_module, "_system_timezone", lambda: zone)
    instants = iter(
        (
            datetime(2026, 3, 8, 6, 50, tzinfo=UTC),
            datetime(2026, 3, 8, 7, 30, tzinfo=UTC),
        )
    )
    clock = AsyncioSchedulerClock(now=lambda: next(instants))

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


def test_standard_library_system_timezone_adapter_resolves_each_instant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_zone = SpringForwardTimezone()

    def host_local(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=host_zone)
        return value.astimezone(host_zone)

    monkeypatch.setattr(scheduler_module, "_host_local", host_local)
    instants = iter(
        (
            datetime(2026, 3, 8, 6, 50, tzinfo=UTC),
            datetime(2026, 3, 8, 7, 30, tzinfo=UTC),
        )
    )
    clock = AsyncioSchedulerClock(now=lambda: next(instants))

    before_transition = clock.now()
    after_transition = clock.now()

    assert (before_transition.hour, before_transition.minute, before_transition.utcoffset()) == (
        1,
        50,
        timedelta(hours=-5),
    )
    assert before_transition.dst() == timedelta(0)
    assert (after_transition.hour, after_transition.minute, after_transition.utcoffset()) == (
        3,
        30,
        timedelta(hours=-4),
    )
    assert after_transition.dst() == timedelta(hours=1)


@pytest.mark.asyncio
async def test_periodic_memory_edit_refreshes_runtime_memory_for_a_later_chat(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    startup_memory = state.long_term_memory_path.read_text(encoding="utf-8")
    legacy_path = agent_home / "memory" / "memory.md"
    legacy_path.parent.mkdir(parents=True)
    legacy_memory = b"# Legacy Agent Home Memory\n"
    legacy_path.write_bytes(legacy_memory)
    updated_memory = startup_memory.replace(
        "## User Preference\n",
        "## User Preference\n\nPrefers concise status reports.\n",
    )
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    summaries = WorkspaceJsonlSummaryStore(state)
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
                        arguments=json.dumps(
                            {
                                "path": str(state.long_term_memory_path),
                                "old_text": startup_memory,
                                "new_text": updated_memory,
                            }
                        ),
                    ),
                ),
            ),
            _response("Memory updated."),
        ),
        streams=(StreamScript(events=(ModelCompleted(response=_response("Old snapshot.")),)),),
    )
    configuration = ConfigLoader(home).load()
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _configuration: provider,
        now=clock.now,
        new_uuid=uuid4,
        memory_scheduler_clock=clock,
    )

    await runtime.start()
    await _wait_until(lambda: clock.sleeps == [5 * 60])
    clock.advance(5 * 60)
    await _wait_until(lambda: len(provider.complete_requests) == 2)
    memory_view = await runtime.management_dispatcher.dispatch("/memory")
    _ = await collect_foreground_outbound(runtime, "What do you remember?")
    await runtime.close()

    first_chat = provider.stream_requests[0]
    assert memory_view.output == updated_memory
    assert startup_memory not in cast(str, first_chat.messages[0]["content"])
    assert updated_memory in cast(str, first_chat.messages[0]["content"])
    assert legacy_path.read_bytes() == legacy_memory

    restarted_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(ModelCompleted(response=_response("New snapshot.")),)),)
    )
    restarted = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _configuration: restarted_provider,
        now=clock.now,
        new_uuid=uuid4,
        memory_scheduler_clock=clock,
    )
    await restarted.start()
    _ = await collect_foreground_outbound(restarted, "What do you remember now?")
    await restarted.close()

    restarted_chat = restarted_provider.stream_requests[0]
    assert updated_memory in cast(str, restarted_chat.messages[0]["content"])


@pytest.mark.asyncio
async def test_memory_refresh_does_not_change_an_active_chat_snapshot(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    startup_memory = state.long_term_memory_path.read_text(encoding="utf-8")
    updated_memory = startup_memory.replace(
        "## User Preference\n",
        "## User Preference\n\nPrefers concise status reports.\n",
    )
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("The user prefers concise status reports.", NOW)
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class SnapshotProvider(ScriptedFakeProvider):
        async def stream(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str = "test-model",
            max_output: int = 1024,
            temperature: float = 0.2,
            reasoning_effort: ReasoningEffort | None = None,
            timeout: int = 30,
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            if messages and messages[0] == {
                "role": "system",
                "content": session_title_prompt(),
            }:
                raise ModelCallError(
                    ErrorInfo(code="model_failed", message="No title response was scripted.")
                )
            self.stream_requests.append(
                ProviderCall(
                    messages=list(messages),
                    tools=tuple(tools),
                    model=model,
                    max_output=max_output,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                    continuation=continuation,
                )
            )
            request_count = len(self.stream_requests)
            if request_count == 1:
                first_started.set()
                await release_first.wait()
            yield ModelCompleted(response=_response(f"Answer {request_count}."))

    provider = SnapshotProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(state.long_term_memory_path),
                                "old_text": startup_memory,
                                "new_text": updated_memory,
                            }
                        ),
                    ),
                ),
            ),
            _response("Memory updated."),
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

    await runtime.start()

    async def collect_first_turn() -> None:
        _ = await collect_foreground_outbound(runtime, "First input.")

    first_turn = asyncio.create_task(collect_first_turn())
    await first_started.wait()
    dream = await runtime.management_dispatcher.dispatch("/dream")
    assert dream.output is not None
    assert updated_memory == state.long_term_memory_path.read_text(encoding="utf-8")
    release_first.set()
    await first_turn
    _ = await collect_foreground_outbound(runtime, "Second input.")
    await runtime.close()

    first_chat = provider.stream_requests[0]
    second_chat = provider.stream_requests[1]
    assert "Prefers concise status reports." not in cast(str, first_chat.messages[0]["content"])
    assert "Prefers concise status reports." in cast(str, second_chat.messages[0]["content"])


@pytest.mark.asyncio
async def test_periodic_failure_is_isolated_and_all_periodic_results_stay_silent(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("A pending summary.", NOW)
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo("model_failed", "Memory model failed.")),
            _response("No update needed."),
        )
    )
    clock = ControlledClock(NOW)
    runtime = prepare_runtime(
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
    assert await WorkspaceFileMemoryStore(state).read_summary_cursor() == 1

    await _wait_until(lambda: clock.sleeps == [5 * 60, 60 * 60])
    clock.advance(60 * 60)
    await asyncio.sleep(0)
    exit_repl.set()
    await repl

    assert await WorkspaceFileMemoryStore(state).read_summary_cursor() == 1
    assert writer.operations == []


@pytest.mark.asyncio
async def test_scheduler_close_cancels_and_awaits_an_active_memory_task(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("A pending summary.", NOW)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class CancellationAwareProvider(ScriptedFakeProvider):
        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str = "test-model",
            max_output: int = 1024,
            temperature: float = 0.2,
            reasoning_effort: ReasoningEffort | None = None,
            timeout: int = 30,
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            _record_complete(
                self,
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )
            if len(self.complete_requests) == 1:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return _response("No update needed.")

    provider = CancellationAwareProvider()
    manager = _manager(home=home, router=_router(provider), summaries=summaries)
    clock = ControlledClock(NOW)
    scheduler = MemoryTaskScheduler(
        manager=manager,
        schedule="0 * * * *",
        clock=clock,
    )
    capture = capture_diagnostics()
    existing_tasks = asyncio.all_tasks()

    scheduler.start()
    await _wait_until(lambda: clock.sleeps == [50 * 60])
    clock.advance(50 * 60)
    await started.wait()
    await scheduler.close()
    await asyncio.sleep(0)
    capture.close()

    assert cancelled.is_set()
    assert await WorkspaceFileMemoryStore(_state(home)).read_summary_cursor() == 1
    assert asyncio.all_tasks() == existing_tasks
    assert not (agent_home / "logs").exists()
    with pytest.raises(RuntimeError, match="scheduler is closed"):
        scheduler.start()
    recovered = await manager.run_manual()
    assert recovered == MemoryTaskResult(
        status="No pending summaries",
        processed_count=0,
        memory_updated=False,
        cursor=1,
    )
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_memory_scheduler_reports_cleanup_failure_without_a_session_log(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("PRIVATE CLEANUP SUMMARY", NOW)
    started = asyncio.Event()

    class CleanupFailingProvider(ScriptedFakeProvider):
        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str = "test-model",
            max_output: int = 1024,
            temperature: float = 0.2,
            reasoning_effort: ReasoningEffort | None = None,
            timeout: int = 30,
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            _record_complete(
                self,
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise RuntimeError("technical Memory Task cleanup failure") from cancellation
            raise AssertionError("cleanup test wait returned unexpectedly")

    provider = CleanupFailingProvider()
    manager = _manager(
        home=home,
        router=_router(provider),
        summaries=summaries,
    )
    clock = ControlledClock(NOW)
    scheduler = MemoryTaskScheduler(
        manager=manager,
        schedule="0 * * * *",
        clock=clock,
    )
    with configured_process_logging():
        scheduler.start()
        await _wait_until(lambda: clock.sleeps == [50 * 60])
        clock.advance(50 * 60)
        await started.wait()
        await scheduler.close()

    assert capsys.readouterr().err == "Memory Task scheduler cleanup failed\n"
    assert not state.logs_directory.exists()
    assert await WorkspaceFileMemoryStore(state).read_summary_cursor() == 1
    recovered = await manager.run_manual()
    assert recovered == MemoryTaskResult(
        status="No pending summaries",
        processed_count=0,
        memory_updated=False,
        cursor=1,
    )
    assert len(provider.complete_requests) == 1
