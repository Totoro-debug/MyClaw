import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterable, Mapping
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
from myclaw.schedule.records import ScheduledWork
from myclaw.session.conversation import ChatModelSettings
from myclaw.session.session import Session
from myclaw.tools.models import ModelToolCall
from myclaw.tools.shell.shell_policy import ShellRequest
from myclaw.tools.shell.shell_process import SubprocessShellBoundary
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.tools.web.web_fetch import PublicWebFetchBoundary
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript, persist_scheduled_work
from tests.fixtures.diagnostic_capture import capture_diagnostics

NOW = datetime(2026, 7, 13, 0, 30, tzinfo=timezone(timedelta(hours=8)))


def _session_id() -> str:
    return f"20260713-003000-000000_{uuid4()}"


class BlockingSchedulerClock:
    def __init__(self) -> None:
        self.sleep_started = asyncio.Event()
        self.sleep_stopped = asyncio.Event()

    def now(self) -> datetime:
        return NOW

    async def sleep(self, seconds: float) -> None:
        del seconds
        self.sleep_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.sleep_stopped.set()


class AdvancingSchedulerClock:
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
        previous_sleep_count = len(self.sleeps)
        self._current += timedelta(seconds=seconds)
        self._release.set()
        for _ in range(100):
            await asyncio.sleep(0)
            if len(self.sleeps) > previous_sleep_count:
                return
        raise AssertionError("Scheduled Work scheduler did not resume sleeping")


class FailingSchedulerClock:
    def now(self) -> datetime:
        raise RuntimeError("scheduled scheduler failed to start")

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


class TerminatingProcess:
    def __init__(self) -> None:
        self.communicate_started = asyncio.Event()
        self._stopped = asyncio.Event()
        self.terminated = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes | None]:
        self.communicate_started.set()
        await self._stopped.wait()
        return b"stopped", None

    async def terminate(self) -> None:
        self.terminated = True
        self._stopped.set()

    async def wait(self) -> None:
        self.waited = True


class OneProcessSpawner:
    def __init__(self, process: TerminatingProcess) -> None:
        self._process = process

    async def spawn(self, command: tuple[str, ...], *, cwd: Path) -> TerminatingProcess:
        del command, cwd
        return self._process


class StaticPublicResolver:
    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        del hostname, port
        return ("93.184.216.34",)


class BlockingHTTPResponse:
    status_code = 200
    headers: Mapping[str, str] = {"content-type": "text/plain; charset=utf-8"}
    peer_ip: str | None = "93.184.216.34"

    def __init__(self) -> None:
        self.body_started = asyncio.Event()
        self.closed = asyncio.Event()

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        self.body_started.set()
        await asyncio.Event().wait()
        if False:
            yield b""

    async def close(self) -> None:
        self.closed.set()


class OneResponseHTTPClient:
    def __init__(self, response: BlockingHTTPResponse) -> None:
        self._response = response

    async def get(
        self,
        url: str,
        *,
        allowed_ips: frozenset[str],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> BlockingHTTPResponse:
        del url, allowed_ips, connect_timeout_seconds, total_timeout_seconds
        return self._response


@pytest.mark.asyncio
async def test_partial_scheduler_start_failure_closes_the_started_memory_loop(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.web]\nenabled = false",
            "[tools.web]\nenabled = false",
        ).replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    memory_clock = BlockingSchedulerClock()
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
        memory_scheduler_clock=memory_clock,
        scheduled_work_scheduler_clock=FailingSchedulerClock(),
    )
    baseline = asyncio.all_tasks()

    try:
        with pytest.raises(RuntimeError, match="scheduled scheduler failed to start"):
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
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
        memory_scheduler_clock=BlockingSchedulerClock(),
        scheduled_work_scheduler_clock=FailingSchedulerClock(),
    )
    baseline = asyncio.all_tasks()
    log_capture = capture_diagnostics()
    state = WorkspaceState(Workspace.from_path(workspace))
    ambient_session_id = _session_id()

    try:
        with session_log(state, ambient_session_id):
            with pytest.raises(RuntimeError, match="scheduled scheduler failed to start"):
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
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
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
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
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
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
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
async def test_runtime_close_still_reaps_shell_when_provider_close_fails(
    agent_home: Path,
    workspace: Path,
) -> None:
    class FailingCloseProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            self.closed = True
            raise RuntimeError("PRIVATE_PROVIDER_CLOSE_BODY_52")

    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = FailingCloseProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Ready."),
                            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    process = TerminatingProcess()
    shell = SubprocessShellBoundary(spawner=OneProcessSpawner(process))
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        shell=shell,
    )
    _ = [event async for event in runtime.conversation.submit("Construct the provider.")]
    shell_execution = asyncio.create_task(
        shell.execute(ShellRequest(command="git status", cwd=workspace, timeout=60))
    )
    await process.communicate_started.wait()
    log_capture = capture_diagnostics()
    state = WorkspaceState(Workspace.from_path(workspace))
    ambient_session_id = _session_id()

    try:
        with session_log(state, ambient_session_id):
            with pytest.raises(RuntimeError, match="PRIVATE_PROVIDER_CLOSE_BODY_52"):
                await runtime.close()

        assert shell_execution.done()
        assert process.terminated
        assert process.waited
    finally:
        await shell.close()
        await asyncio.gather(shell_execution, return_exceptions=True)
        log_capture.close()

    assert not (state.logs_directory / f"{ambient_session_id}.log").exists()
    content = log_capture.text
    assert content.count(" ERROR ") == 1
    marker = "Runtime shutdown failed type=RuntimeError"
    assert content.count(marker) == 1
    assert "RuntimeError: PRIVATE_PROVIDER_CLOSE_BODY_52" in content


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
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
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
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
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
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
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
async def test_runtime_close_waits_for_an_active_web_fetch_response_to_close(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.web]\nenabled = false",
            "[tools.web]\nenabled = true",
        ).replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    response = BlockingHTTPResponse()
    fetch = PublicWebFetchBoundary(
        resolver=StaticPublicResolver(),
        http_client=OneResponseHTTPClient(response),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="",
                                tool_calls=(
                                    ModelToolCall(
                                        id="call-web-fetch",
                                        name="web_fetch",
                                        arguments='{"url":"https://example.com/resource"}',
                                    ),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
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
        web_fetch=fetch,
    )
    baseline = asyncio.all_tasks()
    running = asyncio.create_task(
        runtime.run(
            input_reader=ScriptedInput(("Fetch the resource.",)),
            writer=SilentWriter(),
        )
    )
    await asyncio.wait_for(response.body_started.wait(), timeout=1)

    await runtime.close()
    await asyncio.gather(running, return_exceptions=True)
    await asyncio.sleep(0)

    assert response.closed.is_set()
    assert running.done()
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
        tool_gateway=ToolGateway(),
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
        tool_gateway=ToolGateway(),
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
        tool_gateway=ToolGateway(),
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
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
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
        scheduled_work_scheduler_clock=scheduled_clock,
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


@pytest.mark.asyncio
async def test_runtime_interrupt_keeps_background_work_alive_and_exit_settles_it_before_provider(
    agent_home: Path,
    workspace: Path,
) -> None:
    from types import FrameType

    from myclaw.terminal.interrupts import (
        ForegroundInterruptController,
        SignalDisposition,
    )

    order: list[str] = []

    class LifetimeProvider:
        def __init__(self) -> None:
            self.foreground_started = asyncio.Event()
            self.foreground_stopped = asyncio.Event()
            self.first_background_started = asyncio.Event()
            self.first_background_release = asyncio.Event()
            self.first_background_completed = asyncio.Event()
            self.first_background_cancelled = asyncio.Event()
            self.second_background_started = asyncio.Event()
            self.second_background_stopped = asyncio.Event()
            self.stream_requests: list[ModelRequest] = []
            self.complete_requests: list[ModelRequest] = []
            self._background_calls = 0
            self.closed = False

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.stream_requests.append(request)
            if request.system_prompt == session_title_prompt():
                yield ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(content="Runtime lifetime"),
                        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                        finish_reason="stop",
                    )
                )
                return
            self.foreground_started.set()
            try:
                yield TextDelta(delta="Foreground active.")
                await asyncio.Event().wait()
            finally:
                order.append("foreground-finally")
                self.foreground_stopped.set()

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.complete_requests.append(request)
            if request.system_prompt == session_title_prompt():
                return ModelResponse(
                    message=AssistantModelMessage(content="Runtime lifetime"),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="stop",
                )
            assert request.route == "schedule"
            self._background_calls += 1
            if self._background_calls == 1:
                self.first_background_started.set()
                try:
                    await self.first_background_release.wait()
                except asyncio.CancelledError:
                    self.first_background_cancelled.set()
                    raise
                order.append("first-background-completed")
                self.first_background_completed.set()
                return ModelResponse(
                    message=AssistantModelMessage(content="First background completed."),
                    usage=ModelUsage(input_tokens=2, output_tokens=2, total_tokens=4),
                    finish_reason="stop",
                )
            assert self._background_calls == 2
            self.second_background_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                order.append("second-background-finally")
                self.second_background_stopped.set()
            raise AssertionError("second background work must be cancelled on exit")

        async def close(self) -> None:
            order.append("provider-close")
            self.closed = True

    class ExitAfterInterruptInput:
        def __init__(self) -> None:
            self._calls = 0
            self.waiting_for_exit = asyncio.Event()
            self.release_exit = asyncio.Event()

        async def read(self) -> str:
            self._calls += 1
            if self._calls == 1:
                return "Keep background work alive."
            self.waiting_for_exit.set()
            await self.release_exit.wait()
            return "exit"

    def previous_handler(signum: int, frame: FrameType | None) -> None:
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
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    provider = LifetimeProvider()
    memory_clock = BlockingSchedulerClock()
    scheduled_clock = AdvancingSchedulerClock(NOW)
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
    first_task = ScheduledWork(
        id="11111111-1111-4111-8111-111111111111",
        title="First lifetime task",
        cron="31 0 * * *",
        prompt="Run the first background task.",
        created_at=NOW,
        enabled=True,
        session_id=("20260713-003000-000000_33333333-3333-4333-8333-333333333333"),
    )
    persist_scheduled_work(state.path, (first_task,))
    input_reader = ExitAfterInterruptInput()
    signals = SignalSetter()
    interrupts = ForegroundInterruptController(
        loop=asyncio.get_running_loop(),
        cancel_foreground=runtime.conversation.cancel_active_turn,
        set_signal=signals,
    )
    interrupts.install()
    baseline = asyncio.all_tasks()
    running = asyncio.create_task(runtime.run(input_reader=input_reader, writer=SilentWriter()))
    try:
        await asyncio.wait_for(provider.foreground_started.wait(), timeout=1)
        await scheduled_clock.advance(60)
        await asyncio.wait_for(provider.first_background_started.wait(), timeout=1)

        handler = signals.current
        assert callable(handler)
        handler(2, None)
        await asyncio.wait_for(input_reader.waiting_for_exit.wait(), timeout=1)

        assert provider.foreground_stopped.is_set()
        assert not provider.first_background_cancelled.is_set()
        assert not provider.first_background_completed.is_set()
        provider.first_background_release.set()
        await asyncio.wait_for(provider.first_background_completed.wait(), timeout=1)

        second_task = ScheduledWork(
            id="22222222-2222-4222-8222-222222222222",
            title="Second lifetime task",
            cron="32 0 * * *",
            prompt="Run the second background task.",
            created_at=scheduled_clock.now(),
            enabled=True,
            session_id=("20260713-003100-000000_44444444-4444-4444-8444-444444444444"),
        )
        persist_scheduled_work(state.path, (first_task, second_task))
        await scheduled_clock.advance(60)
        await asyncio.wait_for(provider.second_background_started.wait(), timeout=1)

        input_reader.release_exit.set()
        await asyncio.wait_for(running, timeout=2)
    finally:
        provider.first_background_release.set()
        input_reader.release_exit.set()
        if not running.done():
            await runtime.close()
        await asyncio.gather(running, return_exceptions=True)
        await interrupts.close()
        interrupts.restore()

    await asyncio.sleep(0)
    assert provider.second_background_stopped.is_set()
    assert provider.closed
    assert order.index("first-background-completed") < order.index("second-background-finally")
    assert order.index("second-background-finally") < order.index("provider-close")
    assert asyncio.all_tasks() == baseline
