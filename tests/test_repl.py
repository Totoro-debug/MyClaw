import asyncio
import subprocess
import sys
from collections import deque
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from uuid import UUID

import pytest
from rich.console import Console

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import ManagementViewService
from myclaw.memory.memory_task import WorkspaceFileMemoryStore
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session_store import JsonlSessionStore
from myclaw.terminal.repl import ConsoleProgressiveWriter, ConsoleReplInput, run_repl
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
USER_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
REQUEST_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
ASSISTANT_UUID = UUID("a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34")
SECOND_SESSION_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")


class ScriptedReplInput:
    def __init__(self, values: Iterable[str | None]) -> None:
        self._values = deque(values)

    async def read(self) -> str | None:
        return self._values.popleft()


class RecordingProgressiveWriter:
    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []

    async def write_delta(self, delta: str) -> None:
        self.operations.append(("delta", delta))

    async def finish_turn(self) -> None:
        self.operations.append(("finish", ""))

    async def write_line(self, content: str) -> None:
        self.operations.append(("line", content))


class CancelOnFirstDeltaWriter(RecordingProgressiveWriter):
    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    async def write_delta(self, delta: str) -> None:
        await super().write_delta(delta)
        if not self._cancelled:
            self._cancelled = True
            raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_console_progressive_writer_writes_chunks_then_one_complete_line() -> None:
    output = StringIO()
    writer = ConsoleProgressiveWriter(Console(file=output, force_terminal=False, color_system=None))

    await writer.write_delta("Hello ")
    await writer.write_delta("world")
    await writer.finish_turn()
    await writer.write_line("Management output\n")

    assert output.getvalue() == "Hello world\nManagement output\n"


@pytest.mark.asyncio
async def test_console_repl_input_returns_eof_without_prompting_when_noninteractive() -> None:
    output = StringIO()
    input_reader = ConsoleReplInput(Console(file=output, force_terminal=False, color_system=None))

    assert await input_reader.read() is None
    assert output.getvalue() == ""


@pytest.mark.asyncio
async def test_console_repl_input_cancellation_stops_async_prompt_without_a_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TerminalInput:
        def isatty(self) -> bool:
            return True

    class BlockingPromptSession:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.handle_sigint: bool | None = None

        async def prompt_async(self, message: str, *, handle_sigint: bool) -> str:
            assert message == "You: "
            self.handle_sigint = handle_sigint
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    monkeypatch.setattr("myclaw.terminal.repl.sys.stdin", TerminalInput())
    prompt = BlockingPromptSession()
    input_reader = ConsoleReplInput(
        Console(force_terminal=True),
        prompt_session=prompt,
    )
    reading = asyncio.create_task(input_reader.read())
    await prompt.started.wait()

    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading

    assert prompt.handle_sigint is False


def test_cancelled_console_prompt_does_not_delay_process_exit() -> None:
    script = """
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
import myclaw.terminal.repl as repl_module
from myclaw.terminal.repl import ConsoleReplInput

class TerminalInput:
    def isatty(self):
        return True

async def main():
    repl_module.sys.stdin = TerminalInput()
    with create_pipe_input() as pipe:
        reader = ConsoleReplInput(
            Console(force_terminal=True),
            prompt_session=PromptSession(input=pipe, output=DummyOutput()),
        )
        reading = asyncio.create_task(reader.read())
        await asyncio.sleep(0.05)
        reading.cancel()
        try:
            await reading
        except asyncio.CancelledError:
            print("prompt-cancelled", flush=True)

asyncio.run(main())
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert completed.returncode == 0
    assert "prompt-cancelled" in completed.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize("inputs", [(None,), ("   ", "\t", None)])
async def test_repl_without_nonblank_user_input_leaves_prepared_session_in_memory(
    inputs: tuple[str | None, ...],
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = ScriptedFakeProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(()).__next__,
    )
    writer = RecordingProgressiveWriter()

    await run_repl(
        conversation=conversation,
        input_reader=ScriptedReplInput(inputs),
        writer=writer,
    )

    assert not store.path_for(session.id).parent.exists()
    assert provider.stream_requests == []
    assert provider.complete_requests == []
    assert writer.operations == []


@pytest.mark.asyncio
async def test_repl_exit_and_quit_ignore_case_and_whitespace_without_materializing_messages(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()

    for command, session_uuid in (
        ("  ExIt  ", SESSION_UUID),
        ("\tQuIt\n", SECOND_SESSION_UUID),
    ):
        clock = FakeClock(NOW)
        store = JsonlSessionStore(
            workspace_state=WorkspaceState(Workspace.from_path(workspace)),
            now=clock.now,
            new_uuid=iter((session_uuid,)).__next__,
        )
        session = store.prepare()
        provider = ScriptedFakeProvider()
        conversation = StreamingConversationPort(
            provider=provider,
            sessions=store,
            session_id=session.id,
            settings=ChatModelSettings(
                model="test-model",
                max_output=1024,
                temperature=0.2,
                reasoning_effort=None,
                timeout_seconds=30,
            ),
            now=clock.now,
            new_uuid=iter(()).__next__,
        )
        writer = RecordingProgressiveWriter()

        await run_repl(
            conversation=conversation,
            input_reader=ScriptedReplInput((command, "/after-exit", None)),
            writer=writer,
        )

        assert provider.stream_requests == []
        assert provider.complete_requests == []
        assert not store.path_for(session.id).exists()
        assert writer.operations == []


@pytest.mark.asyncio
async def test_repl_writes_each_text_delta_progressively_then_finishes_once(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    response = ModelResponse(
        message=AssistantModelMessage(content="I will inspect the files."),
        usage=ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        streams=[
            StreamScript(
                events=(
                    TextDelta(delta="I will "),
                    TextDelta(delta="inspect the files."),
                    ModelCompleted(response=response),
                )
            )
        ]
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )
    writer = RecordingProgressiveWriter()

    await run_repl(
        conversation=conversation,
        input_reader=ScriptedReplInput(("Help me inspect this project.", None)),
        writer=writer,
    )

    assert writer.operations == [
        ("delta", "I will "),
        ("delta", "inspect the files."),
        ("finish", ""),
    ]


@pytest.mark.asyncio
async def test_ctrl_c_during_stream_persists_partial_and_repl_runs_the_next_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    second_response = ModelResponse(
        message=AssistantModelMessage(content="Second turn completed."),
        usage=ModelUsage(input_tokens=14, output_tokens=4, total_tokens=18),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(TextDelta(delta="Partial first turn"),)),
            StreamScript(
                events=(
                    TextDelta(delta="Second turn completed."),
                    ModelCompleted(response=second_response),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
                UUID("16fd2706-8baf-433b-82eb-8c7fada847da"),
                UUID("886313e1-3b8a-4a2d-9f7f-77611a4b6f4e"),
                UUID("b3f37212-6f3a-4a1b-8d2e-78ab3f9c4567"),
            )
        ).__next__,
    )
    writer = CancelOnFirstDeltaWriter()

    await run_repl(
        conversation=conversation,
        input_reader=ScriptedReplInput(("First turn.", "Second turn.", None)),
        writer=writer,
    )

    assert writer.operations == [
        ("delta", "Partial first turn"),
        ("finish", ""),
        ("delta", "Second turn completed."),
        ("finish", ""),
    ]
    reloaded = await store.load(session.id)
    assert [
        (message.role, getattr(message, "status", None), message.content)
        for message in reloaded.messages
    ] == [
        ("user", None, "First turn."),
        ("assistant", "interrupted", "Partial first turn"),
        ("user", None, "Second turn."),
        ("assistant", "completed", "Second turn completed."),
    ]
    assert len(provider.stream_requests) == 2


@pytest.mark.asyncio
async def test_task_cancellation_during_foreground_is_cleared_before_next_input(
    agent_home: Path,
    workspace: Path,
) -> None:
    class BlockingProvider:
        def __init__(self) -> None:
            self.waiting = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield TextDelta(delta="Partial foreground")
            self.waiting.set()
            await asyncio.Event().wait()

        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise AssertionError(f"Unexpected completion: {request!r}")

        async def close(self) -> None:
            return None

    class FirstThenExitInput:
        def __init__(self) -> None:
            self._calls = 0
            self.waiting_for_exit = asyncio.Event()
            self.release_exit = asyncio.Event()

        async def read(self) -> str:
            self._calls += 1
            if self._calls == 1:
                return "Start foreground."
            self.waiting_for_exit.set()
            await self.release_exit.wait()
            return "exit"

    home = AgentHome(agent_home)
    home.initialize()
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = BlockingProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )
    input_reader = FirstThenExitInput()
    running = asyncio.create_task(
        run_repl(
            conversation=conversation,
            input_reader=input_reader,
            writer=RecordingProgressiveWriter(),
        )
    )
    await provider.waiting.wait()

    running.cancel()
    await input_reader.waiting_for_exit.wait()

    try:
        assert not running.done()
        assert running.cancelling() == 0
    finally:
        input_reader.release_exit.set()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_repl_dispatches_handled_management_output_and_converses_on_unknown_slash(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    response = ModelResponse(
        message=AssistantModelMessage(content="Ordinary slash input received."),
        usage=ModelUsage(input_tokens=5, output_tokens=4, total_tokens=9),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        streams=[
            StreamScript(
                events=(
                    TextDelta(delta="Ordinary slash input received."),
                    ModelCompleted(response=response),
                )
            )
        ]
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter((TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_store=WorkspaceFileMemoryStore(state))
    )
    writer = RecordingProgressiveWriter()

    await run_repl(
        conversation=conversation,
        input_reader=ScriptedReplInput(("/memory", "/unknown", None)),
        writer=writer,
        management_dispatcher=dispatcher,
    )

    assert writer.operations == [
        (
            "line",
            "# Long-term Memory\n\n## User Info\n\n## User Preference\n\n"
            "## Project Fact\n\n## Lesson\n",
        ),
        ("delta", "Ordinary slash input received."),
        ("finish", ""),
    ]
    assert len(provider.stream_requests) == 1
    request = provider.stream_requests[0]
    assert isinstance(request, ModelRequest)
    assert [message.to_dict() for message in request.messages] == [
        {
            "role": "user",
            "content": (
                "<runtime_context>\n"
                "current_time: 2026-07-11T15:30:12.123+08:00\n"
                f"session_id: {session.id}\n"
                "</runtime_context>\n\n"
                "<user_input>\n"
                "/unknown\n"
                "</user_input>"
            ),
        }
    ]
    reloaded = await store.load(session.id)
    assert [(message.role, message.content) for message in reloaded.messages] == [
        ("user", "/unknown"),
        ("assistant", "Ordinary slash input received."),
    ]
