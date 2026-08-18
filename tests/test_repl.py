import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.context import ContextBuilder
from myclaw.agent.events import (
    AgentEvent,
    ModelCallCompletedPayload,
    TextDeltaPayload,
    TurnCompletedPayload,
    TurnStartedPayload,
)
from myclaw.agent.run import AgentRunModelSettings, AgentRunModelTarget
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import ManagementViewService
from myclaw.memory.memory_task import WorkspaceFileMemoryStore
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.session.conversation import StreamingConversationPort
from myclaw.session.session import Session
from myclaw.terminal.repl import run_repl
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
SECOND_TURN_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
SECOND_SESSION_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")


def _session(
    workspace: Path,
    now: Callable[[], datetime],
    new_uuid: Callable[[], UUID],
) -> Session:
    return Session.create(
        WorkspaceState(Workspace.from_path(workspace)),
        now=now,
        new_uuid=new_uuid,
    )


def _direct_model(provider: ModelProvider) -> AgentRunModelTarget:
    return AgentRunModelTarget.for_provider(
        provider,
        AgentRunModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
    )


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


class ModelCompletionReplConversation:
    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        del text
        yield AgentEvent(
            type="turn_started",
            event_id=0,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=TurnStartedPayload(),
        )
        yield AgentEvent(
            type="text_delta",
            event_id=1,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=TextDeltaPayload(delta="Answer."),
        )
        yield AgentEvent(
            type="model_call_completed",
            event_id=2,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=ModelCallCompletedPayload(
                content="Answer.",
                continues_with_tools=False,
            ),
        )
        yield AgentEvent(
            type="turn_completed",
            event_id=3,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=TurnCompletedPayload(
                content="Answer.",
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            ),
        )

    async def cancel_active_turn(self) -> None:
        pass

    def respond_to_confirmation(self, confirmation_id: UUID, decision: str) -> None:
        del confirmation_id, decision


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
    session = _session(workspace, clock.now, iter((SESSION_UUID,)).__next__)
    provider = ScriptedFakeProvider()
    conversation = StreamingConversationPort(
        model=_direct_model(provider),
        session=session,
        now=clock.now,
        new_uuid=iter(()).__next__,
        context_builder=ContextBuilder(
            Workspace.from_path(workspace),
            "Asia/Shanghai",
            clock=clock.now,
        ),
    )
    writer = RecordingProgressiveWriter()

    await run_repl(
        conversation=conversation,
        input_reader=ScriptedReplInput(inputs),
        writer=writer,
    )

    assert not (workspace / ".myclaw" / "sessions").exists()
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
        session = _session(workspace, clock.now, iter((session_uuid,)).__next__)
        provider = ScriptedFakeProvider()
        conversation = StreamingConversationPort(
            model=_direct_model(provider),
            session=session,
            now=clock.now,
            new_uuid=iter(()).__next__,
            context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
        )
        writer = RecordingProgressiveWriter()

        await run_repl(
            conversation=conversation,
            input_reader=ScriptedReplInput((command, "/after-exit", None)),
            writer=writer,
        )

        assert provider.stream_requests == []
        assert provider.complete_requests == []
        assert not (workspace / ".myclaw" / "sessions").exists()
        assert writer.operations == []


@pytest.mark.asyncio
async def test_repl_writes_each_text_delta_progressively_then_finishes_once(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    session = _session(workspace, clock.now, iter((SESSION_UUID,)).__next__)
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
        model=_direct_model(provider),
        session=session,
        now=clock.now,
        new_uuid=iter((TURN_UUID,)).__next__,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
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
async def test_repl_ignores_nonterminal_model_completion_without_duplicate_output() -> None:
    writer = RecordingProgressiveWriter()

    await run_repl(
        conversation=ModelCompletionReplConversation(),
        input_reader=ScriptedReplInput(("Answer this.", None)),
        writer=writer,
    )

    assert writer.operations == [("delta", "Answer."), ("finish", "")]


@pytest.mark.asyncio
async def test_ctrl_c_during_stream_persists_partial_and_repl_runs_the_next_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    session = _session(workspace, clock.now, iter((SESSION_UUID,)).__next__)
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
        model=_direct_model(provider),
        session=session,
        now=clock.now,
        new_uuid=iter((TURN_UUID, SECOND_TURN_UUID)).__next__,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
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
    assert [
        (message["role"], message.get("status"), message["content"]) for message in session.messages
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

        async def stream(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[object],
            model: str,
            max_output: int,
            temperature: float,
            reasoning_effort: str | None,
            timeout: int,
        ) -> AsyncIterator[ModelStreamEvent]:
            del messages, tools, model, max_output, temperature, reasoning_effort, timeout
            yield TextDelta(delta="Partial foreground")
            self.waiting.set()
            await asyncio.Event().wait()

        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[object],
            model: str,
            max_output: int,
            temperature: float,
            reasoning_effort: str | None,
            timeout: int,
        ) -> ModelResponse:
            raise AssertionError(f"Unexpected completion: {messages!r}")

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
    session = _session(workspace, lambda: NOW, iter((SESSION_UUID,)).__next__)
    provider = BlockingProvider()
    conversation = StreamingConversationPort(
        model=_direct_model(provider),
        session=session,
        now=lambda: NOW,
        new_uuid=iter((TURN_UUID,)).__next__,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
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
    session = _session(workspace, clock.now, iter((SESSION_UUID,)).__next__)
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
        model=_direct_model(provider),
        session=session,
        now=clock.now,
        new_uuid=iter((TURN_UUID,)).__next__,
        context_builder=ContextBuilder(
            Workspace.from_path(workspace),
            "Asia/Shanghai",
            clock=clock.now,
        ),
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
    assert [message for message in request.messages if message["role"] != "system"] == [
        {
            "role": "user",
            "content": (
                "<runtime_context>\n"
                "current_time: 2026-07-11T15:30:12.123+08:00\n"
                f"session_id: {session.session_id}\n"
                "</runtime_context>\n\n"
                "<user_input>\n"
                "/unknown\n"
                "</user_input>"
            ),
        }
    ]
    assert [(message["role"], message["content"]) for message in session.messages] == [
        ("user", "/unknown"),
        ("assistant", "Ordinary slash input received."),
    ]
