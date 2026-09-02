from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from threading import Event as ThreadEvent
from threading import Thread
from types import TracebackType
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

import pytest
from loguru import logger

import myclaw.agent.loop as loop_module
from myclaw.agent.blackboard import Blackboard, FramingResult
from myclaw.agent.loop import AgentLoop, ConfirmationRequestView, SkillContextTooLargeError
from myclaw.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.agent.runner import AgentRunnerResult, AgentRunnerRouter
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log as real_session_log
from myclaw.management.service import RuntimeStatusInput
from myclaw.memory.manager import MemoryManager
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningDelta,
    TextDelta,
)
from myclaw.schedule.model import ScheduleJob
from myclaw.schedule.service import ScheduleService
from myclaw.session.session import Session
from myclaw.skills.catalog import (
    LoadedSkill,
    ManualSkillInvocation,
    SkillLoader,
    SkillMetadata,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.fixtures import (
    BlockingBlackboardGenerator,
    DeterministicBlackboardGenerator,
    collect_foreground_outbound,
)
from tests.fixtures.diagnostic_capture import capture_diagnostics


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


class _Router:
    def __init__(self, outcomes: Sequence[ModelResponse | BaseException]) -> None:
        self._outcomes = deque(outcomes)
        self.calls: list[str] = []

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        marker = str(messages[-1]["content"])
        self.calls.append(marker)

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            outcome = self._outcomes.popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            yield ModelCompleted(response=outcome)

        return replay()

    async def complete(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del route, messages, tools, continuation
        raise AssertionError("unexpected direct completion")


class _BlackboardGenerator(Protocol):
    async def generate(
        self,
        *,
        previous: Blackboard | None,
        last_assistant_content: str,
        current_user_input: str,
    ) -> FramingResult: ...


class _BlackboardGeneratorFake:
    def __init__(
        self,
        outcomes: Sequence[FramingResult | BaseException] = (),
    ) -> None:
        self._outcomes = deque(outcomes)
        self.calls: list[tuple[Blackboard | None, str, str]] = []

    async def generate(
        self,
        *,
        previous: Blackboard | None,
        last_assistant_content: str,
        current_user_input: str,
    ) -> FramingResult:
        self.calls.append((previous, last_assistant_content, current_user_input))
        if self._outcomes:
            outcome = self._outcomes.popleft()
        else:
            outcome = FramingResult(
                blackboard=None,
                usage_delta={
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                status="resolved",
            )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _InvalidBlackboardGenerator:
    async def generate(
        self,
        *,
        previous: Blackboard | None,
        last_assistant_content: str,
        current_user_input: str,
    ) -> FramingResult:
        del previous, last_assistant_content, current_user_input
        return cast(FramingResult, object())


class _MaxRouter(_Router):
    def __init__(self) -> None:
        super().__init__(())

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        self.calls.append(str(messages[-1]["content"]))
        call = ModelToolCall(
            id=f"call-{len(self.calls)}",
            name="unknown_tool",
            arguments="{}",
        )

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(response=_response("working", tool_call=call))

        return replay()


class _BlockingRouter(_Router):
    def __init__(self, started: asyncio.Event) -> None:
        super().__init__(())
        self._started = started

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, messages, tools, continuation
        self.calls.append("call")
        first = len(self.calls) == 1

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            if first:
                self._started.set()
                await asyncio.Event().wait()
            yield ModelCompleted(response=_response("after cancellation"))

        return replay()


class _ConcurrentTitleRouter(_Router):
    def __init__(self) -> None:
        super().__init__(())
        self.chat_started = asyncio.Event()
        self.release_chat = asyncio.Event()

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        is_title = messages[0].get("content") == "Generate a title"
        self.calls.append("title" if is_title else "chat")

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            if is_title:
                yield ModelCompleted(response=_response("Generated while chat blocked"))
                return
            self.chat_started.set()
            await self.release_chat.wait()
            yield ModelCompleted(response=_response("foreground completed"))

        return replay()


class _SlowTitleLogRouter(_Router):
    def __init__(self) -> None:
        super().__init__(())
        self.title_started = asyncio.Event()
        self.release_title = asyncio.Event()
        self.chat_calls = 0

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        is_title = messages[0].get("content") == "Generate a title"

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            if is_title:
                self.title_started.set()
                await self.release_title.wait()
                yield ModelCompleted(response=_response("Slow title"))
                return
            self.chat_calls += 1
            if self.chat_calls == 2:
                logger.warning("Second foreground marker")
            yield ModelCompleted(response=_response(f"foreground {self.chat_calls}"))

        return replay()


class _EventRouter(_Router):
    def __init__(self, streams: Sequence[Sequence[ModelStreamEvent]]) -> None:
        super().__init__(())
        self._streams = deque(tuple(stream) for stream in streams)

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        self.calls.append(str(messages[-1]["content"]))
        events = self._streams.popleft()

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            for event in events:
                yield event

        return replay()


class _TitleBehaviorRouter(_Router):
    def __init__(
        self,
        foreground: Sequence[ModelResponse],
        *,
        title: ModelResponse,
        delay_title: bool = False,
        block_first_foreground: bool = False,
    ) -> None:
        super().__init__(())
        self._foreground = deque(foreground)
        self._title = title
        self._delay_title = delay_title
        self._block_first_foreground = block_first_foreground
        self.title_started = asyncio.Event()
        self.release_title = asyncio.Event()
        self.foreground_started = asyncio.Event()

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        is_title = messages[0].get("content") == "Generate a title"
        self.calls.append("title" if is_title else "chat")

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            if is_title:
                self.title_started.set()
                if self._delay_title:
                    await self.release_title.wait()
                yield ModelCompleted(response=self._title)
                return
            self.foreground_started.set()
            if self._block_first_foreground:
                self._block_first_foreground = False
                await asyncio.Event().wait()
            yield ModelCompleted(response=self._foreground.popleft())

        return replay()


def test_agent_loop_constructor_is_the_generation_composition_boundary() -> None:
    assert tuple(inspect.signature(AgentLoop).parameters) == (
        "workspace_path",
        "workspace_state",
        "agent_home",
        "configuration",
        "bus",
        "schedule_service",
        "model_router",
        "memory_manager",
        "session_id",
        "now",
        "new_uuid",
        "monotonic_now",
    )
    assert tuple(inspect.signature(AgentLoop.close).parameters) == ("self",)


@pytest.mark.asyncio
async def test_agent_loop_constructs_each_generation_collaborator_once_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {
        "session": 0,
        "skill_loader": 0,
        "skill_load": 0,
        "context_builder": 0,
        "summary_manager": 0,
        "tool_gateway": 0,
        "runner": 0,
        "persist": 0,
    }
    constructor_args: dict[str, list[tuple[Any, ...]]] = {
        name: []
        for name in ("context_builder", "summary_manager", "tool_gateway", "runner")
    }

    original_create = Session.create
    original_persist = Session.persist

    def recording_create(*args: Any, **kwargs: Any) -> Session:
        counts["session"] += 1
        return original_create(*args, **kwargs)

    def recording_persist(session: Session) -> None:
        counts["persist"] += 1
        original_persist(session)

    class RecordingSkillLoader(SkillLoader):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            counts["skill_loader"] += 1
            super().__init__(*args, **kwargs)

        def load(
            self,
            *,
            validate: Callable[[tuple[LoadedSkill, ...]], None] | None = None,
        ) -> None:
            counts["skill_load"] += 1
            super().load(validate=validate)

    def record_factory(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def recording(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            constructor_args[name].append(args)
            return original(*args, **kwargs)

        return recording

    monkeypatch.setattr(Session, "create", staticmethod(recording_create))
    monkeypatch.setattr(Session, "persist", recording_persist)
    monkeypatch.setattr(loop_module, "SkillLoader", RecordingSkillLoader)
    for name, attribute in (
        ("context_builder", "ContextBuilder"),
        ("summary_manager", "ConversationSummaryManager"),
        ("tool_gateway", "ToolGateway"),
        ("runner", "AgentRunner"),
    ):
        original = getattr(loop_module, attribute)
        monkeypatch.setattr(loop_module, attribute, record_factory(name, original))

    router = _Router(())
    tasks_before = asyncio.all_tasks()
    loop, session, _bus = _runtime(tmp_path, router)

    assert counts == {
        "session": 1,
        "skill_loader": 1,
        "skill_load": 1,
        "context_builder": 1,
        "summary_manager": 1,
        "tool_gateway": 1,
        "runner": 1,
        "persist": 0,
    }
    assert asyncio.all_tasks() == tasks_before
    assert router.calls == []
    assert constructor_args["runner"][0][0] is router
    assert loop._model_router is router
    assert not (session.workspace_state.sessions_directory / f"{session.session_id}.jsonl").exists()

    await loop.abort()


def _response(
    content: str,
    *,
    tool_call: ModelToolCall | None = None,
    input_tokens: int = 1,
    output_tokens: int = 1,
) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(
            content=content,
            tool_calls=() if tool_call is None else (tool_call,),
        ),
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        finish_reason="stop",
    )


def _runtime(
    tmp_path: Path,
    router: AgentRunnerRouter,
    *,
    context_preparer: Callable[[Session, dict[str, Any]], Awaitable[list[dict[str, Any]]]]
    | None = None,
    context_preparer_with_blackboard: Callable[
        [Session, dict[str, Any], Blackboard | None],
        Awaitable[list[dict[str, Any]]],
    ]
    | None = None,
    context_preparer_with_invocation: Callable[
        [Session, dict[str, Any], Blackboard | None, ManualSkillInvocation | None],
        Awaitable[list[dict[str, Any]]],
    ]
    | None = None,
    blackboard_generator: _BlackboardGenerator | None = None,
    use_default_blackboard_generator: bool = False,
    title_prompt: str | None = None,
    skill_loader: SkillLoader | None = None,
    monotonic_now: Callable[[], float] | None = None,
    config_text: str | None = None,
    use_default_context_preparer: bool = False,
) -> tuple[AgentLoop, Session, MessageBus]:
    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home.path)
    (agent_home.path / "config.toml").write_text(
        MINIMAL_VALID_CONFIG if config_text is None else config_text,
        encoding="utf-8",
    )
    configuration = ConfigLoader(agent_home).load()

    async def execute_user_job(job: ScheduleJob) -> None:
        del job

    async def execute_dream() -> object:
        return None

    schedule = ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    selected_context_preparer = _context if context_preparer is None else context_preparer
    selected_context_preparer_with_blackboard = context_preparer_with_blackboard
    selected_context_preparer_with_invocation = context_preparer_with_invocation

    async def prepare(
        active_session: Session,
        current_user: dict[str, Any],
        blackboard: Blackboard | None = None,
        manual_invocation: ManualSkillInvocation | None = None,
    ) -> list[dict[str, Any]]:
        if selected_context_preparer_with_invocation is not None:
            return await selected_context_preparer_with_invocation(
                active_session,
                current_user,
                blackboard,
                manual_invocation,
            )
        if selected_context_preparer_with_blackboard is not None:
            return await selected_context_preparer_with_blackboard(
                active_session,
                current_user,
                blackboard,
            )
        return await selected_context_preparer(active_session, current_user)

    bus = MessageBus()
    loop = AgentLoop(
        workspace_path=workspace,
        workspace_state=state,
        agent_home=agent_home,
        configuration=configuration,
        bus=bus,
        schedule_service=schedule,
        model_router=router,
        memory_manager=MemoryManager(state),
        session_id=None,
        now=_Clock().now,
        new_uuid=uuid4,
        monotonic_now=(lambda: 0.0) if monotonic_now is None else monotonic_now,
    )
    if not use_default_blackboard_generator:
        generator = blackboard_generator or _BlackboardGeneratorFake()
        object.__setattr__(loop, "_generate_blackboard", generator.generate)
    if title_prompt is None:
        def disable_title(_session: Session, _content: str) -> None:
            return None

        object.__setattr__(loop, "_start_title_if_needed", disable_title)
    else:
        def build_title_messages(content: str) -> list[dict[str, Any]]:
            return [
                {"role": "system", "content": title_prompt},
                {"role": "user", "content": content},
            ]

        object.__setattr__(loop._context_builder, "build_title_messages", build_title_messages)
    if skill_loader is not None:
        loop._skill_loader = skill_loader
        loop._context_builder._skill_loader = skill_loader
    if not use_default_context_preparer:
        object.__setattr__(loop, "_prepare_foreground_context", prepare)
    return loop, loop.session, bus


def _planner_skill_loader(tmp_path: Path) -> SkillLoader:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nFollow the plan.\n",
        encoding="utf-8",
    )
    loader = SkillLoader(
        root=tmp_path / "agent-home" / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    return loader


@pytest.mark.asyncio
async def test_agent_loop_status_projection_starts_uptime_only_after_activation(
    tmp_path: Path,
) -> None:
    monotonic_calls = 0

    def monotonic_now() -> float:
        nonlocal monotonic_calls
        monotonic_calls += 1
        return 42.5

    loop, _session, _bus = _runtime(
        tmp_path,
        _Router(()),
        monotonic_now=monotonic_now,
    )

    assert monotonic_calls == 0
    loop.preflight()
    assert monotonic_calls == 0

    await asyncio.gather(loop.start(), loop.start())

    assert monotonic_calls == 1
    assert loop.runtime_status_input().generation_started_at == 42.5
    await loop.start()
    assert monotonic_calls == 1
    await loop.close()


@pytest.mark.asyncio
async def test_agent_loop_failed_activation_does_not_publish_or_duplicate_consumer(
    tmp_path: Path,
) -> None:
    monotonic_calls = 0

    def monotonic_now() -> float:
        nonlocal monotonic_calls
        monotonic_calls += 1
        if monotonic_calls == 1:
            raise RuntimeError("monotonic clock failed")
        return 84.5

    loop, _session, _bus = _runtime(
        tmp_path,
        _Router(()),
        monotonic_now=monotonic_now,
    )
    loop.preflight()

    with pytest.raises(RuntimeError, match="monotonic clock failed"):
        await loop.start()

    assert loop._consumer_task is None
    assert loop._generation_started_at is None
    assert loop._started is False

    await asyncio.gather(loop.start(), loop.start())

    consumer = loop._consumer_task
    assert consumer is not None
    assert loop._generation_started_at == 84.5
    assert monotonic_calls == 2
    await loop.start()
    assert loop._consumer_task is consumer
    assert monotonic_calls == 2
    await loop.close()


@pytest.mark.asyncio
async def test_replacement_barrier_blocks_a_late_foreground_commit_until_released(
    tmp_path: Path,
) -> None:
    class ObservableCommitLock(asyncio.Lock):
        def __init__(self) -> None:
            super().__init__()
            self.acquire_attempted = asyncio.Event()
            self.commit_completed = asyncio.Event()

        async def acquire(self) -> Literal[True]:
            self.acquire_attempted.set()
            return await super().acquire()

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            del exc_type, exc, tb
            self.release()
            self.commit_completed.set()

    router = _ConcurrentTitleRouter()
    loop, session, _bus = _runtime(tmp_path, router)
    commit_gate = ObservableCommitLock()
    loop._foreground_commit_gate = commit_gate
    session.add_message("user", "Existing turn suppresses title work.")
    initial_messages = tuple(session.messages)
    await loop.start()
    await loop._bus.put_inbound(InboundMessage(content="late foreground"))
    await asyncio.wait_for(router.chat_started.wait(), timeout=1)

    await loop._pause_for_replacement()
    commit_gate.acquire_attempted.clear()
    router.release_chat.set()
    await asyncio.wait_for(commit_gate.acquire_attempted.wait(), timeout=1)

    assert tuple(session.messages) == initial_messages
    assert loop.control.has_active_run

    await loop._release_replacement_barrier(resume_inbound=True)
    await asyncio.wait_for(commit_gate.commit_completed.wait(), timeout=1)

    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "Existing turn suppresses title work.",
        "late foreground",
    ]
    await loop.close()


@pytest.mark.asyncio
async def test_agent_loop_status_projection_is_one_read_immutable_and_side_effect_free(
    tmp_path: Path,
) -> None:
    router = _Router(())
    loop, session, _bus = _runtime(tmp_path, router)
    session.add_message("user", "Status snapshot input")
    session.last_consolidated = 0

    class SessionAccessSpy:
        def __init__(self) -> None:
            self.calls = {
                "session_id": 0,
                "messages": 0,
                "metadata": 0,
                "last_consolidated": 0,
            }

        @property
        def session_id(self) -> str:
            self.calls["session_id"] += 1
            return session.session_id

        @property
        def messages(self) -> list[dict[str, Any]]:
            self.calls["messages"] += 1
            return session.messages

        @property
        def metadata(self) -> dict[str, Any]:
            self.calls["metadata"] += 1
            return session.metadata

        @property
        def last_consolidated(self) -> int:
            self.calls["last_consolidated"] += 1
            return session.last_consolidated

    spy = SessionAccessSpy()
    messages_before = deepcopy(session.messages)
    metadata_before = deepcopy(session.metadata)
    tasks_before = set(asyncio.all_tasks())
    object.__setattr__(loop, "_session", spy)
    try:
        projection = loop.runtime_status_input()
    finally:
        object.__setattr__(loop, "_session", session)

    assert spy.calls == {
        "session_id": 1,
        "messages": 1,
        "metadata": 1,
        "last_consolidated": 1,
    }
    assert session.messages == messages_before
    assert session.metadata == metadata_before
    assert session._pending_persist is None
    assert set(asyncio.all_tasks()) == tasks_before
    assert router.calls == []
    assert loop._consumer_task is None
    assert isinstance(projection.retained_messages, tuple)
    assert isinstance(projection.cumulative_usage, tuple)
    assert all(isinstance(value, str) for value in projection.retained_messages)
    with pytest.raises(FrozenInstanceError):
        projection.session_message_count = 99  # type: ignore[misc]
    await loop.close()


async def _context(
    session: Session,
    current_user: dict[str, Any],
) -> list[dict[str, Any]]:
    del session
    return [
        {"role": "system", "content": "test"},
        {"role": "user", "content": current_user["content"]},
    ]


async def _terminals(bus: MessageBus, count: int) -> list[OutboundMessage]:
    terminals: list[OutboundMessage] = []
    while len(terminals) < count:
        message = await bus.get_outbound()
        if message.metadata.get("_streamed") is True:
            terminals.append(message)
    return terminals


def test_agent_loop_exposes_public_control_seam(tmp_path: Path) -> None:
    loop, _session, _bus = _runtime(tmp_path, _Router((_response("unused"),)))

    assert loop.control is loop
    assert loop.control.has_active_run is False
    assert loop.control.has_pending_confirmation is False


def test_agent_loop_reload_returns_the_current_loader_metadata_and_reuses_generation_state(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )
    loop, session, bus = _runtime(tmp_path, _Router(()))
    loader = loop._skill_loader
    initial_session = loop.session
    initial_context_loader = loop._context_builder._skill_loader
    before_messages = deepcopy(session.messages)
    bus_operations: list[str] = []

    async def record_bus_operation(name: str) -> None:
        bus_operations.append(name)

    object.__setattr__(bus, "reset", lambda: record_bus_operation("reset"))
    object.__setattr__(
        bus,
        "pause_inbound_delivery",
        lambda: record_bus_operation("pause"),
    )
    object.__setattr__(
        bus,
        "resume_inbound_delivery",
        lambda: record_bus_operation("resume"),
    )

    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
        encoding="utf-8",
    )

    metadata = loop.reload_skill()

    assert loop is loop.control
    assert loop.session is initial_session is session
    assert loop._bus is bus
    assert loop._skill_loader is loader is initial_context_loader
    assert session.messages == before_messages
    assert bus_operations == []
    assert metadata == loader.metadata
    assert tuple(item.name for item in metadata) == ("reviewer",)
    assert loader.get("planner") is None
    invocation = loader.resolve_manual("/reviewer request")
    assert invocation is not None
    assert invocation.metadata == metadata[0]
    assert invocation.body.splitlines()[-1] == "new body"


def test_agent_loop_reload_rejects_an_always_loaded_budget_overrun_before_publication(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: always\ndescription: Always loaded\nalways: true\n---\nold body\n",
        encoding="utf-8",
    )
    config = MINIMAL_VALID_CONFIG.replace(
        "[models.providers.primary]",
        "[runtime]\nenable_skill_always_load = true\n\n[models.providers.primary]",
    )
    loop, _session, _bus = _runtime(tmp_path, _Router(()), config_text=config)
    loader = loop._skill_loader
    before_skills = loader.skills
    before_metadata = loader.metadata
    before_invocation = loader.resolve_manual("/always request")
    instruction.write_text(
        "---\nname: always\ndescription: Always loaded\nalways: true\n---\n"
        + ("oversized body\n" * 20_000),
        encoding="utf-8",
    )

    with pytest.raises(SkillContextTooLargeError):
        loop.reload_skill()

    assert loader.skills == before_skills
    assert loader.metadata == before_metadata
    assert loader.get("always") is before_skills[0]
    assert loader.resolve_manual("/always request") == before_invocation


def test_reload_validator_candidate_projection_is_isolated_until_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\nalways: true\n---\nold body\n",
        encoding="utf-8",
    )
    config = MINIMAL_VALID_CONFIG.replace(
        "[models.providers.primary]",
        "[runtime]\nenable_skill_always_load = true\n\n[models.providers.primary]",
    )
    loop, _session, _bus = _runtime(tmp_path, _Router(()), config_text=config)
    loader = loop._skill_loader
    before_skills = loader.skills
    before_invocation = loader.resolve_manual("/planner request")
    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\nalways: true\n---\nnew body\n",
        encoding="utf-8",
    )

    validation_started = ThreadEvent()
    release_validation = ThreadEvent()
    candidate_prompts: list[str] = []

    def block_candidate_estimate(status_input: RuntimeStatusInput) -> int:
        candidate_prompts.append(status_input.system_prompt)
        validation_started.set()
        if not release_validation.wait(timeout=5):
            raise AssertionError("candidate validation was not released")
        return 0

    monkeypatch.setattr(loop_module, "estimate_input_tokens", block_candidate_estimate)
    published: list[tuple[SkillMetadata, ...]] = []
    failures: list[BaseException] = []

    def reload_in_thread() -> None:
        try:
            published.append(loop.reload_skill())
        except BaseException as error:
            failures.append(error)

    reload_thread = Thread(target=reload_in_thread)
    reload_thread.start()
    try:
        assert validation_started.wait(timeout=5)
        assert len(candidate_prompts) == 1
        assert '"name":"reviewer"' in candidate_prompts[0]
        assert '"name":"planner"' not in candidate_prompts[0]
        assert loader.skills == before_skills
        assert loader.resolve_manual("/planner request") == before_invocation
        assert loader.resolve_manual("/reviewer request") is None

        public_messages = loop._context_builder.build_status_messages(
            (),
            session_id=loop.session.session_id,
        )
        assert '"name":"planner"' in str(public_messages[0]["content"])
        assert '"name":"reviewer"' not in str(public_messages[0]["content"])
    finally:
        release_validation.set()
        reload_thread.join(timeout=5)

    assert not reload_thread.is_alive()
    assert failures == []
    assert published == [loader.metadata]
    assert tuple(item.name for item in loader.metadata) == ("reviewer",)
    assert loader.resolve_manual("/planner request") is None
    assert loader.resolve_manual("/reviewer request") is not None


@pytest.mark.asyncio
async def test_reload_during_active_run_preserves_old_request_and_updates_future_run(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )

    class ReloadBarrierRouter(_Router):
        def __init__(self) -> None:
            super().__init__(())
            self.requests: list[list[dict[str, Any]]] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        def stream(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[OpenAIToolSchema],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, tools, continuation
            self.requests.append(deepcopy(list(messages)))
            first = len(self.requests) == 1

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                if first:
                    self.first_started.set()
                    await self.release_first.wait()
                yield ModelCompleted(response=_response("completed"))

            return replay()

    router = ReloadBarrierRouter()
    loop, session, bus = _runtime(
        tmp_path,
        router,
        blackboard_generator=_BlackboardGeneratorFake(),
        use_default_context_preparer=True,
    )
    before_messages = deepcopy(session.messages)
    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("/planner first request"))
        await asyncio.wait_for(router.first_started.wait(), timeout=1)
        first_request = deepcopy(router.requests[0])

        instruction.write_text(
            "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
            encoding="utf-8",
        )
        metadata = loop.reload_skill()

        assert tuple(item.name for item in metadata) == ("reviewer",)
        assert session.messages == before_messages
        assert router.requests[0] == first_request
        assert '"name":"planner"' in str(first_request[0]["content"])
        assert '"name":"reviewer"' not in str(first_request[0]["content"])
        assert "old body" in str(first_request[-1]["content"])
        assert "new body" not in str(first_request[-1]["content"])

        router.release_first.set()
        await _terminals(bus, 1)
        await bus.put_inbound(InboundMessage("/reviewer second request"))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.requests) == 2
    assert '"name":"reviewer"' in str(router.requests[1][0]["content"])
    assert '"name":"planner"' not in str(router.requests[1][0]["content"])
    assert "new body" in str(router.requests[1][-1]["content"])
    assert "old body" not in str(router.requests[1][-1]["content"])


@pytest.mark.parametrize("error_type", (RuntimeError, asyncio.CancelledError))
def test_foreground_projection_scope_restores_published_state_after_failure(
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )
    loop, _session, _bus = _runtime(tmp_path, _Router(()))
    old_skills = loop._skill_loader.skills
    instruction.write_text(
        "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
        encoding="utf-8",
    )
    loop.reload_skill()

    with pytest.raises(error_type):
        with loop._context_builder.foreground_projection_scope(old_skills):
            old_prompt = loop._context_builder.foreground_system_prompt()
            assert '"name":"planner"' in old_prompt
            assert '"name":"reviewer"' not in old_prompt
            raise error_type()

    published_prompt = loop._context_builder.foreground_system_prompt()
    assert '"name":"reviewer"' in published_prompt
    assert '"name":"planner"' not in published_prompt


@pytest.mark.asyncio
async def test_reload_during_context_preparation_keeps_run_skill_snapshot(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\nold body\n",
        encoding="utf-8",
    )

    class ContextPreparationBarrierRouter(_Router):
        def __init__(self) -> None:
            super().__init__(())
            self.requests: list[list[dict[str, Any]]] = []

        def stream(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[OpenAIToolSchema],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, tools, continuation
            self.requests.append(deepcopy(list(messages)))

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                yield ModelCompleted(response=_response("completed"))

            return replay()

    router = ContextPreparationBarrierRouter()
    loop, session, bus = _runtime(
        tmp_path,
        router,
        blackboard_generator=_BlackboardGeneratorFake(),
        use_default_context_preparer=True,
    )
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()
    original_prepare = loop._prepare_foreground_context

    async def blocked_prepare(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        preparation_started.set()
        await release_preparation.wait()
        return await original_prepare(*args, **kwargs)

    object.__setattr__(loop, "_prepare_foreground_context", blocked_prepare)
    before_messages = deepcopy(session.messages)
    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("first request"))
        await asyncio.wait_for(preparation_started.wait(), timeout=1)

        instruction.write_text(
            "---\nname: reviewer\ndescription: Review work\n---\nnew body\n",
            encoding="utf-8",
        )
        metadata = loop.reload_skill()

        assert tuple(item.name for item in metadata) == ("reviewer",)
        assert session.messages == before_messages
        release_preparation.set()
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(router.requests) == 1
    assert "planner" in str(router.requests[0][0]["content"])
    assert "reviewer" not in str(router.requests[0][0]["content"])


@pytest.mark.asyncio
async def test_loop_consumes_foreground_inputs_fifo_and_publishes_one_terminal_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _Router((_response("one"), _response("two"), _response("three")))
    loop, session, _bus = _runtime(tmp_path, router)
    append_calls = 0
    persist_calls = 0
    original_append = Session.append_messages
    original_persist = Session.persist

    def append_messages(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal append_calls
        append_calls += 1
        original_append(active, messages, **kwargs)

    def persist(active: Session) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(active)

    monkeypatch.setattr(Session, "append_messages", append_messages)
    monkeypatch.setattr(Session, "persist", persist)
    await loop.start()
    try:
        for content in ("one", "two", "three"):
            await _bus.put_inbound(InboundMessage(content))

        terminals = await _terminals(_bus, 3)

        assert len(router.calls) == 3
        assert [
            message["content"] for message in session.messages if message["role"] == "user"
        ] == [
            "one",
            "two",
            "three",
        ]
        assert [message.metadata for message in terminals] == [
            {"_streamed": True},
            {"_streamed": True},
            {"_streamed": True},
        ]
        assert append_calls == 3
        assert persist_calls == 3
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_manual_skill_invocation_preserves_raw_order_and_projects_expanded_user(
    tmp_path: Path,
) -> None:
    raw_input = "/planner Do the work"
    body = "Follow the plan.\n"
    document = "---\nname: planner\ndescription: Plan work\n---\n" + body
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(document.encode("utf-8"))
    loader = SkillLoader(
        root=tmp_path / "agent-home" / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    router = _Router((_response("Generated title"), _response("Completed")))
    framer = _BlackboardGeneratorFake()
    observed: list[tuple[dict[str, Any], ManualSkillInvocation | None]] = []

    async def prepare(
        active_session: Session,
        current_user: dict[str, Any],
        blackboard: Blackboard | None,
        manual_invocation: ManualSkillInvocation | None,
    ) -> list[dict[str, Any]]:
        del active_session
        observed.append((deepcopy(current_user), manual_invocation))
        assert blackboard is None
        assert manual_invocation is not None
        return [
            {"role": "system", "content": "test"},
            {
                "role": "user",
                "content": f"<skill>{manual_invocation.body}</skill>"
                f"<request>{manual_invocation.request}</request>",
            },
        ]

    loop, session, _bus = _runtime(
        tmp_path,
        router,
        context_preparer_with_invocation=prepare,
        blackboard_generator=framer,
        title_prompt="Generate a title",
        skill_loader=loader,
    )
    previous_blackboard = {
        "goal": "Preserved goal",
        "completion_boundary": "Preserved boundary",
    }
    session.update_metadata(blackboard=previous_blackboard)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage(raw_input))
        await _terminals(_bus, 1)
    finally:
        await loop.close()

    assert len(router.calls) == 2
    assert raw_input in router.calls
    assert f"<skill>{document}</skill><request>Do the work</request>" in router.calls
    assert framer.calls == []
    assert observed[0][0] == {"role": "user", "content": raw_input}
    assert observed[0][1] is not None
    assert observed[0][1].body == document
    assert observed[0][1].request == "Do the work"
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        raw_input
    ]
    assert session.metadata["blackboard"] == previous_blackboard


@pytest.mark.asyncio
async def test_published_manual_skill_loader_ignores_later_file_changes(
    tmp_path: Path,
) -> None:
    raw_input = "/planner request"
    document = "---\nname: planner\ndescription: Plan work\n---\nPRIVATE MANUAL BODY\n"
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(document.encode("utf-8"))
    loader = SkillLoader(
        root=tmp_path / "agent-home" / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    instruction.unlink()

    router = _Router((_response("Generated title"), _response("Completed")))
    framer = _BlackboardGeneratorFake()

    async def prepare(
        active_session: Session,
        current_user: dict[str, Any],
        blackboard: Blackboard | None,
        manual_invocation: ManualSkillInvocation | None,
    ) -> list[dict[str, Any]]:
        del active_session, current_user
        assert blackboard is None
        assert manual_invocation is not None
        return [
            {"role": "system", "content": "test"},
            {
                "role": "user",
                "content": f"<skill>{manual_invocation.body}</skill>"
                f"<request>{manual_invocation.request}</request>",
            },
        ]

    loop, session, _bus = _runtime(
        tmp_path,
        router,
        context_preparer_with_invocation=prepare,
        blackboard_generator=framer,
        title_prompt="Generate a title",
        skill_loader=loader,
    )
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage(raw_input))
        await _terminals(_bus, 1)
    finally:
        await loop.close()

    assert len(router.calls) == 2
    assert f"<skill>{document}</skill><request>request</request>" in router.calls
    assert framer.calls == []
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        raw_input
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_input", "expected_framing_calls"),
    [
        ("ordinary input", 1),
        ("/unknown", 1),
        (" /planner", 1),
        ("/Planner", 1),
        ("/planner", 0),
        ("/planner\u2003request", 0),
    ],
)
async def test_only_exact_manual_skill_invocations_skip_task_framing(
    tmp_path: Path,
    raw_input: str,
    expected_framing_calls: int,
) -> None:
    loader = _planner_skill_loader(tmp_path)
    framer = _BlackboardGeneratorFake()
    loop, _session, bus = _runtime(
        tmp_path,
        _Router((_response("Completed"),)),
        blackboard_generator=framer,
        skill_loader=loader,
    )

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage(raw_input))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert len(framer.calls) == expected_framing_calls


@pytest.mark.asyncio
async def test_ordinary_turn_after_manual_skill_reuses_preserved_blackboard(
    tmp_path: Path,
) -> None:
    loader = _planner_skill_loader(tmp_path)
    previous = Blackboard(goal="Preserved goal", completion_boundary="Preserved boundary")
    framer = _BlackboardGeneratorFake(
        (
            FramingResult(
                blackboard=previous,
                usage_delta={
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                status="resolved",
            ),
        )
    )
    loop, session, bus = _runtime(
        tmp_path,
        _Router((_response("Manual result"), _response("Ordinary result"))),
        blackboard_generator=framer,
        skill_loader=loader,
    )
    session.update_metadata(
        blackboard={
            "goal": previous.goal,
            "completion_boundary": previous.completion_boundary,
        }
    )

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("/planner do it"))
        await _terminals(bus, 1)
        await bus.put_inbound(InboundMessage("ordinary follow-up"))
        await _terminals(bus, 1)
    finally:
        await loop.close()

    assert framer.calls == [(previous, "Manual result", "ordinary follow-up")]
    assert session.metadata["blackboard"] == {
        "goal": previous.goal,
        "completion_boundary": previous.completion_boundary,
    }


@pytest.mark.asyncio
async def test_manual_skill_context_failure_preserves_blackboard_and_usage(
    tmp_path: Path,
) -> None:
    loader = _planner_skill_loader(tmp_path)
    framer = _BlackboardGeneratorFake()

    async def fail_context(
        active_session: Session,
        current_user: dict[str, Any],
        blackboard: Blackboard | None,
        manual_invocation: ManualSkillInvocation | None,
    ) -> list[dict[str, Any]]:
        del active_session, current_user
        assert blackboard is None
        assert manual_invocation is not None
        raise ModelCallError(ErrorInfo("model_failed", "context failed"))

    loop, session, bus = _runtime(
        tmp_path,
        _Router(()),
        context_preparer_with_invocation=fail_context,
        blackboard_generator=framer,
        skill_loader=loader,
    )
    session.update_metadata(
        blackboard={
            "goal": "Preserved goal",
            "completion_boundary": "Preserved boundary",
        }
    )
    before_metadata = deepcopy(session.metadata)

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("/planner fail"))
        terminal = (await _terminals(bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata["finish_reason"] == "failed"
    assert framer.calls == []
    assert session.metadata == before_metadata
    assert session.messages == []


@pytest.mark.asyncio
async def test_manual_skill_context_cancellation_preserves_blackboard_and_usage(
    tmp_path: Path,
) -> None:
    loader = _planner_skill_loader(tmp_path)
    framer = _BlackboardGeneratorFake()
    started = asyncio.Event()

    async def block_context(
        active_session: Session,
        current_user: dict[str, Any],
        blackboard: Blackboard | None,
        manual_invocation: ManualSkillInvocation | None,
    ) -> list[dict[str, Any]]:
        del active_session, current_user
        assert blackboard is None
        assert manual_invocation is not None
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    loop, session, bus = _runtime(
        tmp_path,
        _Router(()),
        context_preparer_with_invocation=block_context,
        blackboard_generator=framer,
        skill_loader=loader,
    )
    session.update_metadata(
        blackboard={
            "goal": "Preserved goal",
            "completion_boundary": "Preserved boundary",
        }
    )
    before_metadata = deepcopy(session.metadata)

    await loop.start()
    await bus.put_inbound(InboundMessage("/planner wait"))
    await started.wait()
    await loop.cancel_active_run()
    terminal = (await _terminals(bus, 1))[0]
    await loop.close()

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert framer.calls == []
    assert session.metadata == before_metadata
    assert session.messages == []


@pytest.mark.asyncio
async def test_loop_preparation_failure_has_no_session_commit_and_fifo_continues(
    tmp_path: Path,
) -> None:
    router = _Router((_response("after failure"),))
    calls = 0

    async def prepare(
        active: Session,
        current: dict[str, object],
    ) -> list[dict[str, object]]:
        nonlocal calls
        del active, current
        calls += 1
        if calls == 1:
            raise ModelCallError(ErrorInfo("model_failed", "preparation failed"))
        return [{"role": "system", "content": "test"}, {"role": "user", "content": "ok"}]

    loop, session, _bus = _runtime(tmp_path, router, context_preparer=prepare)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("first"))
        await _bus.put_inbound(InboundMessage("second"))
        first = await _bus.get_outbound()
        while first.metadata.get("_streamed") is not True:
            first = await _bus.get_outbound()
        second = await _bus.get_outbound()
        while second.metadata.get("_streamed") is not True:
            second = await _bus.get_outbound()

        assert first.type == "system_control"
        assert first.metadata["error_code"] == "model_failed"
        assert second.metadata == {"_streamed": True}
        assert [
            message["content"] for message in session.messages if message["role"] == "user"
        ] == ["second"]
        assert len(router.calls) == 1
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_loop_commits_failed_runner_result_once_before_one_safe_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = Blackboard(goal="Failed goal", completion_boundary="Failed boundary")
    framing_usage = {
        "model_calls": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    router = _Router((ModelCallError(ErrorInfo("model_failed", "provider failed")),))
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        blackboard_generator=_BlackboardGeneratorFake(
            (FramingResult(blackboard=staged, usage_delta=framing_usage, status="resolved"),)
        ),
    )
    append_calls = 0
    persist_calls = 0
    original_append = Session.append_messages
    original_persist = Session.persist

    def append_messages(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal append_calls
        append_calls += 1
        original_append(active, messages, **kwargs)

    def persist(active: Session) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(active)

    monkeypatch.setattr(Session, "append_messages", append_messages)
    monkeypatch.setattr(Session, "persist", persist)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("failed input"))
        terminal = (await _terminals(_bus, 1))[0]

        assert terminal.type == "system_control"
        assert terminal.content == "provider failed"
        assert terminal.metadata == {
            "finish_reason": "failed",
            "error_code": "model_failed",
            "_streamed": True,
        }
        assert append_calls == 1
        assert persist_calls == 1
        assert [message["role"] for message in session.messages] == ["user", "assistant"]
        assert session.messages[-1]["status"] == "error"
        assert session.metadata["blackboard"] == {
            "goal": staged.goal,
            "completion_boundary": staged.completion_boundary,
        }
        assert session.metadata["token_usage"] == {
            "model_calls": 2,
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
        }
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_loop_commits_max_iteration_repair_once_before_safe_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = Blackboard(goal="Bounded goal", completion_boundary="Bounded boundary")
    framing_usage = {
        "model_calls": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    router = _MaxRouter()
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        blackboard_generator=_BlackboardGeneratorFake(
            (FramingResult(blackboard=staged, usage_delta=framing_usage, status="resolved"),)
        ),
    )
    append_calls = 0
    persist_calls = 0
    original_append = Session.append_messages
    original_persist = Session.persist

    def append_messages(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal append_calls
        append_calls += 1
        original_append(active, messages, **kwargs)

    def persist(active: Session) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(active)

    monkeypatch.setattr(Session, "append_messages", append_messages)
    monkeypatch.setattr(Session, "persist", persist)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("bounded input"))
        terminal = (await _terminals(_bus, 1))[0]

        assert len(router.calls) == 50
        assert terminal.type == "system_control"
        assert terminal.content == (
            "MyClaw 本轮对话已经达到最大循环次数，仍没有输出最终结果。"  # noqa: RUF001
            "可以再次尝试本次请求或者尝试给出更明确的任务目标。"
        )
        assert terminal.metadata == {
            "finish_reason": "max_iterations",
            "error_code": "agent_iteration_limit",
            "_streamed": True,
        }
        assert append_calls == 1
        assert persist_calls == 1
        assert session.metadata["token_usage"] == {
            "model_calls": 51,
            "input_tokens": 54,
            "output_tokens": 52,
            "total_tokens": 106,
        }
        assert session.metadata["blackboard"] == {
            "goal": staged.goal,
            "completion_boundary": staged.completion_boundary,
        }
        assert session.messages[-1]["status"] == "error"
        assert session.messages[-1]["error"]["code"] == "agent_iteration_limit"
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_loop_cancellation_repairs_and_keeps_the_next_queued_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    router = _BlockingRouter(started)
    staged = Blackboard(goal="Cancelled goal", completion_boundary="Cancelled boundary")
    framing_result = FramingResult(
        blackboard=staged,
        usage_delta={
            "model_calls": 1,
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
        },
        status="resolved",
    )
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        blackboard_generator=_BlackboardGeneratorFake((framing_result, framing_result)),
    )
    append_calls = 0
    persist_calls = 0
    original_append = Session.append_messages
    original_persist = Session.persist

    def append_messages(
        active: Session,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        nonlocal append_calls
        append_calls += 1
        original_append(active, messages, **kwargs)

    def persist(active: Session) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(active)

    monkeypatch.setattr(Session, "append_messages", append_messages)
    monkeypatch.setattr(Session, "persist", persist)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("cancelled input"))
        await _bus.put_inbound(InboundMessage("queued input"))
        await started.wait()
        await loop.cancel_active_run()
        terminals = await _terminals(_bus, 2)

        assert router.calls == ["call", "call"]
        assert [terminal.metadata for terminal in terminals] == [
            {
                "finish_reason": "cancelled",
                "error_code": "turn_cancelled",
                "_streamed": True,
            },
            {"_streamed": True},
        ]
        assert append_calls == 2
        assert persist_calls == 2
        assert [
            message["content"] for message in session.messages if message["role"] == "user"
        ] == [
            "cancelled input",
            "queued input",
        ]
        assert session.metadata["blackboard"] == {
            "goal": staged.goal,
            "completion_boundary": staged.completion_boundary,
        }
        assert session.metadata["token_usage"] == {
            "model_calls": 3,
            "input_tokens": 9,
            "output_tokens": 5,
            "total_tokens": 14,
        }
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_loop_confirmation_uses_one_direct_pending_future_and_cancels_it(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    router = _Router(
        (
            _response(
                "",
                tool_call=ModelToolCall(
                    id="call_confirmation",
                    name="read_file",
                    arguments=json.dumps({"path": str(outside)}),
                ),
            ),
        )
    )
    loop, _session, _bus = _runtime(tmp_path, router)
    requests: list[ConfirmationRequestView] = []
    requested = asyncio.Event()

    def on_confirmation(request: ConfirmationRequestView) -> None:
        requests.append(request)
        requested.set()

    loop.bind_confirmation_callback(on_confirmation)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("confirm this"))
        await requested.wait()

        assert len(requests) == 1
        request = requests[0]
        assert loop.has_pending_confirmation
        with pytest.raises(ValueError, match="late or unknown"):
            loop.respond_to_confirmation(uuid4(), "approved")

        await loop.cancel_active_run()
        terminal = (await _terminals(_bus, 1))[0]

        assert terminal.metadata == {
            "finish_reason": "cancelled",
            "error_code": "turn_cancelled",
            "_streamed": True,
        }
        assert not loop.has_pending_confirmation
        with pytest.raises(ValueError, match="late or unknown"):
            loop.respond_to_confirmation(request.confirmation_id, "approved")
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_preparation_cancellation_publishes_the_cancelled_terminal(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    async def prepare(
        active: Session,
        current: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del active, current
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    loop, session, _bus = _runtime(tmp_path, _Router(()), context_preparer=prepare)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("cancel during preparation"))
        await started.wait()

        await loop.cancel_active_run()
        terminal = (await _terminals(_bus, 1))[0]

        assert terminal.type == "system_control"
        assert terminal.content == "MyClaw 已取消本轮对话。"
        assert terminal.metadata == {
            "finish_reason": "cancelled",
            "error_code": "turn_cancelled",
            "_streamed": True,
        }
        assert session.messages == []
        assert session.metadata["token_usage"]["model_calls"] == 0
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_preparation_failure_does_not_start_title_or_accumulate_usage(
    tmp_path: Path,
) -> None:
    async def fail_preparation(
        active: Session,
        current: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del active, current
        raise ModelCallError(ErrorInfo("model_failed", "preparation failed"))

    router = _Router((_response("must not become a title"),))
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        context_preparer=fail_preparation,
        title_prompt="Generate a title",
    )
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("uncommitted input"))
        _ = await _terminals(_bus, 1)
        for _ in range(5):
            await asyncio.sleep(0)

        assert router.calls == []
        assert session.messages == []
        assert session.metadata["title"] == "Untitled session"
        assert session.metadata["token_usage"] == {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_async_preparation_failure_discards_parallel_title_result(
    tmp_path: Path,
) -> None:
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()

    async def fail_after_waiting(
        active: Session,
        current: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del active, current
        preparation_started.set()
        await release_preparation.wait()
        raise ModelCallError(ErrorInfo("model_failed", "preparation failed"))

    router = _ConcurrentTitleRouter()
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        context_preparer=fail_after_waiting,
        title_prompt="Generate a title",
    )
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("uncommitted input"))
        await preparation_started.wait()
        for _ in range(100):
            if "title" in router.calls:
                break
            await asyncio.sleep(0)

        assert router.calls == ["title"]
        assert session.metadata["title"] == "Untitled session"

        release_preparation.set()
        terminal = (await _terminals(_bus, 1))[0]

        assert terminal.metadata == {
            "finish_reason": "failed",
            "error_code": "model_failed",
            "_streamed": True,
        }
        assert session.messages == []
        assert session.metadata["title"] == "Untitled session"
        assert session.metadata["token_usage"] == {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    finally:
        release_preparation.set()
        await loop.close()


@pytest.mark.asyncio
async def test_title_request_uses_context_builder_messages_without_loop_override(
    tmp_path: Path,
) -> None:
    router = _Router(())
    loop, _session, _bus = _runtime(
        tmp_path,
        router,
        title_prompt="Loop-owned title prompt",
    )
    builder_inputs: list[str] = []
    captured_messages: list[dict[str, Any]] = []

    def build_title_messages(content: str) -> list[dict[str, Any]]:
        builder_inputs.append(content)
        return [
            {"role": "system", "content": "Builder-owned title prompt"},
            {"role": "user", "content": content},
        ]

    def stream(
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, tools, continuation
        captured_messages.extend(deepcopy(messages))

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(response=_response("Generated title"))

        return replay()

    object.__setattr__(loop._context_builder, "build_title_messages", build_title_messages)
    object.__setattr__(router, "stream", stream)

    events = loop._router_stream_title("  First title input.  ")
    _ = [event async for event in events]

    assert builder_inputs == ["First title input."]
    assert captured_messages == [
        {"role": "system", "content": "Builder-owned title prompt"},
        {"role": "user", "content": "First title input."},
    ]
    await loop.close()


@pytest.mark.asyncio
async def test_first_message_title_runs_while_foreground_chat_is_blocked(
    tmp_path: Path,
) -> None:
    router = _ConcurrentTitleRouter()
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("first input"))
        await router.chat_started.wait()
        for _ in range(100):
            if session.metadata["title"] != "Untitled session":
                break
            await asyncio.sleep(0)

        assert session.metadata["title"] == "Generated while chat blocked"

        router.release_chat.set()
        assert (await _terminals(_bus, 1))[0].metadata == {"_streamed": True}
    finally:
        router.release_chat.set()
        await loop.close()


@pytest.mark.asyncio
async def test_slow_title_keeps_one_session_log_owner_across_the_next_fifo_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _SlowTitleLogRouter()
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")
    active_contexts = 0
    maximum_contexts = 0
    original_session_log = real_session_log

    @contextmanager
    def observed_session_log(active_session: Session) -> Iterator[None]:
        nonlocal active_contexts, maximum_contexts
        active_contexts += 1
        maximum_contexts = max(maximum_contexts, active_contexts)
        try:
            with original_session_log(active_session):
                yield
        finally:
            active_contexts -= 1

    monkeypatch.setattr("myclaw.agent.loop.session_log", observed_session_log)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("first input"))
        _ = await _terminals(_bus, 1)
        await router.title_started.wait()

        await _bus.put_inbound(InboundMessage("second input"))
        _ = await _terminals(_bus, 1)

        router.release_title.set()
        for _ in range(100):
            if session.metadata["title"] == "Slow title":
                break
            await asyncio.sleep(0)
    finally:
        router.release_title.set()
        await loop.close()

    log_path = session.workspace_state.logs_directory / f"{session.session_id}.log"
    assert maximum_contexts == 1
    assert log_path.read_text(encoding="utf-8").count("Second foreground marker") == 1


@pytest.mark.asyncio
async def test_append_failure_reports_safe_terminal_and_fifo_consumer_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = Blackboard(goal="Previous goal", completion_boundary="Previous boundary")
    staged = Blackboard(goal="Staged goal", completion_boundary="Staged boundary")
    framing_usage = {
        "model_calls": 1,
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    loop, session, _bus = _runtime(
        tmp_path,
        _Router(()),
        blackboard_generator=_BlackboardGeneratorFake(
            (
                FramingResult(
                    blackboard=staged,
                    usage_delta=framing_usage,
                    status="resolved",
                ),
                FramingResult(
                    blackboard=None,
                    usage_delta=None,
                    status="model_failed",
                ),
            )
        ),
    )
    session.update_metadata(
        title="Existing title",
        blackboard={
            "goal": previous.goal,
            "completion_boundary": previous.completion_boundary,
        },
        usage_delta={
            "model_calls": 2,
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
        },
    )
    before_messages = deepcopy(session.messages)
    before_metadata = deepcopy(session.metadata)
    results = deque(
        (
            AgentRunnerResult(
                messages=[{"role": "assistant", "content": "invalid increment"}],
                final_content="invalid increment",
                usage={
                    "model_calls": 1,
                    "input_tokens": 3,
                    "output_tokens": 1,
                    "total_tokens": 4,
                },
            ),
            AgentRunnerResult(
                messages=[
                    {
                        "role": "assistant",
                        "content": "next completed",
                        "tool_calls": [],
                        "status": "completed",
                        "error": None,
                        "token_usage": {
                            "model_calls": 1,
                            "input_tokens": 2,
                            "output_tokens": 1,
                            "total_tokens": 3,
                        },
                    }
                ],
                final_content="next completed",
                usage={
                    "model_calls": 1,
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
            ),
        )
    )

    async def run(*args: Any, **kwargs: Any) -> AgentRunnerResult:
        del args, kwargs
        return results.popleft()

    monkeypatch.setattr(loop._runner, "run", run)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("cannot commit"))
        failed = (await _terminals(_bus, 1))[0]
        assert session.messages == before_messages
        assert session.metadata == before_metadata
        await _bus.put_inbound(InboundMessage("next input"))
        completed = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert failed.type == "system_control"
    assert failed.metadata == {
        "finish_reason": "failed",
        "error_code": "persistence_error",
        "_streamed": True,
    }
    assert completed.type == "model_response"
    assert completed.metadata == {"_streamed": True}
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "next input"
    ]


@pytest.mark.asyncio
async def test_persist_request_failure_is_silent_and_terminal_stays_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, session, _bus = _runtime(tmp_path, _Router((_response("committed in memory"),)))

    def fail_persist() -> None:
        raise OSError("private persistence detail")

    monkeypatch.setattr(session, "persist", fail_persist)
    capture = capture_diagnostics()
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("persist fails"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        capture.close()
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert "Foreground Session persist failed" not in capture.event_text
    assert "private persistence detail" not in capture.text


@pytest.mark.asyncio
async def test_loop_normal_close_saves_only_its_owned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, first, _bus = _runtime(tmp_path, _Router(()))
    closed: list[str] = []
    first_close = first.close

    def close_first() -> None:
        closed.append(first.session_id)
        first_close()

    monkeypatch.setattr(first, "close", close_first)

    await loop.close()

    assert closed == [first.session_id]


@pytest.mark.asyncio
async def test_loop_abort_retains_cancelled_owned_tasks_until_cleanup_finishes(
    tmp_path: Path,
) -> None:
    loop, _session, _bus = _runtime(tmp_path, _Router(()))
    started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def cancellable_work() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_cleanup.wait()
            raise
        finally:
            cleanup_finished.set()

    task = asyncio.create_task(cancellable_work())
    loop._execution_task = task
    await started.wait()

    abort_task = asyncio.create_task(loop.abort())
    await asyncio.sleep(0)

    assert task in loop._aborted_tasks
    release_cleanup.set()
    await cleanup_finished.wait()
    tasks_drained = asyncio.Event()
    task.add_done_callback(lambda _task: tasks_drained.set())
    await tasks_drained.wait()
    await abort_task
    assert loop._aborted_tasks == set()


@pytest.mark.asyncio
async def test_terminal_loop_states_reject_restart(tmp_path: Path) -> None:
    closed_loop, _closed_session, _closed_bus = _runtime(tmp_path / "closed", _Router(()))
    await closed_loop.start()
    await closed_loop.close()

    with pytest.raises(RuntimeError, match="Agent Loop is closed"):
        await closed_loop.start()

    aborted_loop, _aborted_session, _aborted_bus = _runtime(tmp_path / "aborted", _Router(()))
    await aborted_loop.start()
    await aborted_loop.abort()

    with pytest.raises(RuntimeError, match="Agent Loop is closed"):
        await aborted_loop.start()


@pytest.mark.asyncio
async def test_close_transition_blocks_concurrent_preflight_and_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _session, _bus = _runtime(tmp_path, _Router(()))
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def blocked_finish_close() -> None:
        close_started.set()
        await release_close.wait()

    monkeypatch.setattr(loop, "_finish_close", blocked_finish_close)
    closing = asyncio.create_task(loop.close())
    await close_started.wait()

    try:
        with pytest.raises(RuntimeError, match="Agent Loop is closed"):
            loop.preflight()
        with pytest.raises(RuntimeError, match="Agent Loop is closed"):
            await loop.start()
    finally:
        release_close.set()
        await closing


@pytest.mark.asyncio
async def test_abort_wins_before_normal_close_finalizes_the_session(
    tmp_path: Path,
) -> None:
    loop, session, _bus = _runtime(tmp_path, _Router(()))
    session.add_message("user", "preserve this turn")
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def blocked_finish_close() -> None:
        close_started.set()
        await release_close.wait()

    object.__setattr__(loop, "_finish_close", blocked_finish_close)
    closing = asyncio.create_task(loop.close())
    await close_started.wait()

    aborting = asyncio.create_task(loop.abort())
    await asyncio.sleep(0)
    release_close.set()

    await asyncio.gather(closing, aborting)
    assert loop._session_abandoned
    assert not loop._session_closed


@pytest.mark.asyncio
async def test_blank_foreground_input_performs_zero_task_framing_attempts(
    tmp_path: Path,
) -> None:
    framer = DeterministicBlackboardGenerator()
    loop, session, _bus = _runtime(tmp_path, _Router(()), blackboard_generator=framer)
    execution_ready = asyncio.Event()

    committed = await loop._execute_foreground_logged(
        session,
        InboundMessage(" \n\t "),
        title_work=None,
        execution_ready=execution_ready,
    )

    assert not committed
    assert execution_ready.is_set()
    assert framer.calls == 0
    assert session.messages == []
    assert session.metadata["token_usage"] == {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


@pytest.mark.asyncio
async def test_direct_bus_projects_completed_turn_without_session_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, _session, _bus = _runtime(tmp_path, _Router((_response("completed without deltas"),)))

    def reject_session_access(_loop: AgentLoop) -> Session:
        raise AssertionError("Terminal adapter must not access Session")

    monkeypatch.setattr(AgentLoop, "session", property(reject_session_access))

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("foreground input"))
        observed: list[OutboundMessage] = []
        while not observed or observed[-1].metadata.get("_streamed") is not True:
            observed.append(await _bus.get_outbound())
    finally:
        await loop.close()

    assert [(message.type, message.content, message.metadata) for message in observed] == [
        ("model_response", "completed without deltas", {"_stream_delta": True}),
        ("model_response", "", {"_stream_end": True}),
        ("model_response", "", {"_streamed": True}),
    ]


@pytest.mark.asyncio
async def test_loop_publishes_the_exact_sparse_outbound_protocol_without_tool_results(
    tmp_path: Path,
) -> None:
    raw_arguments = '{"missing":"preserved exactly"}'
    tool_call = ModelToolCall(
        id="call_sparse",
        name="unknown_tool",
        arguments=raw_arguments,
    )
    router = _EventRouter(
        (
            (
                ReasoningDelta("reasoning"),
                TextDelta("working"),
                ModelCompleted(response=_response("working", tool_call=tool_call)),
            ),
            (
                TextDelta("done"),
                ModelCompleted(response=_response("done")),
            ),
        )
    )
    loop, session, _bus = _runtime(tmp_path, router)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("sparse output"))
        observed: list[OutboundMessage] = []
        while not observed or observed[-1].metadata.get("_streamed") is not True:
            observed.append(await _bus.get_outbound())

        assert [(message.type, message.content, message.metadata) for message in observed] == [
            ("model_reasoning", "reasoning", {"_stream_delta": True}),
            ("model_reasoning", "", {"_stream_end": True}),
            ("model_response", "working", {"_stream_delta": True}),
            ("model_response", "", {"_stream_end": True}),
            (
                "tool_call",
                "unknown_tool",
                {"tool_call_id": "call_sparse", "arguments": raw_arguments},
            ),
            ("model_response", "done", {"_stream_delta": True}),
            ("model_response", "", {"_stream_end": True}),
            ("model_response", "", {"_streamed": True}),
        ]
        assert [message["role"] for message in session.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_close_normally_cancels_active_run_without_dequeuing_the_next_message(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    router = _BlockingRouter(started)
    loop, session, _bus = _runtime(tmp_path, router)
    queued = InboundMessage("remains queued")
    await loop.start()
    await _bus.put_inbound(InboundMessage("active input"))
    await _bus.put_inbound(queued)
    await started.wait()

    await loop.close()
    terminal = (await _terminals(_bus, 1))[0]

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "active input"
    ]
    assert await _bus.inbound_snapshot() == (queued,)
    assert router.calls == ["call"]


@pytest.mark.asyncio
async def test_foreground_frames_once_with_exact_session_inputs_and_atomic_projection(
    tmp_path: Path,
) -> None:
    previous = Blackboard(goal="Previous goal", completion_boundary="Previous boundary")
    staged = Blackboard(goal="Current goal", completion_boundary="Current boundary")
    framing_usage = {
        "model_calls": 1,
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    framer = _BlackboardGeneratorFake(
        (FramingResult(blackboard=staged, usage_delta=framing_usage, status="resolved"),)
    )
    observed: list[Blackboard | None] = []

    async def prepare(
        active: Session,
        current: dict[str, Any],
        blackboard: Blackboard | None,
    ) -> list[dict[str, Any]]:
        del active
        observed.append(blackboard)
        return [
            {"role": "system", "content": "test"},
            {"role": "user", "content": current["content"]},
        ]

    loop, session, _bus = _runtime(
        tmp_path,
        _Router((_response("answer"),)),
        context_preparer_with_blackboard=prepare,
        blackboard_generator=framer,
    )
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )
    session.add_message(
        "assistant",
        "Latest complete assistant content",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    )

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("raw <blackboard> input"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert framer.calls == [
        (previous, "Latest complete assistant content", "raw <blackboard> input")
    ]
    assert observed == [staged]
    assert observed[0] is staged
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "raw <blackboard> input"
    ]
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 6,
        "output_tokens": 3,
        "total_tokens": 9,
    }


@pytest.mark.asyncio
async def test_tool_iterations_reuse_one_framing_and_context_projection_before_one_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = Blackboard(goal="Stable tool goal", completion_boundary="Stable tool boundary")
    framing_usage = {
        "model_calls": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    framer = _BlackboardGeneratorFake(
        (FramingResult(blackboard=staged, usage_delta=framing_usage, status="resolved"),)
    )

    class ToolLoopRouter(_Router):
        def __init__(self) -> None:
            super().__init__(())
            self.requests: list[list[dict[str, Any]]] = []
            self.responses = deque(
                (
                    _response(
                        "use tool",
                        tool_call=ModelToolCall(
                            id="call_stable_projection",
                            name="unknown_tool",
                            arguments="{}",
                        ),
                        input_tokens=2,
                        output_tokens=1,
                    ),
                    _response("tool loop complete", input_tokens=3, output_tokens=2),
                )
            )

        def stream(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[OpenAIToolSchema],
            continuation: ModelContinuation | None = None,
        ) -> AsyncIterator[ModelStreamEvent]:
            del route, tools, continuation
            self.requests.append(deepcopy(list(messages)))
            response = self.responses.popleft()

            async def replay() -> AsyncIterator[ModelStreamEvent]:
                yield ModelCompleted(response=response)

            return replay()

    router = ToolLoopRouter()
    context_calls: list[Blackboard | None] = []
    projected_blackboard = (
        f"## Task goal\n\n{staged.goal}\n\n## Completion boundary\n\n{staged.completion_boundary}"
    )

    async def prepare(
        active: Session,
        current: dict[str, Any],
        blackboard: Blackboard | None,
    ) -> list[dict[str, Any]]:
        del active
        context_calls.append(blackboard)
        return [
            {"role": "system", "content": "test"},
            {
                "role": "user",
                "content": f"{current['content']}\n\n{projected_blackboard}",
            },
        ]

    loop, session, _bus = _runtime(
        tmp_path,
        router,
        context_preparer_with_blackboard=prepare,
        blackboard_generator=framer,
    )
    append_calls = 0
    original_append = session.append_messages

    def append_messages(messages: list[dict[str, Any]], **kwargs: Any) -> None:
        nonlocal append_calls
        append_calls += 1
        assert kwargs["usage_delta"] == framing_usage
        original_append(messages, **kwargs)

    monkeypatch.setattr(session, "append_messages", append_messages)
    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("raw tool input"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert framer.calls == [(None, "", "raw tool input")]
    assert context_calls == [staged]
    assert context_calls[0] is staged
    assert len(router.requests) == 2
    assert all(request[1]["content"].endswith(projected_blackboard) for request in router.requests)
    assert append_calls == 1
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.messages[0]["content"] == "raw tool input"
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 9,
        "output_tokens": 5,
        "total_tokens": 14,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "usage_delta", "expected_framing_usage"),
    [
        (
            "invalid_response",
            {
                "model_calls": 1,
                "input_tokens": 4,
                "output_tokens": 1,
                "total_tokens": 5,
            },
            {
                "model_calls": 1,
                "input_tokens": 4,
                "output_tokens": 1,
                "total_tokens": 5,
            },
        ),
        (
            "model_failed",
            None,
            {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        ),
    ],
)
async def test_invalid_and_model_failed_framing_statuses_fail_open_and_clear_on_commit(
    tmp_path: Path,
    status: str,
    usage_delta: dict[str, int] | None,
    expected_framing_usage: dict[str, int],
) -> None:
    previous = Blackboard(goal="Old goal", completion_boundary="Old boundary")
    framer = _BlackboardGeneratorFake(
        (
            FramingResult(
                blackboard=None,
                usage_delta=usage_delta,
                status=status,  # type: ignore[arg-type]
            ),
        )
    )
    observed: list[Blackboard | None] = []

    async def prepare(
        active: Session,
        current: dict[str, Any],
        blackboard: Blackboard | None,
    ) -> list[dict[str, Any]]:
        del active, current
        observed.append(blackboard)
        return [{"role": "system", "content": "test"}]

    loop, session, _bus = _runtime(
        tmp_path,
        _Router((_response("raw answer"),)),
        context_preparer_with_blackboard=prepare,
        blackboard_generator=framer,
    )
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("continue without parsed framing"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert framer.calls == [(previous, "", "continue without parsed framing")]
    assert observed == [None]
    assert "blackboard" not in session.metadata
    assert session.metadata["token_usage"] == {
        "model_calls": expected_framing_usage["model_calls"] + 1,
        "input_tokens": expected_framing_usage["input_tokens"] + 1,
        "output_tokens": expected_framing_usage["output_tokens"] + 1,
        "total_tokens": expected_framing_usage["total_tokens"] + 2,
    }


@pytest.mark.asyncio
async def test_framing_cancellation_reclaims_first_title_task_without_commit(
    tmp_path: Path,
) -> None:
    framer = BlockingBlackboardGenerator()
    loop, session, _bus = _runtime(
        tmp_path,
        _ConcurrentTitleRouter(),
        blackboard_generator=framer,
        title_prompt="Generate a title",
    )

    await loop.start()
    await _bus.put_inbound(InboundMessage("cancel before the Agent Run"))
    await framer.started.wait()
    await loop.cancel_active_run()
    terminal = (await _terminals(_bus, 1))[0]
    await loop.close()

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert session.messages == []
    assert session.metadata["title"] == "Untitled session"
    assert session.metadata["token_usage"] == {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert not loop._title_work


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("framer", "error_type", "match"),
    [
        (_BlackboardGeneratorFake((RuntimeError("private framing failure"),)), RuntimeError, "private"),
        (_InvalidBlackboardGenerator(), TypeError, "invalid result"),
    ],
)
async def test_framing_contract_errors_propagate_and_reclaim_first_title_task(
    tmp_path: Path,
    framer: _BlackboardGenerator,
    error_type: type[Exception],
    match: str,
) -> None:
    router = _ConcurrentTitleRouter()
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        blackboard_generator=framer,
        title_prompt="Generate a title",
    )
    before_messages = deepcopy(session.messages)
    before_metadata = deepcopy(session.metadata)
    execution_ready = asyncio.Event()

    with pytest.raises(error_type, match=match):
        await loop._execute_foreground(
            InboundMessage("framing implementation error"),
            execution_ready=execution_ready,
        )

    outbound = asyncio.create_task(_bus.get_outbound())
    done, _ = await asyncio.wait((outbound,), timeout=0)
    assert not done
    outbound.cancel()
    await asyncio.gather(outbound, return_exceptions=True)
    assert execution_ready.is_set()
    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert session.metadata["title"] == "Untitled session"
    assert "chat" not in router.calls
    assert not loop._title_work


@pytest.mark.asyncio
async def test_context_failure_after_framing_preserves_previous_blackboard_and_usage(
    tmp_path: Path,
) -> None:
    previous = Blackboard(goal="Previous goal", completion_boundary="Previous boundary")
    staged = Blackboard(goal="Staged goal", completion_boundary="Staged boundary")
    framer = _BlackboardGeneratorFake(
        (
            FramingResult(
                blackboard=staged,
                usage_delta={
                    "model_calls": 1,
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                },
                status="resolved",
            ),
        )
    )

    async def fail_context(
        active: Session,
        current: dict[str, Any],
        blackboard: Blackboard | None,
    ) -> list[dict[str, Any]]:
        del active, current, blackboard
        raise ModelCallError(ErrorInfo("model_failed", "context failed"))

    loop, session, _bus = _runtime(
        tmp_path,
        _Router(()),
        context_preparer_with_blackboard=fail_context,
        blackboard_generator=framer,
    )
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )
    before_usage = session.metadata["token_usage"]

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("context fails after framing"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata["finish_reason"] == "failed"
    assert session.metadata["blackboard"] == {
        "goal": previous.goal,
        "completion_boundary": previous.completion_boundary,
    }
    assert session.metadata["token_usage"] == before_usage
    assert session.messages == []


@pytest.mark.asyncio
async def test_context_cancellation_after_framing_preserves_previous_blackboard_and_usage(
    tmp_path: Path,
) -> None:
    previous = Blackboard(goal="Previous goal", completion_boundary="Previous boundary")
    staged = Blackboard(goal="Staged goal", completion_boundary="Staged boundary")
    framer = _BlackboardGeneratorFake(
        (
            FramingResult(
                blackboard=staged,
                usage_delta={
                    "model_calls": 1,
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                },
                status="resolved",
            ),
        )
    )
    started = asyncio.Event()

    async def block_context(
        active: Session,
        current: dict[str, Any],
        blackboard: Blackboard | None,
    ) -> list[dict[str, Any]]:
        del active, current, blackboard
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    loop, session, _bus = _runtime(
        tmp_path,
        _Router(()),
        context_preparer_with_blackboard=block_context,
        blackboard_generator=framer,
    )
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )
    before_usage = session.metadata["token_usage"]

    await loop.start()
    await _bus.put_inbound(InboundMessage("context cancellation after framing"))
    await started.wait()
    await loop.cancel_active_run()
    terminal = (await _terminals(_bus, 1))[0]
    await loop.close()

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert session.metadata["blackboard"] == {
        "goal": previous.goal,
        "completion_boundary": previous.completion_boundary,
    }
    assert session.metadata["token_usage"] == before_usage
    assert session.messages == []


@pytest.mark.asyncio
async def test_title_usage_is_preserved_when_title_finishes_before_foreground_commit(
    tmp_path: Path,
) -> None:
    staged = Blackboard(goal="Current goal", completion_boundary="Current boundary")
    framing_usage = {
        "model_calls": 1,
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    framer = _BlackboardGeneratorFake(
        (FramingResult(blackboard=staged, usage_delta=framing_usage, status="resolved"),)
    )
    router = _ConcurrentTitleRouter()
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        blackboard_generator=framer,
        title_prompt="Generate a title",
    )

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("title and foreground race"))
        await router.chat_started.wait()
        for _ in range(100):
            if session.metadata["title"] != "Untitled session":
                break
            await asyncio.sleep(0)

        assert session.metadata["title"] == "Generated while chat blocked"
        assert session.metadata["token_usage"] == {
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        }

        router.release_chat.set()
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        router.release_chat.set()
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 7,
        "output_tokens": 4,
        "total_tokens": 11,
    }


@pytest.mark.asyncio
async def test_foreground_commit_preserves_staged_framing_until_slow_title_finishes(
    tmp_path: Path,
) -> None:
    staged = Blackboard(goal="Current goal", completion_boundary="Current boundary")
    framer = _BlackboardGeneratorFake(
        (
            FramingResult(
                blackboard=staged,
                usage_delta={
                    "model_calls": 1,
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                },
                status="resolved",
            ),
        )
    )
    router = _SlowTitleLogRouter()
    loop, session, _bus = _runtime(
        tmp_path,
        router,
        blackboard_generator=framer,
        title_prompt="Generate a title",
    )

    await loop.start()
    await _bus.put_inbound(InboundMessage("foreground commits first"))
    terminal = (await _terminals(_bus, 1))[0]
    await router.title_started.wait()

    assert terminal.metadata == {"_streamed": True}
    assert session.metadata["title"] == "Untitled session"
    assert session.metadata["blackboard"] == {
        "goal": staged.goal,
        "completion_boundary": staged.completion_boundary,
    }
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 6,
        "output_tokens": 3,
        "total_tokens": 9,
    }

    router.release_title.set()
    for _ in range(100):
        if session.metadata["title"] == "Slow title":
            break
        await asyncio.sleep(0)
    await loop.close()

    assert session.metadata["title"] == "Slow title"
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 7,
        "output_tokens": 4,
        "total_tokens": 11,
    }


@pytest.mark.asyncio
async def test_late_title_is_persisted_by_the_next_completed_turn(tmp_path: Path) -> None:
    router = _TitleBehaviorRouter(
        (_response("First response."), _response("Second response.")),
        title=_response("Generated late title"),
        delay_title=True,
    )
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")

    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "First input.")
        await router.title_started.wait()
        await session.wait_for_pending_persist()
        assert Session.load(session.workspace_state, session.session_id).metadata["title"] == (
            "Untitled session"
        )

        router.release_title.set()
        for _ in range(100):
            if session.metadata["title"] == "Generated late title":
                break
            await asyncio.sleep(0)
        assert session.metadata["title"] == "Generated late title"
        assert Session.load(session.workspace_state, session.session_id).metadata["title"] == (
            "Untitled session"
        )

        await collect_foreground_outbound(_bus, "Second input.")
        await session.wait_for_pending_persist()
    finally:
        router.release_title.set()
        await loop.close()

    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == "Generated late title"
    assert [message["role"] for message in reloaded.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_title", ["tool_call", "empty"])
async def test_invalid_title_uses_first_input_fallback_and_keeps_usage(
    tmp_path: Path,
    invalid_title: str,
) -> None:
    title = (
        ModelResponse(
            message=AssistantModelMessage(
                content="Do not use this title",
                tool_calls=(
                    ModelToolCall(
                        id="invalid-title",
                        name="read_file",
                        arguments='{"path":"README.md"}',
                    ),
                ),
            ),
            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            finish_reason="tool_calls",
        )
        if invalid_title == "tool_call"
        else _response('""', input_tokens=3, output_tokens=1)
    )
    router = _TitleBehaviorRouter((_response("First response.", input_tokens=5),), title=title)
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")

    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "  Meaningful first question.  ")
        for _ in range(100):
            if session.metadata["token_usage"]["model_calls"] == 2:
                break
            await asyncio.sleep(0)
    finally:
        await loop.close()

    title_output_tokens = 2 if invalid_title == "tool_call" else 1
    assert session.metadata["title"] == "Meaningful first question."
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 8,
        "output_tokens": 1 + title_output_tokens,
        "total_tokens": 9 + title_output_tokens,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_turn", [False, True])
async def test_close_applies_first_input_title_fallback_before_final_save(
    tmp_path: Path,
    cancel_turn: bool,
) -> None:
    router = _TitleBehaviorRouter(
        (_response("Foreground response."),),
        title=_response("Unreleased title"),
        delay_title=True,
        block_first_foreground=cancel_turn,
    )
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")

    await loop.start()
    if cancel_turn:
        turn = asyncio.create_task(collect_foreground_outbound(_bus, "  Cancelled first title.  "))
        await router.foreground_started.wait()
        await router.title_started.wait()
        await loop.cancel_active_run()
        terminal = (await turn)[-1]
        assert terminal.metadata["finish_reason"] == "cancelled"
        expected_title = "Cancelled first title."
    else:
        await collect_foreground_outbound(_bus, "  Shutdown fallback title.  ")
        await router.title_started.wait()
        expected_title = "Shutdown fallback title."

    await loop.close()

    assert session.metadata["title"] == expected_title
    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == expected_title


@pytest.mark.asyncio
async def test_loop_close_swallows_final_session_failure(tmp_path: Path) -> None:
    loop, session, _bus = _runtime(tmp_path, _Router(()))
    capture = capture_diagnostics()

    def fail_close() -> None:
        raise OSError("private final Session close failure")

    session.close = fail_close  # type: ignore[method-assign]
    try:
        await loop.close()
    finally:
        capture.close()

    assert "Agent Loop Session close failed type=OSError" in capture.event_text
    assert "private final Session close failure" not in capture.event_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "previous", "expected"),
    [
        (
            {"action": "keep", "task_goal": None, "completion_boundary": None},
            Blackboard(goal="Previous goal", completion_boundary="Previous boundary"),
            Blackboard(goal="Previous goal", completion_boundary="Previous boundary"),
        ),
        (
            {
                "action": "replace",
                "task_goal": "Default goal",
                "completion_boundary": "Default boundary",
            },
            None,
            Blackboard(goal="Default goal", completion_boundary="Default boundary"),
        ),
        (
            {"action": "clear", "task_goal": None, "completion_boundary": None},
            Blackboard(goal="Previous goal", completion_boundary="Previous boundary"),
            None,
        ),
    ],
)
async def test_default_agent_loop_wiring_reduces_keep_replace_and_clear(
    tmp_path: Path,
    decision: dict[str, object],
    previous: Blackboard | None,
    expected: Blackboard | None,
) -> None:
    class DefaultRouter(_Router):
        def __init__(self) -> None:
            super().__init__((_response("main answer"),))
            self.direct_calls: list[
                tuple[str, Sequence[dict[str, Any]], Sequence[OpenAIToolSchema]]
            ] = []

        async def complete(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[OpenAIToolSchema],
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            del continuation
            self.direct_calls.append((route, messages, tools))
            return _response(
                json.dumps(decision, separators=(",", ":")),
                input_tokens=5,
                output_tokens=2,
            )

    router = DefaultRouter()
    loop, session, _bus = _runtime(tmp_path, router, use_default_blackboard_generator=True)
    if previous is not None:
        session.update_metadata(
            blackboard={
                "goal": previous.goal,
                "completion_boundary": previous.completion_boundary,
            }
        )

    await loop.start()
    try:
        await _bus.put_inbound(InboundMessage("default wiring input"))
        terminal = (await _terminals(_bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert len(router.direct_calls) == 1
    assert len(router.calls) == 1
    route, messages, tools = router.direct_calls[0]
    assert route == "chat"
    assert tools == ()
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    expected_last_task = json.dumps(
        (
            None
            if previous is None
            else {
                "task_goal": previous.goal,
                "completion_boundary": previous.completion_boundary,
            }
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    system_content = messages[0]["content"]
    assert isinstance(system_content, str)
    assert "### User input\ndefault wiring input" in system_content
    assert f"### Last Task\n```json\n{expected_last_task}\n```" in system_content
    assert system_content.endswith("### Latest assistant content\n")
    if expected is None:
        assert "blackboard" not in session.metadata
    else:
        assert session.metadata["blackboard"] == {
            "goal": expected.goal,
            "completion_boundary": expected.completion_boundary,
        }
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 6,
        "output_tokens": 3,
        "total_tokens": 9,
    }


@pytest.mark.asyncio
async def test_default_agent_loop_wiring_skips_router_completion_for_manual_skill(
    tmp_path: Path,
) -> None:
    class DefaultRouter(_Router):
        def __init__(self) -> None:
            super().__init__((_response("manual answer"),))
            self.direct_calls: list[tuple[str, Sequence[dict[str, Any]]]] = []

        async def complete(
            self,
            route: Literal["chat", "schedule"],
            *,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[OpenAIToolSchema],
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            del tools, continuation
            self.direct_calls.append((route, messages))
            return _response('{"action":"clear","goal":null,"completion_boundary":null}')

    loader = _planner_skill_loader(tmp_path)
    router = DefaultRouter()
    loop, session, bus = _runtime(
        tmp_path,
        router,
        use_default_blackboard_generator=True,
        skill_loader=loader,
    )

    await loop.start()
    try:
        await bus.put_inbound(InboundMessage("/planner do it"))
        terminal = (await _terminals(bus, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert router.direct_calls == []
    assert len(router.calls) == 1
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "/planner do it"
    ]
