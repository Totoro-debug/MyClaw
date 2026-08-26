from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import pytest
from loguru import logger

from myclaw.agent.blackboard import Blackboard, FramingResult, TaskFramingEvaluator
from myclaw.agent.loop import AgentLoop, ConfirmationRequestView
from myclaw.agent.message_bus import InboundMessage, OutboundMessage
from myclaw.agent.runner import AgentRunnerResult, AgentRunnerRouter
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log as real_session_log
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
from myclaw.schedule.service import ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.session.session import Session
from myclaw.skills.catalog import (
    ManualSkillInvocation,
    SkillCatalog,
    build_runtime_skill_snapshot,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall
from tests.fixtures import BlockingTaskFramingEvaluator, DeterministicTaskFramingEvaluator
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


class _FramingFake:
    def __init__(
        self,
        outcomes: Sequence[FramingResult | BaseException] = (),
    ) -> None:
        self._outcomes = deque(outcomes)
        self.calls: list[tuple[Blackboard | None, str, str]] = []

    async def frame(
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


class _InvalidResultFramer:
    async def frame(
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
    task_framer: TaskFramingEvaluator | None = None,
    use_default_task_framer: bool = False,
    title_prompt: str | None = None,
    skill_catalog: SkillCatalog | None = None,
) -> tuple[AgentLoop, Session]:
    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home.path)
    session = Session.create(state, now=_Clock().now)
    schedule = ScheduleService(store=WorkspaceScheduleStore(state), clock=_Clock())
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

    loop = AgentLoop(
        workspace=workspace,
        skill_catalog=skill_catalog,
        session=session,
        schedule_service=schedule,
        model_router=router,
        context_preparer=prepare,
        now=_Clock().now,
        max_iterations=50,
        task_framer=(None if use_default_task_framer else task_framer or _FramingFake()),
        title_prompt=title_prompt,
    )
    return loop, session


async def _context(
    session: Session,
    current_user: dict[str, Any],
) -> list[dict[str, Any]]:
    del session
    return [
        {"role": "system", "content": "test"},
        {"role": "user", "content": current_user["content"]},
    ]


async def _terminals(loop: AgentLoop, count: int) -> list[OutboundMessage]:
    terminals: list[OutboundMessage] = []
    while len(terminals) < count:
        message = await loop.bus.get_outbound()
        if message.metadata.get("_streamed") is True:
            terminals.append(message)
    return terminals


def test_agent_loop_exposes_public_control_seam(tmp_path: Path) -> None:
    loop, _session = _runtime(tmp_path, _Router((_response("unused"),)))

    assert loop.control is loop
    assert loop.control.has_active_run is False
    assert loop.control.has_pending_confirmation is False


@pytest.mark.asyncio
async def test_loop_consumes_foreground_inputs_fifo_and_publishes_one_terminal_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _Router((_response("one"), _response("two"), _response("three")))
    loop, session = _runtime(tmp_path, router)
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
            await loop.bus.put_inbound(InboundMessage(content))

        terminals = await _terminals(loop, 3)

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
    catalog = build_runtime_skill_snapshot(
        agent_home=AgentHome(tmp_path / "agent-home"),
        reserved_names=(),
        enable_always_load=False,
    ).catalog
    router = _Router((_response("Generated title"), _response("Completed")))
    framer = _FramingFake()
    observed: list[tuple[dict[str, Any], ManualSkillInvocation | None]] = []

    async def prepare(
        active_session: Session,
        current_user: dict[str, Any],
        blackboard: Blackboard | None,
        manual_invocation: ManualSkillInvocation | None,
    ) -> list[dict[str, Any]]:
        del active_session, blackboard
        observed.append((deepcopy(current_user), manual_invocation))
        assert manual_invocation is not None
        return [
            {"role": "system", "content": "test"},
            {
                "role": "user",
                "content": f"<skill>{manual_invocation.body}</skill>"
                f"<request>{manual_invocation.request}</request>",
            },
        ]

    loop, session = _runtime(
        tmp_path,
        router,
        context_preparer_with_invocation=prepare,
        task_framer=framer,
        title_prompt="Generate a title",
        skill_catalog=catalog,
    )
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage(raw_input))
        await _terminals(loop, 1)
    finally:
        await loop.close()

    assert len(router.calls) == 2
    assert raw_input in router.calls
    assert f"<skill>{document}</skill><request>Do the work</request>" in router.calls
    assert framer.calls == [(None, "", raw_input)]
    assert observed[0][0] == {"role": "user", "content": raw_input}
    assert observed[0][1] is not None
    assert observed[0][1].body == document
    assert observed[0][1].request == "Do the work"
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        raw_input
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_kind",
    ("missing", "unreadable", "non_utf8", "metadata_mismatch"),
)
async def test_real_manual_skill_file_failures_short_circuit_then_recover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    body = "PRIVATE MANUAL BODY\n"
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\n" + body,
        encoding="utf-8",
    )
    catalog = build_runtime_skill_snapshot(
        agent_home=AgentHome(tmp_path / "agent-home"),
        reserved_names=(),
        enable_always_load=False,
    ).catalog

    if failure_kind == "missing":
        instruction.unlink()
    elif failure_kind == "unreadable":

        def fail_instruction_open(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise PermissionError("injected unreadable Skill")

        monkeypatch.setattr(Path, "open", fail_instruction_open)
    elif failure_kind == "non_utf8":
        instruction.write_bytes(
            b"---\nname: planner\ndescription: Plan work\n---\n" + b"PRIVATE\xffBODY\n"
        )
    else:
        instruction.write_text(
            "---\nname: planner\ndescription: Changed metadata\n---\n" + body,
            encoding="utf-8",
        )

    router = _Router((_response("Recovered title"), _response("After failure")))
    framer = _FramingFake()
    loop, session = _runtime(
        tmp_path,
        router,
        task_framer=framer,
        title_prompt="Generate a title",
        skill_catalog=catalog,
    )
    before_metadata = deepcopy(session.metadata)
    diagnostics = capture_diagnostics()
    try:
        await loop.start()
        await loop.bus.put_inbound(InboundMessage("/planner request"))
        failure = await _terminals(loop, 1)

        assert failure[0].metadata == {
            "finish_reason": "failed",
            "error_code": "skill_unavailable",
            "_streamed": True,
        }
        assert router.calls == []
        assert framer.calls == []
        assert session.messages == []
        assert session.metadata == before_metadata
        assert body not in diagnostics.text

        await loop.bus.put_inbound(InboundMessage("ordinary input"))
        await _terminals(loop, 1)
    finally:
        await loop.close()
        diagnostics.close()

    assert router.calls == ["ordinary input", "ordinary input"]
    assert framer.calls == [(None, "", "ordinary input")]
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "ordinary input"
    ]


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

    loop, session = _runtime(tmp_path, router, context_preparer=prepare)
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("first"))
        await loop.bus.put_inbound(InboundMessage("second"))
        first = await loop.bus.get_outbound()
        while first.metadata.get("_streamed") is not True:
            first = await loop.bus.get_outbound()
        second = await loop.bus.get_outbound()
        while second.metadata.get("_streamed") is not True:
            second = await loop.bus.get_outbound()

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
    loop, session = _runtime(
        tmp_path,
        router,
        task_framer=_FramingFake(
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
        await loop.bus.put_inbound(InboundMessage("failed input"))
        terminal = (await _terminals(loop, 1))[0]

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
    loop, session = _runtime(
        tmp_path,
        router,
        task_framer=_FramingFake(
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
        await loop.bus.put_inbound(InboundMessage("bounded input"))
        terminal = (await _terminals(loop, 1))[0]

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
    loop, session = _runtime(
        tmp_path,
        router,
        task_framer=_FramingFake((framing_result, framing_result)),
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
        await loop.bus.put_inbound(InboundMessage("cancelled input"))
        await loop.bus.put_inbound(InboundMessage("queued input"))
        await started.wait()
        await loop.cancel_active_run()
        terminals = await _terminals(loop, 2)

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
    loop, _session = _runtime(tmp_path, router)
    requests: list[ConfirmationRequestView] = []
    requested = asyncio.Event()

    def on_confirmation(request: ConfirmationRequestView) -> None:
        requests.append(request)
        requested.set()

    loop.bind_confirmation_callback(on_confirmation)
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("confirm this"))
        await requested.wait()

        assert len(requests) == 1
        request = requests[0]
        assert loop.has_pending_confirmation
        with pytest.raises(ValueError, match="late or unknown"):
            loop.respond_to_confirmation(uuid4(), "approved")

        await loop.cancel_active_run()
        terminal = (await _terminals(loop, 1))[0]

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

    loop, session = _runtime(tmp_path, _Router(()), context_preparer=prepare)
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("cancel during preparation"))
        await started.wait()

        await loop.cancel_active_run()
        terminal = (await _terminals(loop, 1))[0]

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
    loop, session = _runtime(
        tmp_path,
        router,
        context_preparer=fail_preparation,
        title_prompt="Generate a title",
    )
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("uncommitted input"))
        _ = await _terminals(loop, 1)
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
    loop, session = _runtime(
        tmp_path,
        router,
        context_preparer=fail_after_waiting,
        title_prompt="Generate a title",
    )
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("uncommitted input"))
        await preparation_started.wait()
        for _ in range(100):
            if "title" in router.calls:
                break
            await asyncio.sleep(0)

        assert router.calls == ["title"]
        assert session.metadata["title"] == "Untitled session"

        release_preparation.set()
        terminal = (await _terminals(loop, 1))[0]

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
async def test_first_message_title_runs_while_foreground_chat_is_blocked(
    tmp_path: Path,
) -> None:
    router = _ConcurrentTitleRouter()
    loop, session = _runtime(tmp_path, router, title_prompt="Generate a title")
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("first input"))
        await router.chat_started.wait()
        for _ in range(100):
            if session.metadata["title"] != "Untitled session":
                break
            await asyncio.sleep(0)

        assert session.metadata["title"] == "Generated while chat blocked"

        router.release_chat.set()
        assert (await _terminals(loop, 1))[0].metadata == {"_streamed": True}
    finally:
        router.release_chat.set()
        await loop.close()


@pytest.mark.asyncio
async def test_slow_title_keeps_one_session_log_owner_across_the_next_fifo_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _SlowTitleLogRouter()
    loop, session = _runtime(tmp_path, router, title_prompt="Generate a title")
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
        await loop.bus.put_inbound(InboundMessage("first input"))
        _ = await _terminals(loop, 1)
        await router.title_started.wait()

        await loop.bus.put_inbound(InboundMessage("second input"))
        _ = await _terminals(loop, 1)

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
    loop, session = _runtime(
        tmp_path,
        _Router(()),
        task_framer=_FramingFake(
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
        await loop.bus.put_inbound(InboundMessage("cannot commit"))
        failed = (await _terminals(loop, 1))[0]
        assert session.messages == before_messages
        assert session.metadata == before_metadata
        await loop.bus.put_inbound(InboundMessage("next input"))
        completed = (await _terminals(loop, 1))[0]
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
    loop, session = _runtime(tmp_path, _Router((_response("committed in memory"),)))

    def fail_persist() -> None:
        raise OSError("private persistence detail")

    monkeypatch.setattr(session, "persist", fail_persist)
    capture = capture_diagnostics()
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("persist fails"))
        terminal = (await _terminals(loop, 1))[0]
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
    loop, first = _runtime(tmp_path, _Router(()))
    closed: list[str] = []
    first_close = first.close

    def close_first() -> None:
        closed.append(first.session_id)
        first_close()

    monkeypatch.setattr(first, "close", close_first)

    await loop.close()
    loop._close_sessions()

    assert closed == [first.session_id]


@pytest.mark.asyncio
async def test_loop_abort_retains_cancelled_owned_tasks_until_cleanup_finishes(
    tmp_path: Path,
) -> None:
    loop, _session = _runtime(tmp_path, _Router(()))
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

    loop.abort()

    assert task in loop._aborted_tasks
    release_cleanup.set()
    await cleanup_finished.wait()
    tasks_drained = asyncio.Event()
    task.add_done_callback(lambda _task: tasks_drained.set())
    await tasks_drained.wait()
    assert loop._aborted_tasks == set()


@pytest.mark.asyncio
async def test_blank_foreground_input_performs_zero_task_framing_attempts(
    tmp_path: Path,
) -> None:
    framer = DeterministicTaskFramingEvaluator()
    loop, session = _runtime(tmp_path, _Router(()), task_framer=framer)
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
    loop, _session = _runtime(tmp_path, _Router((_response("completed without deltas"),)))

    def reject_session_access(_loop: AgentLoop) -> Session:
        raise AssertionError("Terminal adapter must not access Session")

    monkeypatch.setattr(AgentLoop, "session", property(reject_session_access))

    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("foreground input"))
        observed: list[OutboundMessage] = []
        while not observed or observed[-1].metadata.get("_streamed") is not True:
            observed.append(await loop.bus.get_outbound())
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
    loop, session = _runtime(tmp_path, router)
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("sparse output"))
        observed: list[OutboundMessage] = []
        while not observed or observed[-1].metadata.get("_streamed") is not True:
            observed.append(await loop.bus.get_outbound())

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
        loop._close_sessions()


@pytest.mark.asyncio
async def test_close_normally_cancels_active_run_without_dequeuing_the_next_message(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    router = _BlockingRouter(started)
    loop, session = _runtime(tmp_path, router)
    queued = InboundMessage("remains queued")
    await loop.start()
    await loop.bus.put_inbound(InboundMessage("active input"))
    await loop.bus.put_inbound(queued)
    await started.wait()

    await loop.close()
    terminal = (await _terminals(loop, 1))[0]

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "active input"
    ]
    assert await loop.bus.inbound_snapshot() == (queued,)
    assert router.calls == ["call"]
    loop._close_sessions()


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
    framer = _FramingFake(
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

    loop, session = _runtime(
        tmp_path,
        _Router((_response("answer"),)),
        context_preparer_with_blackboard=prepare,
        task_framer=framer,
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
        await loop.bus.put_inbound(InboundMessage("raw <blackboard> input"))
        terminal = (await _terminals(loop, 1))[0]
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
    framer = _FramingFake(
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
    projected_blackboard = json.dumps(
        {"goal": staged.goal, "completion_boundary": staged.completion_boundary},
        ensure_ascii=False,
        separators=(",", ":"),
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
                "content": f"{current['content']}\n<blackboard>{projected_blackboard}</blackboard>",
            },
        ]

    loop, session = _runtime(
        tmp_path,
        router,
        context_preparer_with_blackboard=prepare,
        task_framer=framer,
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
        await loop.bus.put_inbound(InboundMessage("raw tool input"))
        terminal = (await _terminals(loop, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert framer.calls == [(None, "", "raw tool input")]
    assert context_calls == [staged]
    assert context_calls[0] is staged
    assert len(router.requests) == 2
    assert all(
        request[1]["content"].endswith(f"<blackboard>{projected_blackboard}</blackboard>")
        for request in router.requests
    )
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
    framer = _FramingFake(
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

    loop, session = _runtime(
        tmp_path,
        _Router((_response("raw answer"),)),
        context_preparer_with_blackboard=prepare,
        task_framer=framer,
    )
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )

    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("continue without parsed framing"))
        terminal = (await _terminals(loop, 1))[0]
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
    framer = BlockingTaskFramingEvaluator()
    loop, session = _runtime(
        tmp_path,
        _ConcurrentTitleRouter(),
        task_framer=framer,
        title_prompt="Generate a title",
    )

    await loop.start()
    await loop.bus.put_inbound(InboundMessage("cancel before the Agent Run"))
    await framer.started.wait()
    await loop.cancel_active_run()
    terminal = (await _terminals(loop, 1))[0]
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
        (_FramingFake((RuntimeError("private framing failure"),)), RuntimeError, "private"),
        (_InvalidResultFramer(), TypeError, "invalid result"),
    ],
)
async def test_framing_contract_errors_propagate_and_reclaim_first_title_task(
    tmp_path: Path,
    framer: TaskFramingEvaluator,
    error_type: type[Exception],
    match: str,
) -> None:
    router = _ConcurrentTitleRouter()
    loop, session = _runtime(
        tmp_path,
        router,
        task_framer=framer,
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

    outbound = asyncio.create_task(loop.bus.get_outbound())
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
    framer = _FramingFake(
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

    loop, session = _runtime(
        tmp_path,
        _Router(()),
        context_preparer_with_blackboard=fail_context,
        task_framer=framer,
    )
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )
    before_usage = session.metadata["token_usage"]

    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("context fails after framing"))
        terminal = (await _terminals(loop, 1))[0]
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
    framer = _FramingFake(
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

    loop, session = _runtime(
        tmp_path,
        _Router(()),
        context_preparer_with_blackboard=block_context,
        task_framer=framer,
    )
    session.update_metadata(
        blackboard={"goal": previous.goal, "completion_boundary": previous.completion_boundary}
    )
    before_usage = session.metadata["token_usage"]

    await loop.start()
    await loop.bus.put_inbound(InboundMessage("context cancellation after framing"))
    await started.wait()
    await loop.cancel_active_run()
    terminal = (await _terminals(loop, 1))[0]
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
    framer = _FramingFake(
        (FramingResult(blackboard=staged, usage_delta=framing_usage, status="resolved"),)
    )
    router = _ConcurrentTitleRouter()
    loop, session = _runtime(
        tmp_path,
        router,
        task_framer=framer,
        title_prompt="Generate a title",
    )

    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("title and foreground race"))
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
        terminal = (await _terminals(loop, 1))[0]
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
    framer = _FramingFake(
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
    loop, session = _runtime(
        tmp_path,
        router,
        task_framer=framer,
        title_prompt="Generate a title",
    )

    await loop.start()
    await loop.bus.put_inbound(InboundMessage("foreground commits first"))
    terminal = (await _terminals(loop, 1))[0]
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
@pytest.mark.parametrize(
    ("decision", "previous", "expected"),
    [
        (
            {"action": "keep", "goal": None, "completion_boundary": None},
            Blackboard(goal="Previous goal", completion_boundary="Previous boundary"),
            Blackboard(goal="Previous goal", completion_boundary="Previous boundary"),
        ),
        (
            {
                "action": "replace",
                "goal": "Default goal",
                "completion_boundary": "Default boundary",
            },
            None,
            Blackboard(goal="Default goal", completion_boundary="Default boundary"),
        ),
        (
            {"action": "clear", "goal": None, "completion_boundary": None},
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
    loop, session = _runtime(tmp_path, router, use_default_task_framer=True)
    if previous is not None:
        session.update_metadata(
            blackboard={
                "goal": previous.goal,
                "completion_boundary": previous.completion_boundary,
            }
        )

    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("default wiring input"))
        terminal = (await _terminals(loop, 1))[0]
    finally:
        await loop.close()

    assert terminal.metadata == {"_streamed": True}
    assert len(router.direct_calls) == 1
    route, messages, tools = router.direct_calls[0]
    assert route == "chat"
    assert tools == ()
    assert messages[1]["content"] == json.dumps(
        {
            "previous_blackboard": (
                None
                if previous is None
                else {
                    "goal": previous.goal,
                    "completion_boundary": previous.completion_boundary,
                }
            ),
            "last_assistant_content": "",
            "current_user_input": "default wiring input",
        },
        separators=(",", ":"),
    )
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
