from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pytest
from loguru import logger

from myclaw.agent.loop import AgentLoop, ConfirmationRequestView
from myclaw.agent.message_bus import InboundMessage, OutboundMessage
from myclaw.agent.runner import AgentRunnerRouter
from myclaw.agent.workspace import Workspace
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
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall
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


def _response(content: str, *, tool_call: ModelToolCall | None = None) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(
            content=content,
            tool_calls=() if tool_call is None else (tool_call,),
        ),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


def _runtime(
    tmp_path: Path,
    router: AgentRunnerRouter,
    *,
    context_preparer: Callable[[Session, dict[str, Any]], Awaitable[list[dict[str, Any]]]]
    | None = None,
    title_prompt: str | None = None,
) -> tuple[AgentLoop, Session]:
    agent_home = AgentHome(tmp_path / "agent-home")
    agent_home.initialize()
    workspace = Workspace.from_path(tmp_path / "workspace")
    workspace.path.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home.path)
    session = Session.create(state, now=_Clock().now)
    schedule = ScheduleService(store=WorkspaceScheduleStore(state), clock=_Clock())
    prepare = _context if context_preparer is None else context_preparer
    loop = AgentLoop(
        workspace=workspace,
        session=session,
        schedule_service=schedule,
        model_router=router,
        context_preparer=prepare,
        now=_Clock().now,
        max_iterations=50,
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

    def append_messages(active: Session, messages: list[dict[str, Any]]) -> None:
        nonlocal append_calls
        append_calls += 1
        original_append(active, messages)

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
    router = _Router((ModelCallError(ErrorInfo("model_failed", "provider failed")),))
    loop, session = _runtime(tmp_path, router)
    append_calls = 0
    persist_calls = 0
    original_append = Session.append_messages
    original_persist = Session.persist

    def append_messages(active: Session, messages: list[dict[str, Any]]) -> None:
        nonlocal append_calls
        append_calls += 1
        original_append(active, messages)

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
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_loop_commits_max_iteration_repair_once_before_safe_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _MaxRouter()
    loop, session = _runtime(tmp_path, router)
    append_calls = 0
    persist_calls = 0
    original_append = Session.append_messages
    original_persist = Session.persist

    def append_messages(active: Session, messages: list[dict[str, Any]]) -> None:
        nonlocal append_calls
        append_calls += 1
        original_append(active, messages)

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
            "model_calls": 50,
            "input_tokens": 50,
            "output_tokens": 50,
            "total_tokens": 100,
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
    loop, session = _runtime(tmp_path, router)
    append_calls = 0
    persist_calls = 0
    original_append = Session.append_messages
    original_persist = Session.persist

    def append_messages(active: Session, messages: list[dict[str, Any]]) -> None:
        nonlocal append_calls
        append_calls += 1
        original_append(active, messages)

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
    loop, session = _runtime(
        tmp_path,
        _Router((_response("not committed"), _response("next completed"))),
    )
    append_calls = 0
    original_append = session.append_messages

    def fail_append(messages: list[dict[str, Any]]) -> None:
        nonlocal append_calls
        append_calls += 1
        if append_calls == 1:
            raise OSError("private append detail")
        original_append(messages)

    monkeypatch.setattr(session, "append_messages", fail_append)
    await loop.start()
    try:
        await loop.bus.put_inbound(InboundMessage("cannot commit"))
        failed = (await _terminals(loop, 1))[0]
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
async def test_loop_closes_every_session_it_owned_after_idle_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, first = _runtime(tmp_path, _Router(()))
    second = Session.create(first.workspace_state, now=_Clock().now)
    closed: list[str] = []
    first_close = first.close
    second_close = second.close

    def close_first() -> None:
        closed.append(first.session_id)
        first_close()

    def close_second() -> None:
        closed.append(second.session_id)
        second_close()

    monkeypatch.setattr(first, "close", close_first)
    monkeypatch.setattr(second, "close", close_second)

    loop.switch_session(second)
    await loop.close()
    loop._close_sessions()

    assert closed == [first.session_id, second.session_id]


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
