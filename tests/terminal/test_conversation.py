from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar, Literal, cast
from uuid import UUID, uuid4
from xml.etree import ElementTree

import pytest
from textual.app import App
from textual.driver import Driver
from textual.events import (
    Key,
    MouseDown,
    MouseMove,
    MouseScrollDown,
    MouseScrollRight,
    MouseScrollUp,
    MouseUp,
    Paste,
)
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Button, Markdown, OptionList, Static, TextArea

from myclaw.agent.loop import ConfirmationRequestView, ForegroundConversationProjection
from myclaw.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import PreparedRuntime, RuntimeHost
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.management.commands import ManagementCommandResult
from myclaw.provider.models import (
    ModelCompleted,
    ModelContinuation,
    ModelStreamEvent,
    ReasoningDelta,
    ReasoningEffort,
    TextDelta,
)
from myclaw.session.session import Session
from myclaw.terminal.conversation import (
    TerminalConversationApp,
    TerminalConversationError,
    _format_activity_duration,
    run_terminal_conversation,
)
from myclaw.terminal.conversation import (
    _MessageBusRunProjection as _AgentRunProjection,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
)
from myclaw.utils.json_types import JsonObject
from tests.agent.test_fixed_catalog_runtime import (
    _BlockingClock,
    _response,
    _RuntimeProvider,
)
from tests.agent.test_fixed_catalog_runtime import (
    _runtime as _prepared_runtime,
)
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ProviderCall

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
TURN_ID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
OTHER_TURN_ID = UUID("b3fbe13b-3d1f-4ee6-98e5-dc40a2c4be1c")


@dataclass(frozen=True, slots=True)
class _ResponseSegmentScript:
    content: str


type _ScriptItem = OutboundMessage | ConfirmationRequest | _ResponseSegmentScript


class _ScriptedSource:
    def run(self, text: str) -> AsyncIterator[_ScriptItem]:
        raise NotImplementedError

    async def cancel_active_turn(self) -> bool:
        return False


class ScriptedRunSource(_ScriptedSource):
    def __init__(
        self,
        *,
        pause_before_output: bool = False,
        pause_after_first_delta: bool = False,
        deltas: tuple[str, ...] = ("First ", "answer."),
        completed_content: str = "First answer.",
        deltas_by_submission: tuple[tuple[str, ...], ...] | None = None,
        completed_contents: tuple[str, ...] | None = None,
        outcomes: tuple[Literal["completed", "cancelled", "failed"], ...] = ("completed",),
        cancelled_content: str = "",
        failure_message: str = "The turn failed.",
    ) -> None:
        self.submissions: list[str] = []
        self._deltas = deltas
        self._completed_content = completed_content
        self._deltas_by_submission = deltas_by_submission
        self._completed_contents = completed_contents
        self._outcomes = outcomes
        self._cancelled_content = cancelled_content
        self._failure_message = failure_message
        self.before_output = asyncio.Event()
        self.first_delta_emitted = asyncio.Event()
        self._continue_to_output = asyncio.Event()
        self._continue_after_first_delta = asyncio.Event()
        if not pause_before_output:
            self._continue_to_output.set()
        if not pause_after_first_delta:
            self._continue_after_first_delta.set()

    def continue_to_output(self) -> None:
        self._continue_to_output.set()

    def continue_turn(self) -> None:
        self._continue_after_first_delta.set()

    def pause_after_next_first_delta(self) -> None:
        self.first_delta_emitted.clear()
        self._continue_after_first_delta.clear()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        submission_index = len(self.submissions) - 1
        outcome = self._outcomes[min(submission_index, len(self._outcomes) - 1)]
        deltas = (
            self._deltas
            if self._deltas_by_submission is None
            else self._deltas_by_submission[
                min(submission_index, len(self._deltas_by_submission) - 1)
            ]
        )
        completed_content = (
            self._completed_content
            if self._completed_contents is None
            else self._completed_contents[min(submission_index, len(self._completed_contents) - 1)]
        )
        self.before_output.set()
        await self._continue_to_output.wait()
        for output_index, delta in enumerate(deltas, start=1):
            yield _response_delta(delta)
            if output_index == 1:
                self.first_delta_emitted.set()
                await self._continue_after_first_delta.wait()
        if outcome == "completed":
            if not deltas and completed_content:
                yield _response_delta(completed_content)
            yield OutboundMessage("model_response", "", {"_stream_end": True})
            yield _completed_response()
        elif outcome == "cancelled":
            yield _cancelled_response(self._cancelled_content)
        else:
            yield _failed_response(self._failure_message)


class ToolActivityRunSource(_ScriptedSource):
    def __init__(
        self,
        *,
        start_summary: str = "Running read_file",
    ) -> None:
        self.submissions: list[str] = []
        self.tool_started = asyncio.Event()
        self.complete_tool = asyncio.Event()
        self._start_summary = start_summary

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        yield _tool_call("call-read-file", "read_file", self._start_summary)
        self.tool_started.set()
        await self.complete_tool.wait()
        yield _completed_response()


class ConfirmationRunSource(_ScriptedSource):
    def __init__(
        self,
        *,
        tool_name: str = "read_file",
        details: dict[str, str | int] | None = None,
        reason: str = "The requested path is outside the current Workspace.",
        warnings: tuple[str, ...] = (),
    ) -> None:
        self.submissions: list[str] = []
        self.responses: list[tuple[UUID, ConfirmationDecision]] = []
        self.cancel_calls = 0
        self.confirmation_requested = asyncio.Event()
        self._decision_received = asyncio.Event()
        self._request = ConfirmationRequest(
            confirmation_id=UUID("16fd2706-8baf-4334-8c7f-ada847da0314"),
            tool_call_id="call-confirm",
            tool_name=tool_name,
            summary=f"Confirm {tool_name}",
            details=({"path": "outside.txt"} if details is None else cast(JsonObject, details)),
            warnings=warnings,
            reason=reason,
        )

    async def run(self, text: str) -> AsyncIterator[OutboundMessage | ConfirmationRequest]:
        self.submissions.append(text)
        yield _tool_call("call-confirm", self._request.tool_name, "Running")
        self.confirmation_requested.set()
        yield self._request
        await self._decision_received.wait()
        decision = self.responses[-1][1]
        del decision
        yield _completed_response()

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        assert confirmation_id == self._request.confirmation_id
        self.responses.append((confirmation_id, decision))
        self._decision_received.set()

    async def cancel_active_turn(self) -> bool:
        self.cancel_calls += 1
        return False


class DuplicateLateConfirmationRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.responses: list[tuple[UUID, ConfirmationDecision]] = []
        self.confirmation_requested = asyncio.Event()
        self.request = ConfirmationRequest(
            confirmation_id=UUID("4c9b7d40-8b5d-4a17-8140-0ce4f3511ab1"),
            tool_call_id="call-stale",
            tool_name="read_file",
            summary="Confirm read_file",
            details={"path": "outside.txt"},
            reason="The path resolves outside the current Workspace.",
        )

    async def run(self, text: str) -> AsyncIterator[OutboundMessage | ConfirmationRequest]:
        del text
        self.confirmation_requested.set()
        yield self.request
        yield self.request
        yield _completed_response()

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        self.responses.append((confirmation_id, decision))
        raise ValueError("Confirmation response is late or unknown")

    async def cancel_active_turn(self) -> bool:
        return False


class MultipleConfirmationRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.responses: list[tuple[UUID, ConfirmationDecision]] = []
        self.confirmation_requested = (asyncio.Event(), asyncio.Event())
        self.requests = (
            ConfirmationRequest(
                confirmation_id=UUID("b378d47d-2d73-4670-badc-844245c63c3d"),
                tool_call_id="call-first",
                tool_name="read_file",
                summary="Confirm read_file",
                details={"path": "first.txt"},
                reason="First confirmation.",
            ),
            ConfirmationRequest(
                confirmation_id=UUID("3f1bb452-a8cf-4760-9cde-77b6a4b80ae9"),
                tool_call_id="call-second",
                tool_name="write_file",
                summary="Confirm write_file",
                details={"path": "second.txt", "content": "private"},
                reason="Second confirmation.",
            ),
        )

    async def run(self, text: str) -> AsyncIterator[OutboundMessage | ConfirmationRequest]:
        del text
        for requested, request in zip(self.confirmation_requested, self.requests, strict=True):
            requested.set()
            yield request
        yield _completed_response()

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        self.responses.append((confirmation_id, decision))

    async def cancel_active_turn(self) -> bool:
        return False


class ToolMessageSequenceRunSource(_ScriptedSource):
    def __init__(
        self,
        *sequences: tuple[_ScriptItem, ...],
        pause_before: bool = False,
        pause_after: tuple[int, int] | None = None,
    ) -> None:
        self.submissions: list[str] = []
        self._sequences = sequences
        self._pause_before = pause_before
        self._pause_after = pause_after
        self.paused = asyncio.Event()
        self.continue_events = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        submission_index = len(self.submissions) - 1
        sequence = self._sequences[submission_index]
        response_delta_seen = False
        if self._pause_before:
            self.paused.set()
            await self.continue_events.wait()
        for message_index, script_item in enumerate(sequence):
            messages, response_delta_seen = _expand_script_item(script_item, response_delta_seen)
            for message in messages:
                yield message
            if self._pause_after == (submission_index, message_index):
                self.paused.set()
                await self.continue_events.wait()


class StagedToolMessageSequenceRunSource(_ScriptedSource):
    def __init__(
        self,
        *sequences: tuple[_ScriptItem, ...],
        pause_after: tuple[tuple[int, int], ...],
    ) -> None:
        self.submissions: list[str] = []
        self._sequences = sequences
        self._checkpoints = {
            checkpoint: (asyncio.Event(), asyncio.Event()) for checkpoint in pause_after
        }

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        submission_index = len(self.submissions) - 1
        response_delta_seen = False
        for message_index, script_item in enumerate(self._sequences[submission_index]):
            messages, response_delta_seen = _expand_script_item(script_item, response_delta_seen)
            for message in messages:
                yield message
            checkpoint = self._checkpoints.get((submission_index, message_index))
            if checkpoint is not None:
                reached, continue_events = checkpoint
                reached.set()
                await continue_events.wait()

    async def wait_after(self, submission_index: int, message_index: int) -> None:
        reached, _ = self._checkpoints[(submission_index, message_index)]
        await asyncio.wait_for(reached.wait(), timeout=1)

    def continue_after(self, submission_index: int, message_index: int) -> None:
        _, continue_events = self._checkpoints[(submission_index, message_index)]
        continue_events.set()

    def continue_all(self) -> None:
        for _, continue_events in self._checkpoints.values():
            continue_events.set()


class CancellableRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.submissions: list[str] = []
        self.cancel_calls = 0
        self.first_delta_emitted = asyncio.Event()
        self._active_task: asyncio.Task[object] | None = None

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        self.submissions.append(text)
        active_task = asyncio.current_task()
        assert active_task is not None
        self._active_task = active_task
        try:
            if len(self.submissions) == 1:
                yield _response_delta("partial response")
                self.first_delta_emitted.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    yield _cancelled_response("partial response")
                    return
            else:
                yield _response_delta("Recovered response.")
                yield _completed_response()
        finally:
            self._active_task = None

    async def cancel_active_turn(self) -> bool:
        self.cancel_calls += 1
        if self._active_task is not None:
            self._active_task.cancel()
        return True


class CancellableRuntimeProvider(_RuntimeProvider):
    def __init__(self) -> None:
        super().__init__(())
        self.first_delta_emitted = asyncio.Event()
        self._chat_calls = 0

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
            async for event in super().stream(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            ):
                yield event
            return
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
        self._chat_calls += 1
        if self._chat_calls == 1:
            yield TextDelta(delta="partial runtime response")
            self.first_delta_emitted.set()
            await asyncio.Event().wait()
            return
        yield ModelCompleted(response=_response(content="Recovered runtime response."))


class FailingRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.closed = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            yield _response_delta("partial")
        finally:
            self.closed.set()


class ExplodingRunSource(_ScriptedSource):
    def __init__(self, *, include_tool: bool = True) -> None:
        self.closed = asyncio.Event()
        self.include_tool = include_tool

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            yield _response_delta("Unconfirmed candidate.")
            if self.include_tool:
                yield _tool_call("exploding-call", "read_file", "Running read_file")
            raise RuntimeError("event stream failed")
        finally:
            self.closed.set()


class TerminalThenBlockingRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.terminal_emitted = asyncio.Event()
        self.closed = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            yield _response_delta("process activity")
            yield _tool_call("unfinished-call", "read_file", "Running read_file")
            self.terminal_emitted.set()
            yield _response_delta("final response")
            yield _completed_response()
            await asyncio.Event().wait()
        finally:
            self.closed.set()


class BlockingRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            self.started.set()
            yield _response_delta("partial")
            await asyncio.Event().wait()
        finally:
            self.closed.set()


class FailingCancellationCleanupRunSource(_ScriptedSource):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, text: str) -> AsyncIterator[OutboundMessage]:
        del text
        try:
            self.started.set()
            yield _response_delta("partial")
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("worker cleanup failed")


class FailingMarkdownStream:
    async def write(self, markdown_fragment: str) -> None:
        del markdown_fragment
        raise RuntimeError("markdown write failed")

    async def stop(self) -> None:
        raise RuntimeError("markdown stop failed")


class _ScriptedControl:
    """Final AgentLoop control surface for a scripted MessageBus runtime."""

    def __init__(self, source: _ScriptedSource) -> None:
        self._source = source
        self._active = False
        self._pending_confirmation: UUID | None = None
        self._callback: Callable[[ConfirmationRequestView], None] | None = None
        self._seen_confirmation_ids: set[UUID] = set()
        self._confirmation_released: asyncio.Event | None = None
        self._bridge: asyncio.Task[None] | None = None

    @property
    def has_active_run(self) -> bool:
        return self._active

    @property
    def has_pending_confirmation(self) -> bool:
        return self._pending_confirmation is not None

    def bind_confirmation_callback(
        self, callback: Callable[[ConfirmationRequestView], None]
    ) -> None:
        self._callback = callback

    async def cancel_active_run(self) -> None:
        handled = await self._source.cancel_active_turn()
        if handled:
            return
        bridge = self._bridge
        if bridge is not None and not bridge.done():
            bridge.cancel()
            await asyncio.gather(bridge, return_exceptions=True)

    def project_foreground_conversation(self) -> ForegroundConversationProjection:
        return ForegroundConversationProjection(session_id="test-session", messages=())

    def respond_to_confirmation(
        self, confirmation_id: UUID, decision: ConfirmationDecision
    ) -> None:
        try:
            respond = getattr(self._source, "respond_to_confirmation", None)
            if callable(respond):
                cast(Callable[[UUID, ConfirmationDecision], None], respond)(
                    confirmation_id, decision
                )
        finally:
            if self._pending_confirmation == confirmation_id:
                self._pending_confirmation = None
                released = self._confirmation_released
                self._confirmation_released = None
                if released is not None:
                    released.set()

    async def publish(
        self,
        item: _ScriptItem,
        bus: MessageBus,
    ) -> bool:
        if isinstance(item, ConfirmationRequest):
            if item.confirmation_id in self._seen_confirmation_ids:
                return False
            self._seen_confirmation_ids.add(item.confirmation_id)
            self._pending_confirmation = item.confirmation_id
            released = asyncio.Event()
            self._confirmation_released = released
            callback = self._callback
            if callback is None:
                raise AssertionError("confirmation callback was not bound")
            callback(item)
            await released.wait()
            return False
        if isinstance(item, _ResponseSegmentScript):
            raise TypeError("response segment scripts must be expanded by a run source")
        await bus.put_outbound(item)
        if item.type == "model_response":
            return item.metadata.get("_streamed") is True
        if item.type == "system_control":
            return item.metadata.get("_streamed") is True
        return False


class FakePreparedRuntime:
    def __init__(self, source: _ScriptedSource) -> None:
        self.source = source
        self.bus = MessageBus()
        self.inbound_history: list[InboundMessage] = []
        self.outbound_history: list[OutboundMessage] = []
        put_outbound = self.bus.put_outbound

        async def record_outbound(message: OutboundMessage) -> None:
            self.outbound_history.append(message)
            await put_outbound(message)

        self.bus.put_outbound = record_outbound  # type: ignore[method-assign]
        self.control = _ScriptedControl(source)
        self.management_dispatcher: object | None = None
        self.session_id = "test-session"
        self.start_calls = 0
        self.close_calls = 0
        self._closing = False
        self._bridge: asyncio.Task[None] | None = None

    async def _consume_inbound(self) -> None:
        while not self._closing:
            message = await self.bus.get_inbound()
            self.inbound_history.append(message)
            self.control._active = True
            self.control._bridge = asyncio.current_task()
            terminal = False
            try:
                async for item in self.source.run(message.content):
                    terminal = await self.control.publish(item, self.bus) or terminal
            except asyncio.CancelledError:
                if self._closing:
                    raise
                if not terminal:
                    await self.bus.put_outbound(
                        OutboundMessage(
                            type="system_control",
                            content="Turn cancelled.",
                            metadata={"_streamed": True, "finish_reason": "cancelled"},
                        )
                    )
                terminal = True
            except Exception as error:
                if not terminal:
                    await self.bus.put_outbound(
                        OutboundMessage(
                            type="system_control",
                            content=str(error),
                            metadata={"_streamed": True, "finish_reason": "failed"},
                        )
                    )
                terminal = True
            finally:
                self.control._active = False
                self.control._bridge = None
            if not terminal:
                await self.bus.put_outbound(
                    OutboundMessage(
                        type="system_control",
                        content="Turn failed.",
                        metadata={"_streamed": True, "finish_reason": "failed"},
                    )
                )

    async def start(self) -> None:
        self.start_calls += 1
        self._bridge = asyncio.create_task(self._consume_inbound())

    def bind_confirmation_callback(
        self, callback: Callable[[ConfirmationRequestView], None]
    ) -> None:
        self.control.bind_confirmation_callback(callback)

    async def close(self) -> None:
        self.close_calls += 1
        self._closing = True
        if self._bridge is not None:
            self._bridge.cancel()
            with suppress(asyncio.CancelledError):
                await self._bridge

    def session_messages(self) -> tuple[dict[str, object], ...]:
        return ()


class _DirectControl:
    """Minimal public AgentLoop control seam for direct MessageBus UI tests."""

    def __init__(self) -> None:
        self.confirmation_callback: Callable[[ConfirmationRequestView], None] | None = None

    @property
    def has_active_run(self) -> bool:
        return False

    @property
    def has_pending_confirmation(self) -> bool:
        return False

    async def cancel_active_run(self) -> None:
        return None

    def project_foreground_conversation(self) -> ForegroundConversationProjection:
        return ForegroundConversationProjection(session_id="direct-test", messages=())

    def bind_confirmation_callback(
        self,
        callback: Callable[[ConfirmationRequestView], None],
    ) -> None:
        self.confirmation_callback = callback

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        del confirmation_id, decision
        raise ValueError("Confirmation response is late or unknown")


class KeyboardLifecycleDriver(Driver):
    operations: ClassVar[list[tuple[str, str]]] = []

    def write(self, data: str) -> None:
        self.operations.append(("write", data))

    def flush(self) -> None:
        self.operations.append(("flush", ""))

    def start_application_mode(self) -> None:
        self.write("application:start")
        self.write("\x1b[>1u")
        self.flush()

    def disable_input(self) -> None:
        self.operations.append(("disable_input", ""))

    def stop_application_mode(self) -> None:
        self.write("\x1b[<u")
        self.write("application:stop")
        self.flush()

    def close(self) -> None:
        self.operations.append(("close", ""))


class CloseOrderingRuntime(FakePreparedRuntime):
    def __init__(self, source: BlockingRunSource) -> None:
        super().__init__(source)
        self._blocking_source = source
        self.close_saw_stream_closed = False

    async def close(self) -> None:
        await super().close()
        self.close_saw_stream_closed = self._blocking_source.closed.is_set()


class FailingStartRuntime(FakePreparedRuntime):
    async def start(self) -> None:
        self.start_calls += 1
        raise RuntimeError("runtime startup failed")


class FailingCloseRuntime(FakePreparedRuntime):
    async def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("runtime cleanup failed")


def _runtime(
    source: _ScriptedSource,
    management_dispatcher: object | None = None,
) -> FakePreparedRuntime:
    runtime = FakePreparedRuntime(source)
    runtime.management_dispatcher = management_dispatcher
    return runtime


def _terminal_app(
    runtime: PreparedRuntime | RuntimeHost,
    *,
    app_type: type[TerminalConversationApp] = TerminalConversationApp,
    monotonic: Callable[[], float] | None = None,
) -> TerminalConversationApp:
    runtime_host = runtime if isinstance(runtime, RuntimeHost) else None
    if monotonic is None:
        return app_type(
            bus=runtime.bus,
            control=runtime.control,
            management_dispatcher=runtime.management_dispatcher,
            start_runtime=runtime.start,
            close_runtime=runtime.close,
            runtime_host=runtime_host,
        )
    return app_type(
        bus=runtime.bus,
        control=runtime.control,
        management_dispatcher=runtime.management_dispatcher,
        start_runtime=runtime.start,
        close_runtime=runtime.close,
        monotonic=monotonic,
        runtime_host=runtime_host,
    )


class _GenerationHost(RuntimeHost):
    @property
    def session(self) -> Session:
        return self.generation.session


def _generation_host(
    agent_home: Path,
    workspace: Path,
    provider: _RuntimeProvider,
) -> _GenerationHost:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    return _GenerationHost(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        memory_scheduler_clock=_BlockingClock(),
        schedule_scheduler_clock=_BlockingClock(),
    )


def _tool_call(
    tool_call_id: str,
    tool_name: str,
    arguments: str,
) -> OutboundMessage:
    return OutboundMessage(
        type="tool_call",
        content=tool_name,
        metadata={"tool_call_id": tool_call_id, "arguments": arguments},
    )


def _response_delta(delta: str) -> OutboundMessage:
    return OutboundMessage(
        type="model_response",
        content=delta,
        metadata={"_stream_delta": True},
    )


def _response_segment(content: str) -> _ResponseSegmentScript:
    return _ResponseSegmentScript(content=content)


def _completed_response() -> OutboundMessage:
    return OutboundMessage(
        type="model_response",
        content="",
        metadata={"_streamed": True},
    )


def _cancelled_response(partial_content: str) -> OutboundMessage:
    return OutboundMessage(
        type="system_control",
        content=partial_content,
        metadata={"_streamed": True, "finish_reason": "cancelled"},
    )


def _failed_response(error_message: str) -> OutboundMessage:
    return OutboundMessage(
        type="system_control",
        content=error_message,
        metadata={"_streamed": True, "finish_reason": "failed"},
    )


def _expand_script_item(
    script_item: _ScriptItem,
    response_delta_seen: bool,
) -> tuple[tuple[OutboundMessage, ...], bool]:
    if isinstance(script_item, _ResponseSegmentScript):
        messages: list[OutboundMessage] = []
        if script_item.content and not response_delta_seen:
            messages.append(
                OutboundMessage("model_response", script_item.content, {"_stream_delta": True})
            )
        messages.append(OutboundMessage("model_response", "", {"_stream_end": True}))
        return tuple(messages), False
    if isinstance(script_item, ConfirmationRequest):
        raise TypeError("confirmation scripts must be consumed by the run source")
    is_delta = (
        script_item.type == "model_response" and script_item.metadata.get("_stream_delta") is True
    )
    return (script_item,), is_delta


def _tool_row_texts(app: TerminalConversationApp) -> list[str]:
    rows = app.query(".tool-row")
    assert all(isinstance(row, Static) for row in rows)
    return [str(cast(Static, row).content) for row in rows]


def _visible_screen_text(app: TerminalConversationApp) -> str:
    text = "".join(element.text or "" for element in _screenshot_text_elements(app))
    return text.replace("\xa0", " ")


def _screenshot(app: TerminalConversationApp) -> ElementTree.Element:
    return ElementTree.fromstring(app.export_screenshot(simplify=True))


def _screenshot_text_elements(app: TerminalConversationApp) -> list[ElementTree.Element]:
    return [
        element
        for element in _screenshot(app).iter()
        if element.tag.endswith("text")
        and element.text
        and not element.attrib.get("class", "").endswith("-title")
    ]


def _content_text_nodes(app: TerminalConversationApp, marker: str) -> list[str]:
    return [
        (element.text or "").replace("\xa0", " ")
        for element in _screenshot_text_elements(app)
        if all(character in marker for character in (element.text or "").replace("\xa0", " "))
    ]


def _screenshot_text_nodes(app: TerminalConversationApp) -> list[tuple[str, float, float]]:
    return [
        (
            (element.text or "").replace("\xa0", " "),
            float(element.attrib["x"]),
            float(element.attrib["y"]),
        )
        for element in _screenshot_text_elements(app)
    ]


def _screenshot_width(app: TerminalConversationApp) -> float:
    return float(_screenshot(app).attrib["viewBox"].split()[2])


async def _wait_for_turn(app: TerminalConversationApp) -> None:
    text_area = app.query_one("#conversation-input", TextArea)
    async with asyncio.timeout(2):
        while (
            text_area.read_only
            or getattr(text_area, "active_turn_token", None) is not None
            or app._active_run_projection is not None
        ):
            await asyncio.sleep(0)
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()


def _constant_datetime(value: datetime) -> Callable[[], datetime]:
    return lambda: value


def _constant_uuid(value: UUID) -> Callable[[], UUID]:
    return lambda: value


async def _wait_for_session_picker(
    app: TerminalConversationApp,
    pilot: Pilot[None],
) -> None:
    async with asyncio.timeout(2):
        while app.screen.id != "session-picker" or not app.screen.query("#session-picker-options"):
            await pilot.pause()


async def _wait_for_confirmation(app: TerminalConversationApp, pilot: Pilot[None]) -> None:
    async with asyncio.timeout(5):
        while (
            app.screen.id is None
            or not app.screen.id.startswith("confirmation-")
            or not app.screen.query(".confirmation-details")
        ):
            await pilot.pause()
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()


@pytest.mark.asyncio
async def test_terminal_conversation_starts_blank_and_focuses_input() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)):
        visible_text = _visible_screen_text(app)

        assert "Message MyClaw" in visible_text
        assert "Welcome" not in visible_text
        assert "Session" not in visible_text
        assert "model" not in visible_text.casefold()
        assert isinstance(app.screen.focused, TextArea)
        assert runtime.start_calls == 1

    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_up_drain_keeps_an_inbound_consumed_during_the_atomic_drain() -> None:
    class CoordinatedDrainBus(MessageBus):
        def __init__(self) -> None:
            super().__init__()
            self.drain_started = asyncio.Event()
            self.allow_drain = asyncio.Event()

        async def drain_inbound(self) -> tuple[InboundMessage, ...]:
            self.drain_started.set()
            await self.allow_drain.wait()
            return await super().drain_inbound()

    bus = CoordinatedDrainBus()
    control = _DirectControl()

    async def no_op() -> None:
        return None

    app = TerminalConversationApp(
        bus=bus,
        control=control,
        management_dispatcher=None,
        start_runtime=no_op,
        close_runtime=no_op,
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first"), "enter")
        await pilot.press(*list("second"), "enter")

        drain = asyncio.create_task(pilot.press("up"))
        await asyncio.wait_for(bus.drain_started.wait(), timeout=1)
        consumed = await bus.get_inbound()
        assert consumed == InboundMessage(content="first")
        bus.allow_drain.set()
        await asyncio.wait_for(drain, timeout=1)

        await bus.put_outbound(
            OutboundMessage(
                type="model_response",
                content="done",
                metadata={"_streamed": True},
            )
        )
        async with asyncio.timeout(1):
            while not app.query(".user-message"):
                await pilot.pause()

        assert app.query_one("#conversation-input", TextArea).text == "second"
        user_messages = [
            str(cast(Static, message).content) for message in app.query(".user-message")
        ]
        assert user_messages == ["first"]


@pytest.mark.asyncio
async def test_sparse_protocol_errors_and_duplicate_terminal_do_not_poison_next_run() -> None:
    bus = MessageBus()
    control = _DirectControl()

    async def no_op() -> None:
        return None

    app = TerminalConversationApp(
        bus=bus,
        control=control,
        management_dispatcher=None,
        start_runtime=no_op,
        close_runtime=no_op,
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("malformed"), "enter")
        assert await bus.get_inbound() == InboundMessage(content="malformed")
        await bus.put_outbound(
            OutboundMessage(
                type="tool_call",
                content="read_file",
                metadata={"tool_call_id": "call-malformed"},
            )
        )
        async with asyncio.timeout(1):
            while "Turn failed." not in _visible_screen_text(app):
                await pilot.pause()

        await bus.put_outbound(
            OutboundMessage(
                type="system_control",
                content="duplicate",
                metadata={"_streamed": True, "finish_reason": "failed"},
            )
        )
        await pilot.pause()

        await pilot.press(*list("next"), "enter")
        assert await bus.get_inbound() == InboundMessage(content="next")
        await bus.put_outbound(
            OutboundMessage(
                type="model_response",
                content="next answer",
                metadata={"_stream_delta": True},
            )
        )
        await bus.put_outbound(
            OutboundMessage(
                type="model_response",
                content="",
                metadata={"_streamed": True},
            )
        )
        async with asyncio.timeout(1):
            while "next answer" not in _visible_screen_text(app):
                await pilot.pause()

        assert _visible_screen_text(app).count("next answer") == 1


@pytest.mark.asyncio
async def test_terminal_conversation_inherits_the_terminal_background() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)):
        input_area = app.screen.focused

        assert isinstance(input_area, TextArea)
        assert app.screen.styles.background.a == 0
        assert input_area.styles.background.a == 0


@pytest.mark.asyncio
async def test_nonblank_enter_echoes_user_before_consuming_agent_events() -> None:
    app: TerminalConversationApp
    conversation = ScriptedRunSource(pause_before_output=True)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press("h", "i", "enter"))
        try:
            await asyncio.wait_for(conversation.before_output.wait(), timeout=1)
            async with asyncio.timeout(1):
                while "hi" not in _visible_screen_text(app):
                    await pilot.pause()

            assert conversation.submissions == ["hi"]
            assert "hi" in _visible_screen_text(app)
            assert runtime.inbound_history == [InboundMessage(content="hi")]
        finally:
            conversation.continue_to_output()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_active_turn_keeps_input_editable_and_cancellable_before_a_later_turn() -> None:
    conversation = CancellableRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("first"), "enter"))
        await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
        await asyncio.sleep(0)

        input_area = app.query_one("#conversation-input", TextArea)
        working = app.query_one("#turn-status", Static)
        assert not input_area.read_only
        async with asyncio.timeout(1):
            while not working.display:
                await asyncio.sleep(0)
        assert working.display

        for text in ("queued one", "queued two", "queued three"):
            await pilot.press(*list(text), "enter")
        assert conversation.submissions == ["first"]
        assert "Pending (3): queued one | queued two | queued three" in _visible_screen_text(app)

        await pilot.press("ctrl+c")
        await asyncio.wait_for(submission, timeout=1)
        async with asyncio.timeout(1):
            while conversation.submissions != [
                "first",
                "queued one",
                "queued two",
                "queued three",
            ]:
                await asyncio.sleep(0)
        await _wait_for_turn(app)

        assert conversation.cancel_calls == 1
        assert not input_area.read_only
        assert not working.display
        await pilot.press("ctrl+home")
        visible_text = _visible_screen_text(app)
        assert "partial response" in visible_text
        assert "Turn cancelled." in visible_text

        await pilot.press(*list("again"), "enter")
        await _wait_for_turn(app)
        assert conversation.submissions == [
            "first",
            "queued one",
            "queued two",
            "queued three",
            "again",
        ]
        assert "Recovered response." in _visible_screen_text(app)
        assert all(message.metadata == {} for message in runtime.inbound_history)


@pytest.mark.asyncio
async def test_up_drains_pending_fifo_and_reenter_submits_one_multiline_inbound() -> None:
    conversation = CancellableRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("first"), "enter"))
        await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
        for text in ("queued one", "queued two", "queued three"):
            await pilot.press(*list(text), "enter")

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("up")
        assert input_area.text == "queued one\nqueued two\nqueued three"
        assert "Pending (" not in _visible_screen_text(app)
        assert conversation.submissions == ["first"]
        assert runtime.inbound_history == [InboundMessage(content="first")]

        await pilot.press("ctrl+c")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)
        await pilot.press("enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["first", "queued one\nqueued two\nqueued three"]
    assert runtime.inbound_history == [
        InboundMessage(content="first"),
        InboundMessage(content="queued one\nqueued two\nqueued three"),
    ]
    assert all(message.metadata == {} for message in runtime.inbound_history)


@pytest.mark.asyncio
async def test_repeated_or_delayed_active_ctrl_c_stays_bound_to_the_cancelled_turn() -> None:
    conversation = CancellableRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("first"), "enter"))
        await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("ctrl+c")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)
        await pilot.press("ctrl+c")

        assert conversation.cancel_calls == 1
        assert app.is_running
        assert input_area.text == ""
        assert not input_area.read_only
        assert app.screen.focused is input_area

        await pilot.press("x", "backspace", "ctrl+c")
        await pilot.pause()
        assert not app.is_running


@pytest.mark.asyncio
async def test_ctrl_c_clears_an_idle_draft_without_exiting() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press(*list("draft"))
        await pilot.press("ctrl+c")

        assert input_area.text == ""
        assert app.is_running


@pytest.mark.asyncio
async def test_ctrl_c_on_empty_idle_input_exits() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert not app.is_running


@pytest.mark.asyncio
async def test_ctrl_d_deletes_forward_when_draft_is_nonempty() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press(*list("draft"))
        input_area.move_cursor((0, 0))
        await pilot.press("ctrl+d")

        assert input_area.text == "raft"
        assert app.is_running


@pytest.mark.asyncio
async def test_ctrl_d_on_empty_input_exits() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+d")
        await pilot.pause()

        assert not app.is_running


@pytest.mark.asyncio
async def test_ctrl_d_during_an_active_turn_settles_stream_before_runtime_close() -> None:
    conversation = BlockingRunSource()
    runtime = CloseOrderingRuntime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("active"), "enter"))
        await asyncio.wait_for(conversation.started.wait(), timeout=1)
        await pilot.press("ctrl+d")
        await asyncio.wait_for(submission, timeout=1)
        await pilot.pause()

        assert not app.is_running

    assert conversation.closed.is_set()
    assert runtime.close_saw_stream_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [" EXIT ", " qUiT "])
async def test_exit_and_quit_commands_exit_without_submitting(command: str) -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list(command), "enter")
        await pilot.pause()

        assert conversation.submissions == []
        assert not app.is_running


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["exit now", "quitter"])
async def test_exit_like_text_is_submitted_as_an_ordinary_turn(command: str) -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list(command), "enter")
        await _wait_for_turn(app)

        assert conversation.submissions == [command]
        assert app.is_running


@pytest.mark.asyncio
async def test_multiline_submission_preserves_text_and_ctrl_j_inserts_a_newline() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first line"), "ctrl+j", *list("second line"), "enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["first line\nsecond line"]
    assert runtime.inbound_history == [InboundMessage(content="first line\nsecond line")]


@pytest.mark.asyncio
async def test_supported_modifier_enter_sequences_insert_newlines_without_submitting() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press(*list("first line"))
        input_area.post_message(Key("\x1b[13;66u", None))
        await pilot.pause()
        await pilot.press(*list("second line"))
        input_area.post_message(Key("\x1b[13;67u", None))
        await pilot.pause()
        await pilot.press(*list("third line"), "enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["first line\nsecond line\nthird line"]


@pytest.mark.asyncio
async def test_textual_driver_lifecycle_balances_enhanced_keyboard_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    KeyboardLifecycleDriver.operations = []
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))
    app.driver_class = KeyboardLifecycleDriver

    async def exit_when_ready(_: object) -> None:
        app.exit()

    await app.run_async(headless=False, size=(80, 24), auto_pilot=exit_when_ready)

    keyboard_writes = [
        value
        for operation, value in KeyboardLifecycleDriver.operations
        if operation == "write" and value in {"\x1b[>1u", "\x1b[<u"}
    ]
    assert keyboard_writes == ["\x1b[>1u", "\x1b[<u"]
    assert KeyboardLifecycleDriver.operations[-1] == ("close", "")
    assert runtime.start_calls == 1
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_malformed_enhanced_keyboard_report_does_not_break_ordinary_submission() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.post_message(Key("\x1b[13;" + ("9" * 5000) + "u", None))
        await pilot.pause()
        await pilot.press(*list("still works"), "enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["still works"]


@pytest.mark.asyncio
async def test_whitespace_only_submission_is_ignored() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.post_message(Paste(" \n\t ").stop())
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert conversation.submissions == []
        assert input_area.text == " \n\t "


@pytest.mark.asyncio
async def test_multiline_paste_is_inserted_without_implicit_submission() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        input_area.post_message(Paste("pasted first\npasted second").stop())
        await pilot.pause()

        assert input_area.text == "pasted first\npasted second"
        assert conversation.submissions == []

        await pilot.press("enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["pasted first\npasted second"]


@pytest.mark.asyncio
async def test_multiline_input_grows_to_six_rows_then_scrolls_internally() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        input_region = app.query_one("#conversation-input-region")

        input_area.post_message(Paste("\n".join(f"line {index}" for index in range(6))).stop())
        await pilot.pause()
        six_row_region_height = input_region.size.height

        assert input_area.size.height == 6
        assert input_area.max_scroll_y == 0

        input_area.post_message(Paste("\nline 6\nline 7").stop())
        await pilot.pause()

        assert input_area.size.height == 6
        assert input_area.max_scroll_y > 0
        assert input_region.size.height == six_row_region_height


@pytest.mark.asyncio
async def test_accepted_submissions_are_recalled_only_within_the_runtime_lifetime() -> None:
    conversation = ScriptedRunSource(
        deltas_by_submission=((), (), ()),
        completed_contents=("", "", ""),
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first"), "enter")
        await _wait_for_turn(app)
        await pilot.press(*list("second"), "enter")
        await _wait_for_turn(app)

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("up")
        assert input_area.text == "second"
        await pilot.press("up")
        assert input_area.text == "first"
        await pilot.press("down")
        assert input_area.text == "second"
        await pilot.press("down")
        assert input_area.text == ""

    next_runtime = _runtime(ScriptedRunSource())
    next_app = _terminal_app(cast(PreparedRuntime, next_runtime))
    async with next_app.run_test(size=(80, 24)) as pilot:
        next_input = next_app.query_one("#conversation-input", TextArea)
        await pilot.press("up")
        assert next_input.text == ""


@pytest.mark.asyncio
async def test_scrolling_conversation_keeps_composer_focus_and_draft() -> None:
    content = "\n".join(f"line {index:02d}" for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("seed"), "enter")
        await _wait_for_turn(app)
        await pilot.press(*list("draft"))
        await pilot._post_mouse_events([MouseScrollUp], offset=(10, 5), times=3)

        input_area = app.query_one("#conversation-input", TextArea)
        assert input_area.text == "draft"
        assert isinstance(app.screen.focused, TextArea)


@pytest.mark.asyncio
async def test_text_deltas_update_one_assistant_markdown_and_terminal_marker_closes() -> None:
    conversation = ScriptedRunSource(
        pause_after_first_delta=True,
        completed_content="First answer.",
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press("h", "i", "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            async with asyncio.timeout(1):
                while "First" not in _visible_screen_text(app):
                    await pilot.pause()
            partial_text = _visible_screen_text(app)

            assert partial_text.count("First") == 1
            assert "answer." not in partial_text
        finally:
            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)

        completed_text = _visible_screen_text(app)
        assert completed_text.count("First answer.") == 1
        assert not list(app.query(".agent-run-activity-group"))

    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_intermediate_model_output_and_tools_share_one_activity_group() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment("Planning **the tool call**."),
            _tool_call(
                "call-read-file",
                "read_file",
                "Running read_file",
            ),
            _response_delta("Final **answer**."),
            _response_segment("Final **answer**."),
            _completed_response(),
        )
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        groups = list(app.query(".agent-run-activity-group"))
        assert len(groups) == 1
        group = groups[0]
        activity_content = group.query_one(".agent-run-activity-content")
        assert not activity_content.display
        assert len(list(activity_content.query(".assistant-row"))) == 1
        assert len(list(activity_content.query(".tool-row"))) == 1
        activity_markdown = activity_content.query_one(".assistant-row").query_one(Markdown)
        assert "Planning" in activity_markdown.source

        display = app.query_one("#conversation-display")
        direct_assistant_rows = [
            row for row in display.query(".assistant-row") if row.parent is display
        ]
        if len(direct_assistant_rows) != 1:
            raise AssertionError(
                f"history={[(item.type, item.content, item.metadata) for item in runtime.outbound_history]} "
                f"children={[type(child).__name__ for child in display.children]}"
            )
        assert len(direct_assistant_rows) == 1
        final_markdown = direct_assistant_rows[0].query_one(Markdown)
        assert "Final" in final_markdown.source
        assert _visible_screen_text(app).count("Final") == 1


@pytest.mark.asyncio
async def test_intermediate_completion_reparents_stream_candidate_and_empty_output_is_not_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("candidate"),
            _response_segment(""),
            _tool_call(
                "call-empty",
                "read_file",
                "Running read_file",
            ),
            _response_segment(""),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))
    mounted_assistants: list[tuple[Markdown, Widget | None]] = []
    mount_assistant = app._mount_assistant

    async def record_mounted_assistant(*args: object, **kwargs: object) -> Markdown:
        assistant = await mount_assistant(*args, **kwargs)  # type: ignore[arg-type]
        mounted_assistants.append((assistant, cast(Widget | None, assistant.parent)))
        return assistant

    monkeypatch.setattr(app, "_mount_assistant", record_mounted_assistant)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        activity_content = group.query_one(".agent-run-activity-content")
        activity_markdown = activity_content.query(".assistant-row").first().query_one(Markdown)
        assert activity_markdown.source == "candidate"
        assert len(mounted_assistants) == 1
        streamed_markdown, streamed_row = mounted_assistants[0]
        assert activity_markdown is streamed_markdown
        assert activity_markdown.parent is streamed_row
        assert streamed_row is not None
        assert streamed_row.parent is activity_content
        assert len(list(activity_content.query(".assistant-row"))) == 1
        assert len(list(app.query("#conversation-display > .assistant-row"))) == 0


@pytest.mark.asyncio
async def test_tool_start_reclassifies_a_streamed_candidate_without_model_completion() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("Unclassified candidate."),
            _tool_call("call-read", "read_file", "Running read_file"),
            _response_delta("Final answer."),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        activity_content = group.query_one(".agent-run-activity-content")
        assert activity_content.query_one(Markdown).source == "Unclassified candidate."
        assert _tool_row_texts(app) == ["Running: read_file\nArguments: Running read_file"]
        assert app.query("#conversation-display > .assistant-row").first().query_one(
            Markdown
        ).source == ("Final answer.")
        assert not activity_content.display


@pytest.mark.asyncio
async def test_direct_terminal_marker_preserves_the_current_candidate() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("streamed candidate"),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assert not list(app.query(".agent-run-activity-group"))
        assert app.query("#conversation-display > .assistant-row").first().query_one(
            Markdown
        ).source == ("streamed candidate")
        assert "streamed candidate" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_first_terminal_event_finishes_without_waiting_for_more_events() -> None:
    conversation = TerminalThenBlockingRunSource()
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await asyncio.wait_for(conversation.terminal_emitted.wait(), timeout=1)
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        assert not group.query_one(".agent-run-activity-content").display
        assert "final response" in _visible_screen_text(app)
        assert _tool_row_texts(app) == ["Running: read_file\nArguments: Running read_file"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        (
            _response_segment("process activity"),
            _response_segment("process activity"),
            _tool_call(
                "call-read",
                "read_file",
                "Running read_file",
            ),
            _completed_response(),
        ),
        (
            _response_delta("process activity"),
            _tool_call(
                "call-read",
                "read_file",
                "Running read_file",
            ),
            _response_segment("process activity"),
            _completed_response(),
        ),
    ],
)
async def test_duplicate_or_late_model_completion_reconciles_grouped_candidate(
    events: tuple[_ScriptItem, ...],
) -> None:
    conversation = ToolMessageSequenceRunSource(events)
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        content = app.query_one(".agent-run-activity-content")
        assert [markdown.source for markdown in content.query(Markdown)] == ["process activity"]
        assert _tool_row_texts(app) == ["Running: read_file\nArguments: Running read_file"]


@pytest.mark.asyncio
async def test_reasoning_transition_reopens_the_completed_response_stream() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            OutboundMessage("model_reasoning", "Reasoning A.", {"_stream_delta": True}),
            OutboundMessage("model_reasoning", "", {"_stream_end": True}),
            _response_delta("First answer."),
            _response_segment(""),
            OutboundMessage("model_reasoning", "Reasoning B.", {"_stream_delta": True}),
            OutboundMessage("model_reasoning", "", {"_stream_end": True}),
            _response_delta("Second answer."),
            _response_segment(""),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assistant_rows = list(app.query("#conversation-display > .assistant-row"))
        assert len(assistant_rows) == 1
        assert assistant_rows[0].query_one(Markdown).source == "First answer.Second answer."
        activity = app.query_one(".agent-run-activity-content")
        assert [markdown.source for markdown in activity.query(Markdown)] == [
            "Reasoning A.",
            "Reasoning B.",
        ]


@pytest.mark.asyncio
async def test_late_delta_after_completed_candidate_does_not_reopen_its_stream() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("authoritative complete content"),
            _response_segment(""),
            _response_delta("late fragment"),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assistant = app.query_one("#conversation-display > .assistant-row").query_one(Markdown)
        assert assistant.source == "authoritative complete content"
        assert "late fragment" not in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_successful_empty_terminal_content_shows_status_without_activity() -> None:
    conversation = ScriptedRunSource(deltas=(), completed_content="")
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("empty"), "enter")
        await _wait_for_turn(app)

        assert not list(app.query(".agent-run-activity-group"))
        assert "Completed with no response." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_first_terminal_event_remains_authoritative() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_delta("candidate"),
            _cancelled_response("authoritative partial"),
            _failed_response("Late failure."),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        assert group.query_one(".agent-run-activity-content").query_one(Markdown).source == (
            "candidate"
        )
        visible_text = _visible_screen_text(app)
        assert "Turn cancelled." in visible_text
        assert "Late failure." not in visible_text
        assert "late success" not in visible_text
        assert group.query_one(".agent-run-activity-content").display


@pytest.mark.asyncio
async def test_event_stream_failure_groups_candidate_and_finishes_unfinished_tool() -> None:
    conversation = ExplodingRunSource()
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        activity_content = group.query_one(".agent-run-activity-content")
        assert activity_content.display
        assert activity_content.query_one(Markdown).source == "Unconfirmed candidate."
        assert _tool_row_texts(app) == ["Running: read_file\nArguments: Running read_file"]
        assert "event stream failed" in _visible_screen_text(app)
        assert "Unconfirmed candidate." in _visible_screen_text(app)
        assert not list(app.query("#conversation-display > .assistant-row"))
        assert conversation.closed.is_set()


@pytest.mark.asyncio
async def test_event_stream_failure_without_activity_does_not_create_empty_group() -> None:
    conversation = ToolMessageSequenceRunSource(())
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assert not list(app.query(".agent-run-activity-group"))
        assert "Turn failed." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_failed_no_tool_candidate_becomes_expanded_activity() -> None:
    conversation = ScriptedRunSource(
        deltas=("draft content",),
        outcomes=("failed",),
        failure_message="Model unavailable.",
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        assert group.query_one(".agent-run-activity-content").query_one(Markdown).source == (
            "draft content"
        )
        assert group.query_one(".agent-run-activity-content").display
        assert not list(app.query("#conversation-display > .assistant-row"))
        assert "Model unavailable." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_failure_before_visible_activity_does_not_create_empty_group() -> None:
    conversation = ScriptedRunSource(
        deltas=(),
        outcomes=("failed",),
        failure_message="Model unavailable.",
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        assert not list(app.query(".agent-run-activity-group"))
        assert "Model unavailable." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_empty_cancelled_content_removes_candidate_without_empty_group() -> None:
    conversation = ScriptedRunSource(
        deltas=("streamed candidate",),
        outcomes=("cancelled",),
        cancelled_content="",
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("cancel"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        assert group.query_one(Markdown).source == "streamed candidate"
        assert not list(app.query("#conversation-display > .assistant-row"))
        assert "Turn cancelled." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_replacing_the_display_stops_a_frozen_run_timer(
    agent_home: Path,
    workspace: Path,
) -> None:
    runtime = _prepared_runtime(
        agent_home,
        workspace,
        _RuntimeProvider(()),
    )
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)):
        projection = _AgentRunProjection(app, TURN_ID)
        app._active_run_projection = projection
        projection._start_timing()
        assert projection._timer is not None

        replaced = await app._replace_display_from_session(runtime.session.session_id)

        assert replaced
        assert projection._timer is None


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (0.99, "0s"),
        (59.99, "59s"),
        (60, "1min 0s"),
        (65.9, "1min 5s"),
        (3605, "1h 0min 5s"),
        (24 * 3600 + 5, "24h 0min 5s"),
    ],
)
def test_activity_duration_uses_floored_accumulated_seconds(elapsed: float, expected: str) -> None:
    assert _format_activity_duration(elapsed) == expected


@pytest.mark.asyncio
async def test_activity_timing_includes_wait_before_first_outbound() -> None:
    clock = [0.0]
    conversation = ToolMessageSequenceRunSource(
        (
            _tool_call("call-read", "read_file", "{}"),
            _completed_response(),
        ),
        pause_before=True,
    )
    app = _terminal_app(
        cast(PreparedRuntime, _runtime(conversation)),
        monotonic=lambda: clock[0],
    )

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.paused.wait(), timeout=1)
        clock[0] = 10.9
        conversation.continue_events.set()
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        heading = app.query_one(".agent-run-activity-heading", Static)
        assert str(heading.content) == "\u25b6 10s"


@pytest.mark.asyncio
async def test_activity_heading_starts_with_accumulated_time_and_freezes_on_success() -> None:
    clock = [0.0]
    events = (
        _response_segment("intermediate"),
        _tool_call("call-read", "read_file", "Running read_file"),
        _completed_response(),
    )
    conversation = ToolMessageSequenceRunSource(events, pause_after=(0, 1))
    app = _terminal_app(
        cast(PreparedRuntime, _runtime(conversation)),
        monotonic=lambda: clock[0],
    )
    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.paused.wait(), timeout=1)
        async with asyncio.timeout(1):
            while app._active_run_projection is None:
                await pilot.pause()
        projection = app._active_run_projection
        assert projection is not None
        async with asyncio.timeout(1):
            while not app.query(".agent-run-activity-heading"):
                await pilot.pause()

        clock[0] = 5.9
        projection._refresh_elapsed()
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()
        heading = app.query_one(".agent-run-activity-heading", Static)
        content = app.query_one(".agent-run-activity-content")
        assert not heading.can_focus
        assert str(heading.content) == "\u25bc 5s"
        assert content.display
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)

        await pilot.click(".agent-run-activity-heading")
        assert content.display
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)

        clock[0] = 3605.9
        projection._refresh_elapsed()
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()
        assert str(heading.content) == "\u25bc 1h 0min 5s"

        conversation.continue_events.set()
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        assert str(heading.content) == "\u25b6 1h 0min 5s"
        assert not content.display
        projection._refresh_elapsed()
        assert str(heading.content) == "\u25b6 1h 0min 5s"


@pytest.mark.asyncio
async def test_manual_activity_disclosure_keeps_heading_position_while_content_grows_downward() -> (
    None
):
    intermediate = "\n".join(f"activity {index:02d} " + "a" * 35 for index in range(45))
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment(intermediate),
            _tool_call("call-read", "read_file", "Running read_file"),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)
        heading = app.query_one(".agent-run-activity-heading", Static)
        content = app.query_one(".agent-run-activity-content")
        assert not content.display

        await pilot.click(".agent-run-activity-heading")
        async with asyncio.timeout(1):
            while not app.query_one("#new-content").display:
                await pilot.pause()
        expanded_heading_y = heading.region.y
        assert content.display
        display = app.query_one("#conversation-display")
        assert not display.is_vertical_scroll_end
        assert app.query_one("#new-content").display

        # The heading remains in the viewport after the group shrinks, so the
        # next click can still target the same visible control.
        await pilot.click(".agent-run-activity-heading")
        await pilot.pause()
        assert not content.display
        assert heading.region.y == expanded_heading_y
        assert display.is_vertical_scroll_end

        await pilot.click(".agent-run-activity-heading")
        await pilot.pause()
        assert content.display
        assert heading.region.y == expanded_heading_y
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)


@pytest.mark.asyncio
async def test_failed_activity_group_is_mouse_toggleable_without_moving_composer_focus() -> None:
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment("intermediate"),
            _tool_call("call-read", "read_file", "Running read_file"),
            _failed_response("The Agent Run failed."),
        )
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)

        heading = app.query_one(".agent-run-activity-heading", Static)
        content = app.query_one(".agent-run-activity-content")
        assert not heading.can_focus
        assert str(heading.content).startswith("\u25bc ")
        assert content.display
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)

        await pilot.press("enter")
        assert content.display
        await pilot.click(".agent-run-activity-heading")
        assert not content.display
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)

        await pilot.click(".agent-run-activity-heading")
        assert content.display
        await pilot.click(".tool-row")
        assert content.display


@pytest.mark.asyncio
async def test_tool_confirmation_defaults_to_decline_and_shows_effective_operation() -> None:
    conversation = ConfirmationRunSource(
        details={"path": "outside.txt", "limit": 20},
        reason="The path resolves outside the current Workspace.",
        warnings=("Review the target before allowing access.",),
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        visible_text = _visible_screen_text(app)
        assert str(app.screen.query_one("#confirmation-heading", Static).content) == (
            "Tool Confirmation"
        )
        assert "Tool: Read File" in visible_text
        assert "Reason: The path resolves outside the current" in visible_text
        assert "Workspace." in visible_text
        assert "Warning: Review the target before allowing access." in visible_text
        assert "Path: outside.txt" in visible_text
        assert "Limit: 20" in visible_text
        assert '"path"' not in visible_text
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        assert conversation.responses == []

        await pilot.press("tab")
        assert app.screen.focused is app.screen.query_one("#confirmation-approve", Button)
        await pilot.press("shift+tab")
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("enter")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), "declined"),
    ]
    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
async def test_write_confirmation_hides_content_and_unknown_details() -> None:
    secret = "Authorization: Bearer sk-sensitive-value"
    conversation = ConfirmationRunSource(
        tool_name="write_file",
        details={"path": "outside.txt", "content": secret, "internal": "raw-result"},
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(42, 16)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("write"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        visible_text = _visible_screen_text(app)
        assert str(app.screen.query_one("#confirmation-tool", Static).content) == "Tool: Write File"
        details = [
            str(cast(Static, item).content) for item in app.screen.query(".confirmation-details")
        ]
        assert "Path: outside.txt" in details
        assert f"Content: {len(secret)} characters" in details
        assert "sk-sensitive-value" not in visible_text
        assert "raw-result" not in visible_text
        assert all("sk-sensitive-value" not in detail for detail in details)
        assert all("raw-result" not in detail for detail in details)

        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_web_fetch_confirmation_redacts_url_credentials_and_query_values() -> None:
    conversation = ConfirmationRunSource(
        tool_name="web_fetch",
        details={
            "url": "http://user:password@127.0.0.1/private?token=secret-value",
            "format": "markdown",
        },
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("fetch"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        details = [
            str(cast(Static, item).content) for item in app.screen.query(".confirmation-details")
        ]
        assert "URL: http://127.0.0.1/private?<redacted>" in details
        assert "Format: markdown" in details
        assert all("password" not in detail for detail in details)
        assert all("secret-value" not in detail for detail in details)

        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_exec_confirmation_shows_exact_command_and_arrow_keys_select_approval() -> None:
    command = 'rm -rf "build output" && printf done'
    conversation = ConfirmationRunSource(
        tool_name="exec",
        details={"command": command, "cwd": ".", "timeout": 45},
        reason="The Exec command matches a known destructive operation.",
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(100, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("run"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        visible_text = _visible_screen_text(app)
        assert f"Command: {command}" in visible_text
        assert "CWD: ." in visible_text
        assert "Timeout: 45" in visible_text
        assert "The Exec command matches a known destructive operation." in visible_text
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)

        await pilot.press("right")
        assert app.screen.focused is app.screen.query_one("#confirmation-approve", Button)
        await pilot.press("left")
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("down")
        assert app.screen.focused is app.screen.query_one("#confirmation-approve", Button)
        await pilot.press("up")
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("right")
        assert app.screen.focused is app.screen.query_one("#confirmation-approve", Button)
        await pilot.press("enter")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), "approved"),
    ]
    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("button_id", "decision"),
    [
        ("confirmation-approve", "approved"),
        ("confirmation-decline", "declined"),
    ],
)
async def test_confirmation_buttons_resolve_the_pending_tool_with_mouse(
    button_id: str,
    decision: ConfirmationDecision,
) -> None:
    conversation = ConfirmationRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        assert await pilot.click(f"#{button_id}")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), decision),
    ]
    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("escape", "ctrl+c"))
async def test_confirmation_escape_and_ctrl_c_decline_only_the_pending_tool(key: str) -> None:
    conversation = ConfirmationRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        input_area = app.query_one("#conversation-input", TextArea)

        await pilot.press(key)
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        assert app.is_running
        assert app.screen.focused is input_area
        assert not input_area.read_only

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), "declined"),
    ]
    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
async def test_clicking_outside_confirmation_keeps_it_open_without_a_decision() -> None:
    conversation = ConfirmationRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)

        assert await pilot.click(offset=(1, 1))
        await pilot.pause()
        assert conversation.responses == []
        assert "Tool: Read File" in _visible_screen_text(app)

        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.cancel_calls == 0


@pytest.mark.asyncio
async def test_duplicate_late_confirmation_is_shown_once_and_does_not_fail_the_turn() -> None:
    conversation = DuplicateLateConfirmationRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        await pilot.press("right", "enter")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        assert "A foreground turn failed" not in _visible_screen_text(app)

    assert conversation.responses == [(conversation.request.confirmation_id, "approved")]


@pytest.mark.asyncio
async def test_multiple_confirmations_are_resolved_in_order_with_a_fresh_safe_default() -> None:
    conversation = MultipleConfirmationRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested[0].wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        await pilot.press("right", "enter")

        await asyncio.wait_for(conversation.confirmation_requested[1].wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("enter")

        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (conversation.requests[0].confirmation_id, "approved"),
        (conversation.requests[1].confirmation_id, "declined"),
    ]


@pytest.mark.asyncio
async def test_application_teardown_cancels_an_open_confirmation_without_a_decision() -> None:
    conversation = ConfirmationRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))
    submission: asyncio.Task[None]

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await _wait_for_confirmation(app, pilot)
        assert conversation.responses == []

    await asyncio.gather(submission, return_exceptions=True)
    assert conversation.responses == []


@pytest.mark.asyncio
async def test_tool_activity_renders_raw_arguments_until_terminal_marker() -> None:
    conversation = ToolActivityRunSource(
        start_summary='Running read_file {"arguments":{"path":"C:/private.txt"}}',
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.tool_started.wait(), timeout=1)
        async with asyncio.timeout(1):
            while not app.query(".tool-row"):
                await pilot.pause()
        refreshed = asyncio.Event()
        app.call_after_refresh(refreshed.set)
        await refreshed.wait()

        running_text = _visible_screen_text(app)
        assert "Running: read_file" in running_text
        assert "call-read-file" not in running_text
        assert 'Running read_file {"arguments":{"path":"C:/private.txt"}}' in running_text
        row = app.query_one(".tool-row", Static)
        assert row.outer_size.width == app.query_one(".message-row").outer_size.width
        assert row.parent is app.query_one(".agent-run-activity-content")
        assert not row.has_class("message")
        assert not row.can_focus
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)
        running_row_content = str(row.content)
        await pilot.click(".tool-row")
        assert not row.has_focus
        assert str(row.content) == running_row_content

        conversation.complete_tool.set()
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        final_text = _visible_screen_text(app)
        assert "Running: read_file" not in final_text
        assert str(row.content) == (
            'Running: read_file\nArguments: Running read_file {"arguments":{"path":"C:/private.txt"}}'
        )
        assert not row.parent.display
        assert "Completed with no response." in final_text


@pytest.mark.asyncio
async def test_tool_rows_isolate_calls_and_turns_without_tool_result_projection() -> None:
    first_turn = (
        _tool_call("completion-first", "glob", "Running glob"),
        _tool_call("shared-call", "read_file", "Running read_file"),
        _tool_call("missing-completion", "write_file", "Running write_file"),
        _tool_call("shared-call", "read_file", "Running read_file"),
        _tool_call("shared-call", "read_file", "Running read_file"),
        _completed_response(),
    )
    second_turn = (
        _tool_call("shared-call", "read_file", "Running read_file"),
        _completed_response(),
    )
    conversation = ToolMessageSequenceRunSource(first_turn, second_turn)
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first"), "enter")
        await _wait_for_turn(app)
        assert _tool_row_texts(app) == [
            "Running: glob\nArguments: Running glob",
            "Running: read_file\nArguments: Running read_file",
            "Running: write_file\nArguments: Running write_file",
        ]

        await pilot.press(*list("second"), "enter")
        await _wait_for_turn(app)
        assert _tool_row_texts(app) == [
            "Running: glob\nArguments: Running glob",
            "Running: read_file\nArguments: Running read_file",
            "Running: write_file\nArguments: Running write_file",
            "Running: read_file\nArguments: Running read_file",
        ]
        assert "completion-first" not in _visible_screen_text(app)
        assert "shared-call" not in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_tool_row_updates_preserve_historical_follow_and_resize_state() -> None:
    first_events: list[_ScriptItem] = []
    for index in range(24):
        tool_name = f"tool_{index:02d}"
        tool_call_id = f"call-{index:02d}"
        first_events.append(_tool_call(tool_call_id, tool_name, f"Running {tool_name}"))
    first_events.append(_failed_response("The Agent Run failed."))
    second_events = (
        _tool_call("later-call", "web_search", "Running web_search"),
        _completed_response(),
    )
    conversation = ToolMessageSequenceRunSource(
        tuple(first_events),
        second_events,
        pause_after=(1, 0),
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press(*list("seed"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        async with asyncio.timeout(1):
            while not display.is_vertical_scroll_end:
                await pilot.pause()
        assert display.is_vertical_scroll_end

        await pilot.press("pageup", "pageup")
        assert not display.is_vertical_scroll_end
        submission = asyncio.create_task(pilot.press(*list("later"), "enter"))
        try:
            await asyncio.wait_for(conversation.paused.wait(), timeout=1)
            await pilot.pause()
            historical_scroll_y = display.scroll_y
            async with asyncio.timeout(1):
                while not app.query_one("#new-content").display:
                    await pilot.pause()
            assert app.query_one("#new-content").display
            conversation.continue_events.set()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)
        finally:
            conversation.continue_events.set()

        assert not display.is_vertical_scroll_end
        assert display.scroll_y == historical_scroll_y
        assert app.query_one("#new-content").display

        await pilot.resize_terminal(40, 20)
        await pilot.pause()
        assert not display.is_vertical_scroll_end
        message_width = app.query_one(".message-row").outer_size.width
        assert all(
            row.outer_size.width == message_width
            for row in app.query(".tool-row")
            if row.display and row.parent.display
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reading_history", [False, True])
async def test_activity_layout_changes_preserve_follow_or_historical_anchor_at_each_stage(
    reading_history: bool,
) -> None:
    history = "\n".join(f"history {index:02d} " + "h" * 35 for index in range(70))
    second_events = (
        _response_delta("candidate"),
        _response_segment("candidate"),
        _tool_call(
            "activity-call",
            "read_file",
            "Running read_file",
        ),
        _response_segment("latest answer"),
        _completed_response(),
    )
    conversation = StagedToolMessageSequenceRunSource(
        (
            _response_delta(history),
            _completed_response(),
        ),
        second_events,
        pause_after=tuple((1, event_index) for event_index in range(5)),
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("history"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        await pilot.press("ctrl+end")
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        try:
            await conversation.wait_after(1, 0)
            await pilot.pause()
            refreshed = asyncio.Event()
            app.call_after_refresh(refreshed.set)
            await refreshed.wait()
            async with asyncio.timeout(1):
                while "candidate" not in _visible_screen_text(app):
                    await pilot.pause()
            async with asyncio.timeout(1):
                while not display.is_vertical_scroll_end:
                    await pilot.pause()
            assert display.is_vertical_scroll_end
            historical_position = display.scroll_y
            if reading_history:
                await pilot.press("pageup")
                async with asyncio.timeout(1):
                    while display.is_vertical_scroll_end:
                        await pilot.pause()
                historical_position = display.scroll_y
                assert not display.is_vertical_scroll_end

            expected_tool_rows: dict[int, list[str]] = {
                1: [],
                2: ["Running: read_file\nArguments: Running read_file"],
                3: ["Running: read_file\nArguments: Running read_file"],
                4: ["Running: read_file\nArguments: Running read_file"],
            }
            for previous_event, next_event in zip(range(4), range(1, 5), strict=True):
                conversation.continue_after(1, previous_event)
                await conversation.wait_after(1, next_event)
                await pilot.pause()
                refreshed = asyncio.Event()
                app.call_after_refresh(refreshed.set)
                await refreshed.wait()
                expected_tool_row = expected_tool_rows[next_event][-1:]
                if expected_tool_row:
                    async with asyncio.timeout(1):
                        while _tool_row_texts(app)[-1:] != expected_tool_row:
                            await pilot.pause()
                assert _tool_row_texts(app)[-1:] == expected_tool_row
                if reading_history and expected_tool_row:
                    async with asyncio.timeout(1):
                        while not app.query_one("#new-content").display:
                            await pilot.pause()
                    assert not display.is_vertical_scroll_end
                    assert display.scroll_y == historical_position
                    assert app.query_one("#new-content").display
                elif reading_history:
                    assert not display.is_vertical_scroll_end
                else:
                    async with asyncio.timeout(1):
                        while not display.is_vertical_scroll_end:
                            await pilot.pause()
                    assert not app.query_one("#new-content").display

            conversation.continue_after(1, 4)
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)

            if reading_history:
                assert not display.is_vertical_scroll_end
                assert display.scroll_y == historical_position
                assert app.query_one("#new-content").display
            else:
                assert display.is_vertical_scroll_end
                assert "latest answer" in _visible_screen_text(app)
                assert not app.query_one("#new-content").display
            assert not app.query_one(".agent-run-activity-content").display
        finally:
            conversation.continue_all()
            if not submission.done():
                await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expanded", "reading_history"),
    [(False, False), (False, True), (True, False), (True, True)],
)
async def test_activity_group_resize_preserves_scroll_mode_anchor_and_input_focus(
    expanded: bool,
    reading_history: bool,
) -> None:
    history = "\n".join(f"history {index:02d}" for index in range(80))
    activity = "\n".join(f"activity {index:02d}" for index in range(50))
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment(history),
            _completed_response(),
        ),
        (
            _response_segment(activity),
            _tool_call(
                "activity-call",
                "read_file",
                "Running read_file",
            ),
            _completed_response(),
        ),
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(70, 22)) as pilot:
        await pilot.press(*list("history"), "enter")
        await _wait_for_turn(app)
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        content = app.query_one(".agent-run-activity-content")
        if expanded:
            heading = app.query_one(".agent-run-activity-heading")
            heading.scroll_visible()
            await pilot.pause()
            await pilot.click(heading)
            await pilot.pause()
        assert content.display is expanded

        await pilot.press("pageup" if reading_history else "ctrl+end")
        await pilot.pause()
        scroll_position = display.scroll_y
        assert display.is_vertical_scroll_end is not reading_history

        await pilot.resize_terminal(55, 22)
        await pilot.pause()

        assert content.display is expanded
        assert display.is_vertical_scroll_end is not reading_history
        if reading_history:
            assert display.scroll_y == scroll_position
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)


@pytest.mark.asyncio
async def test_activity_content_uses_main_vertical_scroll_and_keeps_code_horizontal_scroll() -> (
    None
):
    code_line = "very_long_variable_name = " + "0123456789" * 8 + "TAIL"
    intermediate = f"```python\n{code_line}\n```\n\n" + "\n".join(
        f"activity {index:02d}" for index in range(60)
    )
    conversation = ToolMessageSequenceRunSource(
        (
            _response_segment(intermediate),
            _tool_call("activity-call", "read_file", "Running read_file"),
            _completed_response(),
        )
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(48, 20)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)
        await asyncio.sleep(0.05)
        await pilot.press("ctrl+end")
        await pilot.click(".agent-run-activity-heading")
        await pilot.pause()
        display = app.query_one("#conversation-display")
        markdown = app.query_one(".agent-run-activity-content").query_one(Markdown)
        horizontal_scroll_y = display.scroll_y

        assert "TAIL" not in _visible_screen_text(app)
        await pilot._post_mouse_events(
            [MouseScrollRight],
            offset=(10, markdown.region.y + 2),
            times=30,
        )
        assert "TAIL" in _visible_screen_text(app)
        assert display.scroll_y == horizontal_scroll_y

        await pilot._post_mouse_events(
            [MouseScrollDown],
            offset=(10, min(markdown.region.bottom - 1, 15)),
            times=3,
        )
        assert display.scroll_y > horizontal_scroll_y
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)


@pytest.mark.asyncio
async def test_streamed_markdown_preserves_reading_structure_and_link_urls() -> None:
    content = (
        "# Heading\n\n"
        "- first item\n"
        "- second item\n\n"
        "> quoted text\n\n"
        "```python\n"
        "long_value = " + "'x'" * 30 + "\n```\n\n"
        "[documentation](https://example.com/docs)\n"
        "![architecture image](https://example.com/asset.png)\n"
    )
    conversation = ScriptedRunSource(
        deltas=tuple(content),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press(*list("read"), "enter")
        await _wait_for_turn(app)
        await asyncio.sleep(0.1)

        visible_text = _visible_screen_text(app)
        assert "Heading" in visible_text
        assert "first item" in visible_text
        assert "second item" in visible_text
        assert "quoted text" in visible_text
        assert "long_value" in visible_text
        assert "documentation (https://example.com/docs)" in visible_text
        assert "architecture image" in visible_text
        assert "https://example.com/asset.png" in visible_text
        assert "@click" not in app.export_screenshot()


@pytest.mark.asyncio
async def test_markdown_structure_is_visible_while_the_fenced_block_is_incomplete() -> None:
    partial = "# Heading\n\n- first item\n\n> quoted text\n\n```python\nvalue = 1"
    content = partial + "\n```\n"
    conversation = ScriptedRunSource(
        pause_after_first_delta=True,
        deltas=(partial, "\n```\n"),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("progress"), "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            await asyncio.sleep(0.05)
            await pilot.pause()

            visible_text = _visible_screen_text(app)
            assert "Heading" in visible_text
            assert "first item" in visible_text
            assert "quoted text" in visible_text
            assert "value=1" in visible_text
        finally:
            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_high_frequency_deltas_are_preserved_until_the_terminal_marker() -> None:
    streamed_content = "draft-" * 300
    conversation = ScriptedRunSource(
        deltas=tuple(streamed_content),
        completed_content="# Complete\n\nExact final content.",
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await asyncio.wait_for(pilot.press(*list("burst"), "enter"), timeout=5)
        await _wait_for_turn(app)

        visible_text = _visible_screen_text(app)
        assistant = app.query_one("#conversation-display > .assistant-row").query_one(Markdown)
        assert assistant.source == streamed_content
        assert "draft-" in visible_text
        assert "Complete" not in visible_text
        assert "Exact final content." not in visible_text


@pytest.mark.asyncio
async def test_long_markdown_code_lines_remain_unwrapped_in_a_narrow_terminal() -> None:
    code_line = "very_long_variable_name = " + "0123456789" * 8 + "TAIL"
    content = f"```python\n{code_line}\n```"
    conversation = ScriptedRunSource(
        deltas=tuple(content),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(48, 20)) as pilot:
        await pilot.press(*list("code"), "enter")
        await asyncio.sleep(0.1)

        nodes = [
            (text, y)
            for text, _, y in _screenshot_text_nodes(app)
            if text and all(character in code_line for character in text)
        ]
        assert "".join(text for text, _ in nodes).startswith("very_long_variable_name")
        assert any("0" in text for text, _ in nodes)
        assert len({y for _, y in nodes}) == 1

        assert "TAIL" not in _visible_screen_text(app)
        await pilot._post_mouse_events([MouseScrollRight], offset=(10, 5), times=30)
        assert "TAIL" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_incomplete_streamed_markdown_remains_visible_before_completion() -> None:
    content = "[documentation](https://example.com/docs)\n\n```python\nvalue = 1\n```"
    conversation = ScriptedRunSource(
        pause_after_first_delta=True,
        deltas=("[documentation](", "https://example.com/docs)\n\n```python\n", "value = 1\n```"),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("partial"), "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            async with asyncio.timeout(1):
                while "[documentation](" not in _visible_screen_text(app):
                    await pilot.pause()

            partial_text = _visible_screen_text(app)
            assert "[documentation](" in partial_text
            assert "partial" in partial_text
        finally:
            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=1)
            await _wait_for_turn(app)

        await asyncio.sleep(0.05)
        assert "documentation (https://example.com/docs)" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_streamed_markdown_reflows_cjk_content_after_resize() -> None:
    content = "界" * 20
    conversation = ScriptedRunSource(
        deltas=tuple(content),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("cjk"), "enter")
        await asyncio.sleep(0.1)
        assert _content_text_nodes(app, content) == [content]

        await pilot.resize_terminal(40, 18)
        await asyncio.sleep(0.05)
        narrow_lines = _content_text_nodes(app, content)
        assert len(narrow_lines) == 2
        assert "".join(narrow_lines) == content

        await pilot.resize_terminal(80, 24)
        await asyncio.sleep(0.05)
        assert _content_text_nodes(app, content) == [content]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "cancelled_content", "expected_partial", "expected_status"),
    [
        ("cancelled", "Cancelled exact content.", "draft content", "Turn cancelled."),
        ("failed", "", "draft content", "Model unavailable."),
    ],
)
async def test_terminal_outcomes_settle_markdown_and_allow_a_subsequent_turn(
    outcome: Literal["cancelled", "failed"],
    cancelled_content: str,
    expected_partial: str,
    expected_status: str | None,
) -> None:
    conversation = ScriptedRunSource(
        deltas=("draft ", "content"),
        completed_content="Recovered response.",
        outcomes=(outcome, "completed"),
        cancelled_content=cancelled_content,
        failure_message="Model unavailable.",
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first"), "enter")
        await _wait_for_turn(app)
        visible_text = _visible_screen_text(app)
        assert expected_partial in visible_text
        if expected_status is not None:
            assert expected_status in visible_text

        await pilot.press(*list("again"), "enter")
        await _wait_for_turn(app)
        assert conversation.submissions == ["first", "again"]
        assert "draft content" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_cancelled_terminal_marker_renders_the_cancelled_status() -> None:
    conversation = ScriptedRunSource(
        deltas=(),
        outcomes=("cancelled",),
        cancelled_content="Retained partial response.",
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("cancel"), "enter")
        await _wait_for_turn(app)

        visible_text = _visible_screen_text(app)
        assert "Retained partial response." not in visible_text
        assert "Turn cancelled." in visible_text


@pytest.mark.asyncio
async def test_cancelled_terminal_marker_keeps_the_streamed_candidate_in_activity() -> None:
    conversation = ScriptedRunSource(
        deltas=("streamed candidate",),
        outcomes=("cancelled",),
        cancelled_content="authoritative partial",
    )
    app = _terminal_app(cast(PreparedRuntime, _runtime(conversation)))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("cancel"), "enter")
        await _wait_for_turn(app)

        group = app.query_one(".agent-run-activity-group")
        content = group.query_one(".agent-run-activity-content")
        assert content.display
        assert content.query_one(Markdown).source == "streamed candidate"
        assert not app.query("#conversation-display > .assistant-row")
        visible_text = _visible_screen_text(app)
        assert "streamed candidate" in visible_text
        assert "authoritative partial" not in visible_text
        assert visible_text.index("streamed candidate") < visible_text.index("Turn cancelled.")


@pytest.mark.asyncio
async def test_markdown_failure_still_closes_the_conversation_event_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = FailingRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))
    markdown_stream = FailingMarkdownStream()
    monkeypatch.setattr(
        Markdown,
        "get_stream",
        classmethod(lambda cls, markdown: markdown_stream),
    )

    with pytest.raises(RuntimeError, match="markdown write failed") as raised:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press(*list("fail"), "enter")

    assert conversation.closed.is_set()
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "markdown stop failed"


@pytest.mark.asyncio
async def test_runtime_cleanup_failure_does_not_mask_an_application_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = FailingRunSource()
    runtime = FailingCloseRuntime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))
    markdown_stream = FailingMarkdownStream()
    monkeypatch.setattr(
        Markdown,
        "get_stream",
        classmethod(lambda cls, markdown: markdown_stream),
    )

    with pytest.raises(RuntimeError, match="markdown write failed") as raised:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press(*list("fail"), "enter")

    assert isinstance(raised.value.__cause__, BaseExceptionGroup)
    cleanup_messages = {str(error) for error in raised.value.__cause__.exceptions}
    assert cleanup_messages == {"markdown stop failed", "runtime cleanup failed"}
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_terminal_conversation_uses_the_prepared_runtime_lifecycle(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider((_response(content="Prepared runtime answer."),))
    runtime = _prepared_runtime(agent_home, workspace, provider)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("h", "i", "enter")
        await _wait_for_turn(app)

        visible_text = _visible_screen_text(app)
        assert "hi" in visible_text
        assert "Prepared runtime answer." in visible_text

    assert provider.closed


@pytest.mark.asyncio
async def test_terminal_renders_reasoning_from_each_tool_loop_model_call(
    agent_home: Path,
    workspace: Path,
) -> None:
    class MultiReasoningProvider(_RuntimeProvider):
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
            if not (
                messages and messages[0] == {"role": "system", "content": session_title_prompt()}
            ):
                yield ReasoningDelta(delta=f"Reasoning segment {len(self.stream_requests) + 1}.")
            async for event in super().stream(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            ):
                yield event

    (workspace / "note.txt").write_text("note", encoding="utf-8")
    provider = MultiReasoningProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_read",
                    name="read_file",
                    arguments='{"path":"note.txt"}',
                ),
            ),
            _response(content="Final answer."),
        )
    )
    runtime = _prepared_runtime(agent_home, workspace, provider)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("inspect"), "enter")
        await _wait_for_turn(app)
        await pilot.click(".agent-run-activity-heading")
        await pilot.press("ctrl+home")

        visible_text = _visible_screen_text(app)
        assert "Reasoning segment 1." in visible_text
        assert "Reasoning segment 2." in visible_text
        assert "Final answer." in visible_text


@pytest.mark.asyncio
async def test_prepared_runtime_exec_confirmation_preserves_the_exact_long_command(
    agent_home: Path,
    workspace: Path,
) -> None:
    command = f'printf "{"x" * 300}" && rm -rf "build output"'
    provider = _RuntimeProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_long_exec",
                    name="exec",
                    arguments=json.dumps({"command": command, "cwd": ".", "timeout": 45}),
                ),
            ),
            _response(content="The command was declined."),
        )
    )
    runtime = _prepared_runtime(agent_home, workspace, provider)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("run it"), "enter"))
        await _wait_for_confirmation(app, pilot)

        details = [
            str(cast(Static, item).content) for item in app.screen.query(".confirmation-details")
        ]
        assert f"Command: {command}" in details
        assert "CWD: ." in details
        assert "Timeout: 45" in details

        await pilot.press("escape")
        await asyncio.wait_for(submission, timeout=2)
        await _wait_for_turn(app)
        async with asyncio.timeout(2):
            while "The command was declined." not in _visible_screen_text(app):
                await pilot.pause()
        assert "The command was declined." in _visible_screen_text(app)

    assert provider.closed


@pytest.mark.asyncio
async def test_prepared_runtime_close_cancels_the_pending_confirmation_future(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_close",
                    name="exec",
                    arguments=json.dumps({"command": 'rm -rf "build output"', "cwd": "."}),
                ),
            ),
        )
    )
    runtime = _prepared_runtime(agent_home, workspace, provider)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("run it"), "enter"))
        await _wait_for_confirmation(app, pilot)
        assert runtime.control.has_pending_confirmation

    await asyncio.gather(submission, return_exceptions=True)
    assert not runtime.control.has_pending_confirmation
    assert provider.closed


@pytest.mark.asyncio
async def test_prepared_runtime_cancellation_preserves_partial_and_allows_next_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = CancellableRuntimeProvider()
    runtime = _prepared_runtime(agent_home, workspace, provider)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("first"), "enter"))
        await asyncio.wait_for(provider.first_delta_emitted.wait(), timeout=1)
        await pilot.press("ctrl+c")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

        visible_text = _visible_screen_text(app)
        assert "partial runtime response" in visible_text
        assert "Turn cancelled." in visible_text

        await pilot.press(*list("again"), "enter")
        await _wait_for_turn(app)
        assert "Recovered runtime response." in _visible_screen_text(app)

    assert provider.closed


@pytest.mark.asyncio
async def test_narrow_terminal_uses_full_message_width_for_readable_content() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))
    content = "0123456789ABCDEFGHIJ"

    async with app.run_test(size=(30, 16)) as pilot:
        await pilot.press(*list(content), "enter")
        await asyncio.sleep(0.05)

        assert _content_text_nodes(app, content) == [content]


@pytest.mark.asyncio
async def test_wide_terminal_constrains_messages_to_a_comfortable_line_width() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))
    terminal_width = 80
    content = "X" * 100

    async with app.run_test(size=(terminal_width, 24)) as pilot:
        await pilot.press(*list(content), "enter")
        await asyncio.sleep(0.05)

        lines = _content_text_nodes(app, content)
        display_width = terminal_width - 4
        assert "".join(lines) == content
        assert max(map(len, lines)) / display_width == pytest.approx(0.72, abs=0.08)


@pytest.mark.asyncio
async def test_messages_are_side_aligned_with_role_accents_on_wide_terminals() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("user"), "enter")
        await asyncio.sleep(0.05)

        nodes = _screenshot_text_nodes(app)
        user_x = next(x for text, x, _ in nodes if text == "user")
        assistant_x = next(x for text, x, _ in nodes if text == "First answer.")
        accents = [(x, y) for text, x, y in nodes if text == "│"]
        screenshot_width = _screenshot_width(app)

        assert user_x > screenshot_width * 0.65
        assert assistant_x < screenshot_width * 0.25
        assert min(x for x, _ in accents) < screenshot_width * 0.25
        assert max(x for x, _ in accents) > screenshot_width * 0.75


@pytest.mark.asyncio
async def test_cjk_double_width_content_reflows_when_terminal_is_resized() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))
    content = "界" * 20

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list(content), "enter")
        await asyncio.sleep(0.05)
        assert _content_text_nodes(app, content) == [content]

        await pilot.resize_terminal(40, 18)
        await asyncio.sleep(0.05)
        narrow_lines = _content_text_nodes(app, content)
        assert len(narrow_lines) == 2
        assert "".join(narrow_lines) == content

        await pilot.resize_terminal(80, 24)
        await asyncio.sleep(0.05)
        assert _content_text_nodes(app, content) == [content]


@pytest.mark.asyncio
async def test_role_accents_remain_visible_with_limited_ansi_colors() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))
    app.ansi_color = True

    async with app.run_test(size=(40, 18)) as pilot:
        await pilot.press(*list("hello"), "enter")
        await asyncio.sleep(0.05)

        visible_text = _visible_screen_text(app)
        assert app.native_ansi_color
        assert "hello│" in visible_text
        assert "│First answer." in visible_text


@pytest.mark.asyncio
async def test_role_accents_remain_visible_without_terminal_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(40, 18)) as pilot:
        await pilot.press(*list("hello"), "enter")
        await asyncio.sleep(0.05)

        visible_text = _visible_screen_text(app)
        assert "nocolor" in app.screen.pseudo_classes
        assert "hello│" in visible_text
        assert "│First answer." in visible_text


@pytest.mark.asyncio
async def test_conversation_navigation_keys_keep_input_focus() -> None:
    content = "\n".join(f"line {index:02d} " + "x" * 30 for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("navigate"), "enter")
        await asyncio.sleep(0.1)
        display = app.query_one("#conversation-display")
        assert display.max_scroll_y > 0
        assert display.is_vertical_scroll_end

        await pilot.press("ctrl+home")
        assert display.scroll_y == 0
        assert isinstance(app.screen.focused, TextArea)

        await pilot.press("pagedown")
        assert 0 < display.scroll_y < display.max_scroll_y
        assert isinstance(app.screen.focused, TextArea)

        await pilot.press("pageup")
        assert display.scroll_y == 0
        assert isinstance(app.screen.focused, TextArea)

        await pilot.press("ctrl+end")
        assert display.is_vertical_scroll_end
        assert isinstance(app.screen.focused, TextArea)


@pytest.mark.asyncio
async def test_mouse_scroll_keeps_input_focus_and_scrollbar_is_overflow_only() -> None:
    content = "\n".join(f"line {index:02d} " + "x" * 30 for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("scroll"), "enter")
        await asyncio.sleep(0.1)
        display = app.query_one("#conversation-display")
        assert display.max_scroll_y > 0
        assert display.vertical_scrollbar.display

        await pilot._post_mouse_events([MouseScrollUp], offset=(10, 5), times=3)
        assert display.scroll_y < display.max_scroll_y
        assert isinstance(app.screen.focused, TextArea)

        await pilot._post_mouse_events([MouseScrollUp], offset=(10, 5), times=100)
        assert display.scroll_y == 0
        assert isinstance(app.screen.focused, TextArea)

        await pilot._post_mouse_events([MouseScrollDown], offset=(10, 5), times=100)
        assert display.is_vertical_scroll_end
        assert isinstance(app.screen.focused, TextArea)

    empty_runtime = _runtime(ScriptedRunSource())
    empty_app = _terminal_app(cast(PreparedRuntime, empty_runtime))
    async with empty_app.run_test(size=(60, 20)):
        empty_display = empty_app.query_one("#conversation-display")
        assert not empty_display.vertical_scrollbar.display


@pytest.mark.asyncio
async def test_conversation_scrollbar_supports_pointer_dragging() -> None:
    content = "\n".join(f"line {index:02d}" for index in range(100))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("drag"), "enter")
        await asyncio.sleep(0.1)
        display = app.query_one("#conversation-display")
        scrollbar = display.vertical_scrollbar
        start = (scrollbar.region.x, scrollbar.region.bottom - 2)
        end = (scrollbar.region.x, scrollbar.region.bottom - 5)

        await pilot._post_mouse_events([MouseDown], offset=start, button=1)
        await pilot._post_mouse_events([MouseMove], offset=end, button=1)
        await pilot._post_mouse_events([MouseUp], offset=end, button=1)

        assert display.scroll_y > 0
        assert not display.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_dragging_scrollbar_to_bottom_resumes_follow_mode() -> None:
    content = "\n".join(f"line {index:02d}" for index in range(100))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("seed"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        await pilot.press("ctrl+home")

        scrollbar = display.vertical_scrollbar
        start = (scrollbar.region.x, scrollbar.region.y + 1)
        end = (scrollbar.region.x, scrollbar.region.bottom - 2)
        await pilot._post_mouse_events([MouseDown], offset=start, button=1)
        await pilot._post_mouse_events([MouseMove], offset=end, button=1)
        await pilot._post_mouse_events([MouseUp], offset=end, button=1)
        await asyncio.sleep(0.2)
        assert display.is_vertical_scroll_end

        await pilot.press(*list("next"), "enter")
        await _wait_for_turn(app)
        await pilot.pause()

        assert display.is_vertical_scroll_end
        assert not app.query_one("#new-content").display


@pytest.mark.asyncio
async def test_historical_streaming_pauses_follow_and_exposes_new_content() -> None:
    history = " ".join(f"history {index:02d}" for index in range(100))
    conversation = ScriptedRunSource(
        deltas_by_submission=((history,), ("New ", "content.")),
        completed_contents=(history, "New content."),
    )
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("seed"), "enter")
        await _wait_for_turn(app)

        conversation.pause_after_next_first_delta()
        submission = asyncio.create_task(pilot.press(*list("stream"), "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            await asyncio.sleep(0.1)
            await pilot.pause()
            display = app.query_one("#conversation-display")
            assert display.max_scroll_y > 0
            async with asyncio.timeout(1):
                while not display.is_vertical_scroll_end:
                    await pilot.pause()
            assert display.is_vertical_scroll_end

            await pilot.press("ctrl+home")
            await asyncio.sleep(0.05)
            historical_position = display.scroll_y
            assert historical_position == 0
            assert not display.is_vertical_scroll_end

            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=2)
            await _wait_for_turn(app)
            await asyncio.sleep(0.1)

            assert display.scroll_y == historical_position
            assert not display.is_vertical_scroll_end
            assert app.query_one("#new-content").display

            await pilot.press("ctrl+end")
            assert display.is_vertical_scroll_end
            assert not app.query_one("#new-content").display
        finally:
            conversation.continue_turn()
            if not submission.done():
                await asyncio.wait_for(submission, timeout=2)
            await _wait_for_turn(app)


@pytest.mark.asyncio
async def test_historical_resize_preserves_a_visible_message_anchor() -> None:
    content = "\n".join(f"anchor {index:02d} " + "z" * 35 for index in range(50))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("resize"), "enter")
        await asyncio.sleep(0.1)
        display = app.query_one("#conversation-display")
        await pilot.press("ctrl+home")
        await pilot.press("pagedown")
        await asyncio.sleep(0.05)
        before_resize = _visible_screen_text(app)
        anchor = next(
            f"anchor {index:02d}" for index in range(50) if f"anchor {index:02d}" in before_resize
        )

        await pilot.resize_terminal(40, 24)
        await asyncio.sleep(0.1)
        after_resize = _visible_screen_text(app)

        assert anchor in after_resize
        assert not display.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_bottom_follow_is_preserved_when_resize_reflows_content() -> None:
    content = "\n".join(f"line {index:02d} " + "x" * 60 for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("resize"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        async with asyncio.timeout(1):
            while not display.is_vertical_scroll_end:
                await pilot.pause()
        assert display.is_vertical_scroll_end

        await pilot.resize_terminal(40, 24)
        async with asyncio.timeout(1):
            while not display.is_vertical_scroll_end:
                await pilot.pause()

        assert display.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_user_scroll_takes_over_from_rapid_resize_callbacks() -> None:
    content = "\n".join(f"resize {index:03d} " + "x" * 70 for index in range(90))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("resize"), "enter")
        await _wait_for_turn(app)
        await asyncio.sleep(0.05)
        display = app.query_one("#conversation-display")

        for width, height in ((65, 20), (42, 18), (76, 22), (50, 16), (80, 24)):
            await pilot.press("ctrl+home")
            async with asyncio.timeout(1):
                while display.is_vertical_scroll_end:
                    await pilot.pause()
            assert not display.is_vertical_scroll_end
            resize = asyncio.create_task(pilot.resize_terminal(width, height))
            await asyncio.sleep(0)
            await pilot.press("ctrl+end")
            await resize
            async with asyncio.timeout(1):
                while not display.is_vertical_scroll_end:
                    await pilot.pause()
            assert display.is_vertical_scroll_end
            assert display.following


@pytest.mark.asyncio
async def test_historical_resize_preserves_anchor_within_one_long_message() -> None:
    content = "\n".join(f"anchor {index:03d} " + "z" * 55 for index in range(80))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.press(*list("resize"), "enter")
        await _wait_for_turn(app)
        await pilot.press("ctrl+home", "pagedown", "pagedown")
        await pilot.pause()
        before_resize = _visible_screen_text(app)
        visible_anchors = {
            f"anchor {index:03d}" for index in range(80) if f"anchor {index:03d}" in before_resize
        }
        assert visible_anchors

        await pilot.resize_terminal(25, 24)
        await pilot.pause()
        await asyncio.sleep(0.2)
        after_resize = _visible_screen_text(app)

        assert any(anchor in after_resize for anchor in visible_anchors)


@pytest.mark.asyncio
async def test_shutdown_settles_stream_worker_before_runtime_close() -> None:
    conversation = BlockingRunSource()
    runtime = CloseOrderingRuntime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("block"), "enter")
        await asyncio.wait_for(conversation.started.wait(), timeout=1)

    assert conversation.closed.is_set()
    assert runtime.close_saw_stream_closed
    assert not list(app.query(".turn-status"))


@pytest.mark.asyncio
async def test_failed_stream_terminal_still_closes_runtime_once() -> None:
    conversation = FailingCancellationCleanupRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("block"), "enter")
        await asyncio.wait_for(conversation.started.wait(), timeout=1)

    assert runtime.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "undersized"),
    [
        ((19, 10), True),
        ((20, 9), True),
        ((20, 10), False),
    ],
)
async def test_minimum_terminal_size_has_an_exact_20_by_10_boundary(
    size: tuple[int, int],
    undersized: bool,
) -> None:
    app = _terminal_app(cast(PreparedRuntime, _runtime(ScriptedRunSource())))

    async with app.run_test(size=size):
        visible_text = _visible_screen_text(app)
        assert ("Resize to" in visible_text) is undersized
        assert app.query_one("#conversation-display").display is not undersized
        assert app.query_one("#conversation-input-region").display is not undersized


@pytest.mark.asyncio
async def test_undersized_terminal_replaces_presentation_and_recovers_input() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(19, 9)) as pilot:
        size_state = app.query_one("#size-insufficient", Static)
        assert size_state.display
        assert not app.query_one("#conversation-display").display
        assert not app.query_one("#conversation-input-region").display

        await pilot.press(*list("ignored"), "enter")
        await pilot.pause()
        assert conversation.submissions == []

        await pilot.resize_terminal(80, 24)
        await pilot.pause()

        assert not size_state.display
        assert app.query_one("#conversation-display").display
        assert app.query_one("#conversation-input-region").display

        await pilot.press(*list("ready"), "enter")
        await _wait_for_turn(app)

    assert conversation.submissions == ["ready"]


@pytest.mark.asyncio
async def test_undersized_recovery_preserves_completion_draft_history_and_focus() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press(*list("remember"), "enter")
        await _wait_for_turn(app)
        await pilot.press("/")
        assert app.command_completion_visible

        await pilot.resize_terminal(19, 9)
        await pilot.press("down", "enter", *list("ignored"))
        await pilot.pause()
        assert input_area.text == "/"

        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.05)

        assert app.command_completion_visible
        assert "/config" in _visible_screen_text(app)
        assert input_area.text == "/"
        assert app.screen.focused is input_area
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        await pilot.press("up")
        assert input_area.text == "remember"


@pytest.mark.asyncio
async def test_undersized_recovery_preserves_active_turn_state() -> None:
    conversation = BlockingRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("block"), "enter")
        await asyncio.wait_for(conversation.started.wait(), timeout=1)
        input_area = app.query_one("#conversation-input", TextArea)
        assert not input_area.read_only

        await pilot.resize_terminal(19, 9)
        await pilot.press(*list("ignored"), "enter", "ctrl+c")
        await pilot.pause()

        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.05)

        assert not input_area.read_only
        assert "Working" in _visible_screen_text(app)
        assert "partial" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_undersized_terminal_blocks_an_open_confirmation_until_recovery() -> None:
    conversation = ConfirmationRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("inspect"), "enter"))
        await asyncio.wait_for(conversation.confirmation_requested.wait(), timeout=1)
        await pilot.pause()

        await pilot.resize_terminal(19, 9)
        await pilot.pause()

        undersized_text = _visible_screen_text(app)
        assert "Terminal window" in undersized_text
        assert "Resize to" in undersized_text
        assert "Tool Confirmation" not in undersized_text
        await pilot.press("enter")
        await pilot.pause()
        assert conversation.responses == []

        await pilot.resize_terminal(80, 24)
        await _wait_for_confirmation(app, pilot)
        await pilot.pause(0.05)

        assert "Tool Confirmation" in _visible_screen_text(app)
        assert app.screen.focused is app.screen.query_one("#confirmation-decline", Button)
        await pilot.press("enter")
        await asyncio.wait_for(submission, timeout=1)
        await _wait_for_turn(app)

    assert conversation.responses == [
        (UUID("16fd2706-8baf-4334-8c7f-ada847da0314"), "declined"),
    ]


@pytest.mark.asyncio
async def test_resize_back_from_undersized_terminal_restores_historical_anchor() -> None:
    content = "\n".join(f"anchor {index:02d} " + "z" * 35 for index in range(50))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("anchor"), "enter")
        await _wait_for_turn(app)
        await asyncio.sleep(0.1)
        display = app.query_one("#conversation-display")
        await pilot.press("ctrl+home")
        await pilot.pause()
        assert not display.is_vertical_scroll_end
        before_resize = _visible_screen_text(app)
        anchor = next(
            f"anchor {index:02d}" for index in range(50) if f"anchor {index:02d}" in before_resize
        )

        await pilot.resize_terminal(19, 9)
        await pilot.pause()
        assert app.query_one("#size-insufficient", Static).display

        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.05)

        assert not app.query_one("#size-insufficient", Static).display
        assert anchor in _visible_screen_text(app)
        assert not display.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_resize_back_from_undersized_terminal_restores_bottom_follow() -> None:
    content = "\n".join(f"line {index:02d} " + "x" * 50 for index in range(50))
    conversation = ScriptedRunSource(deltas=(content,), completed_content=content)
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("bottom"), "enter")
        await _wait_for_turn(app)
        display = app.query_one("#conversation-display")
        async with asyncio.timeout(2):
            while not display.is_vertical_scroll_end:
                await pilot.pause()

        await pilot.resize_terminal(19, 9)
        await pilot.pause()
        assert app.query_one("#size-insufficient", Static).display

        await pilot.resize_terminal(80, 24)
        async with asyncio.timeout(2):
            while (
                app.query_one("#size-insufficient", Static).display
                or not display.is_vertical_scroll_end
            ):
                await pilot.pause()


@pytest.mark.asyncio
async def test_runtime_start_failure_restores_terminal_before_owner_cleanup() -> None:
    terminal_state = {"restored": False}

    class RecordingDriver(KeyboardLifecycleDriver):
        def start_application_mode(self) -> None:
            terminal_state["restored"] = False
            super().start_application_mode()

        def stop_application_mode(self) -> None:
            super().stop_application_mode()
            terminal_state["restored"] = True

    class RecordingRuntime(FailingStartRuntime):
        def __init__(self, conversation: _ScriptedSource) -> None:
            super().__init__(conversation)
            self.close_saw_terminal_restored = False

        async def close(self) -> None:
            self.close_saw_terminal_restored = terminal_state["restored"]
            await super().close()

    runtime = RecordingRuntime(ScriptedRunSource())
    app = _terminal_app(cast(PreparedRuntime, runtime))
    app.driver_class = RecordingDriver

    with pytest.raises(RuntimeError, match="runtime startup failed"):
        async with app.run_test(headless=False, size=(80, 24)):
            pass

    assert runtime.close_saw_terminal_restored
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_runtime_cleanup_failure_still_restores_terminal_first() -> None:
    terminal_state = {"restored": False}

    class RecordingDriver(KeyboardLifecycleDriver):
        def start_application_mode(self) -> None:
            terminal_state["restored"] = False
            super().start_application_mode()

        def stop_application_mode(self) -> None:
            super().stop_application_mode()
            terminal_state["restored"] = True

    class RecordingRuntime(FailingCloseRuntime):
        def __init__(self, conversation: _ScriptedSource) -> None:
            super().__init__(conversation)
            self.close_saw_terminal_restored = False

        async def close(self) -> None:
            self.close_saw_terminal_restored = terminal_state["restored"]
            await super().close()

    runtime = RecordingRuntime(ScriptedRunSource())
    app = _terminal_app(cast(PreparedRuntime, runtime))
    app.driver_class = RecordingDriver

    with pytest.raises(RuntimeError, match="runtime cleanup failed"):
        async with app.run_test(headless=False, size=(80, 24)) as pilot:
            app.exit()
            await pilot.pause()

    assert runtime.close_saw_terminal_restored


@pytest.mark.asyncio
async def test_runtime_start_and_cleanup_failure_preserves_the_start_error() -> None:
    class FailingStartAndCloseRuntime(FakePreparedRuntime):
        async def start(self) -> None:
            self.start_calls += 1
            raise RuntimeError("runtime startup failed")

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("runtime cleanup failed")

    runtime = FailingStartAndCloseRuntime(ScriptedRunSource())
    app = _terminal_app(cast(PreparedRuntime, runtime))
    app.driver_class = KeyboardLifecycleDriver

    with pytest.raises(RuntimeError, match="runtime startup failed") as raised:
        async with app.run_test(headless=False, size=(80, 24)):
            pass

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "runtime cleanup failed"
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_terminal_start_failure_attempts_application_mode_restore() -> None:
    class FailingStartDriver(KeyboardLifecycleDriver):
        def start_application_mode(self) -> None:
            self.write("application:start")
            self.write("\x1b[>1u")
            self.flush()
            raise RuntimeError("terminal startup failed")

    KeyboardLifecycleDriver.operations = []
    app = _terminal_app(cast(PreparedRuntime, _runtime(ScriptedRunSource())))
    app.driver_class = FailingStartDriver

    with pytest.raises(RuntimeError, match="terminal startup failed"):
        async with app.run_test(headless=False, size=(80, 24)):
            pass

    assert ("write", "application:stop") in KeyboardLifecycleDriver.operations


@pytest.mark.asyncio
async def test_terminal_stop_failure_still_restores_enhanced_keyboard_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")

    class FailingStopDriver(KeyboardLifecycleDriver):
        def stop_application_mode(self) -> None:
            raise RuntimeError("terminal cleanup failed")

    KeyboardLifecycleDriver.operations = []
    app = _terminal_app(cast(PreparedRuntime, _runtime(ScriptedRunSource())))
    app.driver_class = FailingStopDriver

    with pytest.raises(RuntimeError, match="terminal cleanup failed"):
        async with app.run_test(headless=False, size=(80, 24)) as pilot:
            app.exit()
            await pilot.pause()

    keyboard_writes = [
        value
        for operation, value in KeyboardLifecycleDriver.operations
        if operation == "write" and value in {"\x1b[>1u", "\x1b[<u"}
    ]
    assert keyboard_writes == ["\x1b[>1u", "\x1b[<u"]


@pytest.mark.asyncio
async def test_terminal_stop_failure_does_not_mask_an_application_body_failure() -> None:
    class FailingStopDriver(KeyboardLifecycleDriver):
        def stop_application_mode(self) -> None:
            raise RuntimeError("terminal cleanup failed")

    class FailingBodyApp(TerminalConversationApp):
        async def on_mount(self) -> None:
            await super().on_mount()
            raise RuntimeError("application body failed")

    runtime = _runtime(ScriptedRunSource())
    app = _terminal_app(
        cast(PreparedRuntime, runtime),
        app_type=FailingBodyApp,
    )
    app.driver_class = FailingStopDriver

    with pytest.raises(RuntimeError, match="application body failed") as raised:
        async with app.run_test(headless=False, size=(80, 24)):
            pass

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "terminal cleanup failed"


@pytest.mark.asyncio
async def test_partial_terminal_stop_failure_restores_all_modes_before_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    terminal_state = {
        "application_mode": False,
        "alternate_screen": False,
        "mouse_reporting": False,
        "cursor_hidden": False,
        "focus_reporting": False,
        "bracketed_paste": False,
        "kitty_depth": 0,
    }

    class PartialStopDriver(Driver):
        def __init__(
            self,
            app: App[object],
            *,
            debug: bool = False,
            mouse: bool = True,
            size: tuple[int, int] | None = None,
        ) -> None:
            super().__init__(app, debug=debug, mouse=mouse, size=size)
            self._restore_console: Callable[[], None] | None = None

        def write(self, data: str) -> None:
            transitions = {
                "\x1b[?1049h": ("alternate_screen", True),
                "\x1b[?1049l": ("alternate_screen", False),
                "\x1b[?1000h": ("mouse_reporting", True),
                "\x1b[?1000l": ("mouse_reporting", False),
                "\x1b[?25l": ("cursor_hidden", True),
                "\x1b[?25h": ("cursor_hidden", False),
                "\x1b[?1004h": ("focus_reporting", True),
                "\x1b[?1004l": ("focus_reporting", False),
                "\x1b[?2004h": ("bracketed_paste", True),
                "\x1b[?2004l": ("bracketed_paste", False),
            }
            for sequence, (name, enabled) in transitions.items():
                if sequence in data:
                    terminal_state[name] = enabled
            if "\x1b[>1u" in data:
                terminal_state["kitty_depth"] += 1
            if "\x1b[<u" in data:
                terminal_state["kitty_depth"] -= 1

        def flush(self) -> None:
            pass

        def start_application_mode(self) -> None:
            terminal_state["application_mode"] = True

            def restore_console() -> None:
                terminal_state["application_mode"] = False

            self._restore_console = restore_console
            for sequence in (
                "\x1b[?1049h",
                "\x1b[?1000h",
                "\x1b[?25l",
                "\x1b[?1004h",
                "\x1b[>1u",
                "\x1b[?2004h",
            ):
                self.write(sequence)

        def disable_input(self) -> None:
            self.write("\x1b[?1000l")

        def stop_application_mode(self) -> None:
            self.write("\x1b[?2004l")
            raise RuntimeError("terminal cleanup failed")

        def close(self) -> None:
            if self._restore_console is not None:
                self._restore_console()

    class RecordingRuntime(FakePreparedRuntime):
        def __init__(self, conversation: _ScriptedSource) -> None:
            super().__init__(conversation)
            self.close_saw_terminal_restored = False

        async def close(self) -> None:
            self.close_saw_terminal_restored = terminal_state == {
                "application_mode": False,
                "alternate_screen": False,
                "mouse_reporting": False,
                "cursor_hidden": False,
                "focus_reporting": False,
                "bracketed_paste": False,
                "kitty_depth": 0,
            }
            await super().close()

    runtime = RecordingRuntime(ScriptedRunSource())
    app = _terminal_app(cast(PreparedRuntime, runtime))
    app.driver_class = PartialStopDriver

    with pytest.raises(RuntimeError, match="terminal cleanup failed"):
        async with app.run_test(headless=False, size=(80, 24)) as pilot:
            app.exit()
            await pilot.pause()

    assert runtime.close_saw_terminal_restored


@pytest.mark.parametrize(
    "redirected_stream",
    ("stdin", "stdout", "stderr", "__stdin__", "__stdout__", "__stderr__"),
)
def test_non_tty_terminal_streams_are_rejected_before_textual_starts(
    monkeypatch: pytest.MonkeyPatch,
    redirected_stream: str,
) -> None:
    class TerminalStream:
        def __init__(self, interactive: bool) -> None:
            self._interactive = interactive

        def isatty(self) -> bool:
            return self._interactive

    stream_names = ("stdin", "stdout", "stderr", "__stdin__", "__stdout__", "__stderr__")
    for stream_name in stream_names:
        monkeypatch.setattr(sys, stream_name, TerminalStream(stream_name != redirected_stream))

    def app_must_not_start(_: object) -> None:
        raise AssertionError("Textual started for a non-TTY invocation")

    monkeypatch.setattr(TerminalConversationApp, "run", app_must_not_start)

    with pytest.raises(TerminalConversationError, match="interactive stdin, stdout, and stderr"):
        run_terminal_conversation(cast(PreparedRuntime, object()))


@pytest.mark.asyncio
async def test_management_completion_supports_keyboard_filtering_and_escape() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        input_area = app.query_one("#conversation-input", TextArea)

        await pilot.press("/")

        visible_commands = [
            text
            for text, _x, _y in _screenshot_text_nodes(app)
            if text in {"/config", "/status", "/resume", "/memory", "/dream"}
        ]
        assert visible_commands == [
            "/config",
            "/status",
            "/resume",
            "/memory",
            "/dream",
        ]
        assert any(text == "/" for text, _x, _y in _screenshot_text_nodes(app))
        assert app.screen.focused is input_area

        await pilot.press("down", "enter")

        assert input_area.text == "/status"
        assert app.screen.focused is input_area

        await pilot.press("ctrl+c", "/", "down", "up", "enter")

        assert input_area.text == "/config"

        await pilot.press("ctrl+c", "/", "escape")

        assert input_area.text == "/"
        assert not any(
            text in {"/config", "/status", "/resume", "/memory", "/dream"}
            for text, _x, _y in _screenshot_text_nodes(app)
        )
        assert app.screen.focused is input_area

        await pilot.press("m")

        assert [
            text
            for text, _x, _y in _screenshot_text_nodes(app)
            if text in {"/config", "/status", "/resume", "/memory", "/dream"}
        ] == ["/memory"]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((80, 24), (30, 12), (20, 10)))
async def test_management_completion_keeps_the_composer_visible(
    size: tuple[int, int],
) -> None:
    app = _terminal_app(cast(PreparedRuntime, _runtime(ScriptedRunSource())))

    async with app.run_test(size=size) as pilot:
        await pilot.press("/")

        visible_nodes = _screenshot_text_nodes(app)
        assert [
            text
            for text, _x, _y in visible_nodes
            if text in {"/config", "/status", "/resume", "/memory", "/dream"}
        ] == ["/config", "/status", "/resume", "/memory", "/dream"]
        assert any(text == "/" for text, _x, _y in visible_nodes)
        assert isinstance(app.screen.focused, TextArea)


@pytest.mark.asyncio
async def test_management_completion_mouse_selection_updates_the_composer() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/")

        await pilot.click("#command-completion", offset=(2, 2))

        assert app.query_one("#conversation-input", TextArea).text == "/status"
        assert isinstance(app.screen.focused, TextArea)
        assert not any(
            text in {"/config", "/resume", "/memory", "/dream"}
            for text, _x, _y in _screenshot_text_nodes(app)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "output_marker"),
    (
        ("/config", "Path:"),
        ("/status", '"version": "0.1.0"'),
        ("/memory", "# Long-term Memory"),
        ("/dream", "No pending summaries"),
    ),
)
async def test_supported_management_commands_use_the_prepared_runtime_without_session_messages(
    agent_home: Path,
    workspace: Path,
    command: str,
    output_marker: str,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _prepared_runtime(agent_home, workspace, provider)
    original_session = runtime.session
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list(command), "enter")
        await pilot.pause()
        await pilot.press("ctrl+home")

        visible_text = _visible_screen_text(app)
        assert f"Command: {command}" in visible_text
        assert output_marker in visible_text
        assert runtime.session is original_session
        assert original_session.messages == []
        assert provider.stream_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ("/ordinary", "/CONFIG", "/config extra", "/memory "))
async def test_inexact_slash_input_reaches_the_active_message_bus_unchanged(
    agent_home: Path,
    workspace: Path,
    command: str,
) -> None:
    provider = _RuntimeProvider((_response(content="Ordinary slash response."),))
    runtime = _prepared_runtime(agent_home, workspace, provider)
    original_session = runtime.session
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list(command), "enter")
        await _wait_for_turn(app)

        assert runtime.session is original_session
        assert original_session.messages[0]["role"] == "user"
        assert original_session.messages[0]["content"] == command
        assert provider.stream_requests


@pytest.mark.asyncio
async def test_empty_resume_picker_cancellation_preserves_existing_management_rows(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/dream"), "enter")
        await pilot.press("ctrl+home")
        before_resume = _visible_screen_text(app)
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)

        assert "No resumable Conversation Sessions." in _visible_screen_text(app)
        await pilot.click(offset=(1, 1))
        assert app.screen.id == "session-picker"
        await pilot.press("escape", "ctrl+home")
        await pilot.pause()

        visible_text = _visible_screen_text(app)
        assert visible_text == before_resume
        assert "Command: /dream" in visible_text
        assert "No pending summaries" in visible_text
        assert "Command: /resume" not in visible_text
        assert runtime.session.messages == []
        assert provider.stream_requests == []


@pytest.mark.asyncio
async def test_resume_opens_a_picker_with_title_and_local_update_time(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    older = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW.replace(hour=10),
        new_uuid=lambda: UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
    )
    older.update_metadata(title="Older session")
    older.add_message("user", "Older persisted question.")
    older.close()
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )
    target.update_metadata(title="Target session")
    target.add_message("user", "Persisted question.")
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)

        visible_text = _visible_screen_text(app)
        assert app.screen.id == "session-picker"
        assert "Target session" in visible_text
        assert visible_text.index("Target session") < visible_text.index("Older session")
        assert target.session_id not in visible_text
        assert older.session_id not in visible_text
        assert target.updated_at.astimezone().strftime("%Y-%m-%d %H:%M") in visible_text
        assert app.screen.focused is app.screen.query_one("#session-picker-options", OptionList)


@pytest.mark.asyncio
async def test_resume_selection_rebuilds_the_display_from_the_selected_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Restored session")
    target.add_message("user", "Persisted question.")
    target.add_message(
        "assistant",
        "Persisted **answer**.",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
        },
    )
    target.add_message(
        "assistant",
        "",
        tool_calls=[
            {
                "id": "call-restored",
                "name": "read_file",
                "arguments": '{"api_key":"private"}',
            },
            {"id": "call-error", "name": "exec", "arguments": '{"command":"private"}'},
            {"id": "call-refused", "name": "web_fetch", "arguments": '{"url":"private"}'},
        ],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message(
        "tool",
        "private tool result",
        tool_call_id="call-restored",
        name="read_file",
        status="success",
        artifact=None,
    )
    target.add_message(
        "tool",
        "STDERR permission denied; STDOUT secret bytes",
        tool_call_id="call-error",
        name="exec",
        status="error",
        artifact=None,
    )
    target.add_message(
        "tool",
        "private refusal detail",
        tool_call_id="call-refused",
        name="web_fetch",
        status="refused",
        artifact=None,
    )
    target.add_message(
        "assistant",
        "Persisted partial answer.",
        tool_calls=[],
        status="interrupted",
        error={"code": "turn_cancelled", "message": "Turn interrupted by user."},
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message(
        "assistant",
        "",
        tool_calls=[],
        status="error",
        error={"code": "model_failed", "message": "Persisted model failure."},
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 0,
            "total_tokens": 1,
        },
    )
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press(*list("/status"), "enter")
        await pilot.pause()
        await pilot.press("ctrl+home")
        assert "Command: /status" in _visible_screen_text(app)

        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        expected_projection = (
            "Persisted question.",
            "Persisted answer.",
            "Completed: read_file",
            "Failed: exec - The operation did not complete.",
            "Rejected: web_fetch",
            "Persisted partial answer.",
            "Persisted model failure.",
        )
        async with asyncio.timeout(2):
            while runtime.session.session_id != target.session_id or any(
                expected not in _visible_screen_text(app) for expected in expected_projection
            ):
                await pilot.pause()

        visible_text = _visible_screen_text(app)
        assert app.screen.id == "_default"
        assert runtime.session.session_id == target.session_id
        assert "Persisted question." in visible_text
        assert "Persisted answer." in visible_text
        assert "Completed: read_file" in visible_text
        assert "Failed: exec - The operation did not complete." in visible_text
        assert "Rejected: web_fetch" in visible_text
        assert "Persisted partial answer." in visible_text
        assert "Turn cancelled." not in visible_text
        assert "Persisted model failure." in visible_text
        assert "private tool result" not in visible_text
        assert "private refusal detail" not in visible_text
        assert "call-restored" not in visible_text
        assert "call-error" not in visible_text
        assert "call-refused" not in visible_text
        assert "api_key" not in visible_text
        assert "STDERR" not in visible_text
        assert "STDOUT" not in visible_text
        assert "secret bytes" not in visible_text
        assert visible_text.index("Persisted answer.") < visible_text.index("Completed: read_file")
        assert visible_text.index("Completed: read_file") < visible_text.index("Failed: exec")
        assert visible_text.index("Failed: exec") < visible_text.index("Rejected: web_fetch")
        assert visible_text.index("Rejected: web_fetch") < visible_text.index(
            "Persisted partial answer."
        )
        assert "Command: /status" not in visible_text
        assert "Command: /resume" not in visible_text
        assert app.screen.focused is app.query_one("#conversation-input", TextArea)


@pytest.mark.asyncio
async def test_resume_rebuilds_a_successful_tool_run_activity_group(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    timestamps = iter(NOW + timedelta(seconds=offset) for offset in range(10))
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: next(timestamps),
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Restored tool run")
    target.add_message("user", "Read the file.")
    target.add_message(
        "assistant",
        "I will inspect it first.",
        tool_calls=[{"id": "call-restored", "name": "read_file", "arguments": '{"path":"x"}'}],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message(
        "tool",
        "private result",
        tool_call_id="call-restored",
        name="read_file",
        status="success",
        artifact=None,
    )
    target.add_message(
        "assistant",
        "The file is ready.",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while (
                runtime.session.session_id != target.session_id
                or "The file is ready." not in _visible_screen_text(app)
            ):
                await pilot.pause()

        group = app.query_one(".agent-run-activity-group")
        activity = group.query_one(".agent-run-activity-content")
        visible_text = _visible_screen_text(app)

        assert "The file is ready." in visible_text
        assert activity.query_one(Markdown).source == "I will inspect it first."
        assert not activity.display
        assert "\u25b6 3s" in visible_text
        await pilot.click(".agent-run-activity-heading")
        visible_text = _visible_screen_text(app)
        assert "private result" not in visible_text
        assert "Completed: read_file" in visible_text
        assert visible_text.index("I will inspect it first.") < visible_text.index(
            "Completed: read_file"
        )
        assert visible_text.index("Completed: read_file") < visible_text.index("The file is ready.")


@pytest.mark.asyncio
async def test_resume_groups_a_recognizable_tool_result_after_the_final_response(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    timestamps = iter(NOW + timedelta(seconds=offset) for offset in range(10))
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: next(timestamps),
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Late recognizable Tool result")
    target.add_message("user", "Read the file.")
    target.add_message(
        "assistant",
        "I will inspect it first.",
        tool_calls=[{"id": "call-late", "name": "read_file", "arguments": '{"path":"x"}'}],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message(
        "assistant",
        "The file is ready.",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message(
        "tool",
        "private result",
        tool_call_id="call-late",
        name="read_file",
        status="success",
        artifact=None,
    )
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "The file is ready." not in _visible_screen_text(app):
                await pilot.pause()

        group = app.query_one(".agent-run-activity-group")
        activity = group.query_one(".agent-run-activity-content")
        visible_text = _visible_screen_text(app)

        assert not activity.display
        assert "\u25b6 2s" in visible_text
        assert "The file is ready." in visible_text
        assert "I will inspect it first." not in visible_text
        assert "Completed: read_file" not in visible_text

        await pilot.click(".agent-run-activity-heading")
        visible_text = _visible_screen_text(app)
        assert "private result" not in visible_text
        assert visible_text.index("I will inspect it first.") < visible_text.index(
            "Completed: read_file"
        )
        assert visible_text.index("Completed: read_file") < visible_text.index("The file is ready.")


@pytest.mark.asyncio
async def test_resume_uses_the_last_completed_no_tool_assistant_as_final_response(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    timestamps = iter(NOW + timedelta(seconds=offset) for offset in range(12))
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: next(timestamps),
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Multiple final candidates")
    target.add_message("user", "First question")
    target.add_message(
        "assistant",
        "Earlier failed activity.",
        tool_calls=[],
        status="error",
        error={"code": "model_failed", "message": "Stale model failure."},
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message(
        "assistant",
        "Earlier activity.",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message(
        "assistant",
        "Final answer.",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message("user", "Second question")
    target.add_message(
        "assistant",
        "   ",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "Final answer." not in _visible_screen_text(app):
                await pilot.pause()

        groups = list(app.query(".agent-run-activity-group"))
        assert len(groups) == 1
        group = groups[0]
        activity = group.query_one(".agent-run-activity-content")
        visible_text = _visible_screen_text(app)
        assert "Final answer." in visible_text
        assert "\u25b6 3s" in visible_text
        assert not activity.display
        assert "Earlier activity." not in visible_text
        assert "Completed with no response." in visible_text
        assert "Stale model failure." not in visible_text

        await pilot.click(".agent-run-activity-heading")
        visible_text = _visible_screen_text(app)
        assert "Earlier failed activity." in visible_text
        assert "Earlier activity." in visible_text
        assert visible_text.index("Earlier activity.") < visible_text.index("Final answer.")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error", "expected_status"),
    [
        (
            "interrupted",
            {"code": "turn_cancelled", "message": "cancelled"},
            "Turn cancelled.",
        ),
        (
            "error",
            {"code": "model_failed", "message": "Persisted model failure."},
            "Persisted model failure.",
        ),
    ],
)
async def test_resume_expands_cancelled_and_failed_activity_groups(
    agent_home: Path,
    workspace: Path,
    status: Literal["interrupted", "error"],
    error: dict[str, str],
    expected_status: str,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    timestamps = iter(NOW + timedelta(seconds=offset) for offset in range(8))
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: next(timestamps),
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title=f"Restored {status} run")
    target.add_message("user", "Run the operation.")
    stale_status: Literal["interrupted", "error"] = (
        "error" if status == "interrupted" else "interrupted"
    )
    stale_error = (
        {"code": "model_failed", "message": "Stale model failure."}
        if stale_status == "error"
        else {"code": "turn_cancelled", "message": "cancelled"}
    )
    target.add_message(
        "assistant",
        "Earlier terminal activity.",
        tool_calls=[],
        status=stale_status,
        error=stale_error,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message(
        "assistant",
        "Partial activity.",
        tool_calls=[],
        status=status,
        error=error,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "Partial activity." not in _visible_screen_text(app):
                await pilot.pause()

        group = app.query_one(".agent-run-activity-group")
        content = group.query_one(".agent-run-activity-content")
        visible_text = _visible_screen_text(app)
        assert content.display
        assert str(group.query_one(".agent-run-activity-heading", Static).content) == "\u25bc 2s"
        assert expected_status in visible_text
        stale_terminal_status = (
            "Stale model failure." if status == "interrupted" else "Turn cancelled."
        )
        assert stale_terminal_status not in visible_text

        await pilot.click(".agent-run-activity-heading")
        assert not content.display


@pytest.mark.asyncio
async def test_resume_keeps_unknown_outcome_expanded_without_inventing_status(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    timestamps = iter(NOW + timedelta(seconds=offset) for offset in range(6))
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: next(timestamps),
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Unknown outcome")
    target.add_message("user", "Wait for the operation.")
    target.add_message(
        "assistant",
        "Still waiting for a result.",
        tool_calls=[{"id": "call-pending", "name": "read_file", "arguments": "{}"}],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "Still waiting for a result." not in _visible_screen_text(app):
                await pilot.pause()

        group = app.query_one(".agent-run-activity-group")
        content = group.query_one(".agent-run-activity-content")
        visible_text = _visible_screen_text(app)
        assert content.display
        assert str(group.query_one(".agent-run-activity-heading", Static).content) == "\u25bc 1s"
        assert "Turn cancelled." not in visible_text
        assert "Turn failed." not in visible_text
        assert "Completed with no response." not in visible_text

        await pilot.click(".agent-run-activity-heading")
        assert not content.display


@pytest.mark.asyncio
async def test_resume_clamps_reversed_historical_duration_to_zero(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Reversed timestamps")
    target.add_message("user", "Run with reversed timestamps.")
    target.add_message(
        "assistant",
        "Historical activity.",
        tool_calls=[{"id": "call-reversed", "name": "read_file", "arguments": "{}"}],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.messages[0]["timestamp"] = (NOW + timedelta(seconds=5)).isoformat(
        timespec="milliseconds"
    )
    target.messages[1]["timestamp"] = NOW.isoformat(timespec="milliseconds")
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "Historical activity." not in _visible_screen_text(app):
                await pilot.pause()

        heading = app.query_one(".agent-run-activity-heading", Static)
        assert str(heading.content) == "\u25bc 0s"


@pytest.mark.asyncio
async def test_resume_keeps_pre_user_messages_and_unclassifiable_runs_flat(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Unclassifiable history")
    target.add_message(
        "assistant",
        "Before the first user message.",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    target.add_message(
        "tool",
        "private pre-user result",
        tool_call_id="orphan-pre-user",
        name="read_file",
        status="success",
        artifact=None,
    )
    target.add_message("user", "Question with an orphan tool result.")
    target.add_message(
        "tool",
        "private orphan result",
        tool_call_id="orphan-in-run",
        name="read_file",
        status="success",
        artifact=None,
    )
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "Question with an orphan tool result." not in _visible_screen_text(app):
                await pilot.pause()

        visible_text = _visible_screen_text(app)
        assert not list(app.query(".agent-run-activity-group"))
        assert "Before the first user message." in visible_text
        assert "Question with an orphan tool result." in visible_text
        assert visible_text.count("Completed: read_file") == 2
        assert "private pre-user result" not in visible_text
        assert "private orphan result" not in visible_text


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_key", ("escape", "ctrl+c"))
async def test_resume_picker_cancellation_and_outside_click_preserve_current_display(
    agent_home: Path,
    workspace: Path,
    cancel_key: str,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    initial_session_id = runtime.session.session_id
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )
    target.update_metadata(title="Target session")
    target.add_message("user", "Target content.")
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/status"), "enter")
        await pilot.pause()
        await pilot.press("ctrl+home")
        before_resume = _visible_screen_text(app)
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)

        await pilot.click(offset=(1, 1))
        assert app.screen.id == "session-picker"
        await pilot.press("left", "right")
        assert app.screen.id == "session-picker"
        assert runtime.session.session_id == initial_session_id
        assert app.screen.query_one("#session-picker-options", OptionList).has_focus
        await pilot.press(cancel_key)
        await pilot.pause()
        assert app.is_running
        input_area = app.query_one("#conversation-input", TextArea)
        assert not input_area.read_only
        await pilot.press("ctrl+home")

        visible_text = _visible_screen_text(app)
        assert app.screen.id == "_default"
        assert runtime.session.session_id == initial_session_id
        assert visible_text == before_resume
        assert "Command: /status" in visible_text
        assert "Command: /resume" not in visible_text
        assert "Target content." not in visible_text
        assert app.screen.focused is input_area


@pytest.mark.asyncio
async def test_resume_picker_mouse_selection_switches_to_the_clicked_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Mouse target")
    target.add_message("user", "Mouse-selected content.")
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.click("#session-picker-options", offset=(4, 1))

        async with asyncio.timeout(2):
            while (
                runtime.session.session_id != target.session_id
                or "Mouse-selected content." not in _visible_screen_text(app)
            ):
                await pilot.pause()

        assert runtime.session.session_id == target.session_id
        assert "Mouse-selected content." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_resume_failure_after_a_stale_listing_preserves_the_current_display(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    initial_session_id = runtime.session.session_id
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )
    target.update_metadata(title="Stale target")
    target.add_message("user", "Should not be restored.")
    target.close()
    target_path = target.workspace_state.sessions_directory / f"{target.session_id}.jsonl"
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/status"), "enter")
        await pilot.pause()
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        target_path.unlink()
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "model_invalid_request:" not in _visible_screen_text(app):
                await pilot.pause()
        failure_text = _visible_screen_text(app)
        await pilot.press("ctrl+home")

        visible_text = _visible_screen_text(app)
        assert runtime.session.session_id == initial_session_id
        assert "Command: /status" in visible_text
        assert "Should not be restored." not in visible_text
        assert "model_invalid_request:" in failure_text
        assert "not\nresumable." in failure_text


@pytest.mark.asyncio
async def test_resume_picker_scrolls_in_management_order_and_selects_by_keyboard(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    sessions: list[Session] = []
    for index in range(24):
        session_now = NOW.replace(minute=index)
        session_uuid = UUID(f"00000000-0000-4000-8000-{index + 1:012x}")
        session = Session.create(
            runtime.session.workspace_state,
            now=_constant_datetime(session_now),
            new_uuid=_constant_uuid(session_uuid),
        )
        session.update_metadata(title=f"Session {index:02d}")
        session.add_message("user", f"Content {index:02d}.")
        session.close()
        sessions.append(session)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)

        visible_text = _visible_screen_text(app)
        assert visible_text.index("Session 23") < visible_text.index("Session 22")
        options = app.screen.query_one("#session-picker-options", OptionList)
        assert options.max_scroll_y > 0

        await pilot._post_mouse_events([MouseScrollDown], offset=(40, 10), times=3)
        await pilot.pause()
        assert options.scroll_y > 0
        await pilot.press(*(("down",) * 23))
        await pilot.pause()
        assert options.scroll_y > 0
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while runtime.session.session_id != sessions[
                0
            ].session_id or "Content 00." not in _visible_screen_text(app):
                await pilot.pause()
        assert "Content 00." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_resume_picker_reports_corrupt_entries_without_mutating_them(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Valid target")
    target.add_message("user", "Valid restored content.")
    target.close()
    corrupt_path = target.workspace_state.sessions_directory / (
        "20260811-120000-000000_00000000-0000-0000-0000-000000000000.jsonl"
    )
    corrupt_content = b"{not-json\n"
    corrupt_path.write_bytes(corrupt_content)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)

        visible_text = _visible_screen_text(app)
        assert "Skipped 1 corrupt Conversation Session." in visible_text
        assert "Valid target" in visible_text
        assert corrupt_path.read_bytes() == corrupt_content

        await pilot.press("enter")
        async with asyncio.timeout(2):
            while (
                runtime.session.session_id != target.session_id
                or "Valid restored content." not in _visible_screen_text(app)
            ):
                await pilot.pause()

        assert corrupt_path.read_bytes() == corrupt_content
        assert "Valid restored content." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_resume_listing_failure_preserves_session_and_existing_display(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    initial_session = runtime.session
    original_dispatch = runtime.management_dispatcher.dispatch

    async def dispatch_with_listing_failure(command: str) -> ManagementCommandResult:
        if command == "/resume":
            return ManagementCommandResult(
                handled=True,
                output="persistence_error: Conversation Sessions could not be listed.",
            )
        return await original_dispatch(command)

    monkeypatch.setattr(runtime.management_dispatcher, "dispatch", dispatch_with_listing_failure)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/status"), "enter")
        await pilot.press(*list("/resume"), "enter")
        async with asyncio.timeout(2):
            while "could not be listed" not in _visible_screen_text(app):
                await pilot.pause()

        failure_text = _visible_screen_text(app)
        await pilot.press("ctrl+home")
        visible_text = _visible_screen_text(app)
        assert app.screen.id == "_default"
        assert runtime.session is initial_session
        assert "Command: /status" in visible_text
        assert "Conversation Sessions could not be listed." in failure_text


@pytest.mark.asyncio
async def test_resume_requires_result_and_runtime_authority_to_agree(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    initial_session = runtime.session
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Authority target")
    target.add_message("user", "Must not be projected without authority.")
    target.close()

    async def inconsistent_resume(
        session_id: str,
        *,
        force: bool = False,
    ) -> ManagementCommandResult:
        del force
        return ManagementCommandResult(
            handled=True,
            output=f"Resumed session {session_id}.",
            resumed_session_id=session_id,
        )

    monkeypatch.setattr(runtime.management_dispatcher, "resume", inconsistent_resume)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/status"), "enter")
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "did not select" not in _visible_screen_text(app):
                await pilot.pause()
        failure_text = _visible_screen_text(app)
        await pilot.press("ctrl+home")

        visible_text = _visible_screen_text(app)
        assert runtime.session is initial_session
        assert "Command: /status" in visible_text
        assert "Must not be projected without authority." not in visible_text
        assert "Session resume did not select" in failure_text


@pytest.mark.asyncio
async def test_unexpected_resume_exception_preserves_session_display_and_interaction(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    initial_session = runtime.session
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Failing target")
    target.add_message("user", "Must remain hidden after an exception.")
    target.close()

    async def failing_resume(
        session_id: str,
        *,
        force: bool = False,
    ) -> ManagementCommandResult:
        del session_id, force
        raise RuntimeError("private failure detail")

    monkeypatch.setattr(runtime.management_dispatcher, "resume", failing_resume)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/status"), "enter")
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        async with asyncio.timeout(2):
            while "Session resume failed." not in _visible_screen_text(app):
                await pilot.pause()
        failure_text = _visible_screen_text(app)
        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("ctrl+home")

        visible_text = _visible_screen_text(app)
        assert runtime.session is initial_session
        assert app.is_running
        assert not input_area.read_only
        assert "Command: /status" in visible_text
        assert "Must remain hidden after an exception." not in visible_text
        assert "Session resume failed." in failure_text
        assert "private failure detail" not in failure_text


@pytest.mark.asyncio
async def test_resume_selection_serializes_input_until_rebuild_finishes(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Delayed target")
    target.add_message("user", "Delayed restored content.")
    target.close()
    resume_started = asyncio.Event()
    continue_resume = asyncio.Event()
    resume_finished = asyncio.Event()
    resume_errors: list[BaseException] = []
    original_resume = runtime.management_dispatcher.resume

    async def delayed_resume(
        session_id: str,
        *,
        force: bool = False,
    ) -> ManagementCommandResult:
        resume_started.set()
        await continue_resume.wait()
        try:
            return await original_resume(session_id, force=force)
        except BaseException as error:
            resume_errors.append(error)
            raise
        finally:
            resume_finished.set()

    monkeypatch.setattr(runtime.management_dispatcher, "resume", delayed_resume)
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")
        await asyncio.wait_for(resume_started.wait(), timeout=1)

        input_area = app.query_one("#conversation-input", TextArea)
        assert input_area.read_only
        assert app.screen.id == "_default"
        await pilot.press(*list("racing turn"), "enter", *list("/resume"), "enter")
        assert input_area.text == ""
        assert provider.stream_requests == []
        assert app.screen.id == "_default"

        continue_resume.set()
        await asyncio.wait_for(resume_finished.wait(), timeout=1)
        assert resume_errors == []
        async with asyncio.timeout(2):
            while (
                runtime.session.session_id != target.session_id
                or input_area.read_only
                or "Delayed restored content." not in _visible_screen_text(app)
            ):
                await pilot.pause()

        assert "Delayed restored content." in _visible_screen_text(app)
        await pilot.press(*list("/status"), "enter")
        await pilot.press("ctrl+home")
        assert "Command: /status" in _visible_screen_text(app)
        assert provider.stream_requests == []


@pytest.mark.asyncio
async def test_resumed_long_history_starts_latest_and_preserves_runtime_input_history(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _generation_host(agent_home, workspace, provider)
    target = Session.create(
        runtime.session.workspace_state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
    )
    target.update_metadata(title="Long target")
    for index in range(60):
        target.add_message("user", f"Restored line {index:02d} " + "x" * 40)
    target.close()
    app = _terminal_app(runtime)

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        await _wait_for_session_picker(app, pilot)
        await pilot.press("enter")

        display = app.query_one("#conversation-display")
        async with asyncio.timeout(2):
            while (
                runtime.session.session_id != target.session_id
                or not display.is_vertical_scroll_end
                or "Restored line 59" not in _visible_screen_text(app)
            ):
                await pilot.pause()

        assert "Restored line 59" in _visible_screen_text(app)
        assert not app.query_one("#new-content").display
        await pilot.resize_terminal(40, 20)
        await pilot.pause()
        assert display.is_vertical_scroll_end

        input_area = app.query_one("#conversation-input", TextArea)
        assert app.screen.focused is input_area
        await pilot.press("up")
        assert input_area.text == "/resume"


@pytest.mark.asyncio
async def test_management_error_row_preserves_later_command_interaction(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(())
    runtime = _prepared_runtime(agent_home, workspace, provider)
    runtime.session.workspace_state.long_term_memory_path.unlink()
    app = _terminal_app(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/memory"), "enter")
        await pilot.press(*list("/dream"), "enter")
        await pilot.press("ctrl+home")

        visible_text = _visible_screen_text(app)
        assert "persistence_error: Long-term Memory could not be read." in visible_text
        assert "No pending summaries" in visible_text
        assert runtime.session.messages == []
        assert provider.stream_requests == []


@pytest.mark.asyncio
async def test_completion_direction_keys_take_precedence_over_composer_and_input_history() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("previous"), "enter")
        await _wait_for_turn(app)

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("/")
        for direction in ("left", "right"):
            cursor_before = input_area.cursor_location
            await pilot.press(direction)
            assert input_area.cursor_location == cursor_before
            assert input_area.text == "/"
            assert any(text == "/config" for text, _x, _y in _screenshot_text_nodes(app))

        await pilot.press("up", "down")
        assert input_area.text == "/"

        await pilot.press("escape", "ctrl+c", "up")
        assert input_area.text == "previous"


@pytest.mark.asyncio
async def test_completion_ctrl_c_closes_completion_before_idle_draft_behavior() -> None:
    conversation = ScriptedRunSource()
    runtime = _runtime(conversation)
    app = _terminal_app(cast(PreparedRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("previous"), "enter")
        await _wait_for_turn(app)

        input_area = app.query_one("#conversation-input", TextArea)
        await pilot.press("/")
        await pilot.pause()
        assert any(text == "/config" for text, _x, _y in _screenshot_text_nodes(app))

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert input_area.text == "/"
        assert not any(
            text in {"/config", "/status", "/resume", "/memory", "/dream"}
            for text, _x, _y in _screenshot_text_nodes(app)
        )
        assert app.is_running
        assert app.screen.focused is input_area

        await pilot.press("ctrl+c")
        assert input_area.text == ""
        assert app.is_running

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running
