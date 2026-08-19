from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from typing import Any

import pytest

from myclaw.agent.run import (
    AgentRun,
    AgentRunCompletedPayload,
    AgentRunConfirmationRequestedPayload,
    AgentRunModelCallCompletedPayload,
    AgentRunPayload,
    AgentRunStartedPayload,
    AgentRunTextDeltaPayload,
)
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.tools.base import ArtifactReference, BaseTool
from myclaw.tools.tool_gateway import ConfirmationChannel, ModelToolCall, ToolResult
from tests.fixtures import FakeTool, SingleToolGateway


def test_run_interface_accepts_only_execution_inputs() -> None:
    assert tuple(inspect.signature(AgentRun.run).parameters) == (
        "self",
        "messages",
        "current_user",
        "route",
        "emitter",
        "confirmation",
        "externalize_result",
        "cancel_requested",
    )


class _RecordingEmitter:
    def __init__(self) -> None:
        self.payloads: list[AgentRunPayload] = []

    async def emit(self, payload: AgentRunPayload) -> None:
        self.payloads.append(payload)


class _SignallingEmitter(_RecordingEmitter):
    def __init__(self, signal_type: str) -> None:
        super().__init__()
        self._signal_type = signal_type
        self.signalled = asyncio.Event()

    async def emit(self, payload: AgentRunPayload) -> None:
        await super().emit(payload)
        if payload.type == self._signal_type:
            self.signalled.set()


class _DirectRouter:
    def __init__(
        self,
        events: Sequence[ModelStreamEvent],
        *follow_up_events: Sequence[ModelStreamEvent],
    ) -> None:
        self._scripts = [tuple(events), *(tuple(script) for script in follow_up_events)]
        self.calls: list[tuple[str, list[dict[str, Any]], tuple[object, ...]]] = []

    def route_status(self, route: str) -> None:
        del route

    def stream(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[object],
    ) -> AsyncIterator[ModelStreamEvent]:
        captured = deepcopy(list(messages))
        self.calls.append((route, captured, tuple(tools)))
        script = self._scripts.pop(0)
        if len(self.calls) == 2:
            messages[2]["content"] = "provider mutation"

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            for event in script:
                yield event

        return replay()

    async def complete(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[object],
    ) -> ModelResponse:
        del route, messages, tools
        raise AssertionError("Unexpected complete call")

    async def close(self) -> None:
        return None


class _FailingDirectRouter(_DirectRouter):
    def __init__(
        self,
        *,
        error: ErrorInfo | None = None,
        deltas: Sequence[str] = (),
    ) -> None:
        super().__init__(())
        self._error = error or ErrorInfo(code="model_failed", message="provider failed")
        self._deltas = tuple(deltas)

    def stream(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[object],
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, messages, tools

        async def fail() -> AsyncIterator[ModelStreamEvent]:
            for delta in self._deltas:
                yield TextDelta(delta=delta)
            raise ModelCallError(self._error)

        return fail()


class _CompletingDirectRouter:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, list[dict[str, Any]], tuple[object, ...]]] = []

    def route_status(self, route: str) -> None:
        del route

    def stream(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[object],
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, messages, tools
        raise AssertionError("Unexpected stream call")

    async def complete(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[object],
    ) -> ModelResponse:
        self.calls.append((route, deepcopy(list(messages)), tuple(tools)))
        return self.response

    async def close(self) -> None:
        return None


class _BlockingTool(BaseTool):
    name = "blocking_tool"
    description = "Wait until the test releases the Tool."

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def execute(self) -> str:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return "released"


class _ConfirmingTool(BaseTool):
    name = "confirm_action"
    description = "Run a confirmed action."
    required = ("action",)
    action: str

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def check_safety(self, *, action: str) -> str:  # type: ignore[override]
        return f"Confirm action: {action}"

    async def execute(self, *, action: str) -> str:
        self.calls.append(action)
        return f"executed:{action}"


@pytest.mark.asyncio
async def test_awaitable_run_returns_isolated_increment_and_emits_ordered_progress() -> None:
    provider = _DirectRouter(
        (
            TextDelta(delta="Hello"),
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Hello"),
                    usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                    finish_reason="stop",
                )
            ),
        )
    )
    emitter = _RecordingEmitter()
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "runtime context\n\nHello"},
    ]
    current_user = {"role": "user", "content": "Hello"}
    original_messages = deepcopy(messages)
    original_current_user = deepcopy(current_user)
    run = AgentRun(model=provider)

    increment = await run.run(
        messages,
        current_user,
        route="chat",
        emitter=emitter,
    )

    assert messages == original_messages
    assert current_user == original_current_user
    assert increment == [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hello",
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            },
        },
    ]
    assert [payload.type for payload in emitter.payloads] == [
        "started",
        "text_delta",
        "model_call_completed",
        "completed",
    ]
    assert isinstance(emitter.payloads[0], AgentRunStartedPayload)
    assert isinstance(emitter.payloads[1], AgentRunTextDeltaPayload)
    assert isinstance(emitter.payloads[2], AgentRunModelCallCompletedPayload)
    assert isinstance(emitter.payloads[3], AgentRunCompletedPayload)
    assert provider.calls[0][0] == "chat"
    assert provider.calls[0][1] == messages


@pytest.mark.asyncio
async def test_awaitable_run_carries_tool_messages_into_the_next_direct_call() -> None:
    call = ModelToolCall(id="call_read", name="read_file", arguments='{"path":"README.md"}')
    provider = _DirectRouter(
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Reading.", tool_calls=(call,)),
                    usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                    finish_reason="tool_calls",
                )
            ),
        ),
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Finished."),
                    usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                    finish_reason="stop",
                )
            ),
        ),
    )
    gateway = SingleToolGateway(
        (FakeTool(name="read_file", description="Read a file.", outcomes=("contents",)),)
    )
    emitter = _RecordingEmitter()
    run = AgentRun(
        model=provider,
        tool_gateway=gateway,
    )

    increment = await run.run(
        [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "runtime\n\nRead README."},
        ],
        {"role": "user", "content": "Read README."},
        route="chat",
        emitter=emitter,
    )

    assert [payload.type for payload in emitter.payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "tool_completed",
        "model_call_completed",
        "completed",
    ]
    assert [message["role"] for message in increment] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert increment[1]["content"] == "Reading."
    assert increment[2]["content"] == "contents"
    assert increment[3]["content"] == "Finished."
    assert provider.calls[1][1][2]["content"] == "Reading."
    assert provider.calls[1][1][3]["content"] == "contents"
    assert provider.calls[1][1][2] is not increment[1]
    assert provider.calls[1][1][3] is not increment[2]


@pytest.mark.asyncio
async def test_awaitable_run_returns_failed_increment_without_raising_handled_model_failure() -> (
    None
):
    provider = _FailingDirectRouter()
    emitter = _RecordingEmitter()
    run = AgentRun(model=provider)

    increment = await run.run(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "Hello"}],
        {"role": "user", "content": "Hello"},
        route="chat",
        emitter=emitter,
    )

    assert [payload.type for payload in emitter.payloads] == ["started", "failed"]
    assert [message["role"] for message in increment] == ["user", "assistant"]
    assert increment[-1]["status"] == "error"
    assert increment[-1]["error"] == {"code": "model_failed", "message": "provider failed"}


@pytest.mark.asyncio
async def test_awaitable_run_repairs_partial_assistant_on_cooperative_cancellation() -> None:
    provider = _DirectRouter((TextDelta(delta="Partial."),))
    emitter = _RecordingEmitter()
    cancel_requested = iter((False, True)).__next__
    run = AgentRun(
        model=provider,
        cancel_requested=cancel_requested,
    )

    increment = await run.run(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "Hello"}],
        {"role": "user", "content": "Hello"},
        route="chat",
        emitter=emitter,
    )

    assert [payload.type for payload in emitter.payloads] == [
        "started",
        "text_delta",
        "cancelled",
    ]
    assert [message["role"] for message in increment] == ["user", "assistant"]
    assert increment[-1]["content"] == "Partial."
    assert increment[-1]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_awaitable_run_repairs_unfinished_tool_on_cooperative_cancellation() -> None:
    calls = (
        ModelToolCall(id="call_read", name="read_file", arguments='{"path":"README.md"}'),
        ModelToolCall(id="call_other", name="read_file", arguments='{"path":"CONTEXT.md"}'),
    )
    provider = _DirectRouter(
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Reading.", tool_calls=calls),
                    usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                    finish_reason="tool_calls",
                )
            ),
        )
    )
    gateway = SingleToolGateway(
        (FakeTool(name="read_file", description="Read a file.", outcomes=("contents",)),)
    )
    emitter = _RecordingEmitter()
    cancel_requested = iter((False, True)).__next__
    run = AgentRun(
        model=provider,
        tool_gateway=gateway,
        cancel_requested=cancel_requested,
    )

    increment = await run.run(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "Read"}],
        {"role": "user", "content": "Read"},
        route="chat",
        emitter=emitter,
    )

    assert [payload.type for payload in emitter.payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "cancelled",
    ]
    assert [message["role"] for message in increment] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [message["tool_call_id"] for message in increment[2:]] == [
        "call_read",
        "call_other",
    ]
    assert all(message["status"] == "error" for message in increment[2:])
    assert all(
        message["content"] == "Tool call interrupted because the turn was cancelled."
        for message in increment[2:]
    )


@pytest.mark.asyncio
async def test_awaitable_schedule_run_uses_direct_complete_provider_call() -> None:
    provider = _CompletingDirectRouter(
        ModelResponse(
            message=AssistantModelMessage(content="Scheduled."),
            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
            finish_reason="stop",
        )
    )
    emitter = _RecordingEmitter()
    run = AgentRun(model=provider)

    increment = await run.run(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "runtime"}],
        {"role": "user", "content": "Schedule this."},
        route="schedule",
        emitter=emitter,
    )

    assert [payload.type for payload in emitter.payloads] == [
        "started",
        "model_call_completed",
        "completed",
    ]
    assert increment[-1]["content"] == "Scheduled."
    assert provider.calls == [
        (
            "schedule",
            [{"role": "system", "content": "System"}, {"role": "user", "content": "runtime"}],
            (),
        )
    ]


@pytest.mark.asyncio
async def test_awaitable_run_emits_confirmation_before_publishing_confirmed_tool_result() -> None:
    call = ModelToolCall(id="call_confirm", name="confirm_action", arguments='{"action":"write"}')
    provider = _DirectRouter(
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Need approval.", tool_calls=(call,)),
                    usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                    finish_reason="tool_calls",
                )
            ),
        ),
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Done."),
                    usage=ModelUsage(input_tokens=7, output_tokens=2, total_tokens=9),
                    finish_reason="stop",
                )
            ),
        ),
    )
    gateway = SingleToolGateway((_ConfirmingTool(),))
    confirmation = ConfirmationChannel()
    emitter = _SignallingEmitter("confirmation_requested")

    run = AgentRun(
        model=provider,
        tool_gateway=gateway,
    )
    task = asyncio.create_task(
        run.run(
            [{"role": "system", "content": "System"}, {"role": "user", "content": "runtime"}],
            {"role": "user", "content": "Do it."},
            route="chat",
            emitter=emitter,
            confirmation=confirmation,
        )
    )
    await emitter.signalled.wait()
    confirmation_payload = next(
        payload for payload in emitter.payloads if payload.type == "confirmation_requested"
    )
    assert isinstance(confirmation_payload, AgentRunConfirmationRequestedPayload)
    confirmation.respond_to_confirmation(confirmation_payload.request.confirmation_id, "approved")
    increment = await task

    assert [payload.type for payload in emitter.payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "confirmation_requested",
        "tool_completed",
        "model_call_completed",
        "completed",
    ]
    assert increment[2]["status"] == "success"
    assert increment[2]["confirmation"]["decision"] == "approved"


@pytest.mark.asyncio
async def test_awaitable_run_externalizes_tool_result_before_follow_up_model_call() -> None:
    call = ModelToolCall(id="call_large", name="read_file", arguments="{}")
    provider = _DirectRouter(
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Reading.", tool_calls=(call,)),
                    usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                    finish_reason="tool_calls",
                )
            ),
        ),
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Done."),
                    usage=ModelUsage(input_tokens=7, output_tokens=2, total_tokens=9),
                    finish_reason="stop",
                )
            ),
        ),
    )
    gateway = SingleToolGateway(
        (FakeTool(name="read_file", description="Read a file.", outcomes=("large result",)),)
    )
    externalized: list[ToolResult] = []

    def externalize(result: ToolResult) -> ToolResult:
        externalized.append(result)
        return ToolResult(
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            content="preview",
            artifact=ArtifactReference(
                path=(
                    ".myclaw/artifacts/20260711-153012-123456_"
                    "550e8400-e29b-41d4-a716-446655440000/call_large.txt"
                ),
                total_chars=12,
                preview_chars=7,
            ),
        )

    emitter = _RecordingEmitter()
    increment = await AgentRun(
        model=provider,
        tool_gateway=gateway,
        externalize_result=externalize,
    ).run(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "runtime"}],
        {"role": "user", "content": "Read"},
        route="chat",
        emitter=emitter,
    )

    assert [result.content for result in externalized] == ["large result"]
    assert increment[2]["content"] == "preview"
    assert increment[2]["artifact"] == {
        "path": (
            ".myclaw/artifacts/20260711-153012-123456_"
            "550e8400-e29b-41d4-a716-446655440000/call_large.txt"
        ),
        "total_chars": 12,
        "preview_chars": 7,
    }
    assert provider.calls[1][1][3] == increment[2]
    assert provider.calls[1][1][3] is not increment[2]


@pytest.mark.asyncio
async def test_awaitable_run_normalizes_tool_and_externalization_failures() -> None:
    calls = (
        ModelToolCall(id="call_tool_failure", name="read_file", arguments="{}"),
        ModelToolCall(id="call_artifact_failure", name="read_file", arguments="{}"),
    )
    provider = _DirectRouter(
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Reading.", tool_calls=calls),
                    usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                    finish_reason="tool_calls",
                )
            ),
        ),
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Recovered."),
                    usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                    finish_reason="stop",
                )
            ),
        ),
    )
    gateway = SingleToolGateway(
        (
            FakeTool(
                name="read_file",
                description="Read a file.",
                outcomes=(RuntimeError("tool failed"), "large result"),
            ),
        )
    )
    artifact_failures: list[tuple[str, str]] = []
    externalize_calls = 0

    def externalize(result: ToolResult) -> ToolResult:
        nonlocal externalize_calls
        externalize_calls += 1
        if externalize_calls == 2:
            raise OSError("artifact write failed")
        return result

    emitter = _RecordingEmitter()
    increment = await AgentRun(
        model=provider,
        tool_gateway=gateway,
        externalize_result=externalize,
        on_artifact_failure=lambda failure, name: artifact_failures.append((str(failure), name)),
    ).run(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "runtime"}],
        {"role": "user", "content": "Read twice"},
        route="chat",
        emitter=emitter,
    )

    assert [message["status"] for message in increment[2:4]] == ["error", "error"]
    assert increment[2]["content"] == "read_file could not complete the request."
    assert increment[3]["content"] == "read_file result could not be stored."
    assert artifact_failures == [("artifact write failed", "read_file")]
    assert provider.calls[1][1][3:5] == increment[2:4]
    assert [payload.type for payload in emitter.payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
        "model_call_completed",
        "completed",
    ]


@pytest.mark.asyncio
async def test_awaitable_run_repairs_partial_output_on_model_failure() -> None:
    provider = _FailingDirectRouter(deltas=("Partial.",))
    emitter = _RecordingEmitter()

    increment = await AgentRun(model=provider).run(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "runtime"}],
        {"role": "user", "content": "Fail partially"},
        route="chat",
        emitter=emitter,
    )

    assert [payload.type for payload in emitter.payloads] == ["started", "text_delta", "failed"]
    assert increment[-1]["content"] == "Partial."
    assert increment[-1]["status"] == "error"
    assert increment[-1]["error"] == {"code": "model_failed", "message": "provider failed"}


@pytest.mark.asyncio
async def test_awaitable_run_returns_cancelled_increment_for_provider_cancellation() -> None:
    provider = _FailingDirectRouter(
        error=ErrorInfo(code="turn_cancelled", message="provider cancelled"),
        deltas=("Partial.",),
    )
    emitter = _RecordingEmitter()

    increment = await AgentRun(model=provider).run(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "runtime"}],
        {"role": "user", "content": "Cancel"},
        route="chat",
        emitter=emitter,
    )

    assert [payload.type for payload in emitter.payloads] == [
        "started",
        "text_delta",
        "cancelled",
    ]
    assert increment[-1]["content"] == "Partial."
    assert increment[-1]["status"] == "interrupted"
    assert increment[-1]["error"]["code"] == "turn_cancelled"


@pytest.mark.asyncio
async def test_awaitable_run_cancels_active_tool_and_returns_repaired_increment() -> None:
    call = ModelToolCall(id="call_blocking", name="blocking_tool", arguments="{}")
    provider = _DirectRouter(
        (
            ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Waiting.", tool_calls=(call,)),
                    usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
                    finish_reason="tool_calls",
                )
            ),
        )
    )
    tool = _BlockingTool()
    cancel_requested = False

    def is_cancel_requested() -> bool:
        return cancel_requested

    emitter = _RecordingEmitter()
    task = asyncio.create_task(
        AgentRun(
            model=provider,
            tool_gateway=SingleToolGateway((tool,)),
            cancel_requested=is_cancel_requested,
        ).run(
            [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "runtime"},
            ],
            {"role": "user", "content": "Wait"},
            route="chat",
            emitter=emitter,
        )
    )
    await tool.started.wait()

    cancel_requested = True
    task.cancel()
    try:
        increment = await task
    finally:
        tool.release.set()

    assert tool.cancelled.is_set()
    assert [payload.type for payload in emitter.payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "cancelled",
    ]
    assert [message["role"] for message in increment] == ["user", "assistant", "tool"]
    assert increment[-1]["tool_call_id"] == "call_blocking"
    assert increment[-1]["status"] == "error"
    assert increment[-1]["content"] == "Tool call interrupted because the turn was cancelled."
