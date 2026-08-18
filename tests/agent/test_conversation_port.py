import asyncio
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from myclaw.agent.context import ContextBuilder
from myclaw.agent.events import (
    ConfirmationRequestedPayload,
    ModelCallCompletedPayload,
    ToolCompletedPayload,
)
from myclaw.agent.run import (
    AgentRunCompletedPayload,
    AgentRunConfirmationRequestedPayload,
    AgentRunModelCallCompletedPayload,
    AgentRunRoute,
    AgentRunRouter,
    AgentRunStartedPayload,
    AgentRunTextDeltaPayload,
    AgentRunToolCompletedPayload,
    AgentRunToolStartedPayload,
)
from myclaw.agent.run import (
    ConfirmationChannel as AgentRunConfirmationChannel,
)
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    TextDelta,
)
from myclaw.session.conversation import StreamingConversationPort
from myclaw.session.session import Session
from myclaw.tools.base import BaseTool, OpenAIToolSchema
from myclaw.tools.tool_gateway import (
    ConfirmationRequest,
    ModelToolCall,
)
from tests.fixtures import (
    ScriptedFakeProvider,
    ScriptedFakeRouter,
    SingleToolGateway,
    StreamScript,
)

NOW = datetime(2026, 8, 7, 12, 0, 0, 123000, tzinfo=timezone(timedelta(hours=8)))
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
CONFIRMATION_UUID = UUID("16fd2706-8baf-4334-8c7f-ada847da0314")


def _session(workspace: Path, agent_home: Path) -> Session:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    return Session.create(state, now=lambda: NOW, new_uuid=lambda: SESSION_UUID)


def _direct_model(provider: ModelProvider) -> AgentRunRouter:
    return ScriptedFakeRouter(provider)


class _RecordingDirectProvider:
    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []

    def stream(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
    ) -> AsyncIterator[ModelStreamEvent]:
        del tools, model, max_output, temperature, reasoning_effort, timeout
        self.messages.append(deepcopy(list(messages)))

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            yield ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Done"),
                    usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                    finish_reason="stop",
                )
            )

        return replay()

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
    ) -> ModelResponse:
        del messages, tools, model, max_output, temperature, reasoning_effort, timeout
        raise AssertionError("Unexpected complete call")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_awaitable_conversation_builds_context_and_commits_before_terminal(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(workspace, agent_home)
    session.add_message("user", "Earlier question")
    session.add_message(
        "assistant",
        "Earlier answer",
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
    provider = _RecordingDirectProvider()
    context_builder = ContextBuilder(
        Workspace.from_path(workspace),
        "Asia/Shanghai",
        clock=lambda: NOW,
    )
    summary_calls: list[dict[str, Any]] = []
    timeline: list[str] = []

    async def prepare_summary(
        active_session: Session,
        current_user: dict[str, Any],
    ) -> Session:
        summary_calls.append(deepcopy(current_user))
        active_session.last_consolidated = 1
        return active_session

    original_append = session.append_messages

    def append_messages(increment: list[dict[str, Any]]) -> None:
        timeline.append("append")
        original_append(increment)

    def persist() -> None:
        timeline.append("persist")

    monkeypatch.setattr(session, "append_messages", append_messages)
    monkeypatch.setattr(session, "persist", persist)
    conversation = StreamingConversationPort(
        model=_direct_model(cast(ModelProvider, provider)),
        session=session,
        now=lambda: NOW,
        new_uuid=lambda: TURN_UUID,
        foreground_summary_preparer=prepare_summary,
        context_builder=context_builder,
        memory_snapshot=lambda: "memory snapshot",
    )

    observed = []
    async for event in conversation.submit("Current question"):
        observed.append(event)
        if event.type == "turn_completed":
            timeline.append("terminal")

    assert [event.type for event in observed] == [
        "turn_started",
        "model_call_completed",
        "turn_completed",
    ]
    assert summary_calls == [{"role": "user", "content": "Current question"}]
    assert len(provider.messages) == 1
    visible_messages = provider.messages[0]
    assert visible_messages[0]["role"] == "system"
    assert "memory snapshot" in visible_messages[0]["content"]
    assert str(workspace) in visible_messages[0]["content"]
    assert visible_messages[1] == {
        "role": "assistant",
        "content": "Earlier answer",
        "tool_calls": [],
    }
    assert visible_messages[-1]["role"] == "user"
    assert "Current question" in visible_messages[-1]["content"]
    assert "2026-08-07T12:00:00.123+08:00" in visible_messages[-1]["content"]
    assert all("timestamp" not in message for message in visible_messages)
    assert timeline == ["append", "persist", "terminal"]
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


class _ScriptedAgentRun:
    def __init__(self, session: Session) -> None:
        self._session = session
        self.calls: list[tuple[Session, str, AgentRunRoute, object | None]] = []

    async def run(
        self,
        messages: Sequence[dict[str, Any]],
        current_user: dict[str, Any],
        *,
        route: AgentRunRoute,
        emitter: Any,
        confirmation: AgentRunConfirmationChannel | None = None,
    ) -> list[dict[str, Any]]:
        del messages
        self.calls.append((self._session, current_user["content"], route, confirmation))
        request = ConfirmationRequest(
            confirmation_id=CONFIRMATION_UUID,
            tool_call_id="call_confirm",
            tool_name="schedule",
            summary="Add a Schedule Job",
            details={"message": "Remember this"},
            reason="Confirm Schedule Job: Remember this",
        )

        assert confirmation is not None
        pending = asyncio.ensure_future(confirmation(request))
        await asyncio.sleep(0)
        await emitter.emit(AgentRunStartedPayload())
        await emitter.emit(AgentRunTextDeltaPayload(delta="Working"))
        await emitter.emit(
            AgentRunModelCallCompletedPayload(
                content="Working",
                continues_with_tools=True,
            )
        )
        await emitter.emit(
            AgentRunToolStartedPayload(
                tool_call_id="call_confirm",
                tool_name="schedule",
                summary="Running schedule",
            )
        )
        await emitter.emit(AgentRunConfirmationRequestedPayload(request=request))
        await pending
        await emitter.emit(
            AgentRunToolCompletedPayload(
                tool_call_id="call_confirm",
                tool_name="schedule",
                status="success",
                summary="Finished schedule",
            )
        )
        await emitter.emit(
            AgentRunCompletedPayload(
                content="Done",
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        )
        return [
            deepcopy(current_user),
            {
                "role": "assistant",
                "content": "Done",
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
        ]


class _ConfirmedTool(BaseTool):
    name = "schedule"
    description = "Create a Schedule Job."
    required = ("message",)
    message: str

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, *, message: str) -> str:
        self.calls.append(message)
        return "job-created"

    async def check_safety(self, *, message: str) -> str:  # type: ignore[override]
        return f"Confirm Schedule Job: {message}"


class _FailingTool(BaseTool):
    name = "failing_tool"
    description = "Fail one Tool call."

    async def execute(self) -> str:
        raise RuntimeError("injected Tool failure")


class _BlockingPartialProvider:
    def __init__(self) -> None:
        self.partial_emitted = asyncio.Event()
        self.cancelled = asyncio.Event()

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
    ) -> AsyncIterator[ModelStreamEvent]:
        del messages, tools, model, max_output, temperature, reasoning_effort, timeout
        yield TextDelta(delta="Retained partial.")
        self.partial_emitted.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
    ) -> ModelResponse:
        raise AssertionError(
            "Unexpected complete request: "
            f"{messages=}, {tools=}, {model=}, {max_output=}, {temperature=}, "
            f"{reasoning_effort=}, {timeout=}"
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_conversation_port_maps_agent_run_and_accepts_separate_confirmation_response(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = _session(workspace, agent_home)
    agent_run = _ScriptedAgentRun(session)
    conversation = StreamingConversationPort(
        agent_run=agent_run,
        session=session,
        model=_direct_model(ScriptedFakeProvider()),
        now=lambda: NOW,
        new_uuid=lambda: TURN_UUID,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
    )

    events = conversation.submit("Add a schedule.")
    assert (await anext(events)).type == "turn_started"
    assert (await anext(events)).type == "text_delta"
    model_call_completed = await anext(events)
    assert model_call_completed.type == "model_call_completed"
    assert isinstance(model_call_completed.payload, ModelCallCompletedPayload)
    assert model_call_completed.turn_id == TURN_UUID
    assert model_call_completed.event_id == 2
    assert model_call_completed.payload.content == "Working"
    assert model_call_completed.payload.continues_with_tools is True
    tool_started = await anext(events)
    assert tool_started.type == "tool_started"
    assert tool_started.turn_id == TURN_UUID
    assert tool_started.event_id == 3
    confirmation = await anext(events)
    assert confirmation.type == "confirmation_requested"
    assert isinstance(confirmation.payload, ConfirmationRequestedPayload)
    assert confirmation.payload.confirmation_id == CONFIRMATION_UUID
    assert confirmation.payload.turn_id == TURN_UUID
    assert confirmation.payload.tool_call_id == "call_confirm"
    assert confirmation.payload.tool_name == "schedule"
    assert confirmation.payload.reason == "Confirm Schedule Job: Remember this"
    assert confirmation.payload.summary == "Add a Schedule Job"
    assert confirmation.payload.details == {"message": "Remember this"}
    assert confirmation.payload.warnings == ()
    detached_details = confirmation.payload.details
    detached_details["message"] = "changed by consumer"
    assert confirmation.payload.details == {"message": "Remember this"}
    conversation.respond_to_confirmation(confirmation.payload.confirmation_id, "approved")

    assert [event.type async for event in events] == [
        "tool_completed",
        "turn_completed",
    ]
    assert agent_run.calls == [(session, "Add a schedule.", "chat", agent_run.calls[0][3])]
    await events.aclose()


@pytest.mark.asyncio
async def test_foreground_confirmation_reply_is_not_added_as_a_session_user_message(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = _session(workspace, agent_home)
    tool = _ConfirmedTool()
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
                                        id="call_schedule",
                                        name="schedule",
                                        arguments='{"message":"job"}',
                                    ),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Created."),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = SingleToolGateway((tool,))
    ids = iter(
        (
            TURN_UUID,
            UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713"),
            CONFIRMATION_UUID,
            UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
        )
    )
    conversation = StreamingConversationPort(
        model=_direct_model(provider),
        session=session,
        now=lambda: NOW,
        new_uuid=ids.__next__,
        tool_gateway=gateway,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
    )

    events = conversation.submit("Create a job.")
    assert (await anext(events)).type == "turn_started"
    assert (await anext(events)).type == "model_call_completed"
    assert (await anext(events)).type == "tool_started"
    requested = await anext(events)
    assert requested.type == "confirmation_requested"
    assert isinstance(requested.payload, ConfirmationRequestedPayload)
    assert requested.payload.tool_name == "schedule"
    conversation.respond_to_confirmation(requested.payload.confirmation_id, "approved")

    assert [event.type async for event in events] == [
        "tool_completed",
        "model_call_completed",
        "turn_completed",
    ]
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "Create a job."
    ]
    assert tool.calls


@pytest.mark.asyncio
async def test_cancelling_a_foreground_confirmation_emits_cancelled_and_repairs_the_tool_call(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = _session(workspace, agent_home)
    tool = _ConfirmedTool()
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
                                        id="call_schedule",
                                        name="schedule",
                                        arguments='{"message":"job"}',
                                    ),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    gateway = SingleToolGateway((tool,))
    ids = iter((TURN_UUID, UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")))
    conversation = StreamingConversationPort(
        model=_direct_model(provider),
        session=session,
        now=lambda: NOW,
        new_uuid=ids.__next__,
        tool_gateway=gateway,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
    )

    events = conversation.submit("Create a job.")
    assert (await anext(events)).type == "turn_started"
    assert (await anext(events)).type == "model_call_completed"
    assert (await anext(events)).type == "tool_started"
    assert (await anext(events)).type == "confirmation_requested"

    await conversation.cancel_active_turn()

    observed = [event async for event in events]
    assert [event.type for event in observed] == ["tool_completed", "turn_cancelled"]
    assert isinstance(observed[0].payload, ToolCompletedPayload)
    assert observed[0].payload.status == "error"
    assert [message["role"] for message in session.messages] == ["user", "assistant", "tool"]
    assert session.messages[-1]["status"] == "error"
    assert tool.calls == []


@pytest.mark.asyncio
async def test_terminal_already_queued_is_not_lost_to_a_late_cross_task_cancel(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = _session(workspace, agent_home)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done."),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        model=_direct_model(provider),
        session=session,
        now=lambda: NOW,
        new_uuid=lambda: TURN_UUID,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
    )
    model_call_observed = asyncio.Event()
    release_consumer = asyncio.Event()
    observed: list[str] = []

    async def consume() -> None:
        async for event in conversation.submit("Finish before cancellation."):
            observed.append(event.type)
            if event.type == "model_call_completed":
                model_call_observed.set()
                await release_consumer.wait()

    consumer = asyncio.create_task(consume())
    await model_call_observed.wait()
    while len(session.messages) < 2:
        await asyncio.sleep(0)
    await asyncio.sleep(0)

    await conversation.cancel_active_turn()
    release_consumer.set()
    await consumer

    assert observed == ["turn_started", "model_call_completed", "turn_completed"]
    assert [message["role"] for message in session.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_batch_append_failure_raises_safe_error_without_publishing_terminal(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(workspace, agent_home)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Not committed."),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    persist_calls = 0

    def fail_append(messages: list[dict[str, Any]]) -> None:
        del messages
        raise OSError("injected batch append failure")

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "append_messages", fail_append)
    monkeypatch.setattr(session, "persist", persist)
    conversation = StreamingConversationPort(
        model=_direct_model(provider),
        session=session,
        now=lambda: NOW,
        new_uuid=lambda: TURN_UUID,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
    )

    observed = []
    with pytest.raises(ModelCallError) as raised:
        async for event in conversation.submit("Do not publish this turn."):
            observed.append(event)

    assert [event.type for event in observed] == ["turn_started", "model_call_completed"]
    assert raised.value.error.code == "persistence_error"
    assert raised.value.error.message == "The Conversation Session could not be updated."
    assert "injected" not in str(raised.value)
    assert session.messages == []
    assert persist_calls == 0


@pytest.mark.asyncio
async def test_persist_request_failure_remains_silent_and_terminal_stays_ordered(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(workspace, agent_home)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Committed in memory."),
                            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )

    def fail_persist() -> None:
        raise OSError("injected persistence scheduling failure")

    monkeypatch.setattr(session, "persist", fail_persist)
    conversation = StreamingConversationPort(
        model=_direct_model(provider),
        session=session,
        now=lambda: NOW,
        new_uuid=lambda: TURN_UUID,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
    )

    observed = [event async for event in conversation.submit("Keep memory authoritative.")]

    assert [event.type for event in observed] == [
        "turn_started",
        "model_call_completed",
        "turn_completed",
    ]
    assert [message["role"] for message in session.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_streamed_model_failure_commits_partial_error_before_terminal(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = _session(workspace, agent_home)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(TextDelta(delta="Partial failure."),),
                error=RuntimeError("injected provider failure"),
            ),
        )
    )
    conversation = StreamingConversationPort(
        model=_direct_model(provider),
        session=session,
        now=lambda: NOW,
        new_uuid=lambda: TURN_UUID,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
    )

    observed = [event async for event in conversation.submit("Fail after streaming.")]

    assert [event.type for event in observed] == ["turn_started", "text_delta", "turn_failed"]
    assert session.messages[-1]["content"] == "Partial failure."
    assert session.messages[-1]["status"] == "error"
    assert session.messages[-1]["error"]["code"] == "model_failed"


@pytest.mark.asyncio
async def test_explicit_cancellation_commits_streamed_partial_before_terminal(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = _session(workspace, agent_home)
    provider = _BlockingPartialProvider()
    conversation = StreamingConversationPort(
        model=_direct_model(cast(ModelProvider, provider)),
        session=session,
        now=lambda: NOW,
        new_uuid=lambda: TURN_UUID,
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
    )
    events = conversation.submit("Cancel after a partial response.")

    assert (await anext(events)).type == "turn_started"
    assert (await anext(events)).type == "text_delta"
    await provider.partial_emitted.wait()
    await conversation.cancel_active_turn()
    remaining = [event async for event in events]

    assert [event.type for event in remaining] == ["turn_cancelled"]
    assert provider.cancelled.is_set()
    assert session.messages[-1]["content"] == "Retained partial."
    assert session.messages[-1]["status"] == "interrupted"
    assert session.messages[-1]["error"]["code"] == "turn_cancelled"


@pytest.mark.asyncio
async def test_multi_tool_refusal_and_error_are_committed_before_success_terminal(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = _session(workspace, agent_home)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Trying Tools.",
                                tool_calls=(
                                    ModelToolCall(
                                        id="call_refused",
                                        name="schedule",
                                        arguments='{"message":"job"}',
                                    ),
                                    ModelToolCall(
                                        id="call_error",
                                        name="failing_tool",
                                        arguments="{}",
                                    ),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Handled both."),
                            usage=ModelUsage(input_tokens=3, output_tokens=1, total_tokens=4),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        model=_direct_model(provider),
        session=session,
        now=lambda: NOW,
        new_uuid=iter(
            (
                TURN_UUID,
                UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713"),
                CONFIRMATION_UUID,
            )
        ).__next__,
        tool_gateway=SingleToolGateway((_ConfirmedTool(), _FailingTool())),
        context_builder=ContextBuilder(Workspace.from_path(workspace), "Asia/Shanghai"),
    )
    events = conversation.submit("Exercise both Tool outcomes.")

    assert (await anext(events)).type == "turn_started"
    assert (await anext(events)).type == "model_call_completed"
    assert (await anext(events)).type == "tool_started"
    confirmation = await anext(events)
    assert confirmation.type == "confirmation_requested"
    assert isinstance(confirmation.payload, ConfirmationRequestedPayload)
    conversation.respond_to_confirmation(confirmation.payload.confirmation_id, "declined")
    remaining = [event async for event in events]

    assert [event.type for event in remaining] == [
        "tool_completed",
        "tool_started",
        "tool_completed",
        "model_call_completed",
        "turn_completed",
    ]
    tool_events = [event for event in remaining if event.type == "tool_completed"]
    assert [cast(ToolCompletedPayload, event.payload).status for event in tool_events] == [
        "refused",
        "error",
    ]
    assert [message["status"] for message in session.messages if message["role"] == "tool"] == [
        "refused",
        "error",
    ]
