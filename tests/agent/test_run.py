import asyncio
from collections.abc import AsyncIterator
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from myclaw.agent.run import (
    AgentRun,
    AgentRunCancelledPayload,
    AgentRunCompletedPayload,
    AgentRunConfirmationRequestedPayload,
    AgentRunFailedPayload,
    AgentRunModelCallCompletedPayload,
    AgentRunModelSettings,
)
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.session.session import Session
from myclaw.tools.base import BaseTool, OpenAIToolSchema
from myclaw.tools.tool_gateway import (
    ConfirmationChannel,
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
)
from tests.fixtures import FakeTool, ScriptedFakeProvider, SingleToolGateway, StreamScript

NOW = datetime(2026, 7, 18, 18, 30, 12, 123456, tzinfo=timezone(timedelta(hours=8)))
REQUEST_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
FOLLOW_UP_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
TURN_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
CONFIRMATION_UUID = UUID("16fd2706-8baf-4334-8c7f-ada847da0314")


def _settings() -> AgentRunModelSettings:
    return AgentRunModelSettings(
        model="test-model",
        max_output=1024,
        temperature=0.2,
        reasoning_effort=None,
        timeout_seconds=30,
        context_window=4096,
    )


class _ConfirmingTool(BaseTool):
    name = "confirm_action"
    description = "Run one confirmed action."
    required = ("action",)
    action: str

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def check_safety(self, *, action: str) -> str:  # type: ignore[override]
        return f"Confirm action: {action}"

    async def execute(self, *, action: str) -> str:
        self.calls.append(action)
        return f"executed:{action}"


class _ChangingSchemaGateway(SingleToolGateway):
    def __init__(self, tools: tuple[BaseTool, ...]) -> None:
        super().__init__(tools)
        self.schema_reads: list[int] = []

    @property
    def schemas(self) -> list[OpenAIToolSchema]:
        self.schema_reads.append(1)
        return super().schemas if len(self.schema_reads) == 1 else []


class _BlockingStreamProvider:
    def __init__(self) -> None:
        self.closed = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.started.set()
        try:
            yield TextDelta(delta="Partial.")
            await self.release.wait()
        finally:
            self.closed = True

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected completion request: {request!r}")

    async def close(self) -> None:
        return None


class _BlockingConfirmedTool(_ConfirmingTool):
    action: str

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def execute(self, *, action: str) -> str:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return await super().execute(action=action)


class _BlockingConfirmationChannel(ConfirmationChannel):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self._release = asyncio.Event()

    async def __call__(
        self,
        request: ConfirmationRequest,
    ) -> ConfirmationDecision:
        try:
            self.started.set()
            return await super().__call__(request)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.mark.asyncio
async def test_agent_run_streams_chat_and_persists_one_terminal_turn(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="Done"),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done."),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="Frozen system prompt.",
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Say hello.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == [
        "started",
        "text_delta",
        "model_call_completed",
        "completed",
    ]
    model_call = payloads[2]
    assert isinstance(model_call, AgentRunModelCallCompletedPayload)
    assert model_call.content == "Done."
    assert model_call.continues_with_tools is False
    assert {field.name for field in fields(model_call)} == {"content", "continues_with_tools"}
    assert isinstance(payloads[-1], AgentRunCompletedPayload)
    assert payloads[-1].content == "Done."
    assert sum(payload.type in {"completed", "failed", "cancelled"} for payload in payloads) == 1
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert persist_calls == 1
    request = cast(ModelRequest, provider.stream_requests[0])
    assert request.route == "chat"
    assert request.stream is True
    assert request.system_prompt == "Frozen system prompt."


@pytest.mark.asyncio
async def test_agent_run_uses_non_streaming_completion_for_schedule(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Scheduled."),
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                finish_reason="stop",
            ),
        )
    )
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="Schedule prompt.",
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Run later.",
            route="schedule",
            stream=False,
        )
    ]

    assert [payload.type for payload in payloads] == [
        "started",
        "model_call_completed",
        "completed",
    ]
    assert isinstance(payloads[1], AgentRunModelCallCompletedPayload)
    assert payloads[1].content == "Scheduled."
    assert payloads[1].continues_with_tools is False
    assert isinstance(payloads[-1], AgentRunCompletedPayload)
    assert payloads[-1].content == "Scheduled."
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert request.route == "schedule"
    assert request.stream is False


@pytest.mark.asyncio
async def test_agent_run_rejects_a_route_and_stream_mode_mismatch(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider()
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
    )

    with pytest.raises(ValueError, match="must stream"):
        await anext(
            run.run_agent(
                session,
                "Invalid.",
                route="schedule",
                stream=True,
            )
        )


@pytest.mark.asyncio
async def test_agent_run_keeps_tool_and_terminal_order_for_chat(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    call = ModelToolCall(id="call_read", name="read_file", arguments='{"path":"README.md"}')
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Reading.", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Finished."),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = SingleToolGateway(
        (
            FakeTool(
                name="read_file",
                description="Read a test file.",
                required=("path",),
                outcomes=("file contents",),
            ),
        )
    )
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID, UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"))).__next__,
        system_prompt="Tool prompt.",
        tool_gateway=gateway,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Inspect the project.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "tool_completed",
        "model_call_completed",
        "completed",
    ]
    assert isinstance(payloads[1], AgentRunModelCallCompletedPayload)
    assert payloads[1].content == "Reading."
    assert payloads[1].continues_with_tools is True
    assert isinstance(payloads[4], AgentRunModelCallCompletedPayload)
    assert payloads[4].content == "Finished."
    assert payloads[4].continues_with_tools is False
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.messages[2]["content"] == "file contents"
    follow_up = cast(ModelRequest, provider.stream_requests[1])
    assert follow_up.messages[2].content == "file contents"
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_agent_run_emits_confirmation_request_before_waiting_for_approval(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    call = ModelToolCall(
        id="call_confirm",
        name="confirm_action",
        arguments='{"action":"write"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Need approval.", tool_calls=(call,)
                            ),
                            usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Approved."),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    tool = _ConfirmingTool()
    channel = ConfirmationChannel()
    gateway = SingleToolGateway((tool,))
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID, FOLLOW_UP_UUID)).__next__,
        system_prompt="Confirmation prompt.",
        tool_gateway=gateway,
    )
    events = run.run_agent(
        session,
        "Do the action.",
        route="chat",
        stream=True,
        confirmation=channel,
    )

    assert (await anext(events)).type == "started"
    assert (await anext(events)).type == "model_call_completed"
    assert (await anext(events)).type == "tool_started"
    pending_confirmation = asyncio.create_task(anext(events))
    confirmation_payload = await pending_confirmation
    assert confirmation_payload.type == "confirmation_requested"
    assert isinstance(confirmation_payload, AgentRunConfirmationRequestedPayload)
    assert confirmation_payload.request.summary == "Confirm confirm_action"
    assert confirmation_payload.request.details == {"action": "write"}
    assert confirmation_payload.request.reason == "Confirm action: write"
    assert confirmation_payload.request.warnings == ()
    channel.respond_to_confirmation(confirmation_payload.request.confirmation_id, "approved")

    remaining = [payload async for payload in events]
    assert [payload.type for payload in remaining] == [
        "tool_completed",
        "model_call_completed",
        "completed",
    ]
    assert tool.calls == ["write"]
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_agent_run_keeps_declined_confirmation_as_refused_tool_result(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    call = ModelToolCall(
        id="call_decline",
        name="confirm_action",
        arguments='{"action":"delete"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Need approval.", tool_calls=(call,)
                            ),
                            usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="No mutation."),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    tool = _ConfirmingTool()
    channel = ConfirmationChannel()
    gateway = SingleToolGateway((tool,))
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID, FOLLOW_UP_UUID)).__next__,
        tool_gateway=gateway,
    )
    events = run.run_agent(
        session,
        "Decline the action.",
        route="chat",
        stream=True,
        confirmation=channel,
    )

    assert (await anext(events)).type == "started"
    assert (await anext(events)).type == "model_call_completed"
    assert (await anext(events)).type == "tool_started"
    pending_confirmation = asyncio.create_task(anext(events))
    confirmation_payload = await pending_confirmation
    assert isinstance(confirmation_payload, AgentRunConfirmationRequestedPayload)
    channel.respond_to_confirmation(confirmation_payload.request.confirmation_id, "declined")

    remaining = [payload async for payload in events]
    assert [payload.type for payload in remaining] == [
        "tool_completed",
        "model_call_completed",
        "completed",
    ]
    assert tool.calls == []
    assert session.messages[2]["status"] == "refused"
    assert session.messages[2]["content"] == "Tool confirmation was declined."
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_agent_run_freezes_memory_prompt_and_tools_and_prepares_summary_per_request(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="First."),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    memory_reads = 0
    summary_routes: list[str] = []
    summary_budgets: list[tuple[int, int, str, int]] = []

    def read_memory() -> str:
        nonlocal memory_reads
        memory_reads += 1
        return f"memory-{memory_reads}"

    async def prepare_summary(
        active_session: Session,
        route: str,
        context_window: int,
        max_output: int,
        system_prompt: str,
        tools: tuple[OpenAIToolSchema, ...],
    ) -> Session:
        summary_routes.append(route)
        summary_budgets.append((context_window, max_output, system_prompt, len(tools)))
        return active_session

    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="unused",
        memory_snapshot=read_memory,
        system_prompt_for_memory=lambda value: f"system:{value}",
        summary_preparer_for_route=prepare_summary,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Use memory.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == [
        "started",
        "model_call_completed",
        "completed",
    ]
    assert memory_reads == 1
    assert summary_routes == ["chat"]
    assert summary_budgets == [(4096, 1024, "system:memory-1", 0)]
    request = cast(ModelRequest, provider.stream_requests[0])
    assert request.system_prompt == "system:memory-1"


@pytest.mark.asyncio
async def test_agent_run_persists_safe_failed_terminal_once(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(),
                error=ModelCallError(
                    ErrorInfo(code="provider_unavailable", message="Provider is unavailable.")
                ),
            ),
        )
    )
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Fail safely.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == ["started", "failed"]
    assert isinstance(payloads[-1], AgentRunFailedPayload)
    assert payloads[-1].error == ErrorInfo(
        code="provider_unavailable",
        message="Provider is unavailable.",
    )
    assert session.messages[-1]["status"] == "error"
    assert session.messages[-1]["error"] == {
        "code": "provider_unavailable",
        "message": "Provider is unavailable.",
    }
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_agent_run_emits_cancelled_and_persists_before_provider_work(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider()
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        cancel_requested=lambda: True,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Cancel now.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == ["started", "cancelled"]
    assert isinstance(payloads[-1], AgentRunCancelledPayload)
    assert payloads[-1].partial_content == ""
    assert [message["role"] for message in session.messages] == ["user"]
    assert provider.stream_requests == []
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_agent_run_repairs_partial_assistant_output_before_cancelled_terminal(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="Partial."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Complete."),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        cancel_requested=iter((False, True)).__next__,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Start and cancel.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == ["started", "text_delta", "cancelled"]
    assert isinstance(payloads[-1], AgentRunCancelledPayload)
    assert session.messages[-1]["status"] == "interrupted"
    assert session.messages[-1]["content"] == "Partial."
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_agent_run_repairs_unfinished_tool_calls_on_cooperative_cancellation(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    calls = (
        ModelToolCall(id="call_first", name="read_file", arguments="{}"),
        ModelToolCall(id="call_second", name="read_file", arguments="{}"),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Starting.", tool_calls=calls),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    gateway = SingleToolGateway(
        (
            FakeTool(
                name="read_file",
                description="Read a test file.",
                outcomes=("first result",),
            ),
        )
    )
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        tool_gateway=gateway,
        cancel_requested=iter((False, False, True)).__next__,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Run tools and cancel.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "tool_completed",
        "cancelled",
    ]
    assert [message["tool_call_id"] for message in session.messages[2:]] == [
        "call_first",
        "call_second",
    ]
    assert session.messages[-1]["status"] == "error"
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_agent_run_keeps_tool_order_for_schedule_completion(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    call = ModelToolCall(id="call_schedule", name="read_file", arguments="{}")
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Reading.", tool_calls=(call,)),
                usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Finished."),
                usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                finish_reason="stop",
            ),
        )
    )
    gateway = SingleToolGateway(
        (
            FakeTool(
                name="read_file",
                description="Read a test file.",
                outcomes=("schedule contents",),
            ),
        )
    )
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID, FOLLOW_UP_UUID)).__next__,
        tool_gateway=gateway,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Run on schedule.",
            route="schedule",
            stream=False,
        )
    ]

    assert [payload.type for payload in payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "tool_completed",
        "model_call_completed",
        "completed",
    ]
    complete_requests = [cast(ModelRequest, request) for request in provider.complete_requests]
    assert [request.route for request in complete_requests] == [
        "schedule",
        "schedule",
    ]


@pytest.mark.asyncio
async def test_agent_run_freezes_tool_schemas_for_all_model_requests(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    call = ModelToolCall(id="call_schema", name="read_file", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Reading.", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Finished."),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = _ChangingSchemaGateway(
        (
            FakeTool(
                name="read_file",
                description="Read a test file.",
                outcomes=("contents",),
            ),
        )
    )
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID, FOLLOW_UP_UUID)).__next__,
        tool_gateway=gateway,
    )

    [
        payload
        async for payload in run.run_agent(
            session,
            "Use the frozen catalog.",
            route="chat",
            stream=True,
        )
    ]

    assert len(gateway.schema_reads) == 1
    first_request = cast(ModelRequest, provider.stream_requests[0])
    second_request = cast(ModelRequest, provider.stream_requests[1])
    assert first_request.tools == second_request.tools
    assert first_request.tools


@pytest.mark.asyncio
async def test_agent_run_closes_and_repairs_when_consumer_abandons_stream(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = _BlockingStreamProvider()
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
    )
    events = run.run_agent(
        session,
        "Abandon this run.",
        route="chat",
        stream=True,
    )

    assert (await anext(events)).type == "started"
    assert (await anext(events)).type == "text_delta"
    await events.aclose()

    assert provider.closed is True
    assert session.messages[-1]["status"] == "interrupted"
    assert session.messages[-1]["content"] == "Partial."
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_agent_run_normalizes_an_empty_model_stream_without_retrying(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider(streams=(StreamScript(events=()),))
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Handle an empty stream.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == ["started", "failed"]
    assert len(provider.stream_requests) == 1


@pytest.mark.asyncio
async def test_memory_prompt_and_tool_schema_are_frozen_before_started_is_delivered(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider()
    gateway = _ChangingSchemaGateway(
        (FakeTool(name="read_file", description="Read a file.", outcomes=("contents",)),)
    )
    memory_reads = 0
    prompt_inputs: list[str] = []

    def read_memory() -> str:
        nonlocal memory_reads
        memory_reads += 1
        return "snapshot"

    def build_prompt(memory: str) -> str:
        prompt_inputs.append(memory)
        return f"prompt:{memory}"

    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        tool_gateway=gateway,
        memory_snapshot=read_memory,
        system_prompt_for_memory=build_prompt,
    )
    events = run.run_agent(session, "Freeze startup.", route="chat", stream=True)

    assert (await anext(events)).type == "started"
    assert memory_reads == 1
    assert prompt_inputs == ["snapshot"]
    assert len(gateway.schema_reads) == 1
    await events.aclose()


@pytest.mark.asyncio
async def test_noninteractive_schedule_refusal_does_not_emit_confirmation_request(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    call = ModelToolCall(
        id="call_background_confirm",
        name="confirm_action",
        arguments='{"action":"write"}',
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Need approval.", tool_calls=(call,)),
                usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Refused."),
                usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                finish_reason="stop",
            ),
        )
    )
    tool = _ConfirmingTool()
    gateway = SingleToolGateway((tool,))
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=iter((REQUEST_UUID, FOLLOW_UP_UUID)).__next__,
        tool_gateway=gateway,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Run without an interactive host.",
            route="schedule",
            stream=False,
        )
    ]

    assert [payload.type for payload in payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "tool_completed",
        "model_call_completed",
        "completed",
    ]
    assert tool.calls == []
    assert session.messages[2]["status"] == "refused"


@pytest.mark.asyncio
async def test_closing_after_terminal_does_not_repair_completed_state(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="Complete."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Complete."),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
    )
    events = run.run_agent(session, "Finish once.", route="chat", stream=True)

    assert (await anext(events)).type == "started"
    assert (await anext(events)).type == "text_delta"
    assert (await anext(events)).type == "model_call_completed"
    assert (await anext(events)).type == "completed"
    messages_at_terminal = list(session.messages)
    await events.aclose()

    assert session.messages == messages_at_terminal
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_consumer_close_cancels_confirmation_operation_before_returning(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    call = ModelToolCall(
        id="call_abandoned_confirm",
        name="confirm_action",
        arguments='{"action":"write"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Need approval.",
                                tool_calls=(call,),
                            ),
                            usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    channel = _BlockingConfirmationChannel()
    gateway = SingleToolGateway((_ConfirmingTool(),))
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        tool_gateway=gateway,
    )
    events = run.run_agent(
        session,
        "Abandon confirmation.",
        route="chat",
        stream=True,
        confirmation=channel,
    )

    assert (await anext(events)).type == "started"
    assert (await anext(events)).type == "model_call_completed"
    assert (await anext(events)).type == "tool_started"
    assert (await anext(events)).type == "confirmation_requested"
    await channel.started.wait()
    await events.aclose()

    assert channel.cancelled.is_set()
    assert [message["role"] for message in session.messages] == ["user", "assistant", "tool"]
    assert session.messages[-1]["status"] == "error"
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_cancellation_after_approval_propagates_to_the_running_tool(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    call = ModelToolCall(
        id="call_cancel_after_approval",
        name="confirm_action",
        arguments='{"action":"commit"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Commit after approval.",
                                tool_calls=(call,),
                            ),
                            usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    tool = _BlockingConfirmedTool()
    channel = ConfirmationChannel()
    gateway = SingleToolGateway((tool,))
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        tool_gateway=gateway,
    )
    events = run.run_agent(
        session,
        "Commit the action.",
        route="chat",
        stream=True,
        confirmation=channel,
    )

    assert (await anext(events)).type == "started"
    assert (await anext(events)).type == "model_call_completed"
    assert (await anext(events)).type == "tool_started"
    pending_confirmation = asyncio.create_task(anext(events))
    confirmation_payload = await pending_confirmation
    assert confirmation_payload.type == "confirmation_requested"
    assert isinstance(confirmation_payload, AgentRunConfirmationRequestedPayload)
    channel.respond_to_confirmation(confirmation_payload.request.confirmation_id, "approved")
    pending_tool = asyncio.create_task(anext(events))
    await tool.started.wait()

    pending_tool.cancel()
    try:
        for _ in range(5):
            await asyncio.sleep(0)
            if pending_tool.done():
                break
        assert pending_tool.done()
    finally:
        tool.release.set()
        outcomes = await asyncio.gather(pending_tool, return_exceptions=True)

    assert isinstance(outcomes[0], asyncio.CancelledError)
    assert tool.cancelled.is_set()
    assert tool.calls == []
    assert [message["role"] for message in session.messages] == ["user", "assistant", "tool"]
    assert session.messages[-1]["status"] == "error"
    assert session.messages[-1]["content"] == (
        "Tool call interrupted because the turn was cancelled."
    )
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_cooperative_cancellation_closes_provider_before_terminal(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = _BlockingStreamProvider()
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        cancel_requested=iter((False, True)).__next__,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Cancel and close.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == ["started", "text_delta", "cancelled"]
    assert provider.closed is True


@pytest.mark.asyncio
async def test_tool_publication_failure_repairs_provider_order_with_failure_result(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    call = ModelToolCall(id="call_publish_failure", name="read_file", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Reading.", tool_calls=(call,)),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    gateway = SingleToolGateway(
        (FakeTool(name="read_file", description="Read a file.", outcomes=("contents",)),)
    )
    original_add_message = session.add_message
    fail_tool_publication = True

    def add_message(role: str, content: str, **fields: object) -> None:
        nonlocal fail_tool_publication
        if role == "tool" and fail_tool_publication:
            fail_tool_publication = False
            raise OSError("injected Tool publication failure")
        original_add_message(role, content, **fields)

    monkeypatch.setattr(session, "add_message", add_message)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        tool_gateway=gateway,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Repair a failed Tool publication.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == [
        "started",
        "model_call_completed",
        "tool_started",
        "failed",
    ]
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.messages[2]["tool_call_id"] == call.id
    assert session.messages[2]["content"] == "Tool call interrupted because the Agent Run failed."
    assert session.messages[3]["status"] == "error"


@pytest.mark.asyncio
async def test_initial_user_publication_failure_becomes_a_safe_terminal(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider()
    persist_calls = 0

    def fail_add_message(role: str, content: str, **fields: object) -> None:
        del role, content, fields
        raise OSError("injected user publication failure")

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "add_message", fail_add_message)
    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Fail publication safely.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == ["started", "failed"]
    assert session.messages == []
    assert provider.stream_requests == []
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_summary_cannot_replace_the_caller_owned_session(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    replacement = Session.create(state, now=lambda: NOW)
    provider = ScriptedFakeProvider()
    persist_calls = 0

    async def replace_session(
        active_session: Session,
        route: str,
        context_window: int,
        max_output: int,
        system_prompt: str,
        tools: tuple[OpenAIToolSchema, ...],
    ) -> Session:
        del active_session, route, context_window, max_output, system_prompt, tools
        return replacement

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        summary_preparer_for_route=replace_session,
    )

    payloads = [
        payload
        async for payload in run.run_agent(
            session,
            "Keep Session authority.",
            route="chat",
            stream=True,
        )
    ]

    assert [payload.type for payload in payloads] == ["started", "failed"]
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert replacement.messages == []
    assert provider.stream_requests == []
    assert persist_calls == 1


@pytest.mark.asyncio
async def test_process_control_repairs_and_propagates_the_original_exception(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProcessControl(BaseException):
        pass

    control = ProcessControl("stop the process")

    class ProcessControlProvider:
        def __init__(self) -> None:
            self.closed = False

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            try:
                yield TextDelta(delta="Partial.")
                raise control
            finally:
                self.closed = True

        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise AssertionError(f"Unexpected completion request: {request!r}")

        async def close(self) -> None:
            return None

    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW)
    provider = ProcessControlProvider()
    persist_calls = 0

    def persist() -> None:
        nonlocal persist_calls
        persist_calls += 1

    monkeypatch.setattr(session, "persist", persist)
    run = AgentRun(
        provider=provider,
        settings=_settings(),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
    )
    events = run.run_agent(session, "Propagate control.", route="chat", stream=True)

    assert (await anext(events)).type == "started"
    assert (await anext(events)).type == "text_delta"
    with pytest.raises(ProcessControl) as raised:
        await anext(events)

    assert raised.value is control
    assert provider.closed is True
    assert session.messages[-1]["status"] == "interrupted"
    assert session.messages[-1]["content"] == "Partial."
    assert persist_calls == 1
