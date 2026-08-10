import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.events import ConfirmationRequestedPayload, ToolCompletedPayload
from myclaw.agent.run import (
    AgentRunCompletedPayload,
    AgentRunConfirmationRequestedPayload,
    AgentRunInterface,
    AgentRunPayload,
    AgentRunRoute,
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
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session import Session
from myclaw.tools.base import BaseTool
from myclaw.tools.tool_gateway import (
    ConfirmationRequest,
    ModelToolCall,
)
from tests.fixtures import ScriptedFakeProvider, SingleToolGateway, StreamScript

NOW = datetime(2026, 8, 7, 12, 0, 0, 123000, tzinfo=timezone(timedelta(hours=8)))
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
CONFIRMATION_UUID = UUID("16fd2706-8baf-4334-8c7f-ada847da0314")


def _session(workspace: Path, agent_home: Path) -> Session:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    return Session.create(state, now=lambda: NOW, new_uuid=lambda: SESSION_UUID)


class _ScriptedAgentRun(AgentRunInterface):
    def __init__(self) -> None:
        self.calls: list[tuple[Session, str, AgentRunRoute, bool, object | None]] = []

    def run_agent(
        self,
        session: Session,
        input: str,
        route: AgentRunRoute,
        stream: bool,
        confirmation: AgentRunConfirmationChannel | None = None,
    ) -> AsyncIterator[AgentRunPayload]:
        self.calls.append((session, input, route, stream, confirmation))
        request = ConfirmationRequest(
            confirmation_id=CONFIRMATION_UUID,
            tool_call_id="call_confirm",
            tool_name="schedule",
            summary="Add a Schedule Job",
            details={"message": "Remember this"},
        )

        async def payloads() -> AsyncIterator[AgentRunPayload]:
            assert confirmation is not None
            pending = asyncio.ensure_future(confirmation(request))
            await asyncio.sleep(0)
            yield AgentRunStartedPayload()
            yield AgentRunTextDeltaPayload(delta="Working")
            yield AgentRunToolStartedPayload(
                tool_call_id="call_confirm",
                tool_name="schedule",
                summary="Running schedule",
            )
            yield AgentRunConfirmationRequestedPayload(request=request)
            await pending
            yield AgentRunToolCompletedPayload(
                tool_call_id="call_confirm",
                tool_name="schedule",
                status="success",
                summary="Finished schedule",
            )
            yield AgentRunCompletedPayload(
                content="Done",
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return payloads()


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


@pytest.mark.asyncio
async def test_conversation_port_maps_agent_run_and_accepts_separate_confirmation_response(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_run = _ScriptedAgentRun()
    session = _session(workspace, agent_home)
    conversation = StreamingConversationPort(
        agent_run=agent_run,
        session=session,
        provider=ScriptedFakeProvider(),
        settings=ChatModelSettings(
            model="unused",
            max_output=10,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=lambda: TURN_UUID,
    )

    events = conversation.submit("Add a schedule.")
    assert (await anext(events)).type == "turn_started"
    assert (await anext(events)).type == "text_delta"
    assert (await anext(events)).type == "tool_started"
    confirmation = await anext(events)
    assert confirmation.type == "confirmation_requested"
    assert isinstance(confirmation.payload, ConfirmationRequestedPayload)
    assert confirmation.payload.confirmation_id == CONFIRMATION_UUID
    assert confirmation.payload.turn_id == TURN_UUID
    assert confirmation.payload.tool_call_id == "call_confirm"
    assert confirmation.payload.tool_name == "schedule"
    assert confirmation.payload.summary == "Add a Schedule Job"
    assert confirmation.payload.details == {"message": "Remember this"}
    assert confirmation.payload.warnings == ()
    detached_details = confirmation.payload.details
    detached_details["message"] = "changed by consumer"
    assert confirmation.payload.details == {"message": "Remember this"}
    conversation.respond_to_confirmation(confirmation.payload.confirmation_id, "approved")

    assert [event.type async for event in events] == ["tool_completed", "turn_completed"]
    assert agent_run.calls == [(session, "Add a schedule.", "chat", True, agent_run.calls[0][4])]
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
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=10,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=ids.__next__,
        tool_gateway=gateway,
    )

    events = conversation.submit("Create a job.")
    assert (await anext(events)).type == "turn_started"
    assert (await anext(events)).type == "tool_started"
    requested = await anext(events)
    assert requested.type == "confirmation_requested"
    assert isinstance(requested.payload, ConfirmationRequestedPayload)
    assert requested.payload.tool_name == "schedule"
    conversation.respond_to_confirmation(requested.payload.confirmation_id, "approved")

    assert [event.type async for event in events] == ["tool_completed", "turn_completed"]
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
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=10,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=ids.__next__,
        tool_gateway=gateway,
    )

    events = conversation.submit("Create a job.")
    assert (await anext(events)).type == "turn_started"
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
