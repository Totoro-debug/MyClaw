from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Annotated
from uuid import UUID

import pytest

from myclaw.agent.run import model_message_from_session
from myclaw.tools.base import BaseTool
from myclaw.tools.confirmation import (
    ConfirmationChannel,
    ConfirmationDecision,
    ConfirmationPrompt,
    ConfirmationRequest,
    ToolConfirmationMetadata,
)
from myclaw.tools.errors import ToolError
from myclaw.tools.models import ModelToolCall, ToolResult
from myclaw.tools.schema import ToolParam
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.utils.json_types import JsonObject

TURN_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
OTHER_TURN_ID = UUID("6ba7b811-9dad-41d1-80b4-00c04fd430c8")
CONFIRMATION_ID = UUID("6ba7b810-9dad-41d1-80b4-00c04fd430c8")


def _call(
    arguments: str,
    *,
    call_id: str = "call_1",
    name: str = "confirm",
) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=arguments)


class _EffectiveTool(BaseTool):
    name = "effective"
    description = "Exercise effective Tool preparation."
    required = ("action",)

    action: str
    invalid_when_ignored: Annotated[int, ToolParam(minimum=1)] = 1

    def __init__(self) -> None:
        self.prepared: JsonObject | None = None
        self.calls: list[str] = []

    def prepare(self, arguments: JsonObject) -> JsonObject:
        self.prepared = dict(arguments)
        return {"action": arguments["action"]}

    async def execute(self, *, action: str) -> str:
        self.calls.append(action)
        return action


class _ConfirmingTool(BaseTool):
    name = "confirm"
    description = "Exercise Tool Confirmation."
    required = ("action",)

    action: str

    def __init__(self) -> None:
        self.calls: list[str] = []

    def confirmation_request(self, *, action: str) -> ConfirmationPrompt:
        return ConfirmationPrompt(
            summary=f"Run {action}",
            details={"action": action, "nested": {"safe": True}},
            warnings=("This changes state.",),
        )

    async def execute(self, *, action: str) -> str:
        self.calls.append(action)
        return f"executed:{action}"


class _RefusingConfirmingTool(_ConfirmingTool):
    name = "refusing_confirm"
    description = "Refuse before confirmation."
    required = ("action",)
    action: str

    def refusal_reason(self, *, action: str) -> str:
        return f"{action} is refused."

    async def execute(self, *, action: str) -> str:
        return await super().execute(action=action)


class _BlockingConfirmingTool(_ConfirmingTool):
    name = "blocking_confirm"
    description = "Exercise cancellation after approval."
    required = ("action",)
    action: str

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, *, action: str) -> str:
        self.calls.append(action)
        self.started.set()
        await self.release.wait()
        return f"executed:{action}"


def _gateway(
    tool: BaseTool,
    *,
    channel: ConfirmationChannel | None = None,
) -> ToolGateway:
    gateway = ToolGateway(
        turn_id=TURN_ID,
        confirmation=channel,
        new_uuid=lambda: CONFIRMATION_ID,
    )
    gateway.register_tools((tool,))
    return gateway


@pytest.mark.asyncio
async def test_preparation_only_receives_declared_fields_and_can_remove_invalid_fields() -> None:
    tool = _EffectiveTool()
    gateway = ToolGateway()
    gateway.register_tools((tool,))

    result = await gateway.call(
        ModelToolCall(
            id="call_effective",
            name="effective",
            arguments='{"action":"run","invalid_when_ignored":"bad","unknown":1}',
        )
    )

    assert result.status == "success"
    assert result.content == "run"
    assert tool.prepared == {"action": "run", "invalid_when_ignored": "bad"}
    assert tool.calls == ["run"]


@pytest.mark.asyncio
async def test_approved_confirmation_executes_the_frozen_normalized_invocation() -> None:
    tool = _ConfirmingTool()
    channel = ConfirmationChannel(TURN_ID)
    gateway = _gateway(tool, channel=channel)
    task = asyncio.create_task(gateway.call(_call('{"action":"run"}')))

    request = await channel.next_request()
    assert request == ConfirmationRequest(
        confirmation_id=CONFIRMATION_ID,
        turn_id=TURN_ID,
        tool_call_id="call_1",
        tool_name="confirm",
        summary="Run run",
        details={"action": "run", "nested": {"safe": True}},
        warnings=("This changes state.",),
    )
    channel.respond_to_confirmation(request.confirmation_id, "approved")

    result = await task

    assert result.status == "success"
    assert result.content == "executed:run"
    assert result.confirmation is not None
    assert result.confirmation.request == request
    assert result.confirmation.decision == "approved"
    assert tool.calls == ["run"]


@pytest.mark.asyncio
async def test_declined_and_noninteractive_confirmation_are_refused_without_execution() -> None:
    declined_tool = _ConfirmingTool()
    declined_channel = ConfirmationChannel(TURN_ID)
    declined_gateway = _gateway(declined_tool, channel=declined_channel)
    declined_task = asyncio.create_task(declined_gateway.call(_call('{"action":"decline"}')))
    declined_request = await declined_channel.next_request()
    declined_channel.respond_to_confirmation(declined_request.confirmation_id, "declined")

    declined = await declined_task
    assert declined.status == "refused"
    assert declined.content == "Tool confirmation was declined."
    assert declined.confirmation is not None
    assert declined.confirmation.decision == "declined"
    assert declined_tool.calls == []

    noninteractive_tool = _ConfirmingTool()
    noninteractive = await _gateway(noninteractive_tool).call(_call('{"action":"background"}'))
    assert noninteractive.status == "refused"
    assert noninteractive.content == "Tool confirmation is unavailable."
    assert noninteractive.confirmation is not None
    assert noninteractive.confirmation.decision is None
    assert noninteractive_tool.calls == []


@pytest.mark.asyncio
async def test_existing_refusal_happens_before_confirmation() -> None:
    tool = _RefusingConfirmingTool()
    channel = ConfirmationChannel(TURN_ID)
    result = await _gateway(tool, channel=channel).call(
        _call('{"action":"blocked"}', name="refusing_confirm")
    )

    assert result.status == "refused"
    assert result.content == "blocked is refused."
    assert result.confirmation is None
    assert tool.calls == []
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(channel.next_request(), timeout=0.01)


@pytest.mark.asyncio
async def test_protocol_order_confirms_once_before_all_execution_attempts() -> None:
    events: list[str] = []

    class OrderedRetryTool(BaseTool):
        name = "ordered_retry"
        description = "Record the normalized confirmation protocol order."
        required = ("count",)
        max_retries = 1

        count: int

        def __init__(self) -> None:
            self.attempts = 0

        def prepare(self, arguments: JsonObject) -> JsonObject:
            events.append(f"prepare:{arguments['count']}")
            return arguments

        def refusal_reason(self, *, count: int) -> None:
            events.append(f"refusal:{count}")

        def confirmation_request(self, *, count: int) -> ConfirmationPrompt:
            events.append(f"confirmation:{count}")
            return ConfirmationPrompt(summary="Run ordered retry", details={"count": count})

        async def execute(self, *, count: int) -> str:
            self.attempts += 1
            events.append(f"execute:{self.attempts}:{count}")
            if self.attempts == 1:
                raise ToolError("Retry once.")
            return str(count)

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        events.append(f"approval:{request.details['count']}")
        return "approved"

    async def retry(delay: float) -> None:
        events.append(f"retry:{delay}")

    gateway = ToolGateway(
        sleep=retry,
        confirmation=approve,
        turn_id=TURN_ID,
        new_uuid=lambda: CONFIRMATION_ID,
    )
    gateway.register_tools((OrderedRetryTool(),))

    result = await gateway.call(_call('{"count":"2"}', name="ordered_retry"))

    assert result.status == "success"
    assert result.content == "2"
    assert result.confirmation is not None
    assert result.confirmation.decision == "approved"
    assert events == [
        "prepare:2",
        "refusal:2",
        "confirmation:2",
        "approval:2",
        "execute:1:2",
        "retry:1.0",
        "execute:2:2",
    ]


@pytest.mark.asyncio
async def test_cancellation_before_approval_never_executes_and_invalidates_request() -> None:
    tool = _ConfirmingTool()
    channel = ConfirmationChannel(TURN_ID)
    gateway = _gateway(tool, channel=channel)
    task = asyncio.create_task(gateway.call(_call('{"action":"cancelled"}')))
    request = await channel.next_request()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tool.calls == []
    with pytest.raises(ValueError, match="late or unknown"):
        channel.respond_to_confirmation(request.confirmation_id, "approved")


@pytest.mark.asyncio
async def test_cancellation_after_approval_waits_for_execution_and_returns_result() -> None:
    tool = _BlockingConfirmingTool()
    channel = ConfirmationChannel(TURN_ID)
    gateway = _gateway(tool, channel=channel)
    task = asyncio.create_task(
        gateway.call(_call('{"action":"commit"}', name="blocking_confirm"))
    )
    request = await channel.next_request()
    channel.respond_to_confirmation(request.confirmation_id, "approved")
    await tool.started.wait()

    task.cancel()
    tool.release.set()
    result = await task

    assert result.status == "success"
    assert result.content == "executed:commit"
    assert result.confirmation is not None
    assert result.confirmation.decision == "approved"
    assert tool.calls == ["commit"]


@pytest.mark.asyncio
async def test_accepted_approval_survives_cancellation_before_gateway_resumes() -> None:
    tool = _ConfirmingTool()
    channel = ConfirmationChannel(TURN_ID)
    gateway = _gateway(tool, channel=channel)
    task = asyncio.create_task(gateway.call(_call('{"action":"commit"}')))
    request = await channel.next_request()

    channel.respond_to_confirmation(request.confirmation_id, "approved")
    task.cancel()
    result = await task

    assert result.status == "success"
    assert result.confirmation == ToolConfirmationMetadata(
        request=request,
        decision="approved",
    )
    assert tool.calls == ["commit"]


@pytest.mark.asyncio
async def test_confirmation_channel_rejects_wrong_binding_late_and_duplicate_responses() -> None:
    tool = _ConfirmingTool()
    channel = ConfirmationChannel(TURN_ID)
    other_channel = ConfirmationChannel(OTHER_TURN_ID)
    wrong_turn_request = ConfirmationRequest(
        confirmation_id=CONFIRMATION_ID,
        turn_id=OTHER_TURN_ID,
        tool_call_id="call_wrong_turn",
        tool_name="confirm",
        summary="Wrong turn",
        details={},
    )
    with pytest.raises(ValueError, match="another turn"):
        await channel.request_confirmation(wrong_turn_request)

    gateway = _gateway(tool, channel=channel)
    task = asyncio.create_task(gateway.call(_call('{"action":"bound"}')))
    request = await channel.next_request()

    with pytest.raises(ValueError, match="late or unknown"):
        other_channel.respond_to_confirmation(request.confirmation_id, "approved")

    channel.respond_to_confirmation(request.confirmation_id, "approved")
    assert (await task).status == "success"
    with pytest.raises(ValueError, match="late or unknown"):
        channel.respond_to_confirmation(request.confirmation_id, "declined")


@pytest.mark.asyncio
async def test_cancelled_requests_are_not_delivered_and_close_wakes_the_host() -> None:
    channel = ConfirmationChannel(TURN_ID)
    request = ConfirmationRequest(
        confirmation_id=CONFIRMATION_ID,
        turn_id=TURN_ID,
        tool_call_id="call_cancelled",
        tool_name="confirm",
        summary="Cancelled request",
        details={},
    )
    pending = asyncio.create_task(channel.request_confirmation(request))
    await asyncio.sleep(0)

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(channel.next_request(), timeout=0.01)

    host = asyncio.create_task(channel.next_request())
    await asyncio.sleep(0)
    channel.close()
    with pytest.raises(RuntimeError, match="closed"):
        await host


def test_confirmation_values_are_immutable_and_provider_projection_is_content_only() -> None:
    details: JsonObject = {
        "action": "write",
        "nested": {"path": "notes.txt"},
        "steps": ["write"],
    }
    request = ConfirmationRequest(
        confirmation_id=CONFIRMATION_ID,
        turn_id=TURN_ID,
        tool_call_id="call_1",
        tool_name="confirm",
        summary="Write a file",
        details=details,
        warnings=(),
    )
    details["nested"] = {"path": "changed.txt"}

    exposed_details = request.details
    nested = exposed_details["nested"]
    steps = exposed_details["steps"]
    assert isinstance(nested, dict)
    assert isinstance(steps, list)
    nested["path"] = "mutated.txt"
    steps.append("delete")

    assert request.details == {
        "action": "write",
        "nested": {"path": "notes.txt"},
        "steps": ["write"],
    }
    with pytest.raises(FrozenInstanceError):
        request.summary = "changed"  # type: ignore[misc]

    metadata = ToolConfirmationMetadata(request=request, decision="approved")
    tool_result = ToolResult(
        tool_call_id="call_1",
        name="confirm",
        status="success",
        content="done",
        artifact=None,
        confirmation=metadata,
    )
    serialized = tool_result.to_dict()
    assert serialized["confirmation"] == metadata.to_dict()
    projected = model_message_from_session({"role": "tool", **serialized})
    assert projected is not None
    assert projected.to_dict() == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "confirm",
        "content": "done",
    }
