from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.tools.base import (
    BaseTool,
    ToolError,
    is_public_ip,
    normalize_public_ip,
    resolve_workspace_path,
)
from myclaw.tools.schema import Array, Integer, Object, String, parameter
from myclaw.tools.tool_gateway import (
    ConfirmationChannel,
    ModelToolCall,
    ToolConfirmationMetadata,
    ToolGateway,
)

TURN_ID = UUID("11111111-1111-4111-8111-111111111111")


class _PipelineTool(BaseTool):
    name = "pipeline"
    description = "Exercise the final Tool pipeline."

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[int, str]] = []
        self.force_safety = False

    @parameter(
        Object(
            {
                "count": Integer(),
                "payload": String(),
            },
            required=("count", "payload"),
        )
    )
    async def execute(self, *, count: int, payload: str) -> str:
        self.events.append("execute")
        self.calls.append((count, payload))
        return f"{count}:{payload}"

    def validate_arguments(  # type: ignore[override]
        self,
        *,
        count: int,
        payload: str,
    ) -> None:
        del count, payload
        self.events.append("validate")

    async def check_safety(  # type: ignore[override]
        self,
        *,
        count: int,
        payload: str,
    ) -> str | None:
        del count, payload
        self.events.append("safety")
        return "The operation needs confirmation." if self.force_safety else None


@pytest.mark.asyncio
async def test_final_pipeline_orders_preparation_and_preserves_full_execution_values() -> None:
    events: list[str] = []
    tool = _PipelineTool(events)
    gateway = ToolGateway()
    gateway.register_tools((tool,))

    result = await gateway.call(
        ModelToolCall(
            id="call_pipeline",
            name="pipeline",
            arguments='{"count":"2","payload":"full value","unknown":{"kept":true}}',
        )
    )

    assert result.status == "success"
    assert result.content == "2:full value"
    assert events == ["validate", "safety", "execute"]
    assert tool.calls == [(2, "full value")]


@pytest.mark.asyncio
async def test_schema_and_domain_errors_stop_before_safety() -> None:
    events: list[str] = []
    tool = _PipelineTool(events)
    gateway = ToolGateway()
    gateway.register_tools((tool,))

    invalid_schema = await gateway.call(
        ModelToolCall(
            id="call_invalid_schema",
            name="pipeline",
            arguments='{"count":"not-an-integer","payload":3}',
        )
    )
    assert invalid_schema.status == "error"
    assert "$.count" in invalid_schema.content
    assert "$.payload" in invalid_schema.content
    assert events == []

    class DomainErrorTool(_PipelineTool):
        name = "domain_error"

        def validate_arguments(  # type: ignore[override]
            self,
            *,
            count: int,
            payload: str,
        ) -> None:
            del count, payload
            self.events.append("validate")
            raise ToolError("The domain value is not usable.")

    domain_events: list[str] = []
    domain_gateway = ToolGateway()
    domain_gateway.register_tools((DomainErrorTool(domain_events),))
    domain_error = await domain_gateway.call(
        ModelToolCall(
            id="call_domain_error",
            name="domain_error",
            arguments='{"count":2,"payload":"ok"}',
        )
    )

    assert domain_error.status == "error"
    assert domain_error.content == "The domain value is not usable."
    assert domain_events == ["validate"]


@pytest.mark.asyncio
async def test_validation_cannot_change_arguments_seen_by_safety_or_execution() -> None:
    class MutatingValidatorTool(BaseTool):
        name = "mutating_validator"
        description = "Exercise isolated final-pipeline argument snapshots."

        def __init__(self) -> None:
            self.seen: list[tuple[str, list[str]]] = []

        @parameter(Object({"items": Array(String())}, required=("items",)))
        async def execute(self, *, items: list[str]) -> str:
            self.seen.append(("execute", items))
            return "done"

        def validate_arguments(self, *, items: list[str]) -> None:  # type: ignore[override]
            items.clear()

        async def check_safety(self, *, items: list[str]) -> str | None:  # type: ignore[override]
            self.seen.append(("safety", items))
            return "The original arguments need confirmation." if items else None

    tool = MutatingValidatorTool()
    gateway = ToolGateway()
    gateway.register_tools((tool,))

    result = await gateway.call(
        ModelToolCall(
            id="call_mutating_validator",
            name="mutating_validator",
            arguments='{"items":["unsafe"]}',
        )
    )

    assert result.status == "refused"
    assert tool.seen == [("safety", ["unsafe"])]


@pytest.mark.asyncio
async def test_class_decorated_tool_uses_the_final_pipeline() -> None:
    @parameter(Object({"count": Integer()}, required=("count",)))
    class ClassDecoratedTool(BaseTool):
        name = "class_decorated"
        description = "Exercise class-level Schema declarations."

        async def execute(self, *, count: int) -> str:
            return str(count)

    gateway = ToolGateway()
    gateway.register_tools((ClassDecoratedTool(),))

    result = await gateway.call(
        ModelToolCall(
            id="call_class_decorated",
            name="class_decorated",
            arguments='{"count":"not-an-integer"}',
        )
    )

    assert result.status == "error"
    assert "$.count" in result.content


@pytest.mark.asyncio
async def test_confirmation_is_projected_bound_once_and_executes_full_values() -> None:
    events: list[str] = []
    tool = _PipelineTool(events)
    tool.force_safety = True
    gateway = ToolGateway()
    gateway.register_tools((tool,))
    payload = "x" * 300
    call = ModelToolCall(
        id="call_confirm",
        name="pipeline",
        arguments=f'{{"count":2,"payload":"{payload}","unknown":"hidden"}}',
    )

    missing = await gateway.call(call)

    assert missing.status == "refused"
    assert missing.confirmation is not None
    request = missing.confirmation.request
    assert request.reason == "The operation needs confirmation."
    assert request.turn_id is None
    serialized_request = request.to_dict()
    assert "turn_id" not in serialized_request
    detail = request.details["payload"]
    assert detail == {"value": payload[:256], "original_length": 300}
    assert "unknown" not in request.details

    approved = await gateway.call(
        call,
        confirmation=ToolConfirmationMetadata(request=request, decision="approved"),
    )
    assert approved.status == "success"
    assert approved.confirmation is not None
    assert approved.confirmation.decision == "approved"
    assert tool.calls == [(2, payload)]

    duplicate = await gateway.call(
        call,
        confirmation=ToolConfirmationMetadata(request=request, decision="approved"),
    )
    assert duplicate.status == "refused"
    assert duplicate.confirmation is not None
    assert duplicate.confirmation.decision == "approved"
    assert tool.calls == [(2, payload)]

    mismatched = await gateway.call(
        ModelToolCall(id="different_call", name="pipeline", arguments=call.arguments),
        confirmation=ToolConfirmationMetadata(request=request, decision="approved"),
    )
    assert mismatched.status == "refused"
    assert len(tool.calls) == 1


@pytest.mark.asyncio
async def test_bound_confirmation_omits_turn_id_and_propagates_approval_cancel_race() -> None:
    tool = _PipelineTool([])
    tool.force_safety = True
    channel = ConfirmationChannel(TURN_ID)
    gateway = ToolGateway()
    gateway.register_tools((tool,))
    bound_gateway = gateway.for_run(confirmation=channel)
    task = asyncio.create_task(
        bound_gateway.call(
            ModelToolCall(
                id="call_bound_confirmation",
                name="pipeline",
                arguments='{"count":2,"payload":"full value"}',
            )
        )
    )
    request = await channel.next_request()

    assert request.turn_id is None
    channel.respond_to_confirmation(request.confirmation_id, "approved")
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert tool.calls == []


@pytest.mark.asyncio
async def test_tool_error_is_visible_unexpected_failure_is_redacted_and_cancellation_propagates() -> (
    None
):
    class FailingTool(_PipelineTool):
        name = "failing"

        async def execute(self, *, count: int, payload: str) -> str:
            del count, payload
            raise ToolError("Expected domain failure.")

    events: list[str] = []
    failures: list[Exception] = []
    gateway = ToolGateway(on_terminal_failure=failures.append)
    gateway.register_tools((FailingTool(events),))
    expected = await gateway.call(
        ModelToolCall(
            id="call_expected",
            name="failing",
            arguments='{"count":2,"payload":"ok"}',
        )
    )
    assert expected.status == "error"
    assert expected.content == "Expected domain failure."
    assert failures == []

    class UnexpectedTool(_PipelineTool):
        name = "unexpected"

        async def execute(self, *, count: int, payload: str) -> str:
            del count, payload
            raise RuntimeError("private details")

    unexpected_gateway = ToolGateway(on_terminal_failure=failures.append)
    unexpected_gateway.register_tools((UnexpectedTool([]),))
    unexpected = await unexpected_gateway.call(
        ModelToolCall(
            id="call_unexpected",
            name="unexpected",
            arguments='{"count":2,"payload":"ok"}',
        )
    )
    assert unexpected.status == "error"
    assert "private details" not in unexpected.content
    assert len(failures) == 1

    class CancelTool(_PipelineTool):
        name = "cancel"

        def __init__(self, events: list[str]) -> None:
            super().__init__(events)
            self.cancelled = False

        async def execute(self, *, count: int, payload: str) -> str:
            del count, payload
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return "unreachable"

    cancel_tool = CancelTool([])
    cancel_gateway = ToolGateway()
    cancel_gateway.register_tools((cancel_tool,))
    task = asyncio.create_task(
        cancel_gateway.call(
            ModelToolCall(
                id="call_cancel",
                name="cancel",
                arguments='{"count":2,"payload":"ok"}',
            )
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancel_tool.cancelled


def test_workspace_path_and_dns_helpers_fail_closed_across_host_boundaries(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    assert resolve_workspace_path(tmp_path, "nested") == nested.resolve()
    with pytest.raises(ValueError, match="outside the Workspace"):
        resolve_workspace_path(tmp_path, "..")

    assert normalize_public_ip("93.184.216.34") == "93.184.216.34"
    assert normalize_public_ip("::ffff:93.184.216.34") == "93.184.216.34"
    assert is_public_ip("2001:4860:4860::8888")
    assert not is_public_ip("127.0.0.1")
    assert not is_public_ip("fe80::1%12")
    with pytest.raises(ValueError):
        normalize_public_ip("10.0.0.1")
