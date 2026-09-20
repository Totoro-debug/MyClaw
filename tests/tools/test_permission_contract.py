from __future__ import annotations

import json
from typing import Any

import pytest

from myclaw.agent.tools.base import BaseTool, ToolError
from myclaw.agent.tools.permission import (
    PermissionContext,
    PermissionDecision,
    ToolAuthorizationSession,
    ToolInvocationFacts,
    ToolPermissionPolicy,
)
from myclaw.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
    ToolGateway,
)


class _RecordingSession:
    def __init__(self, decision: PermissionDecision) -> None:
        self.decision = decision

    def initial_decision(self) -> PermissionDecision:
        return self.decision

    async def authorize_network_target(
        self,
        target: object,
        resolved_addresses: tuple[str, ...],
    ) -> None:
        del target, resolved_addresses


class _RecordingPolicy(ToolPermissionPolicy):
    def __init__(self) -> None:
        self.facts: list[ToolInvocationFacts] = []
        self.contexts: list[PermissionContext] = []
        self.sessions: list[ToolAuthorizationSession] = []

    def open(
        self,
        facts: ToolInvocationFacts,
        context: PermissionContext,
    ) -> ToolAuthorizationSession:
        session = _RecordingSession("direct")
        self.facts.append(facts)
        self.contexts.append(context)
        self.sessions.append(session)
        return session


class _FixedDecisionPolicy(_RecordingPolicy):
    def __init__(self, decision: PermissionDecision) -> None:
        super().__init__()
        self.decision = decision

    def open(
        self,
        facts: ToolInvocationFacts,
        context: PermissionContext,
    ) -> ToolAuthorizationSession:
        del context
        session = _RecordingSession(self.decision)
        self.facts.append(facts)
        self.sessions.append(session)
        return session


class _PreparingTool(BaseTool):
    name = "preparing"
    description = "A Tool used to inspect the authorization contract."
    required = ("count",)
    count: int

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def check_safety(self, *, count: int) -> str | None:  # type: ignore[override]
        del count
        return "legacy safety reason"

    async def execute(self, *, count: int) -> str:
        self.calls.append(count)
        return str(count)


class _NoSafetyTool(BaseTool):
    name = "no_safety"
    description = "A Tool with no legacy safety reason."
    required = ("value",)
    value: str

    def __init__(self, observed_sessions: list[ToolAuthorizationSession] | None = None) -> None:
        self.calls: list[str] = []
        self.observed_sessions = observed_sessions

    async def execute(self, *, value: str) -> str:
        self.calls.append(value)
        return value

    async def execute_authorized(
        self,
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
    ) -> str:
        if self.observed_sessions is not None:
            self.observed_sessions.append(authorization)
        return await super().execute_authorized(arguments, authorization)


class _EmptySafetyReasonTool(BaseTool):
    name = "empty_safety_reason"
    description = "A Tool with an empty legacy safety reason."

    def __init__(self) -> None:
        self.calls = 0

    async def check_safety(self) -> str | None:  # type: ignore[override]
        return ""

    async def execute(self) -> str:
        self.calls += 1
        return "executed"


class _ValidationTool(BaseTool):
    name = "validation_error"
    description = "A Tool with a validation error."

    def validate_arguments(self) -> str:  # type: ignore[override]
        return "validation failed"

    async def execute(self) -> str:
        raise AssertionError("validation must prevent execution")


class _RefusingTool(BaseTool):
    name = "business_refusal"
    description = "A Tool with a business refusal."
    required = ("value",)
    value: str

    def refusal_reason(self, *, value: str) -> str:
        del value
        return "business refusal"

    async def execute(self, *, value: str) -> str:
        del value
        raise AssertionError("business refusal must prevent execution")


class _ExecutionErrorTool(BaseTool):
    name = "execution_error"
    description = "A Tool with an execution error."

    async def execute(self) -> str:
        raise ToolError("execution failed")


@pytest.mark.asyncio
async def test_gateway_opens_a_fresh_session_with_structured_call_facts() -> None:
    policy = _RecordingPolicy()
    context = PermissionContext(origin="foreground")
    tool = _PreparingTool()
    gateway = ToolGateway._for_memory(
        (tool,),
        permission_policy=policy,
        permission_context=context,
    )

    first = await gateway.call(
        ModelToolCall(id="call-1", name=tool.name, arguments=json.dumps({"count": "7"}))
    )
    second = await gateway.call(
        ModelToolCall(id="call-2", name=tool.name, arguments=json.dumps({"count": "8"}))
    )

    assert (first.status, first.content) == ("success", "7")
    assert (second.status, second.content) == ("success", "8")
    assert [facts.tool_name for facts in policy.facts] == [tool.name, tool.name]
    assert [facts.normalized_arguments for facts in policy.facts] == [
        {"count": 7},
        {"count": 8},
    ]
    assert [facts.legacy_safety_reason for facts in policy.facts] == [
        "legacy safety reason",
        "legacy safety reason",
    ]
    assert policy.contexts == [context, context]
    assert policy.sessions[0] is not policy.sessions[1]
    assert tool.calls == [7, 8]
    assert not any(
        name in vars(tool)
        for name in ("authorization_session", "permission_state", "_authorization_session")
    )


@pytest.mark.asyncio
async def test_policy_confirm_decision_uses_one_confirmation_and_passes_session_to_execution() -> None:
    policy = _FixedDecisionPolicy("confirm")
    observed_sessions: list[ToolAuthorizationSession] = []
    tool = _NoSafetyTool(observed_sessions)
    gateway = ToolGateway._for_memory((tool,), permission_policy=policy)
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(id="call-confirm", name=tool.name, arguments='{"value":"payload"}'),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].reason == "Tool confirmation is required."
    assert requests[0].details == {"value": "payload"}
    assert observed_sessions == []
    assert tool.calls == []


@pytest.mark.asyncio
async def test_direct_policy_session_reaches_the_execution_boundary() -> None:
    policy = _FixedDecisionPolicy("direct")
    observed_sessions: list[ToolAuthorizationSession] = []
    tool = _NoSafetyTool(observed_sessions)
    gateway = ToolGateway._for_memory((tool,), permission_policy=policy)

    result = await gateway.call(
        ModelToolCall(id="call-authorized", name=tool.name, arguments='{"value":"payload"}')
    )

    assert result.status == "success"
    assert len(observed_sessions) == 1
    assert observed_sessions[0] is policy.sessions[0]
    assert tool.calls == ["payload"]


@pytest.mark.asyncio
async def test_approved_confirm_session_reaches_the_execution_boundary() -> None:
    policy = _FixedDecisionPolicy("confirm")
    observed_sessions: list[ToolAuthorizationSession] = []
    tool = _NoSafetyTool(observed_sessions)
    gateway = ToolGateway._for_memory((tool,), permission_policy=policy)

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        return "approved"

    result = await gateway.call(
        ModelToolCall(id="call-approved", name=tool.name, arguments='{"value":"payload"}'),
        confirmation=approve,
    )

    assert result.status == "success"
    assert result.confirmation is not None
    assert result.confirmation.decision == "approved"
    assert observed_sessions == [policy.sessions[0]]
    assert tool.calls == ["payload"]


@pytest.mark.asyncio
async def test_default_policy_preserves_an_empty_legacy_confirmation_reason() -> None:
    tool = _EmptySafetyReasonTool()
    gateway = ToolGateway._for_memory((tool,))
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(id="call-empty-reason", name=tool.name, arguments="{}"),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].reason == ""
    assert tool.calls == 0


@pytest.mark.asyncio
async def test_hard_and_business_errors_do_not_open_permission_or_confirmation() -> None:
    policy = _RecordingPolicy()
    gateway = ToolGateway._for_memory(
        (_ValidationTool(), _RefusingTool(), _ExecutionErrorTool()),
        permission_policy=policy,
    )

    invalid_json = await gateway.call(
        ModelToolCall(id="call-invalid-json", name="validation_error", arguments="not json")
    )
    unknown_tool = await gateway.call(
        ModelToolCall(id="call-unknown", name="unknown", arguments="{}")
    )
    validation = await gateway.call(
        ModelToolCall(id="call-validation", name="validation_error", arguments="{}")
    )
    refusal = await gateway.call(
        ModelToolCall(
            id="call-refusal",
            name="business_refusal",
            arguments='{"value":"payload"}',
        )
    )
    execution = await gateway.call(
        ModelToolCall(id="call-execution", name="execution_error", arguments="{}")
    )

    assert [result.status for result in (invalid_json, unknown_tool, validation, execution)] == [
        "error",
        "error",
        "error",
        "error",
    ]
    assert refusal.status == "refused"
    assert [facts.tool_name for facts in policy.facts] == ["execution_error"]
    assert len(policy.sessions) == 1


@pytest.mark.asyncio
async def test_invalid_policy_decision_is_a_hard_error_without_confirmation() -> None:
    class InvalidSession:
        def initial_decision(self) -> str:
            return "defer"

        async def authorize_network_target(
            self,
            target: object,
            resolved_addresses: tuple[str, ...],
        ) -> None:
            del target, resolved_addresses

    class InvalidPolicy(ToolPermissionPolicy):
        def open(
            self,
            facts: ToolInvocationFacts,
            context: PermissionContext,
        ) -> ToolAuthorizationSession:
            del facts, context
            return InvalidSession()  # type: ignore[return-value]

    tool = _NoSafetyTool()
    gateway = ToolGateway._for_memory((tool,), permission_policy=InvalidPolicy())
    result = await gateway.call(
        ModelToolCall(id="call-invalid-decision", name=tool.name, arguments='{"value":"x"}')
    )

    assert result.status == "error"
    assert result.content == "Tool authorization returned an invalid decision."
    assert tool.calls == []


def test_invocation_facts_detach_normalized_arguments() -> None:
    arguments: dict[str, Any] = {"nested": {"value": 1}}
    facts = ToolInvocationFacts(tool_name="tool", normalized_arguments=arguments)

    arguments["nested"] = {"value": 2}
    detached = facts.normalized_arguments
    detached["nested"]["value"] = 3

    assert facts.normalized_arguments == {"nested": {"value": 1}}
