"""Built-in Permission Policy decisions for Tool calls."""

from dataclasses import dataclass

from myclaw.contracts import ModelToolCall, PermissionDecision, ToolExecutionContext


@dataclass(frozen=True, slots=True)
class PermissionAssessment:
    """A fixed policy decision plus the minimum user-visible confirmation context."""

    decision: PermissionDecision
    action: str = ""
    resource: str = ""
    risk_summary: str = ""


def assess_permission(
    tool_call: ModelToolCall,
    context: ToolExecutionContext,
) -> PermissionAssessment:
    """Return the fixed permission decision for the current execution lane."""
    if tool_call.name in {"write_file", "edit_file"}:
        resource = tool_call.arguments.get("path")
        return PermissionAssessment(
            decision=PermissionDecision.ASK,
            action="write" if tool_call.name == "write_file" else "edit",
            resource=resource if isinstance(resource, str) else "",
            risk_summary="This changes a Workspace file.",
        )
    return PermissionAssessment(decision=PermissionDecision.ALLOW)
