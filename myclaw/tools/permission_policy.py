"""Built-in Permission Policy decisions for Tool calls."""

from dataclasses import dataclass
from enum import StrEnum

from myclaw.tools.models import ModelToolCall, ToolExecutionContext


class PermissionDecision(StrEnum):
    """A Permission Policy decision before execution-context conversion."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


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
    arguments = tool_call.arguments
    if not isinstance(arguments, dict):
        return PermissionAssessment(decision=PermissionDecision.DENY)

    return PermissionAssessment(decision=PermissionDecision.ALLOW)
