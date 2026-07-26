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
    from myclaw.tools.shell.shell_policy import (
        ShellPolicyDenied,
        assess_shell_command,
        parse_shell_request,
    )

    arguments = tool_call.arguments
    if not isinstance(arguments, dict):
        return PermissionAssessment(decision=PermissionDecision.DENY)

    if tool_call.name == "shell":
        try:
            request = parse_shell_request(arguments, context.workspace)
        except ShellPolicyDenied:
            return PermissionAssessment(decision=PermissionDecision.DENY)
        decision = assess_shell_command(
            request.command,
            cwd=request.cwd,
            workspace=request.workspace_root,
        )
        if decision is PermissionDecision.ASK:
            return PermissionAssessment(
                decision=decision,
                action="run",
                resource=request.command,
                risk_summary=(
                    "Approved Shell commands may access paths outside the Workspace or the network."
                ),
            )
        return PermissionAssessment(decision=decision)
    if tool_call.name == "create_scheduled_work":
        title = arguments.get("title")
        cron = arguments.get("cron")
        if not isinstance(title, str) or not title or not isinstance(cron, str) or not cron:
            return PermissionAssessment(decision=PermissionDecision.DENY)
        return PermissionAssessment(
            decision=PermissionDecision.ASK,
            action="schedule",
            resource=f"{title} | {cron}",
            risk_summary="This creates recurring background work.",
        )
    return PermissionAssessment(decision=PermissionDecision.ALLOW)
