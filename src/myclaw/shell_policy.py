"""Fixed first-version Permission Policy for Shell commands."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from unicodedata import category

from myclaw.contracts import JsonObject, PermissionDecision

AUTOMATICALLY_ALLOWED_COMMANDS: Final = frozenset(
    {
        "pwd",
        "git status",
        "git status --short",
        "git diff --stat",
        "git diff --name-only",
    }
)


class ShellPolicyDenied(PermissionError):
    """Raised when a Shell request cannot enter permission evaluation."""


@dataclass(frozen=True, slots=True)
class ShellRequest:
    """One validated request passed to the operating-system Shell boundary."""

    command: str
    cwd: Path
    timeout: int


def parse_shell_request(arguments: JsonObject, workspace: Path) -> ShellRequest:
    """Validate and normalize the Shell arguments covered by the Permission Policy."""
    command = arguments.get("command")
    cwd = arguments.get("cwd", ".")
    timeout = arguments.get("timeout")
    if (
        not isinstance(command, str)
        or not command.strip()
        or not isinstance(cwd, str)
        or not cwd
        or not isinstance(timeout, int)
        or isinstance(timeout, bool)
        or not 60 <= timeout <= 600
        or any(category(character) == "Cc" for character in command)
    ):
        raise ShellPolicyDenied("invalid Shell command parameters")

    try:
        workspace_root = workspace.resolve(strict=True)
        target = Path(cwd)
        if not target.is_absolute():
            target = workspace_root / target
        resolved_cwd = target.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ShellPolicyDenied("Shell cwd cannot be resolved") from exc
    if (
        not workspace_root.is_dir()
        or not resolved_cwd.is_relative_to(workspace_root)
        or not resolved_cwd.is_dir()
    ):
        raise ShellPolicyDenied("Shell cwd must be a Workspace directory")
    return ShellRequest(command=command, cwd=resolved_cwd, timeout=timeout)


def assess_shell_command(command: str) -> PermissionDecision:
    """Allow only a frozen exact command shape; ask for every other command."""
    if command in AUTOMATICALLY_ALLOWED_COMMANDS:
        return PermissionDecision.ALLOW
    return PermissionDecision.ASK
