"""Provider-neutral Shell tool boundary without process lifecycle ownership."""

from pathlib import Path
from typing import Annotated, Protocol

from myclaw.tools.base import BaseTool
from myclaw.tools.errors import ToolError
from myclaw.tools.schema import ToolParam
from myclaw.tools.shell.shell_policy import (
    ShellPolicyDenied,
    ShellRequest,
    parse_shell_request,
    shell_command_is_allowed,
)


class ShellBoundary(Protocol):
    async def execute(self, request: ShellRequest) -> str: ...


class UnavailableShellBoundary:
    """Fail closed until the Shell process lifecycle adapter is composed."""

    async def execute(self, request: ShellRequest) -> str:
        del request
        raise RuntimeError("Shell process execution is unavailable.")


class ShellTool(BaseTool):
    """Expose Shell requests through the Tool protocol."""

    name = "shell"
    description = (
        "Run one of five exact read-only commands from a Workspace directory; this is not an "
        "operating-system filesystem or network sandbox."
    )
    required = ("command", "timeout")

    command: Annotated[str, ToolParam(description="Exact Shell command.", min_length=1)]
    cwd: Annotated[str, ToolParam(description="Workspace-relative working directory.")] = "."
    timeout: Annotated[
        int,
        ToolParam(description="Execution timeout in seconds.", minimum=60, maximum=600),
    ]

    def __init__(self, *, workspace: Path, boundary: ShellBoundary) -> None:
        self._workspace = workspace
        self._boundary = boundary

    def refusal_reason(self, *, command: str, cwd: str, timeout: int) -> str | None:
        request = self._request(command=command, cwd=cwd, timeout=timeout)
        if shell_command_is_allowed(
            request.command,
            cwd=request.cwd,
            workspace=request.workspace_root,
        ):
            return None
        return "Shell command refused because it is not in the safe read-only allowlist."

    async def execute(self, *, command: str, cwd: str, timeout: int) -> str:
        request = self._request(command=command, cwd=cwd, timeout=timeout)
        if not shell_command_is_allowed(
            request.command,
            cwd=request.cwd,
            workspace=request.workspace_root,
        ):
            raise ToolError("Shell command is not in the safe read-only allowlist.")
        try:
            return await self._boundary.execute(request)
        except ShellPolicyDenied as error:
            raise ToolError("Shell process execution was rejected by the safety boundary.") from error

    def _request(self, *, command: str, cwd: str, timeout: int) -> ShellRequest:
        try:
            return parse_shell_request(
                command=command,
                cwd=cwd,
                timeout=timeout,
                workspace=self._workspace,
            )
        except ShellPolicyDenied as error:
            raise ToolError("Shell request parameters or Workspace cwd are invalid.") from error
