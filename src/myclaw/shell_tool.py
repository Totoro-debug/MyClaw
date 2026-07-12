"""Provider-neutral Shell tool boundary without process lifecycle ownership."""

from typing import Protocol

from myclaw.contracts import JsonObject, ToolDefinition, ToolExecutionContext
from myclaw.shell_policy import ShellRequest, parse_shell_request


class ShellBoundary(Protocol):
    async def execute(self, request: ShellRequest) -> str: ...


class UnavailableShellBoundary:
    """Fail closed until the Shell process lifecycle adapter is composed."""

    async def execute(self, request: ShellRequest) -> str:
        del request
        raise RuntimeError("Shell process execution is unavailable.")


class ShellTool:
    """Expose Shell requests through the Tool protocol."""

    _definition = ToolDefinition(
        name="shell",
        description=(
            "Run a command from a Workspace directory; approved commands are not OS filesystem "
            "or network sandboxed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "cwd": {"type": "string", "minLength": 1, "default": "."},
                "timeout": {"type": "integer", "minimum": 60, "maximum": 600},
            },
            "required": ["command", "timeout"],
            "additionalProperties": False,
        },
    )

    def __init__(self, boundary: ShellBoundary) -> None:
        self._boundary = boundary

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str:
        request = parse_shell_request(arguments, context.workspace)
        return await self._boundary.execute(request)
