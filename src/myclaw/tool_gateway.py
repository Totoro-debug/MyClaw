"""Tool Gateway for declared capabilities and normalized results."""

from jsonschema import Draft202012Validator, FormatChecker

from myclaw.contracts import (
    ErrorCode,
    ErrorInfo,
    ModelToolCall,
    Tool,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
)
from myclaw.file_tools import (
    FileToolAccessDenied,
    FileToolArgumentsError,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
)


class ToolGateway:
    """Resolve and execute one Tool catalog once per model call."""

    def __init__(
        self,
        *,
        context: ToolExecutionContext,
        tools: tuple[Tool, ...] | None = None,
    ) -> None:
        self._context = context
        catalog = (ReadFileTool(), ListFilesTool(), SearchFilesTool()) if tools is None else tools
        self._tools = {tool.definition.name: tool for tool in catalog}
        self._definitions = tuple(tool.definition for tool in catalog)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._definitions

    async def execute(self, tool_call: ModelToolCall) -> ToolResult:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return _error_result(
                tool_call,
                code="tool_not_found",
                message="The requested tool is not available.",
            )
        if not Draft202012Validator(
            tool.definition.input_schema,
            format_checker=FormatChecker(),
        ).is_valid(tool_call.arguments):
            return _error_result(
                tool_call,
                code="tool_invalid_arguments",
                message=f"Invalid arguments for {tool_call.name}.",
            )
        try:
            content = await tool.execute(tool_call.arguments, self._context)
        except FileToolAccessDenied:
            return _error_result(
                tool_call,
                code="tool_denied",
                message="The requested path is outside the allowed Workspace.",
            )
        except FileToolArgumentsError:
            return _error_result(
                tool_call,
                code="tool_invalid_arguments",
                message=f"Invalid arguments for {tool_call.name}.",
            )
        except Exception:
            return _error_result(
                tool_call,
                code="tool_failed",
                message=f"{tool_call.name} could not complete the request.",
            )
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            status="success",
            content=content,
            error=None,
            artifact=None,
        )


def _error_result(
    tool_call: ModelToolCall,
    *,
    code: ErrorCode,
    message: str,
) -> ToolResult:
    error = ErrorInfo(code=code, message=message)
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="error",
        content=message,
        error=error,
        artifact=None,
    )
