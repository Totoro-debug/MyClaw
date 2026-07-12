"""Tool Gateway for declared capabilities and normalized results."""

from jsonschema import Draft202012Validator, FormatChecker

from myclaw.contracts import (
    ErrorCode,
    ErrorInfo,
    ModelToolCall,
    PermissionDecision,
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
from myclaw.permission_policy import PermissionAssessment, assess_permission
from myclaw.scheduled_work import ScheduledWorkInvalidError, ScheduledWorkPersistenceError
from myclaw.shell_policy import ShellPolicyDenied
from myclaw.shell_tool import ShellBoundary, ShellTool
from myclaw.tool_artifacts import (
    ArtifactWriteError,
    ArtifactWriter,
    ToolArtifactExternalizer,
)
from myclaw.web_fetch import WebFetchBoundary, WebFetchTool
from myclaw.web_search import WebSearchBoundary, WebSearchTool
from myclaw.workspace_write_tools import EditFileTool, WriteFileTool


class ToolGateway:
    """Resolve and execute one Tool catalog once per model call."""

    def __init__(
        self,
        *,
        context: ToolExecutionContext,
        tools: tuple[Tool, ...] | None = None,
        web_search: WebSearchBoundary | None = None,
        web_fetch: WebFetchBoundary | None = None,
        shell: ShellBoundary | None = None,
        scheduled_work: Tool | None = None,
        max_tool_result_chars: int = 50_000,
        artifact_writer: ArtifactWriter | None = None,
    ) -> None:
        self._context = context
        catalog: tuple[Tool, ...]
        if tools is None:
            catalog = (
                ReadFileTool(),
                ListFilesTool(),
                SearchFilesTool(),
                WriteFileTool(),
                EditFileTool(),
            )
            if web_search is not None:
                catalog += (WebSearchTool(web_search),)
            if web_fetch is not None:
                catalog += (WebFetchTool(web_fetch),)
            if shell is not None:
                catalog += (ShellTool(shell),)
            if scheduled_work is not None:
                catalog += (scheduled_work,)
        else:
            catalog = tools
        self._tools = {tool.definition.name: tool for tool in catalog}
        self._definitions = tuple(tool.definition for tool in catalog)
        self._artifacts = ToolArtifactExternalizer(
            context=context,
            max_tool_result_chars=max_tool_result_chars,
            write_text=artifact_writer,
        )

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._definitions

    def permission_request(self, tool_call: ModelToolCall) -> PermissionAssessment | None:
        """Return an already validated foreground confirmation request, if required."""
        tool = self._tools.get(tool_call.name)
        if tool is None or not Draft202012Validator(
            tool.definition.input_schema,
            format_checker=FormatChecker(),
        ).is_valid(tool_call.arguments):
            return None
        assessment = assess_permission(tool_call, self._context)
        if assessment.decision is PermissionDecision.ASK and self._context.lane == "foreground":
            return assessment
        return None

    def discard_artifact(self, result: ToolResult) -> bool:
        """Roll back one artifact created by this Gateway but not persisted to its Session."""
        return self._artifacts.discard(result)

    def commit_artifact(self, result: ToolResult) -> bool:
        """Release rollback ownership after the artifact reference is persisted."""
        return self._artifacts.commit(result)

    async def execute(
        self,
        tool_call: ModelToolCall,
        *,
        approved: bool | None = None,
    ) -> ToolResult:
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
        assessment = assess_permission(tool_call, self._context)
        if assessment.decision is PermissionDecision.DENY:
            return _error_result(
                tool_call,
                code="tool_denied",
                message="The requested operation is not permitted.",
            )
        if assessment.decision is PermissionDecision.ASK:
            if self._context.lane != "foreground":
                return _refused_result(
                    tool_call,
                    message="Permission confirmation is unavailable in background work.",
                )
            if approved is not True:
                return _refused_result(
                    tool_call,
                    message="Permission denied by user.",
                )
        try:
            content = await tool.execute(tool_call.arguments, self._context)
        except (FileToolAccessDenied, ShellPolicyDenied):
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
        except ScheduledWorkInvalidError:
            return _error_result(
                tool_call,
                code="scheduled_work_invalid",
                message="The Scheduled Work definition is invalid.",
            )
        except ScheduledWorkPersistenceError:
            return _error_result(
                tool_call,
                code="persistence_error",
                message="Scheduled Work could not be read or written.",
            )
        except Exception:
            return _error_result(
                tool_call,
                code="tool_failed",
                message=f"{tool_call.name} could not complete the request.",
            )
        result = ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            status="success",
            content=content,
            error=None,
            artifact=None,
        )
        try:
            return self._artifacts.externalize(result)
        except ArtifactWriteError:
            return _error_result(
                tool_call,
                code="tool_failed",
                message=f"{tool_call.name} result could not be stored.",
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


def _refused_result(tool_call: ModelToolCall, *, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="refused",
        content=message,
        error=ErrorInfo(code="tool_refused", message=message),
        artifact=None,
    )
