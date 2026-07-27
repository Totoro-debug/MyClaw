"""Tool Gateway for declared capabilities and normalized results."""

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Protocol, cast

from jsonschema import Draft202012Validator, FormatChecker

from myclaw.agent.workspace import Workspace
from myclaw.errors import ErrorCode, ErrorInfo
from myclaw.schedule.scheduled_work import (
    ScheduledWorkInvalidError,
    ScheduledWorkPersistenceError,
)
from myclaw.tools.base import BaseTool
from myclaw.tools.errors import ToolError
from myclaw.tools.files.file_tools import (
    FileToolAccessDenied,
    FileToolArgumentsError,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
)
from myclaw.tools.files.workspace_write_tools import EditFileTool, WriteFileTool
from myclaw.tools.models import ModelToolCall, ToolDefinition, ToolExecutionContext, ToolResult
from myclaw.tools.permission_policy import (
    PermissionAssessment,
    PermissionDecision,
    assess_permission,
)
from myclaw.tools.ports import Tool
from myclaw.tools.schema import OpenAIToolSchema
from myclaw.tools.security import Security
from myclaw.tools.shell.shell_policy import ShellPolicyDenied
from myclaw.tools.shell.shell_tool import ShellBoundary, ShellTool
from myclaw.tools.web.web_fetch import WebFetchBoundary, WebFetchTool
from myclaw.tools.web.web_search import WebSearchBoundary, WebSearchTool
from myclaw.utils.json_types import JsonObject, JsonScalar, JsonValue

type Sleep = Callable[[float], Awaitable[None]]

_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")


class _ExecutableTool(Protocol):
    execute: Callable[..., Awaitable[object]]


class ToolGateway:
    """Resolve and execute one Tool catalog once per model call."""

    def __init__(
        self,
        *,
        context: ToolExecutionContext | None = None,
        tools: tuple[Tool, ...] | None = None,
        web_search: WebSearchBoundary | None = None,
        web_fetch: WebFetchBoundary | None = None,
        shell: ShellBoundary | None = None,
        scheduled_work: Tool | None = None,
        max_tool_result_chars: int | None = None,
        artifact_writer: object | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        # Temporary accepted arguments keep legacy constructors running; Gateway ignores them.
        del max_tool_result_chars, artifact_writer
        self._context = context
        catalog: tuple[Tool, ...]
        if context is None:
            if any(
                dependency is not None
                for dependency in (tools, web_search, web_fetch, shell, scheduled_work)
            ):
                msg = "Legacy Tool Gateway dependencies require an execution context"
                raise TypeError(msg)
            catalog = ()
        elif tools is None:
            catalog = ()
            if scheduled_work is not None:
                catalog += (scheduled_work,)
        else:
            catalog = tools
        self._tools = {tool.definition.name: tool for tool in catalog}
        self._definitions = tuple(tool.definition for tool in catalog)
        self._registered = False
        self._registered_tools: dict[str, BaseTool] = {}
        self._schemas: tuple[OpenAIToolSchema, ...] = ()
        self._parameter_schemas: dict[str, JsonObject] = {}
        if context is not None and tools is None:
            # Temporary expand-contract support for direct legacy constructors.
            # Production Runtime replaces this catalog through register_tools(); #48
            # removes context construction entirely.
            workspace = Workspace.from_path(context.workspace)
            security = Security(
                workspace=workspace,
                agent_home=context.agent_home,
                artifact_directory=(
                    context.agent_home
                    / "sessions"
                    / workspace.slug
                    / "artifacts"
                    / context.session_id
                ),
            )
            migrated_tools: list[BaseTool] = [
                ReadFileTool(security=security),
                ListFilesTool(security=security),
                SearchFilesTool(security=security),
                WriteFileTool(security=security),
                EditFileTool(security=security),
            ]
            if web_search is not None:
                migrated_tools.append(WebSearchTool(search=web_search))
            if web_fetch is not None:
                migrated_tools.append(WebFetchTool(fetcher=web_fetch))
            if shell is not None:
                migrated_tools.append(
                    ShellTool(workspace=context.workspace, boundary=shell)
                )
            migrated_schemas = tuple(tool.to_schema() for tool in migrated_tools)
            self._registered_tools = {tool.name: tool for tool in migrated_tools}
            self._schemas = tuple(deepcopy(schema) for schema in migrated_schemas)
            self._parameter_schemas = {
                tool.name: deepcopy(_parameter_schema(schema))
                for tool, schema in zip(migrated_tools, migrated_schemas, strict=True)
            }
        self._sleep = sleep

    def register_tools(self, tools: tuple[BaseTool, ...]) -> None:
        """Register and cache one stable annotation-driven Tool Catalog."""
        if self._registered:
            msg = "Tool Catalog has already been registered"
            raise RuntimeError(msg)
        schemas = tuple(tool.to_schema() for tool in tools)
        parameter_schemas = {
            tool.name: _parameter_schema(schema)
            for tool, schema in zip(tools, schemas, strict=True)
        }
        self._registered_tools = {tool.name: tool for tool in tools}
        self._schemas = tuple(deepcopy(schema) for schema in schemas)
        self._parameter_schemas = {
            name: deepcopy(schema) for name, schema in parameter_schemas.items()
        }
        self._registered = True

    @property
    def schemas(self) -> tuple[OpenAIToolSchema, ...]:
        """Return a defensive snapshot of the registered OpenAI Tool schemas."""
        registered = tuple(deepcopy(schema) for schema in self._schemas)
        legacy = tuple(_legacy_schema(definition) for definition in self._definitions)
        return (*registered, *legacy)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        registered = tuple(_legacy_definition(schema) for schema in self._schemas)
        return (*registered, *self._definitions)

    def permission_request(self, tool_call: ModelToolCall) -> PermissionAssessment | None:
        """Return an already validated foreground confirmation request, if required."""
        context = self._legacy_context()
        tool = self._tools.get(tool_call.name)
        arguments = tool_call.arguments
        if tool is None or not isinstance(arguments, dict) or not Draft202012Validator(
            tool.definition.input_schema,
            format_checker=FormatChecker(),
        ).is_valid(arguments):
            return None
        legacy_call = ModelToolCall(id=tool_call.id, name=tool_call.name, arguments=arguments)
        assessment = assess_permission(legacy_call, context)
        if assessment.decision is PermissionDecision.ASK and context.lane == "foreground":
            return assessment
        return None

    async def call(self, tool_call: ModelToolCall) -> ToolResult:
        """Parse, prepare, refuse, and execute one registered Tool call."""
        raw_arguments = tool_call.arguments
        if isinstance(raw_arguments, dict):
            # Temporary support for directly constructed legacy test calls; removed by #48.
            parsed = raw_arguments
        elif not isinstance(raw_arguments, str):
            return _error_result(
                tool_call,
                code="tool_invalid_arguments",
                message="Tool arguments could not be parsed.",
            )
        else:
            try:
                parsed = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                return _error_result(
                    tool_call,
                    code="tool_invalid_arguments",
                    message="Tool arguments could not be parsed.",
                )
        if not isinstance(parsed, dict):
            return _error_result(
                tool_call,
                code="tool_invalid_arguments",
                message="Tool arguments could not be parsed.",
            )
        arguments = parsed
        if tool_call.name not in self._registered_tools and tool_call.name in self._tools:
            return await self._call_legacy_during_migration(tool_call, arguments)
        tool, normalized, correct = self._prepare(tool_call.name, arguments)
        if tool is None:
            return _error_result(
                tool_call,
                code="tool_not_found",
                message="The requested tool is not available.",
            )
        if not correct:
            return _error_result(
                tool_call,
                code="tool_invalid_arguments",
                message=f"Invalid arguments for {tool_call.name}.",
            )

        refusal = getattr(tool, "refusal_reason", None)
        if refusal is not None:
            try:
                reason = cast(Callable[..., str | None], refusal)(**normalized)
            except Exception as error:
                message = (
                    error.message
                    if isinstance(error, ToolError)
                    else f"{tool_call.name} could not complete the request."
                )
                return _error_result(
                    tool_call,
                    code="tool_failed",
                    message=message,
                )
            if reason is not None:
                if not isinstance(reason, str):
                    return _error_result(
                        tool_call,
                        code="tool_failed",
                        message=f"{tool_call.name} could not complete the request.",
                    )
                return _refused_result(tool_call, message=reason)

        execute = cast(_ExecutableTool, tool).execute
        for attempt in range(tool.max_retries + 1):
            try:
                content = await execute(**normalized)
                if not isinstance(content, str):
                    raise _NonStringToolResult
                return _success_result(tool_call, content)
            except Exception as error:
                if attempt < tool.max_retries:
                    await self._sleep(float(2**attempt))
                    continue
                message = (
                    error.message
                    if isinstance(error, ToolError)
                    else f"{tool_call.name} could not complete the request."
                )
                return _error_result(tool_call, code="tool_failed", message=message)
        raise AssertionError("Tool retry budget exhausted without a terminal result")

    def _prepare(
        self,
        name: str,
        arguments: JsonObject,
    ) -> tuple[BaseTool | None, JsonObject, bool]:
        tool = self._registered_tools.get(name)
        if tool is None:
            return None, {}, False
        schema = self._parameter_schemas[name]
        properties_value = schema.get("properties")
        if not isinstance(properties_value, dict):
            raise AssertionError("cached Tool parameter schema has no properties")
        properties = properties_value
        normalized: JsonObject = {}
        for parameter_name, parameter_schema_value in properties.items():
            if not isinstance(parameter_schema_value, dict):
                raise AssertionError("cached Tool parameter declaration is not an object")
            parameter_schema = parameter_schema_value
            if parameter_name in arguments:
                valid, value = _coerce(arguments[parameter_name], parameter_schema)
                if not valid:
                    return tool, normalized, False
                normalized[parameter_name] = value
            elif "default" in parameter_schema:
                normalized[parameter_name] = deepcopy(parameter_schema["default"])
        correct = Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).is_valid(normalized)
        return tool, normalized, correct

    def _legacy_context(self) -> ToolExecutionContext:
        if self._context is None:
            msg = "The legacy Tool Gateway path requires an execution context"
            raise RuntimeError(msg)
        return self._context

    async def _call_legacy_during_migration(
        self,
        tool_call: ModelToolCall,
        arguments: JsonObject,
    ) -> ToolResult:
        """Run one not-yet-migrated Tool until #48 removes the legacy stack."""
        context = self._legacy_context()
        tool = self._tools[tool_call.name]
        if not Draft202012Validator(
            tool.definition.input_schema,
            format_checker=FormatChecker(),
        ).is_valid(arguments):
            return _error_result(
                tool_call,
                code="tool_invalid_arguments",
                message=f"Invalid arguments for {tool_call.name}.",
            )
        legacy_call = ModelToolCall(
            id=tool_call.id,
            name=tool_call.name,
            arguments=arguments,
        )
        assessment = assess_permission(legacy_call, context)
        if assessment.decision is PermissionDecision.DENY:
            return _error_result(
                tool_call,
                code="tool_denied",
                message="The requested operation is not permitted.",
            )
        if assessment.decision is PermissionDecision.ASK:
            return _refused_result(
                tool_call,
                message="The requested operation requires unavailable user confirmation.",
            )
        try:
            content = await tool.execute(arguments, context)
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
        if not isinstance(content, str):
            return _error_result(
                tool_call,
                code="tool_failed",
                message=f"{tool_call.name} could not complete the request.",
            )
        return _success_result(tool_call, content)

    async def execute(
        self,
        tool_call: ModelToolCall,
        *,
        approved: bool | None = None,
    ) -> ToolResult:
        if tool_call.name in self._registered_tools:
            # Temporary expand-contract forwarding for legacy direct callers. The
            # approval value is intentionally ignored; #48 removes this method.
            del approved
            return await self.call(tool_call)
        context = self._legacy_context()
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return _error_result(
                tool_call,
                code="tool_not_found",
                message="The requested tool is not available.",
            )
        arguments = tool_call.arguments
        if not isinstance(arguments, dict) or not Draft202012Validator(
            tool.definition.input_schema,
            format_checker=FormatChecker(),
        ).is_valid(arguments):
            return _error_result(
                tool_call,
                code="tool_invalid_arguments",
                message=f"Invalid arguments for {tool_call.name}.",
            )
        legacy_call = ModelToolCall(id=tool_call.id, name=tool_call.name, arguments=arguments)
        assessment = assess_permission(legacy_call, context)
        if assessment.decision is PermissionDecision.DENY:
            return _error_result(
                tool_call,
                code="tool_denied",
                message="The requested operation is not permitted.",
            )
        if assessment.decision is PermissionDecision.ASK:
            if context.lane != "foreground":
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
            content = await tool.execute(arguments, context)
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
        return _success_result(tool_call, content)


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


def _success_result(tool_call: ModelToolCall, content: str) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="success",
        content=content,
        error=None,
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


def _parameter_schema(schema: OpenAIToolSchema) -> JsonObject:
    return schema["function"]["parameters"]


def _legacy_schema(definition: ToolDefinition) -> OpenAIToolSchema:
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": deepcopy(definition.input_schema),
        },
    }


def _legacy_definition(schema: OpenAIToolSchema) -> ToolDefinition:
    function = schema["function"]
    return ToolDefinition(
        name=function["name"],
        description=function["description"],
        input_schema=deepcopy(function["parameters"]),
    )


def _coerce(value: JsonValue, schema: JsonObject) -> tuple[bool, JsonScalar]:
    declared = schema.get("type")
    accepted_types: tuple[str, ...]
    if isinstance(declared, str):
        accepted_types = (declared,)
    elif isinstance(declared, list) and all(isinstance(item, str) for item in declared):
        accepted_types = tuple(item for item in declared if isinstance(item, str))
    else:
        raise AssertionError("generated Tool parameter has an invalid type declaration")

    if value is None:
        return "null" in accepted_types, None
    if "string" in accepted_types and isinstance(value, str):
        return True, value
    if "boolean" in accepted_types:
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return True, value.lower() == "true"
    if "integer" in accepted_types:
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            return True, value
        if isinstance(value, str) and _DECIMAL_INTEGER.fullmatch(value):
            return True, int(value)
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return True, int(value)
    return False, None


class _NonStringToolResult(Exception):
    pass
