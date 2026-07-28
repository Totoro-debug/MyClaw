"""Tool Gateway for registered capabilities and normalized results."""

import asyncio
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker

from myclaw.runtime_log import log_sanitized_exception
from myclaw.tools.base import BaseTool
from myclaw.tools.errors import ToolError
from myclaw.tools.models import ModelToolCall, ToolResult
from myclaw.tools.schema import OpenAIToolSchema
from myclaw.utils.json_types import JsonObject, JsonScalar, JsonValue

type Sleep = Callable[[float], Awaitable[None]]

_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")
logger = logging.getLogger(__name__)


class ToolGateway:
    """Resolve and execute one registered Tool Catalog."""

    def __init__(
        self,
        *,
        sleep: Sleep = asyncio.sleep,
        owns_terminal_failures: bool = True,
        on_terminal_failure: Callable[[Exception], None] | None = None,
    ) -> None:
        self._registered = False
        self._tools: dict[str, BaseTool] = {}
        self._schemas: tuple[OpenAIToolSchema, ...] = ()
        self._parameter_schemas: dict[str, JsonObject] = {}
        self._sleep = sleep
        self._owns_terminal_failures = owns_terminal_failures
        self._on_terminal_failure = on_terminal_failure

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
        self._tools = {tool.name: tool for tool in tools}
        self._schemas = tuple(deepcopy(schema) for schema in schemas)
        self._parameter_schemas = {
            name: deepcopy(schema) for name, schema in parameter_schemas.items()
        }
        self._registered = True

    @property
    def schemas(self) -> tuple[OpenAIToolSchema, ...]:
        """Return a defensive snapshot of the registered OpenAI Tool schemas."""
        return tuple(deepcopy(schema) for schema in self._schemas)

    async def call(self, tool_call: ModelToolCall) -> ToolResult:
        """Parse, prepare, refuse, execute, and normalize one Tool call."""
        raw_arguments = tool_call.arguments
        if not isinstance(raw_arguments, str):
            return _error_result(tool_call, message="Tool arguments could not be parsed.")
        try:
            parsed = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError, ValueError):
            return _error_result(tool_call, message="Tool arguments could not be parsed.")
        if not isinstance(parsed, dict):
            return _error_result(tool_call, message="Tool arguments could not be parsed.")

        tool, normalized, correct = self._prepare(tool_call.name, parsed)
        if tool is None:
            return _error_result(tool_call, message="The requested tool is not available.")
        if not correct:
            return _error_result(
                tool_call,
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
                return _error_result(tool_call, message=message)
            if reason is not None:
                if not isinstance(reason, str):
                    return _error_result(
                        tool_call,
                        message=f"{tool_call.name} could not complete the request.",
                    )
                return _refused_result(tool_call, message=reason)

        execute = cast(Callable[..., Awaitable[object]], type(tool).__dict__["execute"])
        for attempt in range(tool.max_retries + 1):
            try:
                content = await execute(tool, **normalized)
                if not isinstance(content, str):
                    raise _NonStringToolResult
                return _success_result(tool_call, content)
            except Exception as error:
                attempt_number = attempt + 1
                total_attempts = tool.max_retries + 1
                if attempt < tool.max_retries:
                    log_sanitized_exception(
                        logger,
                        logging.WARNING,
                        "Tool execution failed "
                        f"name={tool.name} attempt={attempt_number}/{total_attempts} "
                        f"type={type(error).__name__}",
                        error,
                    )
                    await self._sleep(float(2**attempt))
                    continue
                if self._owns_terminal_failures:
                    log_sanitized_exception(
                        logger,
                        logging.ERROR,
                        "Tool execution failed "
                        f"name={tool.name} attempt={attempt_number}/{total_attempts} "
                        f"type={type(error).__name__}",
                        error,
                    )
                if self._on_terminal_failure is not None:
                    self._on_terminal_failure(error)
                message = (
                    error.message
                    if isinstance(error, ToolError)
                    else f"{tool_call.name} could not complete the request."
                )
                return _error_result(tool_call, message=message)
        raise AssertionError("Tool retry budget exhausted without a terminal result")

    def _prepare(
        self,
        name: str,
        arguments: JsonObject,
    ) -> tuple[BaseTool | None, JsonObject, bool]:
        tool = self._tools.get(name)
        if tool is None:
            return None, {}, False
        schema = self._parameter_schemas[name]
        properties_value = schema.get("properties")
        if not isinstance(properties_value, dict):
            raise AssertionError("cached Tool parameter schema has no properties")
        normalized: JsonObject = {}
        for parameter_name, parameter_schema_value in properties_value.items():
            if not isinstance(parameter_schema_value, dict):
                raise AssertionError("cached Tool parameter declaration is not an object")
            if parameter_name in arguments:
                valid, value = _coerce(arguments[parameter_name], parameter_schema_value)
                if not valid:
                    return tool, normalized, False
                normalized[parameter_name] = value
            elif "default" in parameter_schema_value:
                normalized[parameter_name] = deepcopy(parameter_schema_value["default"])
        correct = Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).is_valid(normalized)
        return tool, normalized, correct


def _error_result(tool_call: ModelToolCall, *, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="error",
        content=message,
        artifact=None,
    )


def _success_result(tool_call: ModelToolCall, content: str) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="success",
        content=content,
        artifact=None,
    )


def _refused_result(tool_call: ModelToolCall, *, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="refused",
        content=message,
        artifact=None,
    )


def _parameter_schema(schema: OpenAIToolSchema) -> JsonObject:
    return schema["function"]["parameters"]


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


