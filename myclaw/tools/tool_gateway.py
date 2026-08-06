"""Tool Gateway for registered capabilities and normalized results."""

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import cast
from uuid import UUID, uuid4

from jsonschema import Draft202012Validator, FormatChecker
from loguru import logger

from myclaw.tools.base import BaseTool
from myclaw.tools.confirmation import (
    ConfirmationDecision,
    ConfirmationPrompt,
    ConfirmationRequest,
    ConfirmationRequester,
    ToolConfirmationChannel,
    ToolConfirmationMetadata,
)
from myclaw.tools.errors import ToolError
from myclaw.tools.models import ModelToolCall, ToolResult
from myclaw.tools.schema import OpenAIToolSchema
from myclaw.utils.json_types import JsonObject, JsonScalar, JsonValue
from myclaw.utils.validation import require_uuid4

type Sleep = Callable[[float], Awaitable[None]]
type Confirmation = ToolConfirmationChannel | ConfirmationRequester

_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")


class ToolGateway:
    """Resolve and execute one registered Tool Catalog."""

    def __init__(
        self,
        *,
        sleep: Sleep = asyncio.sleep,
        owns_terminal_failures: bool = True,
        on_terminal_failure: Callable[[Exception], None] | None = None,
        confirmation: Confirmation | None = None,
        confirmation_channel: Confirmation | None = None,
        turn_id: UUID | None = None,
        new_uuid: Callable[[], UUID] = uuid4,
    ) -> None:
        if confirmation is not None and confirmation_channel is not None:
            raise ValueError("Provide only one confirmation channel")
        if turn_id is not None:
            require_uuid4(turn_id, field="turn_id")
        self._registered = False
        self._tools: dict[str, BaseTool] = {}
        self._schemas: tuple[OpenAIToolSchema, ...] = ()
        self._parameter_schemas: dict[str, JsonObject] = {}
        self._sleep = sleep
        self._owns_terminal_failures = owns_terminal_failures
        self._on_terminal_failure = on_terminal_failure
        self._confirmation = confirmation if confirmation is not None else confirmation_channel
        self._turn_id = turn_id
        self._new_uuid = new_uuid

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

        try:
            prompt = self._confirmation_prompt(tool, normalized)
        except Exception as error:
            message = (
                error.message
                if isinstance(error, ToolError)
                else f"{tool_call.name} could not complete the request."
            )
            return _error_result(tool_call, message=message)
        if prompt is not None:
            request = self._confirmation_request(tool_call, prompt)
            if self._confirmation is None:
                return _refused_result(
                    tool_call,
                    message="Tool confirmation is unavailable.",
                    confirmation=ToolConfirmationMetadata(request=request, decision=None),
                )
            decision = await self._request_confirmation(request)
            metadata = ToolConfirmationMetadata(request=request, decision=decision)
            if decision == "declined":
                return _refused_result(
                    tool_call,
                    message="Tool confirmation was declined.",
                    confirmation=metadata,
                )
            return await self._execute_after_approval(
                tool_call,
                tool,
                normalized,
                confirmation=metadata,
            )

        return await self._execute(tool_call, tool, normalized, confirmation=None)

    def _confirmation_prompt(
        self,
        tool: BaseTool,
        normalized: JsonObject,
    ) -> ConfirmationPrompt | None:
        provider = getattr(tool, "confirmation_request", None)
        if provider is None:
            provider = getattr(tool, "confirmation", None)
        if provider is None:
            return None
        prompt = cast(Callable[..., object], provider)(**normalized)
        if prompt is not None and not isinstance(prompt, ConfirmationPrompt):
            raise TypeError("Tool confirmation hook must return a ConfirmationPrompt or None")
        return prompt

    def _confirmation_request(
        self,
        tool_call: ModelToolCall,
        prompt: ConfirmationPrompt,
    ) -> ConfirmationRequest:
        turn_id = self._turn_id
        if turn_id is None:
            channel_turn_id = getattr(self._confirmation, "turn_id", None)
            if isinstance(channel_turn_id, UUID):
                turn_id = channel_turn_id
            else:
                turn_id = self._new_uuid()
            require_uuid4(turn_id, field="turn_id")
            self._turn_id = turn_id
        return ConfirmationRequest(
            confirmation_id=self._new_uuid(),
            turn_id=turn_id,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            summary=prompt.summary,
            details=prompt.details,
            warnings=prompt.warnings,
        )

    async def _request_confirmation(self, request: ConfirmationRequest) -> ConfirmationDecision:
        channel = self._confirmation
        if channel is None:
            raise AssertionError("confirmation channel is required")
        if callable(channel):
            decision = await channel(request)
        else:
            requester = getattr(channel, "request_confirmation", None)
            if requester is None:
                requester = getattr(channel, "request", None)
            if requester is None:
                raise TypeError("confirmation channel cannot receive a request")
            receive = cast(
                Callable[[ConfirmationRequest], Awaitable[ConfirmationDecision]],
                requester,
            )
            decision = await receive(request)
        if decision not in {"approved", "declined"}:
            raise ValueError("confirmation channel returned an invalid decision")
        return decision

    async def _execute_after_approval(
        self,
        tool_call: ModelToolCall,
        tool: BaseTool,
        normalized: JsonObject,
        *,
        confirmation: ToolConfirmationMetadata,
    ) -> ToolResult:
        operation = asyncio.create_task(
            self._execute(tool_call, tool, normalized, confirmation=confirmation)
        )
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            if operation.cancelled():
                raise
            while not operation.done():
                try:
                    await asyncio.shield(operation)
                except asyncio.CancelledError:
                    continue
            if operation.cancelled():
                raise
            return operation.result()

    async def _execute(
        self,
        tool_call: ModelToolCall,
        tool: BaseTool,
        normalized: JsonObject,
        *,
        confirmation: ToolConfirmationMetadata | None,
    ) -> ToolResult:
        for attempt in range(tool.max_retries + 1):
            try:
                execute = cast(Callable[..., Awaitable[object]], type(tool).__dict__["execute"])
                content = await execute(tool, **deepcopy(normalized))
                if not isinstance(content, str):
                    raise _NonStringToolResult
                return _success_result(tool_call, content, confirmation=confirmation)
            except Exception as error:
                attempt_number = attempt + 1
                total_attempts = tool.max_retries + 1
                if attempt < tool.max_retries:
                    logger.opt(exception=error).warning(
                        "Tool execution failed name={} attempt={}/{} type={}",
                        tool.name,
                        attempt_number,
                        total_attempts,
                        type(error).__name__,
                    )
                    await self._sleep(float(2**attempt))
                    continue
                if self._owns_terminal_failures:
                    logger.opt(exception=error).error(
                        "Tool execution failed name={} attempt={}/{} type={}",
                        tool.name,
                        attempt_number,
                        total_attempts,
                        type(error).__name__,
                    )
                if self._on_terminal_failure is not None:
                    self._on_terminal_failure(error)
                message = (
                    error.message
                    if isinstance(error, ToolError)
                    else f"{tool_call.name} could not complete the request."
                )
                return _error_result(tool_call, message=message, confirmation=confirmation)
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
        projected = {
            parameter_name: deepcopy(arguments[parameter_name])
            for parameter_name in properties_value
            if parameter_name in arguments
        }
        custom_preparation = type(tool).prepare is not BaseTool.prepare
        try:
            effective = tool.prepare(projected)
        except Exception:
            return tool, {}, False
        if not isinstance(effective, dict) or not all(
            isinstance(parameter_name, str) for parameter_name in effective
        ):
            return tool, {}, False
        if any(parameter_name not in projected for parameter_name in effective):
            return tool, {}, False
        normalized: JsonObject = {}
        for parameter_name, parameter_schema_value in properties_value.items():
            if not isinstance(parameter_schema_value, dict):
                raise AssertionError("cached Tool parameter declaration is not an object")
            if parameter_name in effective:
                value = effective[parameter_name]
                valid, coerced = _coerce(value, parameter_schema_value)
                if not valid:
                    return tool, normalized, False
                normalized[parameter_name] = coerced
            elif not custom_preparation and "default" in parameter_schema_value:
                normalized[parameter_name] = deepcopy(parameter_schema_value["default"])
        correct = Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).is_valid(normalized)
        return tool, normalized, correct


def _error_result(
    tool_call: ModelToolCall,
    *,
    message: str,
    confirmation: ToolConfirmationMetadata | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="error",
        content=message,
        artifact=None,
        confirmation=confirmation,
    )


def _success_result(
    tool_call: ModelToolCall,
    content: str,
    *,
    confirmation: ToolConfirmationMetadata | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="success",
        content=content,
        artifact=None,
        confirmation=confirmation,
    )


def _refused_result(
    tool_call: ModelToolCall,
    *,
    message: str,
    confirmation: ToolConfirmationMetadata | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="refused",
        content=message,
        artifact=None,
        confirmation=confirmation,
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
