"""The only public Tool invocation seam."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID, uuid4

from jsonschema import Draft202012Validator, FormatChecker
from loguru import logger

from myclaw.tools.base import BaseTool, OpenAIToolSchema, ToolError
from myclaw.utils.json_types import JsonObject, JsonScalar, JsonValue
from myclaw.utils.validation import require_uuid4

if TYPE_CHECKING:
    from myclaw.tools.tool_artifacts import ArtifactReference

type Sleep = Callable[[float], Awaitable[None]]
type ConfirmationDecision = Literal["approved", "declined"]
type ConfirmationOutcome = ConfirmationDecision | None
type ToolResultStatus = Literal["success", "error", "refused"]

_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")


@dataclass(frozen=True, slots=True)
class ConfirmationPrompt:
    """The normalized operation description supplied by a concrete Tool."""

    summary: str
    details: JsonObject
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.summary or len(self.summary) > 240:
            raise ValueError("confirmation summary must contain 1 through 240 characters")
        object.__setattr__(self, "details", deepcopy(self.details))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "details": deepcopy(self.details),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True, init=False)
class ConfirmationRequest:
    """One confirmation request bound to an Agent Run Tool call."""

    confirmation_id: UUID
    turn_id: UUID
    tool_call_id: str
    tool_name: str
    summary: str
    _details: JsonObject = field(repr=False)
    warnings: tuple[str, ...] = ()

    def __init__(
        self,
        confirmation_id: UUID,
        turn_id: UUID,
        tool_call_id: str,
        tool_name: str,
        summary: str,
        details: JsonObject,
        warnings: tuple[str, ...] = (),
    ) -> None:
        require_uuid4(confirmation_id, field="confirmation_id")
        require_uuid4(turn_id, field="turn_id")
        object.__setattr__(self, "confirmation_id", confirmation_id)
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "_details", deepcopy(details))
        object.__setattr__(self, "warnings", tuple(warnings))

    @property
    def details(self) -> JsonObject:
        return deepcopy(self._details)

    def to_dict(self) -> dict[str, object]:
        return {
            "confirmation_id": str(self.confirmation_id),
            "turn_id": str(self.turn_id),
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "summary": self.summary,
            "details": deepcopy(self._details),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ToolConfirmationMetadata:
    """The request and decision carried by a Tool Result."""

    request: ConfirmationRequest
    decision: ConfirmationOutcome

    def to_dict(self) -> dict[str, object]:
        return {"request": self.request.to_dict(), "decision": self.decision}


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """A provider Tool call preserving its raw JSON argument text."""

    id: str
    name: str
    arguments: str

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The normalized result returned by the Tool Gateway."""

    tool_call_id: str
    name: str
    status: ToolResultStatus
    content: str
    artifact: ArtifactReference | None
    confirmation: ToolConfirmationMetadata | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "status": self.status,
            "content": self.content,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
        }
        if self.confirmation is not None:
            result["confirmation"] = self.confirmation.to_dict()
        return result


type ConfirmationRequester = Callable[[ConfirmationRequest], Awaitable[ConfirmationDecision]]
type ConfirmationObserver = Callable[[ConfirmationRequest], None]


class ConfirmationChannel:
    """In-memory interactive channel bound to one Agent Run."""

    def __init__(self, turn_id: UUID) -> None:
        require_uuid4(turn_id, field="turn_id")
        self._turn_id = turn_id
        self._requests: asyncio.Queue[ConfirmationRequest | None] = asyncio.Queue()
        self._pending: dict[UUID, asyncio.Future[ConfirmationDecision]] = {}
        self._consumed: set[UUID] = set()
        self._closed = False

    @property
    def turn_id(self) -> UUID:
        return self._turn_id

    async def __call__(self, request: ConfirmationRequest) -> ConfirmationDecision:
        if self._closed:
            raise RuntimeError("Confirmation channel is closed")
        if request.turn_id != self._turn_id:
            raise ValueError("Confirmation request belongs to another turn")
        if request.confirmation_id in self._pending or request.confirmation_id in self._consumed:
            raise ValueError("Confirmation request is already pending or consumed")

        future: asyncio.Future[ConfirmationDecision] = asyncio.get_running_loop().create_future()
        self._pending[request.confirmation_id] = future
        await self._requests.put(request)
        try:
            try:
                return await future
            except asyncio.CancelledError:
                if future.cancelled() or future.result() != "approved":
                    raise
                return "approved"
        finally:
            self._pending.pop(request.confirmation_id, None)
            self._consumed.add(request.confirmation_id)

    async def next_request(self) -> ConfirmationRequest:
        """Return the next live request for the interactive host."""
        while True:
            if self._closed:
                raise RuntimeError("Confirmation channel is closed")
            request = await self._requests.get()
            if request is None:
                self._requests.put_nowait(None)
                raise RuntimeError("Confirmation channel is closed")
            future = self._pending.get(request.confirmation_id)
            if future is not None and not future.done():
                return request

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        if decision not in {"approved", "declined"}:
            raise ValueError("confirmation decision must be approved or declined")
        future = self._pending.get(confirmation_id)
        if future is None or future.done():
            raise ValueError("Confirmation response is late or unknown")
        future.set_result(decision)

    def close(self) -> None:
        """Invalidate all pending requests without producing a decision."""
        if self._closed:
            return
        self._closed = True
        for future in self._pending.values():
            future.cancel()
        self._requests.put_nowait(None)


class ToolGateway:
    """Resolve and execute one registered Tool Catalog."""

    def __init__(
        self,
        *,
        sleep: Sleep = asyncio.sleep,
        owns_terminal_failures: bool = True,
        on_terminal_failure: Callable[[Exception], None] | None = None,
        confirmation: ConfirmationRequester | None = None,
        turn_id: UUID | None = None,
        new_uuid: Callable[[], UUID] = uuid4,
        on_confirmation_requested: ConfirmationObserver | None = None,
    ) -> None:
        if turn_id is not None:
            require_uuid4(turn_id, field="turn_id")
        self._registered = False
        self._tools: dict[str, BaseTool] = {}
        self._schemas: tuple[OpenAIToolSchema, ...] = ()
        self._parameter_schemas: dict[str, JsonObject] = {}
        self._sleep = sleep
        self._owns_terminal_failures = owns_terminal_failures
        self._on_terminal_failure = on_terminal_failure
        self._confirmation = confirmation
        self._turn_id = turn_id
        self._new_uuid = new_uuid
        self._on_confirmation_requested = on_confirmation_requested

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
        self._schemas = schemas
        self._parameter_schemas = parameter_schemas
        self._registered = True

    @property
    def schemas(self) -> tuple[OpenAIToolSchema, ...]:
        """Return a defensive snapshot of the registered OpenAI Tool schemas."""
        return tuple(deepcopy(schema) for schema in self._schemas)

    def for_run(
        self,
        *,
        confirmation: ConfirmationRequester | None,
        on_confirmation_requested: ConfirmationObserver | None = None,
    ) -> ToolGateway:
        """Bind a detached Gateway view to one Agent Run."""
        bound = copy(self)
        bound._confirmation = confirmation
        channel_turn_id = getattr(confirmation, "turn_id", None)
        if isinstance(channel_turn_id, UUID):
            bound._turn_id = channel_turn_id
        bound._on_confirmation_requested = on_confirmation_requested
        return bound

    async def call(self, tool_call: ModelToolCall) -> ToolResult:
        """Parse, prepare, refuse, execute, and normalize one Tool call."""
        raw_arguments = tool_call.arguments
        if not isinstance(raw_arguments, str):
            return _result(tool_call, "error", "Tool arguments could not be parsed.")
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return _result(tool_call, "error", "Tool arguments could not be parsed.")
        if not isinstance(parsed, dict):
            return _result(tool_call, "error", "Tool arguments could not be parsed.")

        tool, normalized, correct = self._prepare(tool_call.name, parsed)
        if tool is None:
            return _result(tool_call, "error", "The requested tool is not available.")
        if not correct:
            return _result(tool_call, "error", f"Invalid arguments for {tool_call.name}.")

        refusal = getattr(tool, "refusal_reason", None)
        if refusal is not None:
            try:
                reason = cast(Callable[..., str | None], refusal)(**normalized)
            except Exception as error:
                return _result(tool_call, "error", _error_message(tool_call.name, error))
            if reason is not None:
                return _result(tool_call, "refused", reason)

        try:
            prompt = await self._confirmation_prompt(tool, normalized)
        except Exception as error:
            tool.confirmation_finished()
            return _result(tool_call, "error", _error_message(tool_call.name, error))
        if prompt is not None:
            try:
                request = self._confirmation_request(tool_call, prompt)
                if self._confirmation is None:
                    return _result(
                        tool_call,
                        "refused",
                        "Tool confirmation is unavailable.",
                        confirmation=ToolConfirmationMetadata(request=request, decision=None),
                    )
                await self._notify_confirmation_requested(request)
                decision = await self._request_confirmation(request)
                metadata = ToolConfirmationMetadata(request=request, decision=decision)
                if decision == "declined":
                    return _result(
                        tool_call,
                        "refused",
                        "Tool confirmation was declined.",
                        confirmation=metadata,
                    )
                return await self._execute_after_approval(
                    tool_call,
                    tool,
                    normalized,
                    confirmation=metadata,
                )
            finally:
                tool.confirmation_finished()

        return await self._execute(tool_call, tool, normalized, confirmation=None)

    async def _notify_confirmation_requested(self, request: ConfirmationRequest) -> None:
        observer = self._on_confirmation_requested
        if observer is not None:
            observer(request)

    async def _confirmation_prompt(
        self,
        tool: BaseTool,
        normalized: JsonObject,
    ) -> ConfirmationPrompt | None:
        provider = getattr(tool, "confirmation_request", None)
        if provider is None:
            return None
        request = cast(
            Callable[..., Awaitable[ConfirmationPrompt | None]],
            provider,
        )
        return await request(**normalized)

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
        decision = await channel(request)
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
                execute = cast(
                    Callable[..., Awaitable[object]],
                    object.__getattribute__(tool, "execute"),
                )
                content = await execute(**deepcopy(normalized))
                if not isinstance(content, str):
                    raise _NonStringToolResult
                return _result(tool_call, "success", content, confirmation=confirmation)
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
                return _result(
                    tool_call,
                    "error",
                    _error_message(tool_call.name, error),
                    confirmation=confirmation,
                )
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
        properties_value = cast(JsonObject, schema["properties"])
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
        normalized: JsonObject = {}
        for parameter_name, parameter_schema_value in properties_value.items():
            parameter_schema = cast(JsonObject, parameter_schema_value)
            if parameter_name in effective:
                value = effective[parameter_name]
                valid, coerced = _coerce(value, parameter_schema)
                if not valid:
                    return tool, normalized, False
                normalized[parameter_name] = coerced
            elif not custom_preparation and "default" in parameter_schema:
                normalized[parameter_name] = deepcopy(parameter_schema["default"])
        correct = Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).is_valid(normalized)
        return tool, normalized, correct


def _result(
    tool_call: ModelToolCall,
    status: ToolResultStatus,
    content: str,
    confirmation: ToolConfirmationMetadata | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status=status,
        content=content,
        artifact=None,
        confirmation=confirmation,
    )


def _error_message(tool_name: str, error: Exception) -> str:
    if isinstance(error, ToolError):
        return error.message
    return f"{tool_name} could not complete the request."


def _parameter_schema(schema: OpenAIToolSchema) -> JsonObject:
    return schema["function"]["parameters"]


def _coerce(value: JsonValue, schema: JsonObject) -> tuple[bool, JsonScalar]:
    declared = schema.get("type")
    accepted_types = (declared,) if isinstance(declared, str) else tuple(cast(list[str], declared))

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
