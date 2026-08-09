"""The only public Tool invocation seam."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from typing import Literal, cast
from uuid import UUID, uuid4

from jsonschema import Draft202012Validator, FormatChecker
from loguru import logger

from myclaw.tools.base import ArtifactReference, BaseTool, OpenAIToolSchema, ToolError
from myclaw.utils.json_types import JsonObject, JsonScalar, JsonValue
from myclaw.utils.validation import require_uuid4

type Sleep = Callable[[float], Awaitable[None]]
type ConfirmationDecision = Literal["approved", "declined"]
type ConfirmationOutcome = ConfirmationDecision | None
type ToolResultStatus = Literal["success", "error", "refused"]

_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_CONFIRMATION_UNSET = object()


@dataclass(frozen=True, slots=True)
class ConfirmationPrompt:
    """The normalized operation description supplied by a concrete Tool."""

    summary: str
    details: JsonObject
    warnings: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.summary or len(self.summary) > 240:
            raise ValueError("confirmation summary must contain 1 through 240 characters")
        if not isinstance(self.reason, str):
            raise TypeError("confirmation reason must be a string")
        if not isinstance(self.details, dict):
            raise TypeError("confirmation details must be a JSON object")
        if not isinstance(self.warnings, (tuple, list)) or any(
            not isinstance(item, str) for item in self.warnings
        ):
            raise TypeError("confirmation warnings must be a sequence of strings")
        object.__setattr__(self, "details", deepcopy(self.details))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "details": deepcopy(self.details),
            "warnings": list(self.warnings),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True, init=False)
class ConfirmationRequest:
    """One immutable confirmation request bound to one normalized Tool call."""

    confirmation_id: UUID
    tool_call_id: str
    tool_name: str
    reason: str
    summary: str
    _details: JsonObject = field(repr=False)
    warnings: tuple[str, ...] = ()
    _turn_id: UUID | None = field(default=None, repr=False, compare=True)

    def __init__(
        self,
        confirmation_id: UUID,
        turn_id: UUID | None = None,
        tool_call_id: str = "",
        tool_name: str = "",
        summary: str = "",
        details: JsonObject | None = None,
        warnings: tuple[str, ...] = (),
        *,
        reason: str = "",
    ) -> None:
        require_uuid4(confirmation_id, field="confirmation_id")
        if turn_id is not None:
            require_uuid4(turn_id, field="turn_id")
        if not isinstance(reason, str):
            raise TypeError("confirmation reason must be a string")
        if not tool_call_id or not isinstance(tool_call_id, str):
            raise TypeError("confirmation tool_call_id must be a non-empty string")
        if not tool_name or not isinstance(tool_name, str):
            raise TypeError("confirmation tool_name must be a non-empty string")
        if not summary or len(summary) > 240:
            raise ValueError("confirmation summary must contain 1 through 240 characters")
        if details is not None and not isinstance(details, dict):
            raise TypeError("confirmation details must be a JSON object")
        if not isinstance(warnings, (tuple, list)) or any(
            not isinstance(item, str) for item in warnings
        ):
            raise TypeError("confirmation warnings must be a sequence of strings")
        object.__setattr__(self, "confirmation_id", confirmation_id)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "_details", deepcopy({} if details is None else details))
        object.__setattr__(self, "warnings", tuple(warnings))
        object.__setattr__(self, "_turn_id", turn_id)

    @property
    def turn_id(self) -> UUID | None:
        """Return the legacy Agent Run binding when one was supplied."""
        return self._turn_id

    @property
    def details(self) -> JsonObject:
        return deepcopy(self._details)

    def to_dict(self) -> dict[str, object]:
        return {
            "confirmation_id": str(self.confirmation_id),
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "reason": self.reason,
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
    artifact: ArtifactReference | None = None
    confirmation: ToolConfirmationMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("Tool result name must be a non-empty string")
        if self.status not in {"success", "error", "refused"}:
            raise ValueError("Tool result status is invalid")
        if not isinstance(self.content, str):
            raise TypeError("Tool result content must be a string")
        if self.artifact is not None and not isinstance(self.artifact, ArtifactReference):
            raise TypeError("Tool result artifact must be an ArtifactReference")
        if self.status != "success" and self.artifact is not None:
            raise ValueError("only successful Tool results may contain an artifact")

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

    def __init__(self, turn_id: UUID | None = None) -> None:
        if turn_id is not None:
            require_uuid4(turn_id, field="turn_id")
        self._turn_id = turn_id
        self._requests: asyncio.Queue[ConfirmationRequest | None] = asyncio.Queue()
        self._pending: dict[UUID, asyncio.Future[ConfirmationDecision]] = {}
        self._consumed: set[UUID] = set()
        self._closed = False

    @property
    def turn_id(self) -> UUID | None:
        return self._turn_id

    async def __call__(self, request: ConfirmationRequest) -> ConfirmationDecision:
        if self._closed:
            raise RuntimeError("Confirmation channel is closed")
        if (
            self._turn_id is not None
            and request.turn_id is not None
            and request.turn_id != self._turn_id
        ):
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
        if not isinstance(decision, str) or decision not in {"approved", "declined"}:
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
        self._pending_confirmations: dict[
            UUID, tuple[ModelToolCall, JsonObject, ConfirmationRequest]
        ] = {}
        self._used_confirmations: set[UUID] = set()

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
        bound._pending_confirmations = {}
        bound._used_confirmations = set()
        return bound

    async def call(
        self,
        tool_call: ModelToolCall,
        *,
        confirmation: object = _CONFIRMATION_UNSET,
    ) -> ToolResult:
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

        tool = self._tools.get(tool_call.name)
        if tool is None:
            return _result(tool_call, "error", "The requested tool is not available.")
        if self._uses_final_pipeline(tool):
            return await self._call_final_pipeline(
                tool_call,
                tool,
                cast(JsonObject, parsed),
                confirmation=confirmation,
            )

        tool, normalized, correct = self._prepare(tool_call.name, parsed)
        if tool is None:
            return _result(tool_call, "error", "The requested tool is not available.")
        if not correct:
            return _result(tool_call, "error", f"Invalid arguments for {tool_call.name}.")

        try:
            reason = self._evaluate_refusal(tool, normalized)
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

    @staticmethod
    def _uses_final_pipeline(tool: BaseTool) -> bool:
        execute = getattr(type(tool), "execute", None)
        return (
            type(tool).validate_arguments is not BaseTool.validate_arguments
            or type(tool).check_safety is not BaseTool.check_safety
            or getattr(type(tool), "__tool_schema__", None) is not None
            or getattr(execute, "__tool_schema__", None) is not None
        )

    async def _call_final_pipeline(
        self,
        tool_call: ModelToolCall,
        tool: BaseTool,
        arguments: JsonObject,
        *,
        confirmation: object,
    ) -> ToolResult:
        try:
            preparation = await tool.prepare(arguments)
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool_call.name))

        normalized = preparation.arguments
        try:
            reason = self._evaluate_refusal(tool, normalized)
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool_call.name))
        if reason is not None:
            return _result(tool_call, "refused", reason)
        if preparation.safety_reason is None:
            return await self._execute_final(
                tool_call,
                tool,
                normalized,
                confirmation=None,
            )
        try:
            return await self._confirm_final(
                tool_call,
                tool,
                normalized,
                reason=preparation.safety_reason,
                confirmation=confirmation,
            )
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool_call.name))

    @staticmethod
    def _evaluate_refusal(tool: BaseTool, normalized: JsonObject) -> str | None:
        refusal = getattr(tool, "refusal_reason", None)
        if refusal is None:
            return None
        reason = cast(Callable[..., object], refusal)(**deepcopy(normalized))
        if reason is not None and not isinstance(reason, str):
            raise TypeError("Tool refusal checks must return a string reason or None")
        return reason

    async def _confirm_final(
        self,
        tool_call: ModelToolCall,
        tool: BaseTool,
        normalized: JsonObject,
        *,
        reason: str,
        confirmation: object,
    ) -> ToolResult:
        source = self._confirmation if confirmation is _CONFIRMATION_UNSET else confirmation
        prompt = await self._final_confirmation_prompt(tool, normalized, reason)
        projected_details = cast(JsonObject, _project_confirmation_details(prompt.details))
        request = self._confirmation_request(
            tool_call,
            prompt,
            reason=reason,
            details=projected_details,
            bind_turn=False,
        )

        provided_request: ConfirmationRequest | None = None
        provided_decision: ConfirmationDecision | None = None
        if isinstance(source, ToolConfirmationMetadata):
            provided_request = source.request
            provided_decision = source.decision
        elif isinstance(source, ConfirmationRequest):
            provided_request = source
        elif isinstance(source, str) and source in {"approved", "declined"}:
            candidates = [
                item
                for item in self._pending_confirmations.values()
                if item[0].id == tool_call.id
                and item[0].name == tool_call.name
                and item[1] == normalized
            ]
            if len(candidates) == 1:
                provided_request = candidates[0][2]
                provided_decision = cast(ConfirmationDecision, source)

        if provided_request is not None:
            if provided_request.confirmation_id in self._used_confirmations:
                return _refused_confirmation_result(
                    tool_call,
                    "Tool confirmation was already consumed.",
                    request=provided_request,
                    decision=provided_decision,
                )
            if not self._confirmation_matches(
                provided_request,
                tool_call,
                normalized,
                reason=reason,
                details=projected_details,
            ):
                return _refused_confirmation_result(
                    tool_call,
                    "Tool confirmation was expired or did not match this invocation.",
                    request=provided_request,
                    decision=provided_decision,
                )
            if provided_decision not in {"approved", "declined"}:
                return _refused_confirmation_result(
                    tool_call,
                    "Tool confirmation is unavailable.",
                    request=provided_request,
                    decision=None,
                )
            self._pending_confirmations.pop(provided_request.confirmation_id, None)
            self._used_confirmations.add(provided_request.confirmation_id)
            metadata = ToolConfirmationMetadata(
                request=provided_request,
                decision=provided_decision,
            )
            if provided_decision == "declined":
                return _result(
                    tool_call,
                    "refused",
                    "Tool confirmation was declined.",
                    confirmation=metadata,
                )
            return await self._execute_final(
                tool_call,
                tool,
                normalized,
                confirmation=metadata,
            )

        if source is None:
            self._pending_confirmations[request.confirmation_id] = (
                tool_call,
                deepcopy(normalized),
                request,
            )
            return _refused_confirmation_result(
                tool_call,
                "Tool confirmation is unavailable.",
                request=request,
                decision=None,
            )

        try:
            await self._notify_confirmation_requested(request)
            decision = await self._request_confirmation_from(source, request)
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            raise
        except Exception:
            return _refused_confirmation_result(
                tool_call,
                "Tool confirmation was expired or invalid.",
                request=request,
                decision=None,
            )
        if not isinstance(decision, str) or decision not in {"approved", "declined"}:
            return _refused_confirmation_result(
                tool_call,
                "Tool confirmation was expired or invalid.",
                request=request,
                decision=None,
            )
        normalized_decision = cast(ConfirmationDecision, decision)
        self._used_confirmations.add(request.confirmation_id)
        metadata = ToolConfirmationMetadata(request=request, decision=normalized_decision)
        if normalized_decision == "declined":
            return _result(
                tool_call,
                "refused",
                "Tool confirmation was declined.",
                confirmation=metadata,
            )
        return await self._execute_final(
            tool_call,
            tool,
            normalized,
            confirmation=metadata,
        )

    async def _final_confirmation_prompt(
        self,
        tool: BaseTool,
        normalized: JsonObject,
        reason: str,
    ) -> ConfirmationPrompt:
        declared = _declared_arguments(tool, normalized)
        provider = getattr(tool, "confirmation_prompt", None)
        if provider is None:
            provider = getattr(tool, "confirmation_request", None)
        prompt: object | None = None
        if provider is not None:
            prompt = cast(Callable[..., object], provider)(**declared)
            if inspect.isawaitable(prompt):
                prompt = await cast(Awaitable[object], prompt)
            if prompt is not None and not isinstance(prompt, ConfirmationPrompt):
                raise TypeError("Tool confirmation hook must return a ConfirmationPrompt or None")
        if prompt is None:
            summary = tool.confirmation_summary or f"Confirm {tool.name}"
            return ConfirmationPrompt(
                summary=summary[:240],
                details=deepcopy(normalized),
                reason=reason,
            )
        effective = cast(ConfirmationPrompt, prompt)
        if effective.reason:
            return effective
        return replace(effective, reason=reason)

    async def _request_confirmation_from(
        self,
        source: object,
        request: ConfirmationRequest,
    ) -> object:
        if callable(source):
            return await cast(ConfirmationRequester, source)(request)
        requester = getattr(source, "request_confirmation", None)
        if requester is None:
            requester = getattr(source, "request", None)
        if requester is None:
            raise TypeError("confirmation channel cannot receive a request")
        result = cast(Callable[[ConfirmationRequest], object], requester)(request)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    def _confirmation_matches(
        self,
        request: ConfirmationRequest,
        tool_call: ModelToolCall,
        normalized: JsonObject,
        *,
        reason: str,
        details: JsonObject,
    ) -> bool:
        pending = self._pending_confirmations.get(request.confirmation_id)
        if pending is None:
            return False
        expected_call, expected_arguments, expected_request = pending
        return (
            expected_call == tool_call
            and expected_arguments == normalized
            and expected_request == request
            and request.tool_call_id == tool_call.id
            and request.tool_name == tool_call.name
            and request.reason == reason
            and request.details == details
            and request.confirmation_id not in self._used_confirmations
        )

    async def _execute_final(
        self,
        tool_call: ModelToolCall,
        tool: BaseTool,
        normalized: JsonObject,
        *,
        confirmation: ToolConfirmationMetadata | None,
    ) -> ToolResult:
        try:
            content = await cast(
                Callable[..., Awaitable[object]],
                object.__getattribute__(tool, "execute"),
            )(**deepcopy(_declared_arguments(tool, normalized)))
            if not isinstance(content, str):
                raise TypeError("Tool execution must return a string")
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(
                tool_call,
                "error",
                error.message,
                confirmation=confirmation,
            )
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(
                tool_call,
                "error",
                _generic_tool_failure(tool_call.name),
                confirmation=confirmation,
            )
        return _result(tool_call, "success", content, confirmation=confirmation)

    def _record_unexpected_failure(self, tool: BaseTool, error: Exception) -> None:
        if self._owns_terminal_failures:
            logger.opt(exception=error).error(
                "Tool execution failed name={} type={}",
                tool.name,
                type(error).__name__,
            )
        if self._on_terminal_failure is not None:
            self._on_terminal_failure(error)

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
        *,
        reason: str | None = None,
        details: JsonObject | None = None,
        bind_turn: bool = True,
    ) -> ConfirmationRequest:
        turn_id = self._turn_id if bind_turn else None
        if turn_id is None and bind_turn:
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
            reason=prompt.reason if reason is None else reason,
            summary=prompt.summary,
            details=prompt.details if details is None else details,
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
        if custom_preparation:
            try:
                legacy_prepare = cast(
                    Callable[[JsonObject], object],
                    object.__getattribute__(tool, "prepare"),
                )
                effective_value = legacy_prepare(projected)
            except Exception:
                return tool, {}, False
            if not isinstance(effective_value, dict):
                return tool, {}, False
            effective = cast(JsonObject, effective_value)
        else:
            effective = projected
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


def _declared_arguments(tool: BaseTool, arguments: JsonObject) -> JsonObject:
    return {
        name: deepcopy(arguments[name]) for name in tool.parameters.properties if name in arguments
    }


def _project_confirmation_details(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        if len(value) <= 256:
            return value
        return {
            "value": value[:256],
            "original_length": len(value),
        }
    if isinstance(value, list):
        return [_project_confirmation_details(item) for item in value]
    if isinstance(value, dict):
        return {name: _project_confirmation_details(item) for name, item in value.items()}
    return deepcopy(value)


def _generic_tool_failure(tool_name: str) -> str:
    return f"{tool_name} could not complete the request."


def _refused_confirmation_result(
    tool_call: ModelToolCall,
    content: str,
    *,
    request: ConfirmationRequest,
    decision: ConfirmationOutcome,
) -> ToolResult:
    return _result(
        tool_call,
        "refused",
        content,
        confirmation=ToolConfirmationMetadata(request=request, decision=decision),
    )


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
