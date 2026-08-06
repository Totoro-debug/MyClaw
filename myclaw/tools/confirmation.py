"""Host-mediated, one-shot Tool Confirmation values and channels."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID

from myclaw.utils.json_types import JsonObject, JsonValue
from myclaw.utils.validation import require_uuid4

type ConfirmationDecision = Literal["approved", "declined"]
type ConfirmationOutcome = ConfirmationDecision | None


@dataclass(frozen=True, slots=True)
class ConfirmationPrompt:
    """The normalized operation description supplied by a concrete Tool."""

    summary: str
    details: JsonObject
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_summary(self.summary)
        object.__setattr__(self, "details", _copy_json_object(self.details, field="details"))
        object.__setattr__(self, "warnings", _copy_warnings(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "details": _copy_json_object(self.details, field="details"),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True, init=False)
class ConfirmationRequest:
    """The immutable confirmation request bound to one Agent Run Tool call."""

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
        _require_nonempty_string(tool_call_id, field="tool_call_id")
        _require_nonempty_string(tool_name, field="tool_name")
        _require_summary(summary)
        object.__setattr__(self, "confirmation_id", confirmation_id)
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "_details", _copy_json_object(details, field="details"))
        object.__setattr__(self, "warnings", _copy_warnings(warnings))

    @property
    def details(self) -> JsonObject:
        """Return a detached view of the confirmed operation details."""
        return _copy_json_object(self._details, field="details")

    def to_dict(self) -> dict[str, object]:
        return {
            "confirmation_id": str(self.confirmation_id),
            "turn_id": str(self.turn_id),
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "summary": self.summary,
            "details": _copy_json_object(self._details, field="details"),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ToolConfirmationMetadata:
    """The request snapshot and one-shot decision carried by a Tool Result."""

    request: ConfirmationRequest
    decision: ConfirmationOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.request, ConfirmationRequest):
            raise TypeError("confirmation metadata request must be a ConfirmationRequest")
        if self.decision not in {None, "approved", "declined"}:
            raise ValueError("confirmation decision must be approved or declined")

    def to_dict(self) -> dict[str, object]:
        return {"request": self.request.to_dict(), "decision": self.decision}


class ToolConfirmationChannel(Protocol):
    """Receive one request and return its one-shot host decision."""

    async def request_confirmation(self, request: ConfirmationRequest) -> ConfirmationDecision: ...


type ConfirmationRequester = Callable[
    [ConfirmationRequest], Awaitable[ConfirmationDecision]
]


class ConfirmationChannel:
    """In-memory interactive channel bound to exactly one canonical turn."""

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

    async def request_confirmation(self, request: ConfirmationRequest) -> ConfirmationDecision:
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

    async def request(self, request: ConfirmationRequest) -> ConfirmationDecision:
        """Alias for adapters that use a concise request method."""
        return await self.request_confirmation(request)

    async def next_request(self) -> ConfirmationRequest:
        """Return the next pending request for an interactive host."""
        while True:
            if self._closed:
                raise RuntimeError("Confirmation channel is closed")
            request = await self._requests.get()
            if request is None:
                self._requests.put_nowait(None)
                raise RuntimeError("Confirmation channel is closed")
            if self._closed:
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


def _require_nonempty_string(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


def _require_summary(value: str) -> None:
    _require_nonempty_string(value, field="summary")
    if len(value) > 240:
        raise ValueError("summary must not exceed 240 characters")


def _copy_warnings(value: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or any(not isinstance(item, str) for item in value):
        raise TypeError("warnings must be a sequence of strings")
    return tuple(value)


def _copy_json_object(value: JsonObject, *, field: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    copied = _copy_json_value(value, field=field)
    if not isinstance(copied, dict):
        raise AssertionError("JSON object copy lost its object shape")
    return copied


def _copy_json_value(value: object, *, field: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain finite JSON numbers")
        return value
    if isinstance(value, list):
        return [_copy_json_value(item, field=field) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{field} must contain string JSON object keys")
        return {
            key: _copy_json_value(item, field=field)
            for key, item in value.items()
        }
    raise TypeError(f"{field} must contain JSON-native values")
