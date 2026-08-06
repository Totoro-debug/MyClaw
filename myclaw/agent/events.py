"""Typed Agent Events emitted through the Conversation Port."""

from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from myclaw.errors import ErrorInfo
from myclaw.provider.models import ModelUsage
from myclaw.session.session import Session
from myclaw.tools.models import ToolResultStatus
from myclaw.utils.json_types import JsonObject
from myclaw.utils.validation import require_aware_datetime, require_nonnegative_int, require_uuid4

type ConfirmationDecision = Literal["approved", "declined"]

type AgentEventType = Literal[
    "turn_started",
    "text_delta",
    "tool_started",
    "confirmation_requested",
    "tool_completed",
    "turn_completed",
    "turn_failed",
    "turn_cancelled",
    "background_completed",
]


def _require_summary(value: str, *, field: str) -> None:
    if len(value) > 240:
        msg = f"{field} must not exceed 240 characters"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TurnStartedPayload:
    pass


@dataclass(frozen=True, slots=True)
class TextDeltaPayload:
    delta: str

    def __post_init__(self) -> None:
        if not self.delta:
            msg = "delta must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ToolStartedPayload:
    tool_call_id: str
    tool_name: str
    summary: str

    def __post_init__(self) -> None:
        _require_summary(self.summary, field="summary")


@dataclass(frozen=True, slots=True, init=False)
class ConfirmationRequestedPayload:
    confirmation_id: UUID
    turn_id: UUID
    tool_call_id: str
    tool_name: str
    summary: str
    _details: JsonObject = field(repr=False)
    warnings: tuple[str, ...] = ()

    def __init__(
        self,
        *,
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
        _require_summary(summary, field="summary")
        if not isinstance(details, dict):
            raise TypeError("details must be a JSON object")
        if not isinstance(warnings, (tuple, list)) or any(
            not isinstance(item, str) for item in warnings
        ):
            raise TypeError("warnings must be a sequence of strings")
        object.__setattr__(self, "confirmation_id", confirmation_id)
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "_details", deepcopy(details))
        object.__setattr__(self, "warnings", tuple(warnings))

    @property
    def details(self) -> JsonObject:
        """Return a detached view of the normalized operation details."""
        return deepcopy(self._details)


@dataclass(frozen=True, slots=True)
class ToolCompletedPayload:
    tool_call_id: str
    tool_name: str
    status: ToolResultStatus
    summary: str

    def __post_init__(self) -> None:
        _require_summary(self.summary, field="summary")


@dataclass(frozen=True, slots=True)
class TurnCompletedPayload:
    content: str
    usage: ModelUsage


@dataclass(frozen=True, slots=True)
class TurnFailedPayload:
    error: ErrorInfo


@dataclass(frozen=True, slots=True)
class TurnCancelledPayload:
    partial_content: str


@dataclass(frozen=True, slots=True)
class BackgroundCompletedPayload:
    kind: str
    title: str
    session_id: str
    status: str
    summary: str

    def __post_init__(self) -> None:
        Session._require_id(self.session_id)
        _require_summary(self.summary, field="summary")


type AgentEventPayload = (
    TurnStartedPayload
    | TextDeltaPayload
    | ToolStartedPayload
    | ConfirmationRequestedPayload
    | ToolCompletedPayload
    | TurnCompletedPayload
    | TurnFailedPayload
    | TurnCancelledPayload
    | BackgroundCompletedPayload
)

_EVENT_PAYLOAD_TYPES: dict[AgentEventType, type[object]] = {
    "turn_started": TurnStartedPayload,
    "text_delta": TextDeltaPayload,
    "tool_started": ToolStartedPayload,
    "confirmation_requested": ConfirmationRequestedPayload,
    "tool_completed": ToolCompletedPayload,
    "turn_completed": TurnCompletedPayload,
    "turn_failed": TurnFailedPayload,
    "turn_cancelled": TurnCancelledPayload,
    "background_completed": BackgroundCompletedPayload,
}


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """The common event envelope consumed by user-facing adapters."""

    type: AgentEventType
    event_id: int
    turn_id: UUID
    created_at: datetime
    payload: AgentEventPayload

    def __post_init__(self) -> None:
        require_nonnegative_int(self.event_id, field="event_id")
        require_uuid4(self.turn_id, field="turn_id")
        require_aware_datetime(self.created_at, field="created_at")
        expected_payload = _EVENT_PAYLOAD_TYPES[self.type]
        if not isinstance(self.payload, expected_payload):
            msg = f"payload does not match event type {self.type}"
            raise ValueError(msg)
        if (
            isinstance(self.payload, ConfirmationRequestedPayload)
            and self.payload.turn_id != self.turn_id
        ):
            raise ValueError("confirmation payload turn_id does not match event turn_id")


class ConfirmationResponsePort(Protocol):
    """Respond to one pending Tool Confirmation without creating a user message."""

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None: ...


class ConversationPort(ConfirmationResponsePort, Protocol):
    """Submit user input and emit ordered Agent Events."""

    def submit(self, text: str) -> AsyncIterator[AgentEvent]: ...

    async def cancel_active_turn(self) -> None: ...
