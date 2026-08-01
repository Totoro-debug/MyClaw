"""Typed Agent Events emitted through the Conversation Port."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from myclaw.errors import ErrorInfo
from myclaw.provider.models import ModelUsage
from myclaw.session.identifiers import require_session_id
from myclaw.tools.models import ToolResultStatus
from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.utils.validation import require_aware_datetime, require_nonnegative_int, require_uuid4

type AgentEventType = Literal[
    "turn_started",
    "text_delta",
    "progress",
    "tool_started",
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
    def to_dict(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True, slots=True)
class TextDeltaPayload:
    delta: str

    def __post_init__(self) -> None:
        if not self.delta:
            msg = "delta must not be empty"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {"delta": self.delta}


@dataclass(frozen=True, slots=True)
class ProgressPayload:
    status: str
    summary: str

    def __post_init__(self) -> None:
        _require_summary(self.summary, field="summary")

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "summary": self.summary}


@dataclass(frozen=True, slots=True)
class ToolStartedPayload:
    tool_call_id: str
    tool_name: str
    summary: str

    def __post_init__(self) -> None:
        _require_summary(self.summary, field="summary")

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class ToolCompletedPayload:
    tool_call_id: str
    tool_name: str
    status: ToolResultStatus
    summary: str

    def __post_init__(self) -> None:
        _require_summary(self.summary, field="summary")

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class TurnCompletedPayload:
    content: str
    usage: ModelUsage

    def to_dict(self) -> dict[str, object]:
        return {"content": self.content, "usage": self.usage.to_dict()}


@dataclass(frozen=True, slots=True)
class TurnFailedPayload:
    error: ErrorInfo

    def to_dict(self) -> dict[str, object]:
        return {"error": self.error.to_dict()}


@dataclass(frozen=True, slots=True)
class TurnCancelledPayload:
    partial_content: str

    def to_dict(self) -> dict[str, object]:
        return {"partial_content": self.partial_content}


@dataclass(frozen=True, slots=True)
class BackgroundCompletedPayload:
    kind: str
    title: str
    session_id: str
    status: str
    summary: str

    def __post_init__(self) -> None:
        require_session_id(self.session_id)
        _require_summary(self.summary, field="summary")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "title": self.title,
            "session_id": self.session_id,
            "status": self.status,
            "summary": self.summary,
        }


type AgentEventPayload = (
    TurnStartedPayload
    | TextDeltaPayload
    | ProgressPayload
    | ToolStartedPayload
    | ToolCompletedPayload
    | TurnCompletedPayload
    | TurnFailedPayload
    | TurnCancelledPayload
    | BackgroundCompletedPayload
)

_EVENT_PAYLOAD_TYPES: dict[AgentEventType, type[object]] = {
    "turn_started": TurnStartedPayload,
    "text_delta": TextDeltaPayload,
    "progress": ProgressPayload,
    "tool_started": ToolStartedPayload,
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

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "event_id": self.event_id,
            "turn_id": str(self.turn_id),
            "created_at": format_rfc3339_milliseconds(self.created_at),
            "payload": self.payload.to_dict(),
        }


@runtime_checkable
class ConversationPort(Protocol):
    """Submit user input and emit ordered Agent Events."""

    def submit(self, text: str) -> AsyncIterator[AgentEvent]: ...

    async def cancel_active_turn(self) -> None: ...
