"""Conversation Session persisted records."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Literal

from myclaw.errors import ErrorCode
from myclaw.provider.models import ModelUsage
from myclaw.session.identifiers import require_session_id
from myclaw.tools.artifacts import ArtifactReference
from myclaw.tools.models import ModelToolCall, ToolResultStatus
from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.utils.validation import (
    require_aware_datetime,
    require_nonnegative_int,
    require_uuid4_string,
)

type AssistantMessageStatus = Literal["completed", "interrupted", "error"]


def _compact_json_line(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _require_session_title(value: str) -> None:
    if not value:
        msg = "title must not be empty"
        raise ValueError(msg)
    if " ".join(value.split()) != value:
        msg = "title must use normalized whitespace"
        raise ValueError(msg)
    if len(value) > 60:
        msg = "title must not exceed 60 Unicode code points"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CumulativeUsage:
    """Actual model usage accumulated for one Conversation Session."""

    model_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        require_nonnegative_int(self.model_calls, field="model_calls")
        require_nonnegative_int(self.input_tokens, field="input_tokens")
        require_nonnegative_int(self.output_tokens, field="output_tokens")
        require_nonnegative_int(self.total_tokens, field="total_tokens")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            msg = "total_tokens must equal input_tokens + output_tokens"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, int]:
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """The fixed first record in a Conversation Session JSONL file."""

    record_type: ClassVar[Literal["metadata"]] = "metadata"
    schema_version: ClassVar[Literal[1]] = 1

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    consolidation_cursor: int
    cumulative_usage: CumulativeUsage

    def __post_init__(self) -> None:
        require_session_id(self.id, field="id")
        _require_session_title(self.title)
        require_aware_datetime(self.created_at, field="created_at")
        require_aware_datetime(self.updated_at, field="updated_at")
        require_nonnegative_int(self.consolidation_cursor, field="consolidation_cursor")

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "id": self.id,
            "title": self.title,
            "created_at": format_rfc3339_milliseconds(self.created_at),
            "updated_at": format_rfc3339_milliseconds(self.updated_at),
            "consolidation_cursor": self.consolidation_cursor,
            "cumulative_usage": self.cumulative_usage.to_dict(),
        }

    def to_json_line(self) -> str:
        return _compact_json_line(self.to_dict())


@dataclass(frozen=True, slots=True)
class UserSessionMessage:
    """A persisted OpenAI-style user message."""

    record_type: ClassVar[Literal["message"]] = "message"
    role: ClassVar[Literal["user"]] = "user"

    id: str
    created_at: datetime
    content: str

    def __post_init__(self) -> None:
        require_uuid4_string(self.id, field="id")
        require_aware_datetime(self.created_at, field="created_at")
        if not self.content.strip():
            msg = "content must not be empty or whitespace"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "id": self.id,
            "created_at": format_rfc3339_milliseconds(self.created_at),
            "role": self.role,
            "content": self.content,
        }

    def to_json_line(self) -> str:
        return _compact_json_line(self.to_dict())


@dataclass(frozen=True, slots=True)
class SessionError:
    """The intentionally minimal error projection persisted in session history."""

    code: ErrorCode
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class AssistantSessionMessage:
    """A persisted OpenAI-style assistant message."""

    record_type: ClassVar[Literal["message"]] = "message"
    role: ClassVar[Literal["assistant"]] = "assistant"

    id: str
    created_at: datetime
    content: str
    tool_calls: tuple[ModelToolCall, ...]
    status: AssistantMessageStatus
    error: SessionError | None
    usage: ModelUsage

    def __post_init__(self) -> None:
        require_uuid4_string(self.id, field="id")
        require_aware_datetime(self.created_at, field="created_at")
        if self.status == "completed" and self.error is not None:
            msg = "completed assistant must not have an error"
            raise ValueError(msg)
        if self.status != "completed" and self.error is None:
            msg = "non-completed assistant requires an error"
            raise ValueError(msg)
        if self.status != "error" and not self.content and not self.tool_calls:
            msg = "assistant requires content or tool_calls unless status is error"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "id": self.id,
            "created_at": format_rfc3339_milliseconds(self.created_at),
            "role": self.role,
            "content": self.content,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "status": self.status,
            "error": None if self.error is None else self.error.to_dict(),
            "usage": self.usage.to_dict(),
        }

    def to_json_line(self) -> str:
        return _compact_json_line(self.to_dict())


@dataclass(frozen=True, slots=True)
class ToolSessionMessage:
    """A persisted OpenAI-style normalized Tool result."""

    record_type: ClassVar[Literal["message"]] = "message"
    role: ClassVar[Literal["tool"]] = "tool"

    id: str
    created_at: datetime
    tool_call_id: str
    name: str
    content: str
    status: ToolResultStatus
    error: SessionError | None
    artifact: ArtifactReference | None

    def __post_init__(self) -> None:
        require_uuid4_string(self.id, field="id")
        require_aware_datetime(self.created_at, field="created_at")
        if self.status == "success" and self.error is not None:
            msg = "success tool message must not have an error"
            raise ValueError(msg)
        if self.status != "success" and self.error is None:
            msg = "non-success tool message requires an error"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "id": self.id,
            "created_at": format_rfc3339_milliseconds(self.created_at),
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
            "status": self.status,
            "error": None if self.error is None else self.error.to_dict(),
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
        }

    def to_json_line(self) -> str:
        return _compact_json_line(self.to_dict())


type SessionMessage = UserSessionMessage | AssistantSessionMessage | ToolSessionMessage


@dataclass(frozen=True, slots=True)
class ConversationSession:
    """A loaded Conversation Session and its ordered persisted messages."""

    metadata: SessionMetadata
    messages: tuple[SessionMessage, ...]

    def __post_init__(self) -> None:
        if self.metadata.consolidation_cursor > len(self.messages):
            msg = "consolidation_cursor must be a valid message boundary"
            raise ValueError(msg)

    @property
    def short_term_messages(self) -> tuple[SessionMessage, ...]:
        return self.messages[self.metadata.consolidation_cursor :]


@dataclass(frozen=True, slots=True)
class MetadataUpdate:
    """A partial atomic update to the first Session metadata record."""

    title: str | None = None
    updated_at: datetime | None = None
    consolidation_cursor: int | None = None
    cumulative_usage: CumulativeUsage | None = None
    usage_delta: ModelUsage | None = None

    def __post_init__(self) -> None:
        if self.title is not None:
            _require_session_title(self.title)
        if self.updated_at is not None:
            require_aware_datetime(self.updated_at, field="updated_at")
        if self.consolidation_cursor is not None:
            require_nonnegative_int(self.consolidation_cursor, field="consolidation_cursor")
        if self.cumulative_usage is not None and self.usage_delta is not None:
            msg = "cumulative_usage and usage_delta are mutually exclusive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """The only Session fields exposed by the Management Port picker."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

    def __post_init__(self) -> None:
        require_session_id(self.id, field="id")
        _require_session_title(self.title)
        require_aware_datetime(self.created_at, field="created_at")
        require_aware_datetime(self.updated_at, field="updated_at")
        require_nonnegative_int(self.message_count, field="message_count")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": format_rfc3339_milliseconds(self.created_at),
            "updated_at": format_rfc3339_milliseconds(self.updated_at),
            "message_count": self.message_count,
        }
