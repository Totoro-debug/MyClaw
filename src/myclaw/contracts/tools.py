"""Tool boundary records."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal
from urllib.parse import quote, unquote

from myclaw.contracts.common import require_nonnegative_int, require_session_id
from myclaw.contracts.errors import ErrorInfo
from myclaw.contracts.json_types import JsonObject

type ToolResultStatus = Literal["success", "error", "refused"]
type ToolExecutionLane = Literal["foreground", "scheduled_work", "memory_task"]

_WINDOWS_RESERVED_BASENAMES: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def encode_artifact_tool_call_id(tool_call_id: str) -> str:
    """Return the canonical cross-platform filename component for a Tool call ID."""
    basename = tool_call_id.split(".", maxsplit=1)[0].upper()
    if basename in _WINDOWS_RESERVED_BASENAMES:
        return "".join(f"%{byte:02X}" for byte in tool_call_id.encode("utf-8"))
    return quote(tool_call_id, safe="-_.", encoding="utf-8", errors="strict")


class PermissionDecision(StrEnum):
    """A Permission Policy decision before execution-context conversion."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A provider-neutral declared Tool capability."""

    name: str
    description: str
    input_schema: JsonObject

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """A persisted relative reference to an externalized Tool result."""

    path: str
    total_chars: int
    preview_chars: int

    def __post_init__(self) -> None:
        parts = self.path.split("/")
        if len(parts) != 3 or parts[0] != "artifacts":
            msg = "path must match the persisted artifact path contract"
            raise ValueError(msg)
        require_session_id(parts[1])
        filename = parts[2]
        if not filename.endswith(".txt"):
            msg = "artifact filename must end with .txt"
            raise ValueError(msg)
        encoded_tool_call_id = filename.removesuffix(".txt")
        if not encoded_tool_call_id:
            msg = "artifact filename requires a percent-encoded tool call ID"
            raise ValueError(msg)
        try:
            tool_call_id = unquote(encoded_tool_call_id, encoding="utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            msg = "artifact filename must use valid UTF-8 percent-encoding"
            raise ValueError(msg) from exc
        if encode_artifact_tool_call_id(tool_call_id) != encoded_tool_call_id:
            msg = "artifact filename must use canonical UTF-8 percent-encoding"
            raise ValueError(msg)
        require_nonnegative_int(self.total_chars, field="total_chars")
        require_nonnegative_int(self.preview_chars, field="preview_chars")
        if self.preview_chars > self.total_chars:
            msg = "preview_chars must not exceed total_chars"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "total_chars": self.total_chars,
            "preview_chars": self.preview_chars,
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The normalized result returned by the Tool Gateway."""

    tool_call_id: str
    name: str
    status: ToolResultStatus
    content: str
    error: ErrorInfo | None
    artifact: ArtifactReference | None

    def __post_init__(self) -> None:
        if self.status == "success" and self.error is not None:
            msg = "success result must not have an error"
            raise ValueError(msg)
        if self.status != "success" and self.error is None:
            msg = "non-success result requires an error"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "status": self.status,
            "content": self.content,
            "error": None if self.error is None else self.error.to_dict(),
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Runtime scope supplied to a concrete Tool."""

    lane: ToolExecutionLane
    workspace: Path
    agent_home: Path
    session_id: str

    def __post_init__(self) -> None:
        require_session_id(self.session_id)
