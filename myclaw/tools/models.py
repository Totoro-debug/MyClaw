"""Tool boundary values."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from myclaw.errors import ErrorInfo
from myclaw.session.identifiers import require_session_id
from myclaw.tools.artifacts import ArtifactReference
from myclaw.utils.json_types import JsonObject

type ToolResultStatus = Literal["success", "error", "refused"]
type ToolExecutionLane = Literal["foreground", "scheduled_work", "memory_task"]


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
class ModelToolCall:
    """A provider tool call with parsed JSON-object arguments."""

    id: str
    name: str
    arguments: JsonObject

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


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
