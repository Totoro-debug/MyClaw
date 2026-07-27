"""Tool boundary values."""

from dataclasses import dataclass
from typing import Literal

from myclaw.tools.artifacts import ArtifactReference

type ToolResultStatus = Literal["success", "error", "refused"]


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

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "status": self.status,
            "content": self.content,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
        }
