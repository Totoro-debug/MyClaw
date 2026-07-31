"""Persisted Tool artifact references and filename encoding."""

from dataclasses import dataclass
from typing import Final
from urllib.parse import quote, unquote

from myclaw.session.identifiers import require_session_id
from myclaw.utils.validation import require_nonnegative_int

_WINDOWS_RESERVED_BASENAMES: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def encode_artifact_tool_call_id(tool_call_id: str) -> str:
    """Return the canonical Windows filename component for a Tool call ID."""
    basename = tool_call_id.split(".", maxsplit=1)[0].upper()
    if basename in _WINDOWS_RESERVED_BASENAMES:
        return "".join(f"%{byte:02X}" for byte in tool_call_id.encode("utf-8"))
    return quote(tool_call_id, safe="-_.", encoding="utf-8", errors="strict")


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
