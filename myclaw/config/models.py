"""Read-only configuration view values."""

from dataclasses import dataclass
from pathlib import Path

from myclaw.errors import ErrorInfo


@dataclass(frozen=True, slots=True)
class ConfigView:
    """A configuration path, redacted content, and optional safe parse error."""

    path: Path
    redacted_content: str
    error: ErrorInfo | None
