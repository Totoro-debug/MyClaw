"""Fixed Agent Home paths and startup initialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True, slots=True)
class AgentHome:
    """The fixed production Agent Home or an injected composition-root value."""

    path: Path

    @classmethod
    def production(cls) -> Self:
        """Return the canonical production Agent Home."""
        return cls(Path.home() / ".myclaw")

    def initialize(self) -> None:
        """Create the global Agent Home root for configuration and Runtime Log state."""
        self.path.mkdir(parents=True, exist_ok=True)
