"""Fixed Agent Home paths and startup initialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

from myclaw.atomic_files import atomic_replace_text

_LONG_TERM_MEMORY_TEMPLATE: Final = """# Long-term Memory

## User Info

## User Preference

## Project Fact

## Lesson
"""


@dataclass(frozen=True, slots=True)
class AgentHome:
    """The fixed production Agent Home or an injected composition-root value."""

    path: Path

    @classmethod
    def production(cls) -> Self:
        """Return the canonical production Agent Home."""
        return cls(Path.home() / ".myclaw")

    def initialize(self) -> None:
        """Create the base Agent Home directories and Long-term Memory."""
        memory_directory = self.path / "memory"
        memory_directory.mkdir(parents=True, exist_ok=True)
        (self.path / "sessions").mkdir(parents=True, exist_ok=True)
        long_term_memory = memory_directory / "memory.md"
        if not long_term_memory.exists():
            atomic_replace_text(long_term_memory, _LONG_TERM_MEMORY_TEMPLATE)
