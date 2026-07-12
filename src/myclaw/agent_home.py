"""Fixed Agent Home paths and startup initialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
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
        agent_home_root = self.path.resolve(strict=True)
        resolved_memory_directory = memory_directory.resolve(strict=True)
        if (
            not resolved_memory_directory.is_relative_to(agent_home_root)
            or not resolved_memory_directory.is_dir()
        ):
            raise PermissionError("Agent Home memory directory must remain inside Agent Home")
        sessions_directory = self.path / "sessions"
        sessions_directory.mkdir(parents=True, exist_ok=True)
        resolved_sessions_directory = sessions_directory.resolve(strict=True)
        if (
            not resolved_sessions_directory.is_relative_to(agent_home_root)
            or not resolved_sessions_directory.is_dir()
        ):
            raise PermissionError("Agent Home sessions directory must remain inside Agent Home")
        long_term_memory = memory_directory / "memory.md"
        if long_term_memory.exists() or long_term_memory.is_symlink():
            resolved_long_term_memory = long_term_memory.resolve(strict=True)
            status = long_term_memory.lstat()
            if (
                not resolved_long_term_memory.is_relative_to(agent_home_root)
                or not S_ISREG(status.st_mode)
                or status.st_nlink != 1
            ):
                raise PermissionError(
                    "Agent Home Long-term Memory must be an unaliased regular file"
                )
        else:
            atomic_replace_text(long_term_memory, _LONG_TERM_MEMORY_TEMPLATE)
