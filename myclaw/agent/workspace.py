"""Workspace lexical identity."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Self


def normalize_workspace_path(path: Path | PurePath) -> Path:
    """Return the normalized absolute Workspace path without resolving aliases."""
    if isinstance(path, Path):
        normalized = Path(os.path.abspath(path))
    else:
        normalized = Path(os.path.normpath(path))

    if not normalized.is_absolute():
        raise ValueError("Workspace path must be absolute")

    return normalized


@dataclass(frozen=True, slots=True)
class Workspace:
    """A normalized absolute Workspace identity."""

    path: Path

    @classmethod
    def from_path(cls, path: PurePath) -> Self:
        """Build a native Workspace without resolving filesystem aliases."""
        return cls(path=normalize_workspace_path(path))
