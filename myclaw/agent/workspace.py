"""Workspace lexical identity."""

from __future__ import annotations

import ntpath
from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath
from typing import Self


@dataclass(frozen=True, slots=True)
class Workspace:
    """A normalized absolute Workspace identity."""

    path: PureWindowsPath

    @classmethod
    def from_path(cls, path: PurePath) -> Self:
        """Build a Windows Workspace without resolving filesystem aliases."""
        if isinstance(path, Path):
            if not isinstance(path, PureWindowsPath):
                raise ValueError("Workspace path must use Windows syntax")
            normalized = PureWindowsPath(ntpath.abspath(str(path)))
        elif isinstance(path, PureWindowsPath):
            normalized = PureWindowsPath(ntpath.normpath(str(path)))
        else:
            raise ValueError("Workspace path must use Windows syntax")

        if not normalized.is_absolute():
            raise ValueError("Workspace path must be absolute")

        return cls(path=normalized)
