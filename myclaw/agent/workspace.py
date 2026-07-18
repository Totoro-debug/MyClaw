"""Workspace lexical identity and Agent Home slug."""

from __future__ import annotations

import ntpath
import os
import posixpath
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Self


@dataclass(frozen=True, slots=True)
class Workspace:
    """A normalized absolute Workspace identity and its stable slug."""

    path: PurePath
    slug: str

    @classmethod
    def from_path(cls, path: PurePath) -> Self:
        """Build a Workspace without resolving symlinks or other filesystem aliases."""
        if isinstance(path, Path):
            path = Path(os.path.abspath(path))

        if isinstance(path, PureWindowsPath):
            windows_path = (
                path if isinstance(path, Path) else PureWindowsPath(ntpath.normpath(str(path)))
            )
            normalized: PurePath = windows_path
            if windows_path.drive.startswith("\\\\"):
                unc_drive = windows_path.drive.lstrip("\\").split("\\")
                segments = ["unc", *unc_drive, *windows_path.parts[1:]]
            else:
                segments = [windows_path.drive.removesuffix(":"), *windows_path.parts[1:]]
        else:
            normalized = (
                path if isinstance(path, Path) else PurePosixPath(posixpath.normpath(str(path)))
            )
            segments = list(normalized.parts[1:])

        if not normalized.is_absolute():
            raise ValueError("Workspace path must be absolute")

        slug = "-".join(segment.replace("-", "_").lower() for segment in segments)
        return cls(path=normalized, slug=slug)
