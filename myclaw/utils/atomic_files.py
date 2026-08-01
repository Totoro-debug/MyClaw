"""Compatibility facade for host filesystem persistence operations."""

from __future__ import annotations

from os import stat_result
from pathlib import Path

from myclaw.utils.host_filesystem import HOST_FILESYSTEM, FileIdentity


def path_for_io(path: Path) -> Path:
    """Return the host-native path used for filesystem I/O."""
    return HOST_FILESYSTEM.path_for_io(path)


def atomic_replace_text(target: Path, content: str) -> None:
    """Replace *target* atomically with exact UTF-8 text content."""
    HOST_FILESYSTEM.atomic_replace_text(target, content)


def atomic_create_text(target: Path, content: str) -> bool:
    """Atomically create *target* without replacing an existing file."""
    return HOST_FILESYSTEM.atomic_create_text(target, content)


def atomic_create_text_with_identity(target: Path, content: str) -> FileIdentity | None:
    """Atomically create *target* and return its stable ownership identity."""
    return HOST_FILESYSTEM.atomic_create_text_with_identity(target, content)


def atomic_create_bytes(target: Path, content: bytes) -> bool:
    """Atomically create *target* with complete content, or preserve its value."""
    return HOST_FILESYSTEM.atomic_create_bytes(target, content)


def atomic_create_bytes_with_identity(target: Path, content: bytes) -> FileIdentity | None:
    """Atomically create *target* and return its stable ownership identity."""
    return HOST_FILESYSTEM.atomic_create_bytes_with_identity(target, content)


def file_identity(status: stat_result) -> FileIdentity:
    """Return fields stable across the temporary hard-link publication step."""
    return HOST_FILESYSTEM.file_identity(status)


def atomic_replace_bytes(target: Path, content: bytes) -> None:
    """Replace *target* atomically with complete byte content."""
    HOST_FILESYSTEM.atomic_replace_bytes(target, content)
