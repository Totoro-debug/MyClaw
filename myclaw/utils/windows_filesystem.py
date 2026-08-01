"""Compatibility facade for Windows filesystem predicates."""

from os import stat_result
from pathlib import Path

from myclaw.utils.host_filesystem import WINDOWS_HOST_FILESYSTEM


def is_windows_directory(status: stat_result) -> bool:
    """Return whether *status* is an ordinary, unredirected Windows directory."""
    return WINDOWS_HOST_FILESYSTEM.is_directory(status)


def is_windows_regular_file(status: stat_result) -> bool:
    """Return whether *status* is an ordinary, unredirected Windows file."""
    return WINDOWS_HOST_FILESYSTEM.is_regular_file(status)


def require_owned_directory(path: Path, *, within: Path) -> Path:
    """Return the normalized owned directory or reject an unsafe path."""
    return WINDOWS_HOST_FILESYSTEM.require_owned_directory(path, within=within)


def require_owned_regular_file(path: Path, *, within: Path) -> Path:
    """Return the normalized singly linked owned file or reject an unsafe path."""
    return WINDOWS_HOST_FILESYSTEM.require_owned_regular_file(path, within=within)
