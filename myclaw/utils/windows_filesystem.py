"""Windows file-attribute predicates for persistent filesystem boundaries."""

from errno import EACCES
from os import stat_result
from pathlib import Path
from stat import (
    FILE_ATTRIBUTE_DEVICE,
    FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_REPARSE_POINT,
)

_NON_REGULAR_ATTRIBUTES = (
    FILE_ATTRIBUTE_DEVICE | FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT
)


def is_windows_directory(status: stat_result) -> bool:
    """Return whether *status* is an ordinary, unredirected Windows directory."""
    attributes = status.st_file_attributes
    return bool(attributes & FILE_ATTRIBUTE_DIRECTORY) and not bool(
        attributes & FILE_ATTRIBUTE_REPARSE_POINT
    )


def is_windows_regular_file(status: stat_result) -> bool:
    """Return whether *status* is an ordinary, unredirected Windows file."""
    return not bool(status.st_file_attributes & _NON_REGULAR_ATTRIBUTES)


def require_owned_directory(path: Path, *, within: Path) -> Path:
    """Return the normalized owned directory or reject an unsafe filesystem path."""
    owned_root = _require_owned_root(within)
    status = path.lstat()
    resolved = _resolved_for_comparison(path)
    if not is_windows_directory(status) or not resolved.is_relative_to(owned_root):
        _raise_unsafe(path)
    return resolved


def require_owned_regular_file(path: Path, *, within: Path) -> Path:
    """Return the normalized singly linked owned file or reject an unsafe path."""
    owned_root = _require_owned_root(within)
    status = path.lstat()
    resolved = _resolved_for_comparison(path)
    if (
        not is_windows_regular_file(status)
        or status.st_nlink != 1
        or not resolved.is_relative_to(owned_root)
    ):
        _raise_unsafe(path)
    return resolved


def _require_owned_root(path: Path) -> Path:
    status = path.lstat()
    resolved = _resolved_for_comparison(path)
    if not is_windows_directory(status):
        _raise_unsafe(path)
    return resolved


def _resolved_for_comparison(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    native = str(resolved)
    if native.startswith("\\\\?\\UNC\\"):
        return Path(f"\\\\{native.removeprefix('\\\\?\\UNC\\')}")
    return Path(native.removeprefix("\\\\?\\"))


def _raise_unsafe(path: Path) -> None:
    raise PermissionError(EACCES, "Owned path is unavailable or unsafe", str(path))
