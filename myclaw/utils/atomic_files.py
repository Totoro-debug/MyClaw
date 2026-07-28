"""Shared path and atomic-write primitives for local persistence."""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path
from typing import Final

type FileIdentity = tuple[int, int, int, int]

_UNSUPPORTED_FSYNC_ERRNOS: Final = frozenset(
    {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
)


def path_for_io(path: Path) -> Path:
    """Return a path suitable for I/O, including Windows extended paths."""
    if os.name != "nt":
        return path
    native = str(path.absolute())
    if native.startswith("\\\\?\\"):
        return path
    if native.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{native.lstrip('\\')}")
    return Path(f"\\\\?\\{native}")


def atomic_replace_text(target: Path, content: str) -> None:
    """Replace *target* atomically with exact UTF-8 text content."""
    atomic_replace_bytes(target, content.encode("utf-8"))


def atomic_create_text(target: Path, content: str) -> bool:
    """Atomically create *target* without replacing an existing file."""
    return atomic_create_text_with_identity(target, content) is not None


def atomic_create_text_with_identity(target: Path, content: str) -> FileIdentity | None:
    """Atomically create *target* and return its stable ownership identity."""
    return atomic_create_bytes_with_identity(target, content.encode("utf-8"))


def atomic_create_bytes(target: Path, content: bytes) -> bool:
    """Atomically create *target* with complete content, or preserve its current value."""
    return atomic_create_bytes_with_identity(target, content) is not None


def atomic_create_bytes_with_identity(target: Path, content: bytes) -> FileIdentity | None:
    """Atomically create *target* and return its stable ownership identity."""
    target = Path(target)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)

    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            written = stream.write(content)
            if written != len(content):
                raise OSError("atomic creation did not write the complete content")
            stream.flush()
            _fsync_file(stream.fileno())
            identity = file_identity(os.fstat(stream.fileno()))

        try:
            os.link(temporary, target)
        except FileExistsError:
            return None
        _fsync_parent_best_effort(target.parent)
        return identity
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def file_identity(status: os.stat_result) -> FileIdentity:
    """Return fields stable across the temporary hard-link publication step."""
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def atomic_replace_bytes(target: Path, content: bytes) -> None:
    """Replace *target* atomically with complete byte content."""
    target = Path(target)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)

    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            written = stream.write(content)
            if written != len(content):
                raise OSError("atomic replacement did not write the complete content")
            stream.flush()
            _fsync_file(stream.fileno())

        os.replace(temporary, target)
        _fsync_parent_best_effort(target.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _fsync_file(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise


def _fsync_parent_best_effort(parent: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(parent, flags)
    except OSError:
        return

    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
