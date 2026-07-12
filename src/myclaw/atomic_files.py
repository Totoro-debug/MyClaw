"""Crash-safe whole-file replacement for MyClaw state."""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path
from typing import Final

_UNSUPPORTED_FSYNC_ERRNOS: Final = frozenset(
    {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
)


def atomic_replace_text(target: Path, content: str) -> None:
    """Replace *target* atomically with exact UTF-8 text content."""
    atomic_replace_bytes(target, content.encode("utf-8"))


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
