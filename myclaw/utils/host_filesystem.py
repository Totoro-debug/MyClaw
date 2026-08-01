"""Host filesystem operations for safe local persistence."""

from __future__ import annotations

import errno
import os
import tempfile
from errno import EACCES
from os import stat_result
from pathlib import Path
from stat import (
    FILE_ATTRIBUTE_DEVICE,
    FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_REPARSE_POINT,
)
from typing import Final, Protocol

type FileIdentity = tuple[int, int, int, int]


class UnsafeFilesystemPath(PermissionError):
    """An existing filesystem object violates owned-path safety rules."""


class FilesystemAdapter(Protocol):
    """Native path behavior required by the host filesystem seam."""

    def path_for_io(self, path: Path) -> Path: ...

    def is_directory(self, status: stat_result) -> bool: ...

    def is_regular_file(self, status: stat_result) -> bool: ...

    def resolved_for_comparison(self, path: Path) -> Path: ...

    def sync_parent_directory(self, path: Path) -> None: ...


class WindowsFilesystemAdapter:
    """Windows native path behavior."""

    def path_for_io(self, path: Path) -> Path:
        native = str(path.absolute())
        if native.startswith("\\\\?\\"):
            return path
        if native.startswith("\\\\"):
            return Path(f"\\\\?\\UNC\\{native.lstrip('\\')}")
        return Path(f"\\\\?\\{native}")

    def is_directory(self, status: stat_result) -> bool:
        attributes = status.st_file_attributes
        return bool(attributes & FILE_ATTRIBUTE_DIRECTORY) and not bool(
            attributes & FILE_ATTRIBUTE_REPARSE_POINT
        )

    def is_regular_file(self, status: stat_result) -> bool:
        non_regular = (
            FILE_ATTRIBUTE_DEVICE | FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT
        )
        return not bool(status.st_file_attributes & non_regular)

    def resolved_for_comparison(self, path: Path) -> Path:
        resolved = path.resolve(strict=True)
        native = str(resolved)
        if native.startswith("\\\\?\\UNC\\"):
            return Path(f"\\\\{native.removeprefix('\\\\?\\UNC\\')}")
        return Path(native.removeprefix("\\\\?\\"))

    def sync_parent_directory(self, path: Path) -> None:
        del path


class HostFilesystem:
    """Deep boundary for native persistent filesystem behavior."""

    def __init__(self, adapter: FilesystemAdapter) -> None:
        self._adapter = adapter

    def path_for_io(self, path: Path) -> Path:
        """Return the host-native path used for filesystem I/O."""
        return self._adapter.path_for_io(Path(path))

    def is_directory(self, status: stat_result) -> bool:
        """Return whether a status identifies an ordinary host directory."""
        return self._adapter.is_directory(status)

    def is_regular_file(self, status: stat_result) -> bool:
        """Return whether a status identifies an ordinary host regular file."""
        return self._adapter.is_regular_file(status)

    def require_owned_directory(self, path: Path, *, within: Path) -> Path:
        """Return an ordinary contained directory or reject an unsafe path."""
        owned_root = self._require_owned_root(within)
        status = path.lstat()
        resolved = self._adapter.resolved_for_comparison(path)
        if not self._adapter.is_directory(status) or not resolved.is_relative_to(owned_root):
            _raise_unsafe(path)
        return resolved

    def require_contained_directory(self, path: Path, *, within: Path) -> Path:
        """Validate a directory beneath an already-resolved owned root."""
        status = path.lstat()
        resolved = path.resolve(strict=True)
        if not self._adapter.is_directory(status) or not resolved.is_relative_to(within):
            _raise_unsafe(path)
        return resolved

    def require_owned_regular_file(self, path: Path, *, within: Path) -> Path:
        """Return an ordinary singly linked contained file or reject an unsafe path."""
        owned_root = self._require_owned_root(within)
        status = path.lstat()
        resolved = self._adapter.resolved_for_comparison(path)
        if (
            not self._adapter.is_regular_file(status)
            or status.st_nlink != 1
            or not resolved.is_relative_to(owned_root)
        ):
            _raise_unsafe(path)
        return resolved

    def require_contained_regular_file(self, path: Path, *, within: Path) -> Path:
        """Validate a regular file beneath an already-resolved owned root."""
        status = path.lstat()
        resolved = path.resolve(strict=True)
        if (
            not self._adapter.is_regular_file(status)
            or status.st_nlink != 1
            or not resolved.is_relative_to(within)
        ):
            _raise_unsafe(path)
        return resolved

    def require_opened_owned_regular_file(
        self, descriptor: int, path: Path, *, within: Path
    ) -> Path:
        """Require an open descriptor to match its stable owned regular-file path."""
        owned_root = self._require_owned_root(within)
        opened = os.fstat(descriptor)
        current = path.lstat()
        resolved = self._adapter.resolved_for_comparison(path)
        if (
            not self._adapter.is_regular_file(opened)
            or opened.st_nlink != 1
            or not self._adapter.is_regular_file(current)
            or current.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            or not resolved.is_relative_to(owned_root)
        ):
            _raise_unsafe(path)
        return resolved

    def require_opened_contained_regular_file(
        self, descriptor: int, path: Path, *, within: Path
    ) -> Path:
        """Validate an open file beneath an already-resolved owned root."""
        opened = os.fstat(descriptor)
        current = path.lstat()
        resolved = path.resolve(strict=True)
        if (
            not self._adapter.is_regular_file(opened)
            or opened.st_nlink != 1
            or not self._adapter.is_regular_file(current)
            or current.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            or not resolved.is_relative_to(within)
        ):
            _raise_unsafe(path)
        return resolved

    def atomic_create_text(self, target: Path, content: str) -> bool:
        """Create exact UTF-8 content without replacing an existing target."""
        return self.atomic_create_text_with_identity(target, content) is not None

    def atomic_create_text_with_identity(self, target: Path, content: str) -> FileIdentity | None:
        """Create exact UTF-8 content and return its publication identity."""
        return self.atomic_create_bytes_with_identity(target, content.encode("utf-8"))

    def atomic_create_bytes(self, target: Path, content: bytes) -> bool:
        """Create complete byte content without replacing an existing target."""
        return self.atomic_create_bytes_with_identity(target, content) is not None

    def atomic_create_bytes_with_identity(
        self, target: Path, content: bytes
    ) -> FileIdentity | None:
        """Create complete byte content and return its publication identity."""
        io_target = self.path_for_io(target)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=io_target.parent,
            prefix=f".{io_target.name}.",
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
                identity = self.file_identity(os.fstat(stream.fileno()))
            try:
                os.link(temporary, io_target)
            except FileExistsError:
                return None
            self._adapter.sync_parent_directory(io_target.parent)
            return identity
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def atomic_replace_text(self, target: Path, content: str) -> None:
        """Replace a target atomically with exact UTF-8 content."""
        self.atomic_replace_bytes(target, content.encode("utf-8"))

    def atomic_replace_bytes(self, target: Path, content: bytes) -> None:
        """Replace a target atomically with complete byte content."""
        io_target = self.path_for_io(target)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=io_target.parent,
            prefix=f".{io_target.name}.",
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
            os.replace(temporary, io_target)
            self._adapter.sync_parent_directory(io_target.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def file_identity(status: stat_result) -> FileIdentity:
        """Return identity fields stable across hard-link publication."""
        return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)

    def _require_owned_root(self, path: Path) -> Path:
        status = path.lstat()
        resolved = self._adapter.resolved_for_comparison(path)
        if not self._adapter.is_directory(status):
            _raise_unsafe(path)
        return resolved


def _raise_unsafe(path: Path) -> None:
    raise UnsafeFilesystemPath(EACCES, "Owned path is unavailable or unsafe", str(path))


def _fsync_file(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno != errno.EINVAL:
            raise


WINDOWS_HOST_FILESYSTEM: Final = HostFilesystem(WindowsFilesystemAdapter())
HOST_FILESYSTEM: Final = WINDOWS_HOST_FILESYSTEM
