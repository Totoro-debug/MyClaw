"""Standard-library cross-process locking for the Runtime Log."""

from __future__ import annotations

import errno
import os
from typing import Final, Protocol, cast


class RuntimeLogLock(Protocol):
    """Small lock boundary used by the Runtime Log writer."""

    def try_acquire(self, descriptor: int) -> bool: ...

    def release(self, descriptor: int) -> None: ...


class FcntlCapability(Protocol):
    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, descriptor: int, operation: int) -> object: ...


class MsvcrtCapability(Protocol):
    LK_NBLCK: int
    LK_UNLCK: int

    def locking(self, descriptor: int, mode: int, bytes_to_lock: int) -> object: ...


class WindowsRuntimeLogLock:
    """Nonblocking one-byte Runtime Log locking through Windows msvcrt."""

    def __init__(self, capability: MsvcrtCapability) -> None:
        self._capability = capability

    def try_acquire(self, descriptor: int) -> bool:
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            self._capability.locking(descriptor, self._capability.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    def release(self, descriptor: int) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        self._capability.locking(descriptor, self._capability.LK_UNLCK, 1)


class PosixRuntimeLogLock:
    """Nonblocking exclusive Runtime Log locking through POSIX fcntl."""

    def __init__(self, capability: FcntlCapability) -> None:
        self._capability = capability

    def try_acquire(self, descriptor: int) -> bool:
        try:
            self._capability.flock(
                descriptor,
                self._capability.LOCK_EX | self._capability.LOCK_NB,
            )
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True

    def release(self, descriptor: int) -> None:
        self._capability.flock(descriptor, self._capability.LOCK_UN)


def _platform_runtime_log_lock() -> RuntimeLogLock:
    if os.name == "nt":
        import msvcrt

        return WindowsRuntimeLogLock(cast(MsvcrtCapability, msvcrt))

    import fcntl

    return PosixRuntimeLogLock(cast(FcntlCapability, fcntl))


PLATFORM_RUNTIME_LOG_LOCK: Final[RuntimeLogLock] = _platform_runtime_log_lock()
