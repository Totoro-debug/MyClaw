import errno

import pytest

from myclaw.runtime_log_lock import PosixRuntimeLogLock


class FakeFcntl:
    LOCK_EX = 1
    LOCK_NB = 2
    LOCK_UN = 4

    def __init__(self) -> None:
        self.locked: set[int] = set()

    def flock(self, descriptor: int, operation: int) -> None:
        if operation == self.LOCK_EX | self.LOCK_NB:
            self.locked.add(descriptor)
        elif operation == self.LOCK_UN:
            self.locked.remove(descriptor)
        else:
            raise AssertionError(f"unsupported fake lock operation: {operation}")


class FailingFcntl(FakeFcntl):
    def __init__(self, error_number: int) -> None:
        super().__init__()
        self._error_number = error_number

    def flock(self, descriptor: int, operation: int) -> None:
        del descriptor, operation
        raise OSError(self._error_number, "injected POSIX lock failure")


def test_posix_runtime_log_lock_acquires_and_releases_exclusive_ownership() -> None:
    capability = FakeFcntl()
    lock = PosixRuntimeLogLock(capability)

    assert lock.try_acquire(17) is True
    assert capability.locked == {17}

    lock.release(17)

    assert capability.locked == set()


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EAGAIN])
def test_posix_runtime_log_lock_reports_contention(error_number: int) -> None:
    lock = PosixRuntimeLogLock(FailingFcntl(error_number))

    assert lock.try_acquire(17) is False


def test_posix_runtime_log_lock_propagates_unexpected_errors() -> None:
    lock = PosixRuntimeLogLock(FailingFcntl(errno.EIO))

    with pytest.raises(OSError, match="injected POSIX lock failure"):
        lock.try_acquire(17)
