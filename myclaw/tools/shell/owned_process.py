"""Host-specific process-tree ownership behind one asynchronous interface."""

import asyncio
import ctypes
import os
import signal
from collections.abc import Awaitable, Callable, Coroutine
from ctypes import wintypes
from pathlib import Path
from typing import Any, Final, Never, Protocol, cast

_POSIX_SIGKILL = getattr(signal, "SIGKILL", 9)
_WINDOWS_PROCESS_CREATION_FLAGS: Final = 0x08000204


class OwnedProcess(Protocol):
    async def communicate(self) -> tuple[bytes, bytes | None]: ...

    async def terminate(self) -> None: ...

    async def wait(self) -> None: ...


class OwnedProcessSpawner(Protocol):
    async def spawn(self, command: tuple[str, ...], *, cwd: Path) -> OwnedProcess: ...


async def _spawn_owned_process(
    creation_awaitable: Coroutine[Any, Any, asyncio.subprocess.Process],
    *,
    own: Callable[[asyncio.subprocess.Process], Awaitable[OwnedProcess]],
    activate: Callable[[asyncio.subprocess.Process], None] | None = None,
    release_unowned: Callable[[], None] | None = None,
) -> OwnedProcess:
    creation = asyncio.create_task(creation_awaitable)
    try:
        process = await asyncio.shield(creation)
    except BaseException as primary_error:
        try:
            process = await _join_task(creation)
        except BaseException as creation_error:
            if release_unowned is not None:
                release_unowned()
            if creation_error is primary_error:
                raise primary_error from None
            raise primary_error from creation_error
        try:
            owned = await own(process)
        except BaseException as ownership_error:
            raise primary_error from ownership_error
        await _terminate_then_raise(owned, primary_error)
    owned = await own(process)
    if activate is not None:
        try:
            activate(process)
        except BaseException as ownership_error:
            await _terminate_then_raise(owned, ownership_error)
    return owned


def default_owned_process_spawner(*, host_name: str = os.name) -> OwnedProcessSpawner:
    if host_name == "nt":
        return WindowsOwnedProcessSpawner()
    return PosixOwnedProcessSpawner()


def _signal_process_group(process_group: int, sent_signal: int) -> None:
    function_name = "killpg"
    kill_group = cast(Callable[[int, int], None], getattr(os, function_name))
    kill_group(process_group, sent_signal)


class _ProcessTree(Protocol):
    async def terminate(self) -> None: ...


class _WindowsJob(Protocol):
    def assign(self, pid: int) -> None: ...

    def resume(self, pid: int) -> None: ...

    async def terminate(self) -> None: ...

    def close(self) -> None: ...


class _AsyncioOwnedProcess:
    def __init__(self, process: asyncio.subprocess.Process, *, tree: _ProcessTree) -> None:
        self._process = process
        self._tree = tree

    async def communicate(self) -> tuple[bytes, bytes | None]:
        stdout, stderr = await self._process.communicate()
        if self._process.returncode != 0:
            raise RuntimeError("Shell command failed")
        return stdout, stderr

    async def terminate(self) -> None:
        await self._tree.terminate()

    async def wait(self) -> None:
        await self._process.wait()


class _PosixProcessTree:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        termination_timeout: float,
        signal_process_group: Callable[[int, int], None],
    ) -> None:
        self._process = process
        self._termination_timeout = termination_timeout
        self._signal_process_group = signal_process_group
        self._termination: asyncio.Task[None] | None = None

    async def terminate(self) -> None:
        if self._termination is None:
            self._termination = asyncio.create_task(self._terminate())
        await asyncio.shield(self._termination)

    async def _terminate(self) -> None:
        root_wait = asyncio.create_task(self._process.wait())
        try:
            self._signal_group(signal.SIGTERM)
            if await self._wait_for_exit(root_wait):
                return
            self._signal_group(_POSIX_SIGKILL)
            if await self._wait_for_exit(root_wait):
                return
            raise TimeoutError("POSIX process group did not terminate")
        except BaseException as ownership_error:
            try:
                await self._kill_and_wait_root(root_wait)
            except BaseException as root_error:
                raise ownership_error from root_error
            raise
        finally:
            if not root_wait.done():
                root_wait.cancel()
                await asyncio.gather(root_wait, return_exceptions=True)

    async def _kill_and_wait_root(self, root_wait: asyncio.Task[int]) -> None:
        if self._process.returncode is None:
            self._process.kill()
        await asyncio.wait_for(
            asyncio.shield(root_wait),
            timeout=self._termination_timeout,
        )

    async def _wait_for_exit(self, root_wait: asyncio.Task[int]) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._termination_timeout
        while True:
            group_alive = self._group_is_alive()
            if root_wait.done() and not group_alive:
                root_wait.result()
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.01, remaining))

    def _group_is_alive(self) -> bool:
        try:
            self._kill_group(0)
        except ProcessLookupError:
            return False
        return True

    def _signal_group(self, sent_signal: int) -> None:
        try:
            self._kill_group(sent_signal)
        except ProcessLookupError:
            pass

    def _kill_group(self, sent_signal: int) -> None:
        self._signal_process_group(self._process.pid, sent_signal)


class _WindowsProcessTree:
    def __init__(self, process: asyncio.subprocess.Process, *, job: _WindowsJob) -> None:
        self._process = process
        self._job = job
        self._termination: asyncio.Task[None] | None = None

    async def terminate(self) -> None:
        if self._termination is None:
            self._termination = asyncio.create_task(self._terminate())
        await asyncio.shield(self._termination)

    async def _terminate(self) -> None:
        cleanup_error: BaseException | None = None
        try:
            await self._job.terminate()
        except BaseException as error:
            cleanup_error = error
        try:
            await self._process.wait()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error


class WindowsOwnedProcessSpawner:
    async def spawn(self, command: tuple[str, ...], *, cwd: Path) -> OwnedProcess:
        job = WindowsJob.create()
        owned = await _spawn_owned_process(
            asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=_WINDOWS_PROCESS_CREATION_FLAGS,
            ),
            own=lambda process: self._assign(job=job, process=process),
            activate=lambda process: job.resume(process.pid),
            release_unowned=job.close,
        )
        return owned

    @staticmethod
    async def _assign(
        *,
        job: _WindowsJob,
        process: asyncio.subprocess.Process,
    ) -> OwnedProcess:
        try:
            job.assign(process.pid)
        except BaseException as ownership_error:
            job.close()
            cleanup = asyncio.create_task(_kill_and_wait(process))
            try:
                await _join_task(cleanup)
            except BaseException as cleanup_error:
                raise ownership_error from cleanup_error
            raise
        return _AsyncioOwnedProcess(
            process,
            tree=_WindowsProcessTree(process, job=job),
        )


class PosixOwnedProcessSpawner:
    def __init__(
        self,
        *,
        termination_timeout: float = 5.0,
        signal_process_group: Callable[[int, int], None] = _signal_process_group,
    ) -> None:
        self._termination_timeout = termination_timeout
        self._signal_process_group = signal_process_group

    async def spawn(self, command: tuple[str, ...], *, cwd: Path) -> OwnedProcess:
        return await _spawn_owned_process(
            asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            ),
            own=self._owned,
        )

    async def _owned(self, process: asyncio.subprocess.Process) -> OwnedProcess:
        return _AsyncioOwnedProcess(
            process,
            tree=_PosixProcessTree(
                process,
                termination_timeout=self._termination_timeout,
                signal_process_group=self._signal_process_group,
            ),
        )


async def _terminate_then_raise(owned: OwnedProcess, primary_error: BaseException) -> Never:
    cleanup = asyncio.create_task(owned.terminate())
    try:
        await _join_task(cleanup)
    except BaseException as cleanup_error:
        raise primary_error from cleanup_error
    raise primary_error


async def _join_task[T](task: asyncio.Task[T]) -> T:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                break
        except BaseException:
            break
    return task.result()


async def _kill_and_wait(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.kill()
    await process.wait()


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Final = 0x00002000
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS: Final = 1
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: Final = 9
_PROCESS_TERMINATE: Final = 0x0001
_PROCESS_SET_QUOTA: Final = 0x0100
_THREAD_SUSPEND_RESUME: Final = 0x0002
_TH32CS_SNAPTHREAD: Final = 0x00000004
_INVALID_HANDLE_VALUE: Final[int] = cast(int, ctypes.c_void_p(-1).value)
_INVALID_DWORD: Final = 0xFFFFFFFF


class _ThreadEntry32(ctypes.Structure):
    _fields_ = (
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    )


class _JobBasicLimitInformation(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = (
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    )


class _JobExtendedLimitInformation(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _JobBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


class _JobBasicAccountingInformation(ctypes.Structure):
    _fields_ = (
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    )


def _windows_kernel32() -> ctypes.CDLL:
    return cast(
        ctypes.CDLL,
        # WinDLL is absent from non-Windows type stubs.
        getattr(ctypes, "WinDLL")("kernel32", use_last_error=True),  # noqa: B009
    )


def _windows_last_error() -> int:
    return cast(int, getattr(ctypes, "get_last_error")())  # noqa: B009


def _windows_error(operation: str) -> OSError:
    code = _windows_last_error()
    return OSError(code, f"{operation} failed with Windows error {code}")


class WindowsJob:
    """Own one process tree until every associated process has exited."""

    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle
        kernel32 = _windows_kernel32()
        self._set_information = kernel32.SetInformationJobObject
        self._set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        self._set_information.restype = wintypes.BOOL
        self._open_process = kernel32.OpenProcess
        self._open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        self._open_process.restype = wintypes.HANDLE
        self._assign_process = kernel32.AssignProcessToJobObject
        self._assign_process.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self._assign_process.restype = wintypes.BOOL
        self._terminate_job = kernel32.TerminateJobObject
        self._terminate_job.argtypes = (wintypes.HANDLE, wintypes.UINT)
        self._terminate_job.restype = wintypes.BOOL
        self._query_information = kernel32.QueryInformationJobObject
        self._query_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        self._query_information.restype = wintypes.BOOL
        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = (wintypes.HANDLE,)
        self._close_handle.restype = wintypes.BOOL

    @classmethod
    def create(cls) -> "WindowsJob":
        kernel32 = _windows_kernel32()
        create_job = kernel32.CreateJobObjectW
        create_job.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        create_job.restype = wintypes.HANDLE
        raw_handle = create_job(None, None)
        if not raw_handle:
            raise _windows_error("CreateJobObjectW")
        job = cls(int(raw_handle))
        limits = _JobExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        handle = job._required_handle()
        if not job._set_information(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = _windows_error("SetInformationJobObject")
            job.close()
            raise error
        return job

    def assign(self, pid: int) -> None:
        process_handle = self._open_process(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
            False,
            pid,
        )
        if not process_handle:
            raise _windows_error("OpenProcess")
        try:
            if not self._assign_process(self._required_handle(), process_handle):
                raise _windows_error("AssignProcessToJobObject")
        finally:
            self._close_handle(process_handle)

    def resume(self, pid: int) -> None:
        kernel32 = _windows_kernel32()
        create_snapshot = kernel32.CreateToolhelp32Snapshot
        create_snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        create_snapshot.restype = wintypes.HANDLE
        thread_first = kernel32.Thread32First
        thread_first.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32))
        thread_first.restype = wintypes.BOOL
        thread_next = kernel32.Thread32Next
        thread_next.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32))
        thread_next.restype = wintypes.BOOL
        open_thread = kernel32.OpenThread
        open_thread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_thread.restype = wintypes.HANDLE
        resume_thread = kernel32.ResumeThread
        resume_thread.argtypes = (wintypes.HANDLE,)
        resume_thread.restype = wintypes.DWORD

        snapshot = create_snapshot(_TH32CS_SNAPTHREAD, 0)
        if int(snapshot) == _INVALID_HANDLE_VALUE:
            raise _windows_error("CreateToolhelp32Snapshot")
        try:
            entry = _ThreadEntry32()
            entry.dwSize = ctypes.sizeof(entry)
            if not thread_first(snapshot, ctypes.byref(entry)):
                raise _windows_error("Thread32First")
            while int(entry.th32OwnerProcessID) != pid:
                if not thread_next(snapshot, ctypes.byref(entry)):
                    raise RuntimeError("Created Windows process has no resumable thread")
            thread_handle = open_thread(
                _THREAD_SUSPEND_RESUME,
                False,
                entry.th32ThreadID,
            )
            if not thread_handle:
                raise _windows_error("OpenThread")
            try:
                if resume_thread(thread_handle) == _INVALID_DWORD:
                    raise _windows_error("ResumeThread")
            finally:
                self._close_handle(thread_handle)
        finally:
            self._close_handle(snapshot)

    async def terminate(self) -> None:
        handle = self._required_handle()
        try:
            if self._active_processes(handle) > 0 and not self._terminate_job(handle, 1):
                raise _windows_error("TerminateJobObject")
            deadline = asyncio.get_running_loop().time() + 5
            while self._active_processes(handle) > 0:
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("Windows Job processes did not terminate")
                await asyncio.sleep(0.01)
        finally:
            self.close()

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        self._close_handle(handle)

    def _active_processes(self, handle: int) -> int:
        accounting = _JobBasicAccountingInformation()
        returned_length = wintypes.DWORD()
        if not self._query_information(
            handle,
            _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            ctypes.byref(returned_length),
        ):
            raise _windows_error("QueryInformationJobObject")
        return int(accounting.ActiveProcesses)

    def _required_handle(self) -> int:
        if self._handle is None:
            raise RuntimeError("Windows Job is closed")
        return self._handle
