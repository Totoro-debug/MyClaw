"""Minimal Windows Job Object ownership for Shell process trees."""

import asyncio
import ctypes
from ctypes import wintypes
from typing import Final

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Final = 0x00002000
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS: Final = 1
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: Final = 9
_PROCESS_TERMINATE: Final = 0x0001
_PROCESS_SET_QUOTA: Final = 0x0100


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
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _windows_last_error() -> int:
    return ctypes.get_last_error()


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
