import asyncio
import ctypes
import shutil
import subprocess
import sys
from ctypes import wintypes
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from myclaw.agent.events import AgentEvent
from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
)
from myclaw.tools.models import ModelToolCall
from myclaw.tools.shell.shell_policy import ShellRequest
from myclaw.tools.shell.shell_process import (
    AsyncioShellProcessSpawner,
    ShellProcess,
    SubprocessShellBoundary,
)
from myclaw.tools.shell.shell_tool import ShellTool
from myclaw.tools.shell.windows_job import WindowsJob
from myclaw.tools.tool_gateway import ToolGateway
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript

WINDOWS_NEW_GROUP_NO_WINDOW = 0x08000200


@pytest.mark.asyncio
async def test_windows_process_spawner_executes_argv_directly_and_assigns_the_job(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    commands: list[tuple[str, ...]] = []
    options: dict[str, object] = {}

    class FakeProcess:
        pid = 123
        returncode = 0

    class FakeJob:
        def __init__(self) -> None:
            self.assigned: list[int] = []

        def assign(self, pid: int) -> None:
            self.assigned.append(pid)

        def close(self) -> None:
            return None

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        commands.append(command)
        options.update(kwargs)
        return cast(asyncio.subprocess.Process, FakeProcess())

    job = FakeJob()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(WindowsJob, "create", staticmethod(lambda: job))

    command = ("trusted-git.exe", "-c", "core.fsmonitor=false", "--no-pager", "status")

    await AsyncioShellProcessSpawner().spawn(command, cwd=workspace)

    assert commands == [command]
    assert options["cwd"] == workspace
    assert options["stdin"] is asyncio.subprocess.DEVNULL
    assert options["creationflags"] == WINDOWS_NEW_GROUP_NO_WINDOW
    assert job.assigned == [123]


def _windows_kernel32() -> ctypes.CDLL:
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _windows_last_error() -> int:
    return ctypes.get_last_error()


def _windows_process_exists(pid: int) -> bool:
    synchronize = 0x00100000
    wait_timeout = 0x00000102
    kernel32 = _windows_kernel32()
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait_for_single_object.restype = wintypes.DWORD
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = open_process(synchronize, False, pid)
    if not handle:
        return False
    try:
        return int(wait_for_single_object(handle, 0)) == wait_timeout
    finally:
        close_handle(handle)


def _terminate_windows_process_tree(pid: int) -> None:
    subprocess.run(
        ("taskkill", "/PID", str(pid), "/T", "/F"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _windows_handle_count() -> int:
    kernel32 = _windows_kernel32()
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_handle_count = kernel32.GetProcessHandleCount
    get_process_handle_count.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    get_process_handle_count.restype = wintypes.BOOL
    count = wintypes.DWORD()
    if not get_process_handle_count(get_current_process(), ctypes.byref(count)):
        code = _windows_last_error()
        raise OSError(code, f"GetProcessHandleCount failed with Windows error {code}")
    return int(count.value)


async def _wait_for_path(path: Path) -> None:
    deadline = asyncio.get_running_loop().time() + 5
    while True:
        if path.exists():
            return
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {path}")


async def _wait_for_process_exit(pid: int) -> bool:
    for _ in range(300):
        if not _windows_process_exists(pid):
            return True
        await asyncio.sleep(0.01)
    return False


@pytest.mark.asyncio
async def test_process_liveness_probe_does_not_terminate_a_live_process() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        assert _windows_process_exists(process.pid)
        await asyncio.sleep(0.05)
        assert process.returncode is None
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()


@dataclass(frozen=True, slots=True)
class ProcessStart:
    command: tuple[str, ...]
    cwd: Path


class FakeShellProcess:
    def __init__(self, output: bytes) -> None:
        self._output = output
        self.communicated = False
        self.terminate_calls = 0
        self.wait_calls = 0

    async def communicate(self) -> tuple[bytes, bytes | None]:
        self.communicated = True
        return self._output, None

    async def terminate(self) -> None:
        self.terminate_calls += 1

    async def wait(self) -> None:
        self.wait_calls += 1


class FakeProcessSpawner:
    def __init__(self, process: ShellProcess) -> None:
        self._process = process
        self.starts: list[ProcessStart] = []

    async def spawn(self, command: tuple[str, ...], *, cwd: Path) -> ShellProcess:
        self.starts.append(ProcessStart(command=command, cwd=cwd))
        return self._process


class SequenceProcessSpawner:
    def __init__(self, processes: tuple[ShellProcess, ...]) -> None:
        self._processes = iter(processes)

    async def spawn(self, command: tuple[str, ...], *, cwd: Path) -> ShellProcess:
        del command, cwd
        return next(self._processes)


class DelayedProcessSpawner:
    def __init__(self, process: ShellProcess) -> None:
        self._process = process
        self.created = asyncio.Event()
        self.release = asyncio.Event()

    async def spawn(self, command: tuple[str, ...], *, cwd: Path) -> ShellProcess:
        del command, cwd
        self.created.set()
        await self.release.wait()
        return self._process


class DirectCommandSpawner:
    def __init__(self, command: tuple[str, ...]) -> None:
        self._command = command

    async def spawn(self, command: tuple[str, ...], *, cwd: Path) -> ShellProcess:
        del command
        return await AsyncioShellProcessSpawner().spawn(self._command, cwd=cwd)


class BlockingFakeShellProcess:
    def __init__(self) -> None:
        self._released = asyncio.Event()
        self.communicate_started = asyncio.Event()
        self.terminated = False
        self.terminate_calls = 0
        self.reaped = False

    async def communicate(self) -> tuple[bytes, bytes | None]:
        self.communicate_started.set()
        await self._released.wait()
        self.reaped = True
        return b"partial output", None

    async def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_calls > 1:
            raise AssertionError("A Shell process must be terminated only once")
        self.terminated = True
        self._released.set()

    async def wait(self) -> None:
        await self._released.wait()
        self.reaped = True


class ControlledTerminateProcess:
    def __init__(
        self,
        *,
        release_termination: asyncio.Event | None = None,
        failure: OSError | None = None,
    ) -> None:
        self._released = asyncio.Event()
        self._release_termination = release_termination
        self._failure = failure
        self.communicate_started = asyncio.Event()
        self.terminate_started = asyncio.Event()
        self.reaped = False

    async def communicate(self) -> tuple[bytes, bytes | None]:
        self.communicate_started.set()
        await self._released.wait()
        self.reaped = True
        return b"stopped", None

    async def terminate(self) -> None:
        self.terminate_started.set()
        if self._release_termination is not None:
            await self._release_termination.wait()
        if self._failure is not None:
            raise self._failure
        self._released.set()

    def release(self) -> None:
        self._released.set()

    async def wait(self) -> None:
        await self._released.wait()
        self.reaped = True


class CompletedOutputControlledTerminateProcess:
    def __init__(self, release_termination: asyncio.Event) -> None:
        self._release_termination = release_termination
        self.terminate_started = asyncio.Event()
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes | None]:
        return b"completed output", None

    async def terminate(self) -> None:
        self.terminate_started.set()
        await self._release_termination.wait()

    async def wait(self) -> None:
        self.waited = True


class FailingCommunicateProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.reaped = False

    async def communicate(self) -> tuple[bytes, bytes | None]:
        raise OSError("pipe read failed")

    async def terminate(self) -> None:
        self.terminated = True

    async def wait(self) -> None:
        self.reaped = True


class FaultingTerminateProcess:
    def __init__(self) -> None:
        self._released = asyncio.Event()
        self.communicate_started = asyncio.Event()
        self.communicate_finished = False
        self.communicate_cancelled = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes | None]:
        self.communicate_started.set()
        try:
            await self._released.wait()
        except asyncio.CancelledError:
            self.communicate_cancelled = True
            raise
        self.communicate_finished = True
        return b"cleanup output", None

    async def terminate(self) -> None:
        self._released.set()
        raise OSError("terminate failed")

    async def wait(self) -> None:
        self.waited = True


class TransientCleanupFailureProcess:
    def __init__(self) -> None:
        self._released = asyncio.Event()
        self.communicate_started = asyncio.Event()
        self.terminate_calls = 0
        self.wait_calls = 0
        self.reaped = False

    async def communicate(self) -> tuple[bytes, bytes | None]:
        self.communicate_started.set()
        await self._released.wait()
        return b"stopped", None

    async def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_calls == 1:
            raise OSError("transient terminate failure")
        self._released.set()

    async def wait(self) -> None:
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise OSError("transient wait failure")
        await self._released.wait()
        self.reaped = True


class ImmediateTimeoutWaiter:
    def __init__(self) -> None:
        self.timeouts: list[int] = []

    async def wait(
        self,
        output: asyncio.Task[tuple[bytes, bytes | None]],
        *,
        timeout: int,
    ) -> tuple[bytes, bytes | None]:
        self.timeouts.append(timeout)
        await asyncio.sleep(0)
        raise TimeoutError


class PathTimeoutWaiter:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def wait(
        self,
        output: asyncio.Task[tuple[bytes, bytes | None]],
        *,
        timeout: int,
    ) -> tuple[bytes, bytes | None]:
        del output, timeout
        await _wait_for_path(self._path)
        await asyncio.sleep(0.1)
        raise TimeoutError


@pytest.mark.asyncio
async def test_shell_boundary_runs_in_the_resolved_cwd_and_returns_output(
    workspace: Path,
) -> None:
    process = FakeShellProcess(b"workspace output\n")
    spawner = FakeProcessSpawner(process)
    shell = SubprocessShellBoundary(spawner=spawner)

    output = await shell.execute(
        ShellRequest(command="test-operation", cwd=workspace.resolve(), timeout=60)
    )

    assert output == "workspace output\n"
    assert spawner.starts == [ProcessStart(command=("test-operation",), cwd=workspace.resolve())]
    assert process.communicated is True
    assert process.terminate_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_repeated_successful_commands_release_process_tree_ownership_before_close(
    workspace: Path,
) -> None:
    processes = tuple(FakeShellProcess(f"output {index}\n".encode()) for index in range(12))
    shell = SubprocessShellBoundary(spawner=SequenceProcessSpawner(processes))

    try:
        for index in range(12):
            assert (
                await shell.execute(
                    ShellRequest(command="test-operation", cwd=workspace.resolve(), timeout=60)
                )
                == f"output {index}\n"
            )
        assert [(process.terminate_calls, process.wait_calls) for process in processes] == [
            (1, 1)
        ] * 12

        await shell.close()

        assert [(process.terminate_calls, process.wait_calls) for process in processes] == [
            (1, 1)
        ] * 12
    finally:
        await shell.close()


@pytest.mark.asyncio
async def test_repeated_successful_commands_do_not_accumulate_windows_handles(
    workspace: Path,
) -> None:
    subprocess.run(("git", "init", "-q", str(workspace)), check=True)
    shell = SubprocessShellBoundary()

    try:
        await shell.execute(
            ShellRequest(command="git status --short", cwd=workspace.resolve(), timeout=60)
        )
        await asyncio.sleep(0)
        baseline = _windows_handle_count()

        for _ in range(12):
            assert (
                await shell.execute(
                    ShellRequest(
                        command="git status --short",
                        cwd=workspace.resolve(),
                        timeout=60,
                    )
                )
                == ""
            )
        await asyncio.sleep(0)

        assert _windows_handle_count() <= baseline + 4
    finally:
        await shell.close()


@pytest.mark.parametrize(
    ("requested", "arguments"),
    (
        (
            "git status",
            ("-c", "core.fsmonitor=false", "--no-pager", "status"),
        ),
        (
            "git status --short",
            ("-c", "core.fsmonitor=false", "--no-pager", "status", "--short"),
        ),
        (
            "git diff --stat",
            (
                "-c",
                "core.fsmonitor=false",
                "--no-pager",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--stat",
            ),
        ),
        (
            "git diff --name-only",
            (
                "-c",
                "core.fsmonitor=false",
                "--no-pager",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--name-only",
            ),
        ),
    ),
)
@pytest.mark.asyncio
async def test_allowlisted_git_commands_disable_pager_external_diff_and_textconv(
    requested: str,
    arguments: tuple[str, ...],
    workspace: Path,
) -> None:
    discovered_git = shutil.which("git")
    assert discovered_git is not None
    executed = (str(Path(discovered_git).resolve(strict=True)), *arguments)
    spawner = FakeProcessSpawner(FakeShellProcess(b"git output\n"))
    shell = SubprocessShellBoundary(spawner=spawner)

    output = await shell.execute(
        ShellRequest(command=requested, cwd=workspace.resolve(), timeout=60)
    )

    assert output == "git output\n"
    assert spawner.starts == [ProcessStart(command=executed, cwd=workspace.resolve())]


@pytest.mark.asyncio
async def test_shell_timeout_terminates_and_awaits_the_child_process(
    workspace: Path,
) -> None:
    process = BlockingFakeShellProcess()
    waiter = ImmediateTimeoutWaiter()
    shell = SubprocessShellBoundary(
        spawner=FakeProcessSpawner(process),
        waiter=waiter,
    )

    with pytest.raises(TimeoutError):
        await shell.execute(
            ShellRequest(command="long-running", cwd=workspace.resolve(), timeout=60)
        )

    assert waiter.timeouts == [60]
    assert process.communicate_started.is_set()
    assert process.terminated is True
    assert process.reaped is True


@pytest.mark.asyncio
async def test_shell_cancellation_terminates_and_awaits_the_child_process(
    workspace: Path,
) -> None:
    process = BlockingFakeShellProcess()
    shell = SubprocessShellBoundary(spawner=FakeProcessSpawner(process))
    execution = asyncio.create_task(
        shell.execute(ShellRequest(command="long-running", cwd=workspace.resolve(), timeout=60))
    )
    await process.communicate_started.wait()

    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert process.terminated is True
    assert process.reaped is True


@pytest.mark.asyncio
async def test_shell_cancellation_joins_spawn_before_terminating_the_created_process(
    workspace: Path,
) -> None:
    process = BlockingFakeShellProcess()
    spawner = DelayedProcessSpawner(process)
    shell = SubprocessShellBoundary(spawner=spawner)
    execution = asyncio.create_task(
        shell.execute(ShellRequest(command="starting", cwd=workspace.resolve(), timeout=60))
    )
    await spawner.created.wait()

    execution.cancel()
    execution.cancel()
    await asyncio.sleep(0)
    try:
        assert not execution.done()
    finally:
        spawner.release.set()
        if execution.done():
            await process.terminate()
            await process.wait()
        else:
            with pytest.raises(asyncio.CancelledError):
                await execution
        await shell.close()

    assert process.terminated
    assert process.reaped


@pytest.mark.asyncio
async def test_cancellation_during_successful_finalization_surfaces_after_cleanup(
    workspace: Path,
) -> None:
    release_termination = asyncio.Event()
    process = CompletedOutputControlledTerminateProcess(release_termination)
    shell = SubprocessShellBoundary(spawner=FakeProcessSpawner(process))
    execution = asyncio.create_task(
        shell.execute(ShellRequest(command="completed", cwd=workspace.resolve(), timeout=60))
    )
    await process.terminate_started.wait()

    execution.cancel()
    await asyncio.sleep(0)
    release_termination.set()

    try:
        with pytest.raises(asyncio.CancelledError):
            await execution
    finally:
        release_termination.set()
        await asyncio.gather(execution, return_exceptions=True)
        await shell.close()
    assert process.waited is True


@pytest.mark.asyncio
async def test_shell_io_failure_terminates_and_reaps_before_propagating(
    workspace: Path,
) -> None:
    process = FailingCommunicateProcess()
    shell = SubprocessShellBoundary(spawner=FakeProcessSpawner(process))

    with pytest.raises(OSError, match="pipe read failed"):
        await shell.execute(ShellRequest(command="failing", cwd=workspace.resolve(), timeout=60))
    await shell.close()

    assert process.terminated is True
    assert process.reaped is True


@pytest.mark.asyncio
async def test_shell_close_terminates_and_awaits_an_active_child_process(
    workspace: Path,
) -> None:
    process = BlockingFakeShellProcess()
    shell = SubprocessShellBoundary(spawner=FakeProcessSpawner(process))
    execution = asyncio.create_task(
        shell.execute(ShellRequest(command="long-running", cwd=workspace.resolve(), timeout=60))
    )
    await process.communicate_started.wait()

    await shell.close()

    assert process.terminated is True
    assert process.reaped is True
    assert await execution == "partial output"


@pytest.mark.asyncio
async def test_shell_close_is_idempotent_and_rejects_new_processes(
    workspace: Path,
) -> None:
    spawner = FakeProcessSpawner(FakeShellProcess(b"must not run"))
    shell = SubprocessShellBoundary(spawner=spawner)

    await shell.close()
    await shell.close()

    with pytest.raises(RuntimeError, match="closed"):
        await shell.execute(ShellRequest(command="pwd", cwd=workspace.resolve(), timeout=60))
    assert spawner.starts == []


@pytest.mark.asyncio
async def test_shell_close_awaits_every_process_before_propagating_a_cleanup_failure(
    workspace: Path,
) -> None:
    release_other = asyncio.Event()
    failing = ControlledTerminateProcess(failure=OSError("terminate failed"))
    other = ControlledTerminateProcess(release_termination=release_other)
    shell = SubprocessShellBoundary(
        spawner=SequenceProcessSpawner((failing, other)),
    )
    executions = (
        asyncio.create_task(
            shell.execute(ShellRequest(command="first", cwd=workspace.resolve(), timeout=60))
        ),
        asyncio.create_task(
            shell.execute(ShellRequest(command="second", cwd=workspace.resolve(), timeout=60))
        ),
    )
    await failing.communicate_started.wait()
    await other.communicate_started.wait()
    close = asyncio.create_task(shell.close())
    await failing.terminate_started.wait()
    await other.terminate_started.wait()
    try:
        await asyncio.wait_for(asyncio.shield(close), timeout=0.1)
    except TimeoutError:
        close_returned_before_other_cleanup = False
    except OSError:
        close_returned_before_other_cleanup = True

    failing.release()
    release_other.set()
    with pytest.raises(OSError, match="terminate failed"):
        await close
    await asyncio.gather(*executions, return_exceptions=True)

    assert close_returned_before_other_cleanup is False
    assert other.reaped is True


@pytest.mark.asyncio
async def test_repeated_cancellation_remains_tracked_until_process_cleanup_finishes(
    workspace: Path,
) -> None:
    release_termination = asyncio.Event()
    process = ControlledTerminateProcess(release_termination=release_termination)
    shell = SubprocessShellBoundary(spawner=FakeProcessSpawner(process))
    execution = asyncio.create_task(
        shell.execute(ShellRequest(command="slow-stop", cwd=workspace.resolve(), timeout=60))
    )
    await process.communicate_started.wait()

    execution.cancel()
    await process.terminate_started.wait()
    execution.cancel()
    await asyncio.sleep(0)
    close = asyncio.create_task(shell.close())
    try:
        await asyncio.wait_for(asyncio.shield(close), timeout=0.1)
    except TimeoutError:
        close_returned_before_cleanup = False
    else:
        close_returned_before_cleanup = True

    release_termination.set()
    with pytest.raises(asyncio.CancelledError):
        await execution
    await close

    assert close_returned_before_cleanup is False
    assert process.reaped is True


@pytest.mark.parametrize("primary", ("cancellation", "timeout"))
@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_cancellation_or_timeout(
    primary: str,
    workspace: Path,
) -> None:
    process = FaultingTerminateProcess()
    waiter = ImmediateTimeoutWaiter() if primary == "timeout" else None
    shell = SubprocessShellBoundary(
        spawner=FakeProcessSpawner(process),
        waiter=waiter,
    )

    if primary == "cancellation":
        execution = asyncio.create_task(
            shell.execute(ShellRequest(command="cancelled", cwd=workspace.resolve(), timeout=60))
        )
        await process.communicate_started.wait()
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
    else:
        with pytest.raises(TimeoutError):
            await shell.execute(
                ShellRequest(command="timed-out", cwd=workspace.resolve(), timeout=60)
            )

    assert process.communicate_finished or process.communicate_cancelled
    assert process.waited is True


@pytest.mark.asyncio
async def test_close_retries_transient_failed_cleanup_and_reaps_the_process(
    workspace: Path,
) -> None:
    process = TransientCleanupFailureProcess()
    shell = SubprocessShellBoundary(
        spawner=FakeProcessSpawner(process),
        waiter=ImmediateTimeoutWaiter(),
    )

    with pytest.raises(TimeoutError) as raised:
        await shell.execute(ShellRequest(command="timed-out", cwd=workspace.resolve(), timeout=60))

    assert isinstance(raised.value.__cause__, OSError)
    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert process.reaped is False

    await shell.close()
    await shell.close()

    assert process.terminate_calls == 2
    assert process.wait_calls == 2
    assert process.reaped is True


@pytest.mark.asyncio
async def test_close_retries_cleanup_failed_after_cancellation_during_spawn(
    workspace: Path,
) -> None:
    process = TransientCleanupFailureProcess()
    spawner = DelayedProcessSpawner(process)
    shell = SubprocessShellBoundary(spawner=spawner)
    execution = asyncio.create_task(
        shell.execute(ShellRequest(command="starting", cwd=workspace.resolve(), timeout=60))
    )
    await spawner.created.wait()

    execution.cancel()
    spawner.release.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await execution
    assert isinstance(raised.value.__cause__, OSError)
    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert process.reaped is False

    await shell.close()

    assert process.terminate_calls == 2
    assert process.wait_calls == 2
    assert process.reaped is True


@pytest.mark.asyncio
async def test_production_shell_boundary_executes_a_real_git_operation_directly(
    workspace: Path,
) -> None:
    subprocess.run(("git", "init", "-q", str(workspace)), check=True)
    shell = SubprocessShellBoundary()
    try:
        output = await shell.execute(
            ShellRequest(
                command="git status",
                cwd=workspace.resolve(),
                timeout=60,
            )
        )
    finally:
        await shell.close()

    assert "On branch" in output


@pytest.mark.asyncio
async def test_production_shell_cancellation_leaves_no_grandchild_process(
    workspace: Path,
) -> None:
    pid_path = workspace / "shell-child.pid"
    script = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
        "time.sleep(60)"
    )
    shell = SubprocessShellBoundary(spawner=DirectCommandSpawner((sys.executable, "-c", script)))
    child_pid: int | None = None
    execution = asyncio.create_task(
        shell.execute(ShellRequest(command="test-operation", cwd=workspace.resolve(), timeout=60))
    )
    try:
        await _wait_for_path(pid_path)
        child_pid = int(pid_path.read_text(encoding="utf-8"))

        execution.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=3)
        assert await _wait_for_process_exit(child_pid)
    finally:
        if child_pid is not None and _windows_process_exists(child_pid):
            _terminate_windows_process_tree(child_pid)
            await _wait_for_process_exit(child_pid)
        await shell.close()


@pytest.mark.asyncio
async def test_shell_timeout_terminates_descendants_after_the_root_process_exits(
    workspace: Path,
) -> None:
    pid_path = workspace / "orphaned-shell-child.pid"
    child_script = "import time; time.sleep(10)"
    outer_script = (
        "import pathlib, subprocess, sys, time; "
        "time.sleep(0.1); "
        f"child = subprocess.Popen([sys.executable, '-c', {child_script!r}]); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid), encoding='utf-8')"
    )
    shell = SubprocessShellBoundary(
        spawner=DirectCommandSpawner((sys.executable, "-c", outer_script)),
        waiter=PathTimeoutWaiter(pid_path),
    )
    execution = asyncio.create_task(
        shell.execute(ShellRequest(command="test-operation", cwd=workspace.resolve(), timeout=60))
    )
    child_pid: int | None = None
    try:
        await _wait_for_path(pid_path)
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        try:
            await asyncio.wait_for(asyncio.shield(execution), timeout=3)
        except TimeoutError:
            assert execution.done(), "Shell cleanup did not finish after its root process exited"
        else:
            raise AssertionError("The injected Shell timeout did not propagate")
        with pytest.raises(TimeoutError):
            await execution
        assert await _wait_for_process_exit(child_pid)
    finally:
        if child_pid is not None and _windows_process_exists(child_pid):
            _terminate_windows_process_tree(child_pid)
            await _wait_for_process_exit(child_pid)
        if not execution.done():
            execution.cancel()
        with pytest.raises((asyncio.CancelledError, TimeoutError)):
            await execution
        await shell.close()


@pytest.mark.asyncio
async def test_shell_execute_terminates_descendants_after_normal_root_completion(
    workspace: Path,
) -> None:
    pid_path = workspace / "detached-shell-child.pid"
    child_script = "import time; time.sleep(10)"
    outer_script = (
        "import pathlib, subprocess, sys, time; "
        "time.sleep(0.1); "
        "child = subprocess.Popen("
        f"[sys.executable, '-c', {child_script!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid), encoding='utf-8')"
    )
    shell = SubprocessShellBoundary(
        spawner=DirectCommandSpawner((sys.executable, "-c", outer_script))
    )
    child_pid: int | None = None
    try:
        await shell.execute(
            ShellRequest(
                command="test-operation",
                cwd=workspace.resolve(),
                timeout=60,
            )
        )
        await _wait_for_path(pid_path)
        child_pid = int(pid_path.read_text(encoding="utf-8"))

        assert await _wait_for_process_exit(child_pid)
        await shell.close()
    finally:
        if child_pid is not None and _windows_process_exists(child_pid):
            _terminate_windows_process_tree(child_pid)
            await _wait_for_process_exit(child_pid)


@pytest.mark.asyncio
async def test_tool_gateway_returns_validated_native_pwd_without_starting_a_process(
    agent_home: Path,
    workspace: Path,
) -> None:
    spawner = FakeProcessSpawner(FakeShellProcess(b"process output must not be used"))
    shell = SubprocessShellBoundary(spawner=spawner)
    gateway = ToolGateway()
    gateway.register_tools((ShellTool(workspace=workspace, boundary=shell),))
    try:
        result = await gateway.call(
            ModelToolCall(
                id="call_pwd",
                name="shell",
                arguments='{"command":"pwd","timeout":60}',
            )
        )
    finally:
        await shell.close()

    assert result.status == "success"
    assert result.content == str(workspace.resolve())
    assert spawner.starts == []


@pytest.mark.asyncio
async def test_runtime_close_terminates_and_awaits_its_active_shell_process(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    process = BlockingFakeShellProcess()
    shell = SubprocessShellBoundary(spawner=FakeProcessSpawner(process))
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: ScriptedFakeProvider(),
        now=lambda: datetime(2026, 7, 12, 12, 0, tzinfo=UTC),
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
        shell=shell,
    )
    execution = asyncio.create_task(
        shell.execute(ShellRequest(command="long-running", cwd=workspace.resolve(), timeout=60))
    )
    await process.communicate_started.wait()

    await runtime.close()

    assert process.terminated is True
    assert process.reaped is True
    assert await execution == "partial output"


@pytest.mark.asyncio
async def test_runtime_shutdown_cancels_a_shell_turn_without_double_termination(
    agent_home: Path,
    workspace: Path,
) -> None:
    subprocess.run(("git", "init", "-q", str(workspace)), check=True)
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    process = BlockingFakeShellProcess()
    shell = SubprocessShellBoundary(spawner=FakeProcessSpawner(process))
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="",
                                tool_calls=(
                                    ModelToolCall(
                                        id="call_shell",
                                        name="shell",
                                        arguments='{"command":"git status","timeout":60}',
                                    ),
                                ),
                            ),
                            usage=ModelUsage(
                                input_tokens=4,
                                output_tokens=2,
                                total_tokens=6,
                            ),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: datetime(2026, 7, 12, 12, 0, tzinfo=UTC),
        new_uuid=uuid4,
        shell=shell,
    )

    async def collect_events() -> list[AgentEvent]:
        return [event async for event in runtime.conversation.submit("Run pwd.")]

    turn = asyncio.create_task(collect_events())
    await process.communicate_started.wait()

    await runtime.close()
    events = await turn

    assert process.terminate_calls == 1
    assert process.reaped is True
    assert events[-1].type == "turn_cancelled"
