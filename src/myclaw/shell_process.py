"""Production Shell execution behind an injectable operating-system process boundary."""

import asyncio
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from myclaw.shell_policy import ShellPolicyDenied, ShellRequest, trusted_git_executable

if sys.platform == "win32":
    from subprocess import CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW

    _WINDOWS_SHELL_CREATION_FLAGS: Final = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
else:
    _WINDOWS_SHELL_CREATION_FLAGS = 0

_HARDENED_GIT_ARGUMENTS: Final = {
    "git status": ("-c", "core.fsmonitor=false", "--no-pager", "status"),
    "git status --short": (
        "-c",
        "core.fsmonitor=false",
        "--no-pager",
        "status",
        "--short",
    ),
    "git diff --stat": (
        "-c",
        "core.fsmonitor=false",
        "--no-pager",
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--stat",
    ),
    "git diff --name-only": (
        "-c",
        "core.fsmonitor=false",
        "--no-pager",
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-only",
    ),
}
_WINDOWS_SHELL_BOOTSTRAP: Final = (
    "import subprocess, sys; "
    "gate = sys.stdin.buffer.read(1); "
    "raise SystemExit(125 if gate != b'1' else subprocess.call(sys.argv[1], shell=True))"
)


class ShellProcess(Protocol):
    async def communicate(self) -> tuple[bytes, bytes | None]: ...

    async def terminate(self) -> None: ...

    async def wait(self) -> None: ...


class ShellProcessSpawner(Protocol):
    async def spawn(self, command: str, *, cwd: Path) -> ShellProcess: ...


class _ProcessTree(Protocol):
    async def terminate(self) -> None: ...


class _AsyncioShellProcess:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        tree: _ProcessTree | None = None,
    ) -> None:
        self._process = process
        self._tree = tree

    async def communicate(self) -> tuple[bytes, bytes | None]:
        stdout, stderr = await self._process.communicate()
        if self._process.returncode != 0:
            raise RuntimeError("Shell command failed")
        return stdout, stderr

    async def wait(self) -> None:
        await self._process.wait()

    async def terminate(self) -> None:
        if self._tree is not None:
            await self._tree.terminate()
            return
        try:
            os.kill(-self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        await asyncio.sleep(0)
        try:
            os.kill(-self._process.pid, 9)
        except ProcessLookupError:
            return
        await self._process.wait()
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            try:
                os.kill(-self._process.pid, 0)
            except ProcessLookupError:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("POSIX Shell process group did not terminate")
            await asyncio.sleep(0.01)


class AsyncioShellProcessSpawner:
    async def spawn(self, command: str, *, cwd: Path) -> ShellProcess:
        if os.name == "nt":
            return await self._spawn_windows(command, cwd=cwd)
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        return _AsyncioShellProcess(process)

    @staticmethod
    async def _spawn_windows(command: str, *, cwd: Path) -> ShellProcess:
        from myclaw.windows_job import WindowsJob

        platform_command = "cd" if command == "pwd" else command
        job = WindowsJob.create()
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-S",
                "-c",
                _WINDOWS_SHELL_BOOTSTRAP,
                platform_command,
                cwd=cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=_WINDOWS_SHELL_CREATION_FLAGS,
            )
            job.assign(process.pid)
            if process.stdin is None:
                raise RuntimeError("Windows Shell bootstrap has no stdin gate")
            process.stdin.write(b"1")
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()
        except BaseException:
            job.close()
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                if process.returncode is None:
                    process.kill()
                await asyncio.shield(process.wait())
            raise
        return _AsyncioShellProcess(process, tree=job)


class ShellProcessWaiter(Protocol):
    async def wait(
        self,
        output: asyncio.Task[tuple[bytes, bytes | None]],
        *,
        timeout: int,
    ) -> tuple[bytes, bytes | None]: ...


class AsyncioShellProcessWaiter:
    async def wait(
        self,
        output: asyncio.Task[tuple[bytes, bytes | None]],
        *,
        timeout: int,
    ) -> tuple[bytes, bytes | None]:
        return await asyncio.wait_for(asyncio.shield(output), timeout=timeout)


@dataclass(slots=True)
class _ActiveProcess:
    process: ShellProcess
    output: asyncio.Task[tuple[bytes, bytes | None]]
    stop: asyncio.Task[None] | None = None


class SubprocessShellBoundary:
    """Run validated Shell requests through an injected process spawner."""

    def __init__(
        self,
        *,
        spawner: ShellProcessSpawner | None = None,
        waiter: ShellProcessWaiter | None = None,
    ) -> None:
        self._spawner = AsyncioShellProcessSpawner() if spawner is None else spawner
        self._waiter = AsyncioShellProcessWaiter() if waiter is None else waiter
        self._active: dict[int, _ActiveProcess] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def execute(self, request: ShellRequest) -> str:
        command = _resolved_shell_command(request)
        async with self._lock:
            if self._closed:
                raise RuntimeError("Shell process boundary is closed")
            spawn = asyncio.create_task(self._spawner.spawn(command, cwd=request.cwd))
            try:
                process = await asyncio.shield(spawn)
            except BaseException as primary_error:
                try:
                    process = await _join_spawn(spawn)
                except BaseException as cleanup_error:
                    if cleanup_error is primary_error:
                        raise primary_error from None
                    raise primary_error from cleanup_error
                output = asyncio.create_task(process.communicate())
                active = _ActiveProcess(process=process, output=output)
                active.stop = asyncio.create_task(self._stop(active))
                stop = active.stop
                self._active[id(process)] = active
                try:
                    await _await_stop(stop)
                except BaseException as cleanup_error:
                    if _stop_succeeded(stop):
                        self._active.pop(id(process), None)
                    else:
                        active.stop = None
                    raise primary_error from cleanup_error
                self._active.pop(id(process), None)
                raise primary_error
            output = asyncio.create_task(process.communicate())
            active = _ActiveProcess(process=process, output=output)
            self._active[id(process)] = active
        try:
            stdout, _ = await self._waiter.wait(output, timeout=request.timeout)
        except BaseException as primary_error:
            try:
                await self._terminate_and_wait(active)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        else:
            await self._terminate_and_wait(active)
        finally:
            async with self._lock:
                if _stop_succeeded(active.stop):
                    self._active.pop(id(process), None)
        return stdout.decode("utf-8", errors="replace")

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            active = tuple(self._active.values())
        results = await asyncio.gather(
            *(self._terminate_and_wait(item) for item in active),
            return_exceptions=True,
        )
        async with self._lock:
            for item in active:
                if _stop_succeeded(item.stop):
                    self._active.pop(id(item.process), None)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _terminate_and_wait(self, active: _ActiveProcess) -> None:
        async with self._lock:
            if active.stop is None:
                active.stop = asyncio.create_task(self._stop(active))
            stop = active.stop
        try:
            await _await_stop(stop)
        except BaseException:
            async with self._lock:
                if active.stop is stop and not _stop_succeeded(stop):
                    active.stop = None
            raise

    @staticmethod
    async def _stop(active: _ActiveProcess) -> None:
        cleanup_error: BaseException | None = None
        try:
            await active.process.terminate()
        except BaseException as error:
            cleanup_error = error
        try:
            await active.process.wait()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None and not active.output.done():
            active.output.cancel()
        try:
            await asyncio.shield(active.output)
        except BaseException:
            pass
        if cleanup_error is not None:
            raise cleanup_error


def _resolved_shell_command(request: ShellRequest) -> str:
    arguments = _HARDENED_GIT_ARGUMENTS.get(request.command)
    if arguments is None:
        return request.command
    executable = trusted_git_executable(workspace=request.workspace_root)
    if executable is None:
        raise ShellPolicyDenied("trusted Git executable is unavailable")
    command = (os.fspath(executable), *arguments)
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


async def _join_spawn(task: asyncio.Task[ShellProcess]) -> ShellProcess:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                break
        except BaseException:
            break
    return task.result()


def _stop_succeeded(stop: asyncio.Task[None] | None) -> bool:
    return stop is not None and stop.done() and not stop.cancelled() and stop.exception() is None


async def _await_stop(stop: asyncio.Task[None]) -> None:
    cancellation: asyncio.CancelledError | None = None
    while not stop.done():
        try:
            await asyncio.shield(stop)
        except asyncio.CancelledError as error:
            if stop.cancelled():
                raise
            cancellation = error
        except BaseException:
            break
    try:
        stop.result()
    except BaseException as cleanup_error:
        if cancellation is not None:
            raise cancellation from cleanup_error
        raise
    if cancellation is not None:
        raise cancellation
