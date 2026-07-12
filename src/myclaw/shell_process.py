"""Production Shell execution behind an injectable operating-system process boundary."""

import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from myclaw.shell_policy import ShellRequest

_HARDENED_GIT_COMMANDS: Final = {
    "git status": "git --no-pager status",
    "git status --short": "git --no-pager status --short",
    "git diff --stat": "git --no-pager diff --no-ext-diff --no-textconv --stat",
    "git diff --name-only": "git --no-pager diff --no-ext-diff --no-textconv --name-only",
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
        return await self._process.communicate()

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
        command = _HARDENED_GIT_COMMANDS.get(request.command, request.command)
        async with self._lock:
            if self._closed:
                raise RuntimeError("Shell process boundary is closed")
            process = await self._spawner.spawn(command, cwd=request.cwd)
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
                if active.stop is not None and active.stop.done():
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
                if item.stop is not None and item.stop.done():
                    self._active.pop(id(item.process), None)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _terminate_and_wait(self, active: _ActiveProcess) -> None:
        async with self._lock:
            if active.stop is None:
                active.stop = asyncio.create_task(self._stop(active))
            stop = active.stop
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
