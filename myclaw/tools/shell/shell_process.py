"""Production Shell execution behind an injectable operating-system process boundary."""

import asyncio
import os
from dataclasses import dataclass
from typing import Final, Protocol

from myclaw.tools.shell.owned_process import (
    OwnedProcess,
    OwnedProcessSpawner,
    default_owned_process_spawner,
)
from myclaw.tools.shell.shell_policy import ShellPolicyDenied, ShellRequest, trusted_git_executable

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
    process: OwnedProcess
    output: asyncio.Task[tuple[bytes, bytes | None]]
    stop: asyncio.Task[None] | None = None


class SubprocessShellBoundary:
    """Run validated Shell requests through an injected process spawner."""

    def __init__(
        self,
        *,
        spawner: OwnedProcessSpawner | None = None,
        waiter: ShellProcessWaiter | None = None,
    ) -> None:
        self._spawner = default_owned_process_spawner() if spawner is None else spawner
        self._waiter = AsyncioShellProcessWaiter() if waiter is None else waiter
        self._active: dict[int, _ActiveProcess] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def execute(self, request: ShellRequest) -> str:
        if request.command == "pwd":
            async with self._lock:
                if self._closed:
                    raise RuntimeError("Shell process boundary is closed")
            return os.fspath(request.cwd)
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


def _resolved_shell_command(request: ShellRequest) -> tuple[str, ...]:
    arguments = _HARDENED_GIT_ARGUMENTS.get(request.command)
    if arguments is None:
        raise ShellPolicyDenied("Shell command is not permitted")
    executable = trusted_git_executable(workspace=request.workspace_root)
    if executable is None:
        raise ShellPolicyDenied("trusted Git executable is unavailable")
    return (os.fspath(executable), *arguments)


async def _join_spawn(task: asyncio.Task[OwnedProcess]) -> OwnedProcess:
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
