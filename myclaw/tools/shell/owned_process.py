"""Host-specific process-tree ownership behind one asynchronous interface."""

import asyncio
import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Final, Protocol, cast

_POSIX_SIGKILL = getattr(signal, "SIGKILL", 9)
_WINDOWS_PROCESS_CREATION_FLAGS: Final = 0x08000204


class OwnedProcess(Protocol):
    async def communicate(self) -> tuple[bytes, bytes | None]: ...

    async def terminate(self) -> None: ...

    async def wait(self) -> None: ...


class OwnedProcessSpawner(Protocol):
    async def spawn(self, command: tuple[str, ...], *, cwd: Path) -> OwnedProcess: ...


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
        from myclaw.tools.shell.windows_job import WindowsJob

        job = WindowsJob.create()
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=_WINDOWS_PROCESS_CREATION_FLAGS,
            )
        )
        try:
            process = await asyncio.shield(creation)
        except BaseException as primary_error:
            try:
                process = await _join_task(creation)
            except BaseException as creation_error:
                job.close()
                if creation_error is primary_error:
                    raise primary_error from None
                raise primary_error from creation_error
            try:
                owned = await self._assign(job=job, process=process)
            except BaseException as ownership_error:
                raise primary_error from ownership_error
            cleanup = asyncio.create_task(owned.terminate())
            try:
                await _join_task(cleanup)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise primary_error
        owned = await self._assign(job=job, process=process)
        try:
            job.resume(process.pid)
        except BaseException as ownership_error:
            cleanup = asyncio.create_task(owned.terminate())
            try:
                await _join_task(cleanup)
            except BaseException as cleanup_error:
                raise ownership_error from cleanup_error
            raise
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
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        )
        try:
            process = await asyncio.shield(creation)
        except BaseException as primary_error:
            try:
                process = await _join_task(creation)
            except BaseException as creation_error:
                if creation_error is primary_error:
                    raise primary_error from None
                raise primary_error from creation_error
            owned = self._owned(process)
            cleanup = asyncio.create_task(owned.terminate())
            try:
                await _join_task(cleanup)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise primary_error
        return self._owned(process)

    def _owned(self, process: asyncio.subprocess.Process) -> OwnedProcess:
        return _AsyncioOwnedProcess(
            process,
            tree=_PosixProcessTree(
                process,
                termination_timeout=self._termination_timeout,
                signal_process_group=self._signal_process_group,
            ),
        )


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
