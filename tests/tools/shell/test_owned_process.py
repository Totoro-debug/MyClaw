import asyncio
import ctypes
import signal
from pathlib import Path
from typing import cast

import pytest

import myclaw.tools.shell.windows_job as windows_job
from myclaw.tools.shell.owned_process import (
    PosixOwnedProcessSpawner,
    WindowsOwnedProcessSpawner,
    default_owned_process_spawner,
)
from myclaw.tools.shell.windows_job import WindowsJob


def test_windows_job_rejects_a_pointer_sized_invalid_thread_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFunction:
        argtypes: object = None
        restype: object = None

        def __init__(self, result: int) -> None:
            self._result = result

        def __call__(self, *_args: object) -> int:
            return self._result

    class FakeKernel32:
        CreateToolhelp32Snapshot = FakeFunction(cast(int, ctypes.c_void_p(-1).value))
        Thread32First = FakeFunction(0)
        Thread32Next = FakeFunction(0)
        OpenThread = FakeFunction(0)
        ResumeThread = FakeFunction(0)

    closed: list[int] = []
    job = object.__new__(WindowsJob)
    monkeypatch.setattr(job, "_close_handle", closed.append, raising=False)
    monkeypatch.setattr(windows_job, "_windows_kernel32", FakeKernel32)

    with pytest.raises(OSError, match="CreateToolhelp32Snapshot"):
        job.resume(123)

    assert closed == []


@pytest.mark.asyncio
async def test_posix_spawner_executes_exact_argv_in_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    commands: list[tuple[str, ...]] = []
    options: dict[str, object] = {}

    class FakeProcess:
        pid = 123
        returncode = 0

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        commands.append(command)
        options.update(kwargs)
        return cast(asyncio.subprocess.Process, FakeProcess())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    command = ("git", "-c", "core.fsmonitor=false", "--no-pager", "status")

    await PosixOwnedProcessSpawner().spawn(command, cwd=workspace)

    assert commands == [command]
    assert options == {
        "cwd": workspace,
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
        "start_new_session": True,
    }


@pytest.mark.asyncio
async def test_non_windows_host_attempts_posix_process_ownership_without_an_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    options: dict[str, object] = {}

    class FakeProcess:
        pid = 234
        returncode = 0

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        del command
        options.update(kwargs)
        return cast(asyncio.subprocess.Process, FakeProcess())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    await default_owned_process_spawner(host_name="unlisted-host").spawn(
        ("git", "status"), cwd=workspace
    )

    assert options["start_new_session"] is True


@pytest.mark.asyncio
async def test_posix_owned_process_terminates_the_group_waits_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    sent_signals: list[int] = []
    group_alive = True

    class FakeProcess:
        pid = 456
        returncode: int | None = None
        wait_calls = 0

        async def wait(self) -> int:
            self.wait_calls += 1
            self.returncode = -signal.SIGTERM
            return self.returncode

    process = FakeProcess()

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        del command, kwargs
        return cast(asyncio.subprocess.Process, process)

    def kill_group(process_group: int, sent_signal: int) -> None:
        nonlocal group_alive
        assert process_group == process.pid
        if sent_signal == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        sent_signals.append(sent_signal)
        group_alive = False

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    owned = await PosixOwnedProcessSpawner(signal_process_group=kill_group).spawn(
        ("git", "status"), cwd=workspace
    )

    await owned.terminate()
    await owned.terminate()

    assert sent_signals == [signal.SIGTERM]
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_posix_owned_process_escalates_and_fully_waits_after_the_grace_period(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    sent_signals: list[int] = []
    killed = asyncio.Event()
    group_alive = True

    class FakeProcess:
        pid = 789
        returncode: int | None = None
        wait_calls = 0

        async def wait(self) -> int:
            self.wait_calls += 1
            await killed.wait()
            self.returncode = -9
            return self.returncode

    process = FakeProcess()

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        del command, kwargs
        return cast(asyncio.subprocess.Process, process)

    def kill_group(process_group: int, sent_signal: int) -> None:
        nonlocal group_alive
        assert process_group == process.pid
        if sent_signal == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        sent_signals.append(sent_signal)
        if sent_signal == 9:
            group_alive = False
            killed.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    owned = await PosixOwnedProcessSpawner(
        termination_timeout=0.01,
        signal_process_group=kill_group,
    ).spawn(("git", "status"), cwd=workspace)

    await owned.terminate()

    assert sent_signals == [signal.SIGTERM, 9]
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_posix_spawn_cancellation_cleans_the_created_process_before_returning(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    creation_started = asyncio.Event()
    release_creation = asyncio.Event()
    sent_signals: list[int] = []
    group_alive = True

    class FakeProcess:
        pid = 987
        returncode: int | None = None
        wait_calls = 0

        async def wait(self) -> int:
            self.wait_calls += 1
            self.returncode = -signal.SIGTERM
            return self.returncode

    process = FakeProcess()

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        del command, kwargs
        creation_started.set()
        await release_creation.wait()
        return cast(asyncio.subprocess.Process, process)

    def kill_group(process_group: int, sent_signal: int) -> None:
        nonlocal group_alive
        assert process_group == process.pid
        if sent_signal == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        sent_signals.append(sent_signal)
        group_alive = False

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    spawn = asyncio.create_task(
        PosixOwnedProcessSpawner(signal_process_group=kill_group).spawn(
            ("git", "status"), cwd=workspace
        )
    )
    await creation_started.wait()

    spawn.cancel()
    await asyncio.sleep(0)

    assert not spawn.done()
    release_creation.set()
    with pytest.raises(asyncio.CancelledError):
        await spawn
    assert sent_signals == [signal.SIGTERM]
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_posix_spawn_cancellation_chains_group_cleanup_failure_after_root_reap(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    creation_started = asyncio.Event()
    release_creation = asyncio.Event()
    killed = asyncio.Event()

    class FakeProcess:
        pid = 654
        returncode: int | None = None
        kill_calls = 0
        wait_calls = 0

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9
            killed.set()

        async def wait(self) -> int:
            self.wait_calls += 1
            await killed.wait()
            assert self.returncode is not None
            return self.returncode

    process = FakeProcess()

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        del command, kwargs
        creation_started.set()
        await release_creation.wait()
        return cast(asyncio.subprocess.Process, process)

    def fail_group_signal(process_group: int, sent_signal: int) -> None:
        del process_group, sent_signal
        raise OSError("group signal failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    spawn = asyncio.create_task(
        PosixOwnedProcessSpawner(signal_process_group=fail_group_signal).spawn(
            ("git", "status"), cwd=workspace
        )
    )
    await creation_started.wait()

    spawn.cancel()
    release_creation.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await spawn
    assert isinstance(raised.value.__cause__, OSError)
    assert process.kill_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_windows_spawn_cancellation_assigns_the_job_and_fully_cleans_the_process(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    creation_started = asyncio.Event()
    release_creation = asyncio.Event()
    terminated = asyncio.Event()

    class FakeProcess:
        pid = 321
        returncode: int | None = None
        wait_calls = 0

        async def wait(self) -> int:
            self.wait_calls += 1
            await terminated.wait()
            assert self.returncode is not None
            return self.returncode

    process = FakeProcess()

    class FakeJob:
        def __init__(self) -> None:
            self.assigned: list[int] = []
            self.terminate_calls = 0

        def assign(self, pid: int) -> None:
            self.assigned.append(pid)

        def resume(self, pid: int) -> None:
            del pid
            raise AssertionError("A cancelled suspended process must not be resumed")

        async def terminate(self) -> None:
            self.terminate_calls += 1
            process.returncode = 1
            terminated.set()

        def close(self) -> None:
            return None

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        del command, kwargs
        creation_started.set()
        await release_creation.wait()
        return cast(asyncio.subprocess.Process, process)

    job = FakeJob()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(WindowsJob, "create", staticmethod(lambda: job))
    spawn = asyncio.create_task(
        WindowsOwnedProcessSpawner().spawn(("git.exe", "status"), cwd=workspace)
    )
    await creation_started.wait()

    spawn.cancel()
    await asyncio.sleep(0)

    assert not spawn.done()
    release_creation.set()
    with pytest.raises(asyncio.CancelledError):
        await spawn
    assert job.assigned == [process.pid]
    assert job.terminate_calls == 1
    assert process.wait_calls == 1


@pytest.mark.asyncio
async def test_windows_resume_failure_chains_cleanup_failure_after_waiting_for_root(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    class FakeProcess:
        pid = 432
        returncode: int | None = None
        wait_calls = 0

        async def wait(self) -> int:
            self.wait_calls += 1
            self.returncode = 1
            return self.returncode

    process = FakeProcess()

    class FakeJob:
        def __init__(self) -> None:
            self.assigned: list[int] = []
            self.terminate_calls = 0

        def assign(self, pid: int) -> None:
            self.assigned.append(pid)

        def resume(self, pid: int) -> None:
            del pid
            raise OSError("resume failed")

        async def terminate(self) -> None:
            self.terminate_calls += 1
            raise RuntimeError("job cleanup failed")

        def close(self) -> None:
            return None

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        del command, kwargs
        return cast(asyncio.subprocess.Process, process)

    job = FakeJob()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(WindowsJob, "create", staticmethod(lambda: job))

    with pytest.raises(OSError, match="resume failed") as raised:
        await WindowsOwnedProcessSpawner().spawn(("git.exe", "status"), cwd=workspace)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert job.assigned == [process.pid]
    assert job.terminate_calls == 1
    assert process.wait_calls == 1
