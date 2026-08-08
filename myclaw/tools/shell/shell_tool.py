"""Shell Tool policy, execution, and model-facing declaration."""

import asyncio
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Final, Protocol
from unicodedata import category

from myclaw.tools.base import BaseTool, ToolError, ToolParam
from myclaw.tools.shell.owned_process import (
    OwnedProcess,
    OwnedProcessSpawner,
    default_owned_process_spawner,
)
from myclaw.utils.host_filesystem import HOST_FILESYSTEM, HostFilesystem

AUTOMATICALLY_ALLOWED_COMMANDS: Final = frozenset(
    {
        "pwd",
        "git status",
        "git status --short",
        "git diff --stat",
        "git diff --name-only",
    }
)
_GIT_FILTER_CONFIG_PATTERN: Final = r"^filter\..*\.(clean|smudge|process)$"


@dataclass(frozen=True, slots=True)
class _TrustedGitExecutable:
    path: Path
    identity: tuple[int, int, int, int]


def _capture_git_executable(
    *,
    discover: Callable[[str], str | None] = shutil.which,
    host_filesystem: HostFilesystem = HOST_FILESYSTEM,
) -> _TrustedGitExecutable | None:
    discovered = discover("git")
    if discovered is None:
        return None
    try:
        path = Path(discovered).resolve(strict=True)
        status = path.stat()
    except OSError:
        return None
    if not path.is_file() or not host_filesystem.accepts_native_executable_name(path):
        return None
    return _TrustedGitExecutable(
        path=path,
        identity=(status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns),
    )


_TRUSTED_GIT_EXECUTABLE: Final = _capture_git_executable()


class ShellPolicyDenied(PermissionError):
    """Raised when a Shell request cannot be normalized safely."""


@dataclass(frozen=True, slots=True)
class ShellRequest:
    """One validated request passed to the operating-system Shell boundary."""

    command: str
    cwd: Path
    timeout: int
    workspace_root: Path | None = field(default=None, repr=False, compare=False)


def parse_shell_request(
    *,
    command: str,
    cwd: str,
    timeout: int,
    workspace: Path,
) -> ShellRequest:
    """Validate and normalize one Shell request within its Workspace cwd boundary."""
    if (
        not command.strip()
        or not cwd
        or not 60 <= timeout <= 600
        or any(category(character) == "Cc" for character in command)
    ):
        raise ShellPolicyDenied("invalid Shell command parameters")

    try:
        workspace_root = workspace.resolve(strict=True)
        target = Path(cwd)
        if not target.is_absolute():
            target = workspace_root / target
        resolved_cwd = target.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ShellPolicyDenied("Shell cwd cannot be resolved") from exc
    if (
        not workspace_root.is_dir()
        or not resolved_cwd.is_relative_to(workspace_root)
        or not resolved_cwd.is_dir()
    ):
        raise ShellPolicyDenied("Shell cwd must be a Workspace directory")
    return ShellRequest(
        command=command,
        cwd=resolved_cwd,
        timeout=timeout,
        workspace_root=workspace_root,
    )


def shell_command_is_allowed(
    command: str,
    *,
    cwd: Path | None = None,
    workspace: Path | None = None,
) -> bool:
    """Return whether one command matches the frozen safe read-only policy."""
    if command not in AUTOMATICALLY_ALLOWED_COMMANDS:
        return False
    if not command.startswith("git ") or cwd is None:
        return True
    git_executable = trusted_git_executable(workspace=workspace)
    if git_executable is None or _git_filters_may_run(cwd, git_executable):
        return False
    return True


def trusted_git_executable(*, workspace: Path | None = None) -> Path | None:
    captured = _TRUSTED_GIT_EXECUTABLE
    if captured is None:
        return None
    try:
        status = captured.path.stat()
        identity = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
        workspace_root = workspace.resolve(strict=True) if workspace is not None else None
    except OSError:
        return None
    if identity != captured.identity:
        return None
    if workspace_root is not None and captured.path.is_relative_to(workspace_root):
        return None
    return captured.path


def _git_filters_may_run(cwd: Path, git_executable: Path) -> bool:
    try:
        completed = subprocess.run(
            (
                os.fspath(git_executable),
                "-C",
                str(cwd),
                "config",
                "--includes",
                "--null",
                "--name-only",
                "--get-regexp",
                _GIT_FILTER_CONFIG_PATTERN,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if completed.returncode == 1:
        return False
    if completed.returncode != 0:
        return True
    configured_filters = _configured_filter_names(completed.stdout)
    if configured_filters and "GIT_ATTR_SOURCE" in os.environ:
        return True
    selected_filters = _repository_attribute_filters(cwd, git_executable)
    return (
        configured_filters is None
        or selected_filters is None
        or not configured_filters.isdisjoint(selected_filters)
    )


def _configured_filter_names(output: bytes) -> frozenset[str] | None:
    names: set[str] = set()
    suffixes = (".clean", ".smudge", ".process")
    for raw_key in output.split(b"\0"):
        if not raw_key:
            continue
        key = os.fsdecode(raw_key)
        suffix = next((item for item in suffixes if key.endswith(item)), None)
        if not key.startswith("filter.") or suffix is None:
            return None
        name = key[len("filter.") : -len(suffix)]
        if not name:
            return None
        names.add(name)
    return frozenset(names)


def _repository_attribute_filters(
    cwd: Path,
    git_executable: Path,
) -> frozenset[str] | None:
    try:
        root_result = subprocess.run(
            (
                os.fspath(git_executable),
                "-C",
                str(cwd),
                "rev-parse",
                "--show-toplevel",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if root_result.returncode != 0:
        return frozenset()

    try:
        root = Path(os.fsdecode(root_result.stdout.strip())).resolve(strict=True)
    except (OSError, ValueError):
        return None

    try:
        global_attributes_result = subprocess.run(
            (
                os.fspath(git_executable),
                "-C",
                str(cwd),
                "config",
                "--includes",
                "--null",
                "--path",
                "--get-all",
                "core.attributesFile",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if global_attributes_result.returncode not in (0, 1):
        return None

    selected_filters: set[str] = set()
    for raw_path in global_attributes_result.stdout.split(b"\0"):
        if not raw_path:
            continue
        attributes_path = Path(os.fsdecode(raw_path))
        if not attributes_path.is_absolute():
            return None
        if attributes_path.exists():
            global_filters = _attribute_file_filters(attributes_path)
            if global_filters is None:
                return None
            selected_filters.update(global_filters)

    try:
        info_result = subprocess.run(
            (
                os.fspath(git_executable),
                "-C",
                str(cwd),
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "info/attributes",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if info_result.returncode != 0 or not info_result.stdout.strip():
        return None
    info_attributes = Path(os.fsdecode(info_result.stdout.strip()))
    if info_attributes.exists():
        info_filters = _attribute_file_filters(info_attributes)
        if info_filters is None:
            return None
        selected_filters.update(info_filters)

    walk_failed = False

    def record_walk_failure(_: OSError) -> None:
        nonlocal walk_failed
        walk_failed = True

    for directory, directories, files in os.walk(
        root,
        onerror=record_walk_failure,
        followlinks=False,
    ):
        directories[:] = [name for name in directories if name != ".git"]
        if ".gitattributes" not in files:
            continue
        file_filters = _attribute_file_filters(Path(directory) / ".gitattributes")
        if file_filters is None:
            return None
        selected_filters.update(file_filters)
    if walk_failed:
        return None
    return frozenset(selected_filters)


def _attribute_file_filters(path: Path) -> frozenset[str] | None:
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        return None
    selected_filters: set[str] = set()
    for line in lines:
        if line.lstrip().startswith(b"#"):
            continue
        attributes = line.split()[1:]
        for attribute in attributes:
            if attribute == b"filter":
                selected_filters.add("true")
            elif attribute.startswith(b"filter="):
                raw_name = attribute[len(b"filter=") :]
                if not raw_name or b'"' in raw_name:
                    return None
                selected_filters.add(os.fsdecode(raw_name))
    return frozenset(selected_filters)


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
        self._waiter = waiter
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
            if self._waiter is None:
                stdout, _ = await asyncio.wait_for(
                    asyncio.shield(output),
                    timeout=request.timeout,
                )
            else:
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


class ShellBoundary(Protocol):
    async def execute(self, request: ShellRequest) -> str: ...


class ShellTool(BaseTool):
    """Expose Shell requests through the Tool protocol."""

    name = "shell"
    description = (
        "Run one of five exact read-only commands from a Workspace directory; this is not an "
        "operating-system filesystem or network sandbox."
    )
    required = ("command", "timeout")

    command: Annotated[str, ToolParam(description="Exact Shell command.", min_length=1)]
    cwd: Annotated[str, ToolParam(description="Workspace-relative working directory.")] = "."
    timeout: Annotated[
        int,
        ToolParam(description="Execution timeout in seconds.", minimum=60, maximum=600),
    ]

    def __init__(self, *, workspace: Path, boundary: ShellBoundary) -> None:
        self._workspace = workspace
        self._boundary = boundary

    def refusal_reason(self, *, command: str, cwd: str, timeout: int) -> str | None:
        request = self._request(command=command, cwd=cwd, timeout=timeout)
        if shell_command_is_allowed(
            request.command,
            cwd=request.cwd,
            workspace=request.workspace_root,
        ):
            return None
        return "Shell command refused because it is not in the safe read-only allowlist."

    async def execute(self, *, command: str, cwd: str, timeout: int) -> str:
        request = self._request(command=command, cwd=cwd, timeout=timeout)
        if not shell_command_is_allowed(
            request.command,
            cwd=request.cwd,
            workspace=request.workspace_root,
        ):
            raise ToolError("Shell command is not in the safe read-only allowlist.")
        try:
            return await self._boundary.execute(request)
        except ShellPolicyDenied as error:
            raise ToolError(
                "Shell process execution was rejected by the safety boundary."
            ) from error

    def _request(self, *, command: str, cwd: str, timeout: int) -> ShellRequest:
        try:
            return parse_shell_request(
                command=command,
                cwd=cwd,
                timeout=timeout,
                workspace=self._workspace,
            )
        except ShellPolicyDenied as error:
            raise ToolError("Shell request parameters or Workspace cwd are invalid.") from error
