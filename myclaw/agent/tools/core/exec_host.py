"""Portable Host Exec adapters with one process-lifetime shell resolution."""

from __future__ import annotations

import asyncio
import base64
import json
import ntpath
import os
import posixpath
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from myclaw.agent.tools.core.exec_policy import (
    ExecAssessment,
    ExecOutcome,
    ExecPlatform,
    ExecShellFamily,
    ExecShellSelector,
    ResolvedExecShell,
    assess_command,
)
from myclaw.utils.async_tasks import await_task_preserving_cancellation

EXEC_CAPABILITY_ERROR: Final = "Exec capability is unavailable because the selected shell is missing."
_PROCESS_REAP_TIMEOUT: Final[float] = 5.0
_INSPECTION_TIMEOUT: Final[int] = 5
_POSIX_ENVIRONMENT: Final[tuple[str, ...]] = ("HOME", "LANG", "TERM", "PATH")
_WINDOWS_ENVIRONMENT: Final[tuple[str, ...]] = (
    "HOME",
    "LANG",
    "TERM",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "USERPROFILE",
)
_PS_FLAGS: Final[tuple[str, ...]] = ("-NoLogo", "-NoProfile", "-NonInteractive")
_BASH_FLAGS: Final[tuple[str, ...]] = ("--noprofile", "--norc")
_BACKGROUND_CLEANUPS: Final[set[asyncio.Task[None]]] = set()
_PS_VERSION_PREFIX: Final[str] = "MYCLAW_PS_VERSION:"
_PS_VERSION_COMMAND: Final[str] = rf"""
$text = '{_PS_VERSION_PREFIX}' + $PSVersionTable.PSVersion.ToString()
$bytes = [Text.Encoding]::UTF8.GetBytes($text)
$stdout = [Console]::OpenStandardOutput()
$stdout.Write($bytes, 0, $bytes.Length)
"""
_PS_INSPECTOR_SCRIPT: Final[str] = r"""
param([string]$encodedSource)
$source = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encodedSource))
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $source,
    [ref]$tokens,
    [ref]$errors
)
$commands = @($ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.CommandAst] }, $true) |
    ForEach-Object { $_.GetCommandName() } | Where-Object { $_ })
$json = [ordered]@{
    syntax_ok = ($errors.Count -eq 0)
    command_names = $commands
} | ConvertTo-Json -Compress
$bytes = [Text.Encoding]::UTF8.GetBytes($json)
$stdout = [Console]::OpenStandardOutput()
$stdout.Write($bytes, 0, $bytes.Length)
"""


class ExecHostError(Exception):
    """A host-level execution failure that is not a Tool Result."""


class ExecCapabilityUnavailable(ExecHostError):
    """The selected shell is absent and cannot execute any command."""

    def __init__(self) -> None:
        super().__init__(EXEC_CAPABILITY_ERROR)


class ExecProcess(Protocol):
    @property
    def returncode(self) -> int | None: ...

    async def communicate(self) -> tuple[bytes | None, bytes | None]: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ExecProcessSpec:
    """Shared process inputs used by both Host inspection and execution."""

    executable: str
    flags: tuple[str, ...]
    cwd: Path
    environment: tuple[tuple[str, str], ...]

    @property
    def env(self) -> dict[str, str]:
        return dict(self.environment)


class ExecHost(Protocol):
    """The raw Host boundary consumed by the Exec Tool adapter."""

    resolved_shell: ResolvedExecShell

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment: ...

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome: ...

    def process_spec(self, cwd: Path) -> ExecProcessSpec: ...


def resolve_exec_shell(
    selector: ExecShellSelector,
    *,
    platform: str | None = None,
    which: Callable[[str], str | None] | None = None,
    version_probe: Callable[[str, dict[str, str]], tuple[int, ...] | None] | None = None,
    environment: Mapping[str, str] | None = None,
) -> ResolvedExecShell:
    """Resolve one effective selector without ever crossing an explicit fallback."""
    if selector not in {"auto", "powershell", "pwsh"}:
        raise ValueError("Exec shell selector is invalid")
    host_platform = _normalize_platform(os.name if platform is None else platform)
    env = _minimal_environment(
        os.environ if environment is None else environment,
        platform=host_platform,
    )
    locator = which or (lambda name: shutil.which(name, path=env.get("PATH")))

    def locate(name: str) -> str | None:
        return _canonical_shell_path(
            locator(name),
            expected=name,
            platform=host_platform,
            require_exists=which is None,
        )

    if host_platform == "posix":
        executable = locate("bash")
        return _resolved(
            selector=selector,
            platform="posix",
            family="bash",
            executable=executable,
            flags=_BASH_FLAGS,
            environment=env,
        )

    probe = version_probe or _probe_powershell_version
    if selector == "pwsh":
        executable = locate("pwsh")
        version = None if executable is None else probe(executable, env)
        if executable is None or not _is_pwsh_version(version):
            return _unavailable(
                selector=selector,
                platform="windows",
                family="pwsh",
                flags=_PS_FLAGS,
                environment=env,
            )
        return _resolved(
            selector=selector,
            platform="windows",
            family="pwsh",
            executable=executable,
            flags=_PS_FLAGS,
            environment=env,
            version=version,
        )

    if selector == "powershell":
        executable = locate("powershell")
        version = None if executable is None else probe(executable, env)
        if executable is None or not _is_windows_powershell_version(version):
            return _unavailable(
                selector=selector,
                platform="windows",
                family="powershell",
                flags=_PS_FLAGS,
                environment=env,
            )
        return _resolved(
            selector=selector,
            platform="windows",
            family="powershell",
            executable=executable,
            flags=_PS_FLAGS,
            environment=env,
            version=version,
        )

    pwsh = locate("pwsh")
    if pwsh is not None:
        version = probe(pwsh, env)
        if _is_pwsh_version(version):
            return _resolved(
                selector=selector,
                platform="windows",
                family="pwsh",
                executable=pwsh,
                flags=_PS_FLAGS,
                environment=env,
                version=version,
            )

    powershell = locate("powershell")
    version = None if powershell is None else probe(powershell, env)
    if powershell is not None and _is_windows_powershell_version(version):
        return _resolved(
            selector=selector,
            platform="windows",
            family="powershell",
            executable=powershell,
            flags=_PS_FLAGS,
            environment=env,
            version=version,
        )
    return _unavailable(
        selector=selector,
        platform="windows",
        family="powershell",
        flags=_PS_FLAGS,
        environment=env,
    )


def create_exec_host(resolved_shell: ResolvedExecShell) -> ExecHost:
    """Construct the adapter for an already resolved process-lifetime shell."""
    if not resolved_shell.available:
        return _UnavailableExecHost(resolved_shell)
    if resolved_shell.family == "bash":
        return BashExecHost(resolved_shell)
    return PowerShellExecHost(resolved_shell)


def _normalize_platform(value: str) -> ExecPlatform:
    if value in {"nt", "windows", "win32"}:
        return "windows"
    if value in {"posix", "linux", "darwin", "macos"}:
        return "posix"
    raise ValueError("Exec host platform is unsupported")


def _resolved(
    *,
    selector: ExecShellSelector,
    platform: ExecPlatform,
    family: ExecShellFamily,
    executable: str | None,
    flags: tuple[str, ...],
    environment: dict[str, str],
    version: tuple[int, ...] | None = None,
) -> ResolvedExecShell:
    if executable is None:
        return _unavailable(
            selector=selector,
            platform=platform,
            family=family,
            flags=flags,
            environment=environment,
        )
    return ResolvedExecShell(
        selector=selector,
        platform=platform,
        family=family,
        executable=executable,
        flags=flags,
        environment=tuple(environment.items()),
        available=True,
        version=version,
    )


def _unavailable(
    *,
    selector: ExecShellSelector,
    platform: ExecPlatform,
    family: ExecShellFamily,
    flags: tuple[str, ...],
    environment: dict[str, str],
) -> ResolvedExecShell:
    return ResolvedExecShell(
        selector=selector,
        platform=platform,
        family=family,
        executable=None,
        flags=flags,
        environment=tuple(environment.items()),
        available=False,
        diagnostic=EXEC_CAPABILITY_ERROR,
    )


def _minimal_environment(
    source: Mapping[str, str],
    *,
    platform: ExecPlatform,
) -> dict[str, str]:
    result: dict[str, str] = {}
    allowed = _WINDOWS_ENVIRONMENT if platform == "windows" else _POSIX_ENVIRONMENT
    for expected in allowed:
        for name, value in source.items():
            if name.upper() == expected:
                result[expected] = value
                break
    return result


def _canonical_shell_path(
    executable: str | None,
    *,
    expected: str,
    platform: ExecPlatform,
    require_exists: bool,
) -> str | None:
    if executable is None:
        return None
    path_module = ntpath if platform == "windows" else posixpath
    normalized: str = path_module.normpath(executable)
    if not path_module.isabs(normalized):
        return None
    basename = path_module.basename(normalized).lower()
    allowed_names = {expected.lower()}
    if platform == "windows":
        allowed_names.add(f"{expected.lower()}.exe")
    if basename not in allowed_names:
        return None
    current_platform = _normalize_platform(os.name)
    if require_exists and current_platform == platform:
        try:
            normalized = str(Path(normalized).resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            return None
        if path_module.basename(normalized).lower() not in allowed_names:
            return None
    return normalized


def _is_pwsh_version(version: tuple[int, ...] | None) -> bool:
    return version is not None and version >= (7,)


def _is_windows_powershell_version(version: tuple[int, ...] | None) -> bool:
    return version is not None and version[:1] == (5,) and version >= (5, 1)


def _probe_powershell_version(executable: str, environment: dict[str, str]) -> tuple[int, ...] | None:
    try:
        completed = subprocess.run(
            [executable, *_PS_FLAGS, "-Command", _PS_VERSION_COMMAND],
            check=False,
            cwd=None,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    match = re.fullmatch(
        rf"\s*{re.escape(_PS_VERSION_PREFIX)}(\d+(?:\.\d+){{1,3}})\s*",
        completed.stdout,
    )
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


class _BaseExecHost:
    def __init__(self, resolved_shell: ResolvedExecShell) -> None:
        if not resolved_shell.available or resolved_shell.executable is None:
            raise ValueError("An available shell is required for a concrete Exec Host")
        self.resolved_shell = resolved_shell

    def process_spec(self, cwd: Path) -> ExecProcessSpec:
        return ExecProcessSpec(
            executable=cast(str, self.resolved_shell.executable),
            flags=self.resolved_shell.flags,
            cwd=Path(cwd),
            environment=self.resolved_shell.environment,
        )

    async def _run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout: int,
    ) -> ExecOutcome:
        process: ExecProcess | None = None
        communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None = None
        try:
            process = await _spawn_process(
                argv=argv,
                cwd=cwd,
                environment=self.resolved_shell.env,
            )
            communication = asyncio.create_task(process.communicate())
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=timeout,
                )
            except TimeoutError:
                stdout, stderr = await _cleanup_preserving_cancellation(process, communication)
                return ExecOutcome(
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=True,
                )
            except asyncio.CancelledError:
                await _cleanup_without_replacing_cancellation(process, communication)
                raise
            except Exception as error:
                await _cleanup_without_replacing_cancellation(process, communication)
                raise ExecHostError("Exec failed while reading process output.") from error
            return ExecOutcome(
                exit_code=process.returncode,
                stdout=_as_bytes(stdout),
                stderr=_as_bytes(stderr),
            )
        except asyncio.CancelledError:
            if process is not None and communication is None:
                await _cleanup_without_replacing_cancellation(process, None)
            raise
        except ExecHostError:
            raise
        except Exception as error:
            raise ExecHostError(
                f"Exec failed to start {self.resolved_shell.family}: {error}"
            ) from error

class BashExecHost(_BaseExecHost):
    """POSIX Bash Host using a no-profile, no-rc process policy."""

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        del cwd
        try:
            import tree_sitter_bash
            from tree_sitter import Language, Parser

            parser = Parser(Language(tree_sitter_bash.language()))
            tree = parser.parse(command.encode("utf-8"))
        except (ImportError, OSError, TypeError, ValueError):
            return ExecAssessment.uncertain_result(
                "Bash AST inspection is unavailable.",
                status="uncertain",
            )
        except Exception:
            return ExecAssessment.uncertain_result(
                "Bash AST inspection failed.",
                status="failed",
            )

        root = tree.root_node
        if root.has_error:
            return assess_command(
                command,
                family="bash",
                syntax_confidence="unknown",
                syntax_uncertain=True,
                diagnostics=("Bash AST syntax was malformed.",),
                inspector_status="uncertain",
            )
        return assess_command(
            command,
            family="bash",
            syntax_confidence="high",
            syntax_uncertain=False,
            command_names=_tree_command_names(root, command),
        )

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
        spec = self.process_spec(cwd)
        return await self._run(
            (spec.executable, *spec.flags, "-c", command),
            cwd=spec.cwd,
            timeout=timeout,
        )


class PowerShellExecHost(_BaseExecHost):
    """Windows PowerShell Host with a static Parser.ParseInput inspector."""

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
        inspector_command = f"& {{\n{_PS_INSPECTOR_SCRIPT}\n}} '{encoded}'"
        spec = self.process_spec(cwd)
        try:
            outcome = await self._run(
                (spec.executable, *spec.flags, "-Command", inspector_command),
                cwd=spec.cwd,
                timeout=_INSPECTION_TIMEOUT,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ExecAssessment.uncertain_result(
                "PowerShell AST inspection failed.",
                status="failed",
            )
        if outcome.timed_out:
            return ExecAssessment.uncertain_result(
                "PowerShell AST inspection timed out.",
                status="timeout",
            )
        if outcome.exit_code != 0 or outcome.stderr:
            return ExecAssessment.uncertain_result(
                "PowerShell AST inspection failed.",
                status="failed",
            )
        try:
            payload = json.loads(outcome.stdout.decode("utf-8"))
            if not isinstance(payload, dict) or set(payload) != {"syntax_ok", "command_names"}:
                raise ValueError("parser output is not an object")
            syntax_ok = payload.get("syntax_ok")
            raw_names = payload.get("command_names")
            if not isinstance(syntax_ok, bool) or not isinstance(raw_names, list):
                raise ValueError("parser output is malformed")
            if any(not isinstance(name, str) or not name for name in raw_names):
                raise ValueError("parser command names are malformed")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return ExecAssessment.uncertain_result(
                "PowerShell AST inspection returned malformed output.",
                status="uncertain",
            )
        if not syntax_ok:
            return assess_command(
                command,
                family="powershell",
                syntax_confidence="unknown",
                syntax_uncertain=True,
                command_names=tuple(cast(list[str], raw_names)),
                diagnostics=("PowerShell AST syntax was malformed.",),
                inspector_status="uncertain",
            )
        return assess_command(
            command,
            family="powershell",
            syntax_confidence="high",
            syntax_uncertain=False,
            command_names=tuple(cast(list[str], raw_names)),
        )

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
        spec = self.process_spec(cwd)
        return await self._run(
            (spec.executable, *spec.flags, "-Command", command),
            cwd=spec.cwd,
            timeout=timeout,
        )


class _UnavailableExecHost:
    def __init__(self, resolved_shell: ResolvedExecShell) -> None:
        self.resolved_shell = resolved_shell

    def process_spec(self, cwd: Path) -> ExecProcessSpec:
        del cwd
        raise ExecCapabilityUnavailable

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        del command, cwd
        return ExecAssessment.uncertain_result(
            EXEC_CAPABILITY_ERROR,
            status="uncertain",
        )

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
        del command, cwd, timeout
        raise ExecCapabilityUnavailable


async def _spawn_process(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    environment: dict[str, str],
) -> ExecProcess:
    spawning = asyncio.create_task(
        _create_process(
            argv=argv,
            cwd=cwd,
            environment=environment,
        )
    )
    try:
        return await asyncio.shield(spawning)
    except asyncio.CancelledError as cancellation:
        cleanup = asyncio.create_task(_cleanup_cancelled_spawn(spawning))
        try:
            await await_task_preserving_cancellation(cleanup)
        except asyncio.CancelledError:
            raise
        except BaseException:
            pass
        raise cancellation


async def _create_process(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    environment: dict[str, str],
) -> ExecProcess:
    return await asyncio.create_subprocess_exec(
        *argv,
        cwd=os.fspath(cwd),
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _cleanup_cancelled_spawn(spawning: asyncio.Task[ExecProcess]) -> None:
    try:
        process = await asyncio.wait_for(
            asyncio.shield(spawning),
            timeout=_PROCESS_REAP_TIMEOUT,
        )
    except TimeoutError:
        _defer_spawn_cleanup(spawning)
        return
    except BaseException:
        return
    await _cleanup_process(process, None)


def _defer_spawn_cleanup(spawning: asyncio.Task[ExecProcess]) -> None:
    spawning.cancel()
    cleanup = asyncio.create_task(_cleanup_late_spawn(spawning))
    _BACKGROUND_CLEANUPS.add(cleanup)
    cleanup.add_done_callback(_BACKGROUND_CLEANUPS.discard)


async def _cleanup_late_spawn(spawning: asyncio.Task[ExecProcess]) -> None:
    try:
        process = await spawning
        await _cleanup_process(process, None)
    except BaseException:
        pass


async def _cleanup_process(
    process: ExecProcess,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None,
) -> tuple[bytes, bytes]:
    try:
        if process.returncode is None:
            process.kill()
    except Exception:
        pass

    wait_task = asyncio.create_task(process.wait())
    tasks: tuple[asyncio.Task[object], ...] = (
        wait_task,
        *((communication,) if communication is not None else ()),
    )
    joined = asyncio.create_task(_join_cleanup_tasks(tasks))
    try:
        await asyncio.wait_for(asyncio.shield(joined), timeout=_PROCESS_REAP_TIMEOUT)
    except TimeoutError:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if not joined.done():
            joined.cancel()
        await asyncio.gather(joined, return_exceptions=True)
    return _communication_output(communication)


async def _join_cleanup_tasks(tasks: tuple[asyncio.Task[object], ...]) -> None:
    await asyncio.gather(*tasks, return_exceptions=True)


async def _cleanup_without_replacing_cancellation(
    process: ExecProcess,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None,
) -> None:
    cleanup = asyncio.create_task(_cleanup_process(process, communication))
    try:
        await await_task_preserving_cancellation(cleanup)
    except asyncio.CancelledError:
        raise
    except BaseException:
        pass


async def _cleanup_preserving_cancellation(
    process: ExecProcess,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None,
) -> tuple[bytes, bytes]:
    cleanup = asyncio.create_task(_cleanup_process(process, communication))
    return await await_task_preserving_cancellation(cleanup)


def _communication_output(
    communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None,
) -> tuple[bytes, bytes]:
    if communication is None or not communication.done() or communication.cancelled():
        return b"", b""
    try:
        stdout, stderr = communication.result()
    except BaseException:
        return b"", b""
    return _as_bytes(stdout), _as_bytes(stderr)


def _as_bytes(value: bytes | None) -> bytes:
    return b"" if value is None else value


def _tree_command_names(root: Any, source: str) -> tuple[str, ...]:
    names: list[str] = []
    stack = [root]
    encoded = source.encode("utf-8")
    while stack:
        node = stack.pop()
        if getattr(node, "type", None) == "command":
            for child in getattr(node, "children", ()):
                if getattr(child, "type", None) == "command_name":
                    name = encoded[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
                    if name and name not in names:
                        names.append(name)
                    break
        stack.extend(reversed(tuple(getattr(node, "children", ()))))
    return tuple(names)


__all__ = [
    "EXEC_CAPABILITY_ERROR",
    "BashExecHost",
    "ExecCapabilityUnavailable",
    "ExecHost",
    "ExecHostError",
    "ExecOutcome",
    "ExecProcess",
    "ExecProcessSpec",
    "PowerShellExecHost",
    "ResolvedExecShell",
    "create_exec_host",
    "resolve_exec_shell",
]
