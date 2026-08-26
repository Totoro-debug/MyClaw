"""Exec Core Catalog Tool."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Annotated, Final, Protocol
from urllib.parse import urlsplit

from myclaw.tools.base import BaseTool, ToolError, ToolParam, truncate_text
from myclaw.tools.network_safety import DNSResolver, SocketDNSResolver, assess_target
from myclaw.utils.async_tasks import await_task_preserving_cancellation

_BASH: Final[str] = "bash"
_OUTPUT_LIMIT: Final[int] = 4000
_PROCESS_REAP_TIMEOUT: Final[float] = 5.0
_ALLOWED_ENVIRONMENT: Final[tuple[str, ...]] = ("HOME", "LANG", "TERM", "PATH")
_BACKGROUND_CLEANUPS: Final[set[asyncio.Task[None]]] = set()
_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"https?://[^\s\"'`<>]+",
    re.IGNORECASE,
)
_URL_TRAILING_CHARACTERS: Final[str] = ".,;:!?)]}"
_DESTRUCTIVE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?<![\w-])rm\b(?=[^\n;&|]*(?:--(?:force|recursive)\b|(?<![\w-])-[^\s;&|]*[rf]))",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![\w-])(?:del|erase)\b(?=[^\n;&|]*/[^\s;&|]*f\b)", re.IGNORECASE),
    re.compile(
        r"(?<![\w-])(?:rd|rmdir)\b(?=[^\n;&|]*(?:/[^\s;&|]*s\b|(?<![\w-])-[^\s;&|]*r))",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![\w-])format(?:\s|$)", re.IGNORECASE),
    re.compile(r"(?<![\w-])(?:mkfs(?:\.[\w-]+)?|diskpart)(?:\s|$)", re.IGNORECASE),
    re.compile(r"(?<![\w-])dd\b[^\n;]*\bif\s*=", re.IGNORECASE),
    re.compile(r"(?:>\s*|\bof\s*=\s*)/dev/", re.IGNORECASE),
    re.compile(
        r"(?:>\s*|\bof\s*=\s*)\\\\\.\\(?:PhysicalDrive\d*|[A-Za-z]:)",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![\w-])(?:shutdown|reboot|poweroff)(?:\s|$)", re.IGNORECASE),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:", re.IGNORECASE),
)


class ExecProcess(Protocol):
    """The direct Bash process operations required by Exec."""

    @property
    def returncode(self) -> int | None: ...

    async def communicate(self) -> tuple[bytes | None, bytes | None]: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


class ExecTool(BaseTool):
    """Run one user-confirmable command through a direct Bash login shell."""

    name = "exec"
    description = "Run one Bash login-shell command with captured output in the current Workspace."
    required = ("command",)

    command: Annotated[str, ToolParam(description="Bash command to execute.", min_length=1)]
    cwd: Annotated[str, ToolParam(description="Working directory.", min_length=1)] = "."
    timeout: Annotated[
        int,
        ToolParam(description="Execution timeout in seconds.", minimum=1, maximum=600),
    ] = 60

    def __init__(self, *, workspace: Path, resolver: DNSResolver | None = None) -> None:
        self._workspace = workspace
        self._resolver = SocketDNSResolver() if resolver is None else resolver

    def validate_arguments(  # type: ignore[override]
        self,
        *,
        command: str,
        cwd: str,
        timeout: int,
    ) -> str | None:
        del timeout
        if not command.strip():
            return "Exec command must not be blank."
        if "\x00" in command:
            return "Exec command must not contain a NUL character."
        try:
            target = self.resolve_path_argument(workspace=self._workspace, requested=cwd)
        except ToolError as error:
            return error.message
        if not target.is_dir():
            return "Exec working directory must be a directory."
        return None

    async def check_safety(  # type: ignore[override]
        self,
        *,
        command: str,
        cwd: str,
        timeout: int,
    ) -> str | None:
        del timeout
        reasons: list[str] = []
        if _matches_destructive_pattern(command):
            reasons.append(
                "The Exec command matches a known destructive operation and requires confirmation."
            )
        cwd_reason = self.workspace_path_safety_reason(
            workspace=self._workspace,
            requested=cwd,
        )
        if cwd_reason is not None:
            reasons.append(cwd_reason)
        url_reason = await self._url_safety_reason(command)
        if url_reason is not None:
            reasons.append(url_reason)
        return " ".join(dict.fromkeys(reasons)) or None

    async def execute(self, *, command: str, cwd: str, timeout: int) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=cwd)
        if not target.is_dir():
            raise ToolError("Exec working directory must be a directory.")

        process: ExecProcess | None = None
        communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None = None
        try:
            process = await _spawn_process(
                command=command,
                cwd=target,
            )
            communication = asyncio.create_task(process.communicate())
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=timeout,
                )
            except TimeoutError as error:
                stdout, stderr = await _cleanup_preserving_cancellation(process, communication)
                raise ToolError(
                    _format_timeout(timeout=timeout, stdout=stdout, stderr=stderr)
                ) from error
            except asyncio.CancelledError:
                await _cleanup_without_replacing_cancellation(process, communication)
                raise
            except Exception as error:
                await _cleanup_without_replacing_cancellation(process, communication)
                raise ToolError(f"Exec failed while reading process output: {error}") from error
            return _format_result(
                exit_code=process.returncode,
                stdout=_as_bytes(stdout),
                stderr=_as_bytes(stderr),
            )
        except asyncio.CancelledError:
            if process is not None and communication is None:
                await _cleanup_without_replacing_cancellation(process, None)
            raise
        except ToolError:
            raise
        except Exception as error:
            raise ToolError(f"Exec failed to start Bash: {error}") from error

    async def _url_safety_reason(self, command: str) -> str | None:
        reasons: list[str] = []
        for raw_url in _URL_PATTERN.findall(command):
            url = raw_url.rstrip(_URL_TRAILING_CHARACTERS)
            reason = await self._check_url(url)
            if reason is not None and reason not in reasons:
                reasons.append(reason)
        return " ".join(reasons) or None

    async def _check_url(self, url: str) -> str | None:
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return "An Exec URL could not be verified and requires confirmation."
        if hostname is None:
            return "An Exec URL has no hostname and requires confirmation."
        effective_port = (
            port if port is not None else (443 if parsed.scheme.lower() == "https" else 80)
        )

        assessment = await assess_target(hostname, effective_port, self._resolver)
        reasons = {
            "literal_non_global": (
                "An Exec URL uses a private or non-global address and requires confirmation."
            ),
            "dns_failure": "An Exec URL has a DNS failure and requires confirmation.",
            "dns_empty": "An Exec URL has no DNS result and requires confirmation.",
            "dns_non_global": (
                "An Exec URL resolves to a private or non-global address and requires confirmation."
            ),
        }
        return None if assessment.risk is None else reasons[assessment.risk]


async def _spawn_process(*, command: str, cwd: os.PathLike[str]) -> ExecProcess:
    spawning = asyncio.create_task(
        _create_process(
            command=command,
            cwd=cwd,
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


async def _create_process(*, command: str, cwd: os.PathLike[str]) -> ExecProcess:
    process = await asyncio.create_subprocess_exec(
        _BASH,
        "--login",
        "-c",
        command,
        cwd=os.fspath(cwd),
        env=_minimal_environment(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return process


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


def _minimal_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for expected in _ALLOWED_ENVIRONMENT:
        for name, value in os.environ.items():
            if name.upper() == expected:
                environment[expected] = value
                break
    return environment


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


def _format_result(*, exit_code: int | None, stdout: bytes, stderr: bytes) -> str:
    return _format_streams(
        heading=f"Exit code: {exit_code}",
        stdout=stdout,
        stderr=stderr,
    )


def _format_timeout(*, timeout: int, stdout: bytes, stderr: bytes) -> str:
    return _format_streams(
        heading=f"Exec timed out after {timeout} seconds.",
        stdout=stdout,
        stderr=stderr,
    )


def _format_streams(*, heading: str, stdout: bytes, stderr: bytes) -> str:
    blocks = [heading]
    decoded_stdout = stdout.decode("utf-8", errors="replace")
    decoded_stderr = stderr.decode("utf-8", errors="replace")
    if decoded_stdout:
        blocks.append(f"stdout:\n{decoded_stdout}")
    if decoded_stderr:
        blocks.append(f"stderr:\n{decoded_stderr}")
    return truncate_text("\n".join(blocks), limit=_OUTPUT_LIMIT)


def _matches_destructive_pattern(command: str) -> bool:
    return any(pattern.search(command) is not None for pattern in _DESTRUCTIVE_PATTERNS)


__all__ = ["ExecProcess", "ExecTool"]
