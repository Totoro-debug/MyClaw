"""Exec Core Catalog Tool."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Annotated, Any, Final
from urllib.parse import urlsplit

from myclaw.agent.tools.base import BaseTool, ToolError, ToolParam, truncate_text
from myclaw.agent.tools.core.exec_host import (
    ExecCapabilityUnavailable,
    ExecHost,
    ExecHostError,
    ExecProcess,
    create_exec_host,
    resolve_exec_shell,
)
from myclaw.agent.tools.core.exec_policy import (
    ExecAssessment,
    requires_legacy_destructive_confirmation,
)
from myclaw.agent.tools.network_safety import DNSResolver, SocketDNSResolver, assess_target
from myclaw.agent.tools.permission import ToolInvocationFacts

_OUTPUT_LIMIT: Final[int] = 4000
_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"https?://[^\s\"'`<>]+",
    re.IGNORECASE,
)
_URL_TRAILING_CHARACTERS: Final[str] = ".,;:!?)]}"


class ExecTool(BaseTool):
    """Adapt one normalized Tool call to the process-lifetime Host Exec boundary."""

    name = "exec"
    description = "Run one host-shell command with captured output in the selected directory."
    required = ("command",)

    command: Annotated[str, ToolParam(description="Shell command to execute.", min_length=1)]
    cwd: Annotated[str, ToolParam(description="Working directory.", min_length=1)] = "."
    timeout: Annotated[
        int,
        ToolParam(description="Execution timeout in seconds.", minimum=1, maximum=600),
    ] = 60

    def __init__(
        self,
        *,
        workspace: Path,
        resolver: DNSResolver | None = None,
        host: ExecHost | None = None,
    ) -> None:
        self._workspace = workspace
        self._resolver = SocketDNSResolver() if resolver is None else resolver
        self._host = (
            create_exec_host(resolve_exec_shell("auto"))
            if host is None
            else host
        )

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Freeze one canonical cwd for validation, inspection, confirmation, and execution."""
        prepared = await super().prepare_arguments(arguments)
        if not self._host.resolved_shell.available:
            return prepared
        cwd = prepared.get("cwd")
        if not isinstance(cwd, str):
            raise ToolError("Exec working directory is invalid.")
        prepared["cwd"] = str(
            self.resolve_path_argument(workspace=self._workspace, requested=cwd)
        )
        return prepared

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
        if not self._host.resolved_shell.available:
            return self._host.resolved_shell.diagnostic or (
                "Exec capability is unavailable because the selected shell is missing."
            )
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
        if requires_legacy_destructive_confirmation(command):
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
        try:
            outcome = await self._host.execute(command, target, timeout)
        except asyncio.CancelledError:
            raise
        except ExecCapabilityUnavailable as error:
            raise ToolError(str(error)) from error
        except ExecHostError as error:
            raise ToolError(str(error)) from error
        except Exception as error:
            raise ToolError("Exec failed to start the selected shell.") from error
        if outcome.timed_out:
            raise ToolError(
                _format_timeout(timeout=timeout, stdout=outcome.stdout, stderr=outcome.stderr)
            )
        return _format_result(
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
        )

    async def collect_invocation_facts(
        self,
        prepared_arguments: dict[str, Any],
        *,
        safety_reason: str | None,
    ) -> ToolInvocationFacts:
        """Attach one detached Host assessment to the shared authorization facts."""
        command = prepared_arguments["command"]
        cwd = prepared_arguments["cwd"]
        if not isinstance(command, str) or not isinstance(cwd, str):
            raise ToolError("Exec arguments are invalid.")
        target = self.resolve_path_argument(workspace=self._workspace, requested=cwd)
        try:
            assessment = await self._host.inspect(command, target)
        except asyncio.CancelledError:
            raise
        except Exception:
            assessment = ExecAssessment.uncertain_result(
                "Exec inspection failed.",
                status="failed",
            )
        reasons = [reason for reason in (safety_reason, assessment.confirmation_reason) if reason]
        return ToolInvocationFacts(
            tool_name=self.name,
            normalized_arguments=prepared_arguments,
            legacy_safety_reason=" ".join(dict.fromkeys(reasons)) or None,
            exec_assessment=assessment,
        )

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

__all__ = ["ExecProcess", "ExecTool"]
