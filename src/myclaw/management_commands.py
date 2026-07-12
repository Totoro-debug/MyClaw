"""Standalone dispatch for read-only Management Commands."""

import json
from dataclasses import dataclass
from typing import Protocol

from myclaw.contracts import SessionSummary, format_rfc3339_milliseconds
from myclaw.contracts.management import (
    ConfigView,
    MemoryTaskResult,
    ResumeResult,
    RuntimeStatus,
)
from myclaw.management import ManagementError
from myclaw.session_store import SessionListingReport


class _ManagementViewPort(Protocol):
    async def config_view(self) -> ConfigView: ...

    async def status(self) -> RuntimeStatus: ...

    async def memory_view(self) -> str: ...

    async def dream(self) -> MemoryTaskResult: ...

    async def resumable_sessions(self) -> tuple[SessionSummary, ...]: ...

    async def resumable_listing(self) -> SessionListingReport: ...

    async def resume(self, session_id: str) -> ResumeResult: ...


@dataclass(frozen=True, slots=True)
class ManagementCommandResult:
    """Renderable output and whether a Management Command was recognized."""

    handled: bool
    output: str | None
    resume_sessions: tuple[SessionSummary, ...] | None = None


class ManagementCommandDispatcher:
    """Dispatch exact built-in commands without entering conversation flow."""

    def __init__(self, management: _ManagementViewPort) -> None:
        self._management = management

    async def dispatch(self, command: str) -> ManagementCommandResult:
        """Return rendered output for a recognized Management Command."""
        if command == "/resume":
            try:
                listing = await self._management.resumable_listing()
            except ManagementError as management_error:
                return ManagementCommandResult(
                    handled=True,
                    output=f"{management_error.error.code}: {management_error.error.message}",
                )
            sessions = listing.sessions
            lines: list[str] = []
            if listing.skipped_count:
                lines.append(
                    f"Warning: Skipped {listing.skipped_count} corrupt Conversation "
                    f"{'Session' if listing.skipped_count == 1 else 'Sessions'}."
                )
            if not sessions:
                lines.append("No resumable Conversation Sessions.")
            else:
                lines.append("Resumable sessions:")
                lines.extend(
                    f"{index}. {session.title} | "
                    f"{format_rfc3339_milliseconds(session.updated_at)} | "
                    f"{session.message_count} "
                    f"{'message' if session.message_count == 1 else 'messages'}"
                    for index, session in enumerate(sessions, start=1)
                )
            return ManagementCommandResult(
                handled=True,
                output="\n".join(lines),
                resume_sessions=sessions,
            )
        if command == "/status":
            try:
                status = await self._management.status()
                output = json.dumps(status.to_dict(), ensure_ascii=False, indent=2)
            except ManagementError as management_error:
                output = f"{management_error.error.code}: {management_error.error.message}"
            return ManagementCommandResult(handled=True, output=output)
        if command == "/memory":
            try:
                output = await self._management.memory_view()
            except ManagementError as management_error:
                output = f"{management_error.error.code}: {management_error.error.message}"
            return ManagementCommandResult(
                handled=True,
                output=output,
            )
        if command == "/dream":
            try:
                result = await self._management.dream()
            except ManagementError as management_error:
                output = f"{management_error.error.code}: {management_error.error.message}"
            else:
                if result.error is None and result.status == "No pending summaries":
                    output = result.status
                else:
                    headline = (
                        result.status
                        if result.error is None
                        else f"{result.error.code}: {result.error.message}"
                    )
                    output = (
                        f"{headline}\n"
                        f"processed_count: {result.processed_count}\n"
                        f"memory_updated: {str(result.memory_updated).lower()}\n"
                        f"cursor: {result.cursor}"
                    )
            return ManagementCommandResult(handled=True, output=output)
        if command != "/config":
            return ManagementCommandResult(handled=False, output=None)
        try:
            view = await self._management.config_view()
        except ManagementError as management_error:
            return ManagementCommandResult(
                handled=True,
                output=f"{management_error.error.code}: {management_error.error.message}",
            )
        prefix = f"Path: {view.path}\n"
        if view.error is not None:
            prefix = f"{view.error.code}: {view.error.message}\n{prefix}"
        return ManagementCommandResult(
            handled=True,
            output=f"{prefix}{view.redacted_content}",
        )

    async def resume(self, session_id: str) -> ManagementCommandResult:
        try:
            result = await self._management.resume(session_id)
        except ManagementError as management_error:
            output = f"{management_error.error.code}: {management_error.error.message}"
        else:
            output = f"Resumed session {result.session_id}."
        return ManagementCommandResult(handled=True, output=output)
