"""Standalone dispatch for read-only Management Commands."""

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from loguru import logger

from myclaw.config.config import ConfigView
from myclaw.logging.session import without_session_log
from myclaw.management.service import (
    ManagementError,
    ResumeResult,
    RuntimeStatus,
    SessionListingEntry,
    SessionListingReport,
)
from myclaw.memory.memory_task import MemoryTaskResult
from myclaw.utils.time import format_rfc3339_milliseconds


class _ManagementCommand(StrEnum):
    CONFIG = "/config"
    STATUS = "/status"
    RESUME = "/resume"
    MEMORY = "/memory"
    DREAM = "/dream"


SUPPORTED_MANAGEMENT_COMMANDS = tuple(command.value for command in _ManagementCommand)


class ManagementPort(Protocol):
    async def config_view(self) -> ConfigView: ...

    async def status(self) -> RuntimeStatus: ...

    async def memory_view(self) -> str: ...

    async def dream(self) -> MemoryTaskResult: ...

    async def resumable_listing(self) -> SessionListingReport: ...

    async def resume(self, session_id: str) -> ResumeResult: ...


@dataclass(frozen=True, slots=True)
class ManagementCommandResult:
    """Renderable output and whether a Management Command was recognized."""

    handled: bool
    output: str | None
    resume_sessions: tuple[SessionListingEntry, ...] | None = None


class ManagementCommandDispatcher:
    """Dispatch exact built-in commands without entering conversation flow."""

    def __init__(self, management: ManagementPort) -> None:
        self._management = management

    async def dispatch(self, command: str) -> ManagementCommandResult:
        """Return rendered output for a recognized Management Command."""
        with without_session_log():
            try:
                parsed_command = _ManagementCommand(command)
            except ValueError:
                return ManagementCommandResult(handled=False, output=None)
            return await self._dispatch(parsed_command)

    async def _dispatch(self, command: _ManagementCommand) -> ManagementCommandResult:
        if command is _ManagementCommand.RESUME:
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
        if command is _ManagementCommand.STATUS:
            try:
                status = await self._management.status()
                output = json.dumps(status.to_dict(), ensure_ascii=False, indent=2)
            except ManagementError as management_error:
                output = f"{management_error.error.code}: {management_error.error.message}"
            return ManagementCommandResult(handled=True, output=output)
        if command is _ManagementCommand.MEMORY:
            try:
                output = await self._management.memory_view()
            except ManagementError as management_error:
                output = f"{management_error.error.code}: {management_error.error.message}"
            return ManagementCommandResult(
                handled=True,
                output=output,
            )
        if command is _ManagementCommand.DREAM:
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
        if command is not _ManagementCommand.CONFIG:
            raise RuntimeError(f"Supported Management Command has no handler: {command}")
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
        with without_session_log():
            try:
                result = await self._management.resume(session_id)
            except ManagementError as management_error:
                output = f"{management_error.error.code}: {management_error.error.message}"
            except Exception as error:
                logger.opt(exception=error).error(
                    "Management command failed command=/resume type={}", type(error).__name__
                )
                raise
            else:
                output = f"Resumed session {result.session_id}."
            return ManagementCommandResult(handled=True, output=output)
