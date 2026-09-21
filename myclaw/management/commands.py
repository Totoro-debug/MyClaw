"""Standalone dispatch for read-only Management Commands."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from loguru import logger

from myclaw.agent.memory.dream import DreamResult
from myclaw.agent.permission import ToolPermissionLevel
from myclaw.config.config import ConfigView
from myclaw.logging.session import without_session_log
from myclaw.management.service import (
    FatalManagementError,
    ManagementError,
    ResumeResult,
    RuntimeStatus,
    SessionListingEntry,
    SessionListingReport,
)
from myclaw.provider.models import ReasoningEffort
from myclaw.skills.catalog import SkillMetadata
from myclaw.utils.time import format_rfc3339_milliseconds


@dataclass(frozen=True, slots=True)
class ManagementCommandDefinition:
    """Canonical completion and dispatch facts for one Management Command."""

    token: str
    description: str


_CONFIG_COMMAND = ManagementCommandDefinition("/config", "View User Configuration")
_STATUS_COMMAND = ManagementCommandDefinition("/status", "View Runtime Status")
_EFFORT_COMMAND = ManagementCommandDefinition("/effort", "Set Chat Reasoning Effort")
_PERMISSION_COMMAND = ManagementCommandDefinition(
    "/permission",
    "Set Foreground Tool Permission Level",
)
RESUME_MANAGEMENT_COMMAND = ManagementCommandDefinition("/resume", "Resume a Conversation Session")
_MEMORY_COMMAND = ManagementCommandDefinition("/memory", "View Long-term Memory")
_DREAM_COMMAND = ManagementCommandDefinition(
    "/dream",
    "Process pending Conversation Summaries",
)
RELOAD_SKILL_MANAGEMENT_COMMAND = ManagementCommandDefinition(
    "/reload_skill",
    "Reload Skills",
)
MANAGEMENT_COMMANDS = (
    _CONFIG_COMMAND,
    _STATUS_COMMAND,
    _EFFORT_COMMAND,
    _PERMISSION_COMMAND,
    RESUME_MANAGEMENT_COMMAND,
    _MEMORY_COMMAND,
    _DREAM_COMMAND,
    RELOAD_SKILL_MANAGEMENT_COMMAND,
)
_MANAGEMENT_COMMAND_BY_TOKEN: Mapping[str, ManagementCommandDefinition] = MappingProxyType(
    {command.token: command for command in MANAGEMENT_COMMANDS}
)


class ManagementPort(Protocol):
    async def config_view(self) -> ConfigView: ...

    async def status(self) -> RuntimeStatus: ...

    async def reasoning_effort(self) -> ReasoningEffort: ...

    async def update_reasoning_effort(self, effort: ReasoningEffort) -> ReasoningEffort: ...

    async def permission_level(self) -> ToolPermissionLevel: ...

    async def update_permission_level(self, level: ToolPermissionLevel) -> ToolPermissionLevel: ...

    async def memory_view(self) -> str: ...

    async def dream(self) -> DreamResult: ...

    async def reload_skill(self) -> tuple[SkillMetadata, ...]: ...

    async def resumable_listing(self) -> SessionListingReport: ...

    async def resume(self, session_id: str, *, force: bool = False) -> ResumeResult: ...


@dataclass(frozen=True, slots=True)
class ManagementCommandResult:
    """Renderable output and whether a Management Command was recognized."""

    handled: bool
    output: str | None
    effort_selection: ReasoningEffort | None = None
    permission_selection: ToolPermissionLevel | None = None
    resume_sessions: tuple[SessionListingEntry, ...] | None = None
    resumed_session_id: str | None = None
    resume_skipped_count: int = 0
    skill_metadata: tuple[SkillMetadata, ...] | None = None


class ManagementCommandDispatcher:
    """Dispatch exact built-in commands without entering conversation flow."""

    def __init__(self, management: ManagementPort) -> None:
        self._management = management

    @staticmethod
    def _skill_reload_failure() -> ManagementCommandResult:
        return ManagementCommandResult(
            handled=True,
            output="skill_reload_failed: Skill reload failed.",
        )

    async def dispatch(self, command: str) -> ManagementCommandResult:
        """Return rendered output for a recognized Management Command."""
        with without_session_log():
            parsed_command = _MANAGEMENT_COMMAND_BY_TOKEN.get(command)
            if parsed_command is None:
                return ManagementCommandResult(handled=False, output=None)
            management = self._management
            if parsed_command is _EFFORT_COMMAND:
                try:
                    effort = await management.reasoning_effort()
                except ManagementError as management_error:
                    return ManagementCommandResult(
                        handled=True,
                        output=f"{management_error.error.code}: {management_error.error.message}",
                    )
                return ManagementCommandResult(
                    handled=True,
                    output=None,
                    effort_selection=effort,
                )
            if parsed_command is _PERMISSION_COMMAND:
                try:
                    level = await management.permission_level()
                except ManagementError as management_error:
                    return ManagementCommandResult(
                        handled=True,
                        output=f"{management_error.error.code}: {management_error.error.message}",
                    )
                return ManagementCommandResult(
                    handled=True,
                    output=None,
                    permission_selection=level,
                )
            if parsed_command is RESUME_MANAGEMENT_COMMAND:
                try:
                    listing = await management.resumable_listing()
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
                    resume_skipped_count=listing.skipped_count,
                )
            if parsed_command is _STATUS_COMMAND:
                try:
                    status = await management.status()
                    output = json.dumps(status.to_dict(), ensure_ascii=False, indent=2)
                except ManagementError as management_error:
                    output = f"{management_error.error.code}: {management_error.error.message}"
                return ManagementCommandResult(handled=True, output=output)
            if parsed_command is _MEMORY_COMMAND:
                try:
                    output = await management.memory_view()
                except ManagementError as management_error:
                    output = f"{management_error.error.code}: {management_error.error.message}"
                return ManagementCommandResult(
                    handled=True,
                    output=output,
                )
            if parsed_command is _DREAM_COMMAND:
                try:
                    result = await management.dream()
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
            if parsed_command is RELOAD_SKILL_MANAGEMENT_COMMAND:
                try:
                    metadata = await management.reload_skill()
                except ManagementError:
                    return self._skill_reload_failure()
                except Exception as error:
                    logger.warning(
                        "Management command failed command=/reload_skill type={}",
                        type(error).__name__,
                    )
                    return self._skill_reload_failure()
                return ManagementCommandResult(
                    handled=True,
                    output=f"Skill count: {len(metadata)}",
                    skill_metadata=metadata,
                )
            if parsed_command is not _CONFIG_COMMAND:
                raise RuntimeError(f"Supported Management Command has no handler: {parsed_command}")
            try:
                view = await management.config_view()
            except ManagementError as management_error:
                return ManagementCommandResult(
                    handled=True,
                    output=f"{management_error.error.code}: {management_error.error.message}",
                )
            return ManagementCommandResult(
                handled=True,
                output=f"{view.header_text()}{view.redacted_content}",
            )

    async def update_reasoning_effort(
        self,
        effort: ReasoningEffort,
    ) -> ManagementCommandResult:
        """Commit a selected Runtime-Lifetime Reasoning Effort."""
        with without_session_log():
            try:
                published = await self._management.update_reasoning_effort(effort)
            except ManagementError as management_error:
                return ManagementCommandResult(
                    handled=True,
                    output=f"{management_error.error.code}: {management_error.error.message}",
                )
            return ManagementCommandResult(
                handled=True,
                output=f"Chat reasoning effort: {published}",
            )

    async def update_permission_level(
        self,
        level: ToolPermissionLevel,
    ) -> ManagementCommandResult:
        """Commit a selected process-local foreground Tool Permission Level."""
        with without_session_log():
            try:
                published = await self._management.update_permission_level(level)
            except ManagementError as management_error:
                return ManagementCommandResult(
                    handled=True,
                    output=f"{management_error.error.code}: {management_error.error.message}",
                )
            return ManagementCommandResult(
                handled=True,
                output=f"Foreground permission level: {published}",
            )

    async def resume(self, session_id: str, *, force: bool = False) -> ManagementCommandResult:
        with without_session_log():
            try:
                result = await self._management.resume(session_id, force=force)
            except FatalManagementError:
                raise
            except ManagementError as management_error:
                return ManagementCommandResult(
                    handled=True,
                    output=f"{management_error.error.code}: {management_error.error.message}",
                )
            except Exception as error:
                logger.opt(exception=error).error(
                    "Management command failed command=/resume type={}", type(error).__name__
                )
                raise
            else:
                output = f"Resumed session {result.session_id}."
                return ManagementCommandResult(
                    handled=True,
                    output=output,
                    resumed_session_id=result.session_id,
                )
