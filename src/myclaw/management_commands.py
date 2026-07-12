"""Standalone dispatch for read-only Management Commands."""

import json
from dataclasses import dataclass
from typing import Protocol

from myclaw.contracts.management import ConfigView, RuntimeStatus
from myclaw.management import ManagementError


class _ManagementViewPort(Protocol):
    async def config_view(self) -> ConfigView: ...

    async def status(self) -> RuntimeStatus: ...

    async def memory_view(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ManagementCommandResult:
    """Renderable output and whether a Management Command was recognized."""

    handled: bool
    output: str | None


class ManagementCommandDispatcher:
    """Dispatch exact built-in commands without entering conversation flow."""

    def __init__(self, management: _ManagementViewPort) -> None:
        self._management = management

    async def dispatch(self, command: str) -> ManagementCommandResult:
        """Return rendered output for a recognized Management Command."""
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
