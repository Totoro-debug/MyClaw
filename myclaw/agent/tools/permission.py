"""Run-local Tool authorization contracts.

This module owns the authorization boundary without depending on concrete Tools.  The
legacy safety reason remains an input to the default policy during the expand phase;
later policies can replace it with structured classification facts.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

type PermissionDecision = Literal["direct", "confirm"]
type ToolPermissionLevel = Literal["read-only", "workspace-write", "full-access"]
type ToolRunOrigin = Literal["foreground", "schedule", "memory"]
type ResolvedExecShell = str
type FileAccess = object
type ExecAssessment = object
type ScheduleAction = object
type NetworkAssessment = object
type NormalizedNetworkTarget = object
type IPAddress = str


@dataclass(frozen=True, slots=True)
class PermissionSnapshot:
    """Immutable permission values captured for one Agent Run."""

    level: ToolPermissionLevel
    exec_shell: ResolvedExecShell


@dataclass(frozen=True, slots=True)
class PermissionContext:
    """Run facts supplied to a Tool Permission Policy."""

    level: ToolPermissionLevel | None = None
    configured_schedule_level: ToolPermissionLevel | None = None
    origin: ToolRunOrigin = "foreground"
    workspace_root: Path | None = None
    exec_shell: ResolvedExecShell | None = None


@dataclass(frozen=True, slots=True, init=False)
class ToolInvocationFacts:
    """Detached facts for one normalized Tool invocation."""

    tool_name: str
    _normalized_arguments: dict[str, Any] = field(repr=False)
    legacy_safety_reason: str | None
    file_accesses: tuple[FileAccess, ...]
    exec_assessment: ExecAssessment | None
    schedule_action: ScheduleAction | None
    network_targets: tuple[NetworkAssessment, ...]

    def __init__(
        self,
        tool_name: str,
        normalized_arguments: Mapping[str, Any],
        *,
        legacy_safety_reason: str | None = None,
        file_accesses: tuple[FileAccess, ...] = (),
        exec_assessment: ExecAssessment | None = None,
        schedule_action: ScheduleAction | None = None,
        network_targets: tuple[NetworkAssessment, ...] = (),
    ) -> None:
        if not isinstance(tool_name, str) or not tool_name:
            raise TypeError("Tool invocation tool_name must be a non-empty string")
        if not isinstance(normalized_arguments, Mapping):
            raise TypeError("Tool invocation normalized_arguments must be a mapping")
        if any(not isinstance(name, str) for name in normalized_arguments):
            raise TypeError("Tool invocation argument names must be strings")
        if legacy_safety_reason is not None and not isinstance(legacy_safety_reason, str):
            raise TypeError("Tool invocation legacy_safety_reason must be a string or None")
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "_normalized_arguments", deepcopy(dict(normalized_arguments)))
        object.__setattr__(self, "legacy_safety_reason", legacy_safety_reason)
        object.__setattr__(self, "file_accesses", tuple(deepcopy(file_accesses)))
        object.__setattr__(self, "exec_assessment", deepcopy(exec_assessment))
        object.__setattr__(self, "schedule_action", deepcopy(schedule_action))
        object.__setattr__(self, "network_targets", tuple(deepcopy(network_targets)))

    @property
    def normalized_arguments(self) -> dict[str, Any]:
        """Return a detached view of the normalized argument object."""
        return deepcopy(self._normalized_arguments)

    @property
    def safety_reason(self) -> str | None:
        """Expose the legacy reason while the expand-phase bridge is active."""
        return self.legacy_safety_reason


class ToolAuthorizationSession(Protocol):
    """One authorization state machine owned by one Gateway call."""

    def initial_decision(self) -> PermissionDecision: ...

    async def authorize_network_target(
        self,
        target: NormalizedNetworkTarget,
        resolved_addresses: tuple[IPAddress, ...],
    ) -> None: ...


class _LegacyAuthorizationSession:
    def __init__(self, safety_reason: str | None) -> None:
        self._safety_reason = safety_reason

    def initial_decision(self) -> PermissionDecision:
        return "confirm" if self._safety_reason is not None else "direct"

    async def authorize_network_target(
        self,
        target: NormalizedNetworkTarget,
        resolved_addresses: tuple[IPAddress, ...],
    ) -> None:
        del target, resolved_addresses


class ToolPermissionPolicy:
    """Map normalized invocation facts to a per-call authorization session."""

    def open(
        self,
        facts: ToolInvocationFacts,
        context: PermissionContext,
    ) -> ToolAuthorizationSession:
        """Bridge the existing safety-reason behavior during contract expansion."""
        del context
        return _LegacyAuthorizationSession(facts.legacy_safety_reason)


__all__ = [
    "ExecAssessment",
    "FileAccess",
    "IPAddress",
    "NetworkAssessment",
    "NormalizedNetworkTarget",
    "PermissionContext",
    "PermissionDecision",
    "PermissionSnapshot",
    "ResolvedExecShell",
    "ScheduleAction",
    "ToolAuthorizationSession",
    "ToolInvocationFacts",
    "ToolPermissionLevel",
    "ToolPermissionPolicy",
    "ToolRunOrigin",
]
