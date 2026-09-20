"""Run-local Tool authorization contracts.

This module owns the authorization boundary without depending on concrete Tools.  The
legacy safety reason remains an input to the default policy during the expand phase;
later policies can replace it with structured classification facts.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from myclaw.agent.permission import (
    PERMISSION_LEVELS,
    PermissionSnapshot,
    RuntimePermissionControl,
    ToolPermissionLevel,
    validate_permission_level,
)
from myclaw.agent.tools.core.exec_policy import ExecAssessment, ResolvedExecShell

type PermissionDecision = Literal["direct", "confirm"]
type ToolRunOrigin = Literal["foreground", "schedule", "memory"]
type FileAccessRole = Literal["read", "write"]
type ScheduleAction = object
type NetworkAssessment = object
type NormalizedNetworkTarget = object
type IPAddress = str


@dataclass(frozen=True, slots=True)
class FileAccess:
    """One canonical host path access observed by a File Tool."""

    path: Path
    role: FileAccessRole
    base: Path
    workspace_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("File access path must be a Path")
        if self.role not in {"read", "write"}:
            raise ValueError("File access role is invalid")
        if not isinstance(self.base, Path) or not isinstance(self.workspace_root, Path):
            raise TypeError("File access roots must be Paths")


def canonicalize_file_access(
    *,
    workspace: Path,
    base: Path,
    requested: str | Path,
    role: FileAccessRole,
) -> FileAccess:
    """Resolve a model path using host links and the nearest existing ancestor."""
    if not isinstance(workspace, Path) or not isinstance(base, Path):
        raise TypeError("File access workspace and base must be Paths")
    if not isinstance(requested, (str, Path)):
        raise TypeError("File access requested path must be a string or Path")

    workspace_root = workspace.resolve(strict=True)
    base_root = base.resolve(strict=True)
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = base_root / candidate
    candidate = Path(os.path.abspath(str(candidate)))
    canonical = _resolve_with_missing_ancestor(candidate)
    return FileAccess(
        path=canonical,
        role=role,
        base=base_root,
        workspace_root=workspace_root,
    )


def _resolve_with_missing_ancestor(candidate: Path) -> Path:
    missing: list[str] = []
    current = candidate
    while not os.path.lexists(str(current)):
        parent = current.parent
        if parent == current:
            raise OSError(f"could not resolve path ancestor: {candidate}")
        missing.append(current.name)
        current = parent

    resolved = current.resolve(strict=True)
    for name in reversed(missing):
        resolved /= name
    return resolved


@dataclass(frozen=True, slots=True)
class PermissionContext:
    """Run facts supplied to a Tool Permission Policy."""

    level: ToolPermissionLevel | None = None
    configured_schedule_level: ToolPermissionLevel | None = None
    origin: ToolRunOrigin = "foreground"
    workspace_root: Path | None = None
    exec_shell: ResolvedExecShell | None = None
    snapshot: PermissionSnapshot | None = None

    def __post_init__(self) -> None:
        if self.level is not None:
            validate_permission_level(self.level)
        if self.configured_schedule_level is not None:
            validate_permission_level(self.configured_schedule_level)
        if self.origin not in {"foreground", "schedule", "memory"}:
            raise ValueError("Tool run origin is invalid")
        if self.workspace_root is not None and not isinstance(self.workspace_root, Path):
            raise TypeError("Permission context workspace_root must be a Path or None")
        if self.exec_shell is not None and not isinstance(self.exec_shell, ResolvedExecShell):
            raise TypeError("Permission context exec_shell must be resolved or None")
        if self.snapshot is not None:
            if self.level is not None and self.level != self.snapshot.level:
                raise ValueError("Permission context level must match its snapshot")
            snapshot_shell = cast(ResolvedExecShell, self.snapshot.exec_shell)
            if self.exec_shell is not None and self.exec_shell != snapshot_shell:
                raise ValueError("Permission context Exec shell must match its snapshot")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PermissionSnapshot,
        *,
        workspace_root: Path,
        origin: ToolRunOrigin = "foreground",
        configured_schedule_level: ToolPermissionLevel | None = None,
    ) -> PermissionContext:
        if not isinstance(snapshot, PermissionSnapshot):
            raise TypeError("Permission context requires a PermissionSnapshot")
        return cls(
            level=snapshot.level,
            configured_schedule_level=configured_schedule_level,
            origin=origin,
            workspace_root=workspace_root,
            exec_shell=cast(ResolvedExecShell, snapshot.exec_shell),
            snapshot=snapshot,
        )


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

    def confirmation_reason(self) -> str:
        return (
            self._safety_reason
            if self._safety_reason is not None
            else "Tool confirmation is required."
        )

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
        """Classify one detached invocation without retaining mutable run state."""
        if facts.file_accesses and context.origin != "memory":
            if context.origin == "schedule" and context.level is None:
                return _LegacyAuthorizationSession(facts.legacy_safety_reason)
            decision, reason = _classify_file_accesses(facts.file_accesses, context)
            return _FileAuthorizationSession(decision, reason)
        return _LegacyAuthorizationSession(facts.legacy_safety_reason)


class _FileAuthorizationSession:
    def __init__(self, decision: PermissionDecision, reason: str | None = None) -> None:
        self._decision = decision
        self._reason = reason or "Tool confirmation is required."

    def initial_decision(self) -> PermissionDecision:
        return self._decision

    def confirmation_reason(self) -> str:
        return self._reason

    async def authorize_network_target(
        self,
        target: NormalizedNetworkTarget,
        resolved_addresses: tuple[IPAddress, ...],
    ) -> None:
        del target, resolved_addresses


def _classify_file_accesses(
    accesses: tuple[FileAccess, ...],
    context: PermissionContext,
) -> tuple[PermissionDecision, str | None]:
    level = context.level
    has_external = any(
        not _is_host_path_within(access.path, access.workspace_root) for access in accesses
    )
    has_write = any(access.role == "write" for access in accesses)

    if level == "full-access":
        return "direct", None
    if level == "read-only" and has_write:
        return "confirm", "Write access requires confirmation in read-only mode."
    if has_external:
        return "confirm", (
            "The requested path resolves outside the Workspace and requires confirmation."
        )
    if level is None:
        return "direct", None
    return "direct", None


def _is_host_path_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.path.normcase(str(path)), os.path.normcase(str(root)))
        ) == os.path.normcase(str(root))
    except ValueError:
        return False


__all__ = [
    "PERMISSION_LEVELS",
    "ExecAssessment",
    "FileAccess",
    "FileAccessRole",
    "IPAddress",
    "NetworkAssessment",
    "NormalizedNetworkTarget",
    "PermissionContext",
    "PermissionDecision",
    "PermissionSnapshot",
    "ResolvedExecShell",
    "RuntimePermissionControl",
    "ScheduleAction",
    "ToolAuthorizationSession",
    "ToolInvocationFacts",
    "ToolPermissionLevel",
    "ToolPermissionPolicy",
    "ToolRunOrigin",
    "canonicalize_file_access",
]
