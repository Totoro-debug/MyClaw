"""Runtime-Lifetime state for configured MCP Server connections."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from myclaw.config.config import MCPServerConfiguration
from myclaw.tools.base import BaseTool
from myclaw.tools.mcp import MCPServerConnection, MCPTool
from myclaw.utils.async_tasks import await_task_preserving_cancellation

_MAX_MODEL_TOOL_NAME_LENGTH = 64
_MODEL_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

type MCPToolSnapshot = tuple[MCPTool, ...]


class MCPConnectionAdapter(Protocol):
    """The connection seam owned by the MCP Runtime Manager."""

    @property
    def tools(self) -> tuple[MCPTool, ...]: ...

    @property
    def unavailable(self) -> bool: ...

    async def connect(self) -> tuple[MCPTool, ...]: ...

    async def close(self) -> None: ...


type MCPConnectionFactory = Callable[[MCPServerConfiguration, Path], MCPConnectionAdapter]


@dataclass(frozen=True, slots=True)
class MCPStartupReport:
    """The initial generation's connected Servers and MCP Tool Snapshot."""

    snapshot: MCPToolSnapshot
    connected_servers: tuple[str, ...]
    failed_servers: tuple[str, ...]

    @property
    def tools(self) -> MCPToolSnapshot:
        """Return the generation's MCP Tool Snapshot."""
        return self.snapshot


@dataclass(frozen=True, slots=True)
class MCPSnapshotReport:
    """A candidate MCP Tool Snapshot prepared for a later generation."""

    snapshot: MCPToolSnapshot
    reused_servers: tuple[str, ...]
    retried_servers: tuple[str, ...]
    failed_servers: tuple[str, ...]

    @property
    def tools(self) -> MCPToolSnapshot:
        """Return the candidate generation's MCP Tool Snapshot."""
        return self.snapshot


def allocate_mcp_tool_name(mcp_name: str, remote_name: str) -> str | None:
    """Allocate the documented provider-safe name for one remote Tool.

    The fallback is considered only when the preferred name exceeds the provider's
    maximum length. Invalid candidates are rejected without character replacement
    or truncation; collisions are handled by the Runtime Manager.
    """
    if not isinstance(mcp_name, str) or not mcp_name:
        return None
    if not isinstance(remote_name, str) or not remote_name:
        return None

    preferred = f"mcp_{mcp_name}_{remote_name}"
    candidate = f"mcp_{remote_name}" if len(preferred) > _MAX_MODEL_TOOL_NAME_LENGTH else preferred
    return candidate if _MODEL_TOOL_NAME_PATTERN.fullmatch(candidate) else None


class MCPRuntimeManager:
    """Connect configured MCP Servers and prepare immutable generation snapshots."""

    def __init__(
        self,
        workspace: Path,
        *,
        connection_factory: MCPConnectionFactory | None = None,
        built_in_tools: Sequence[BaseTool] = (),
        built_in_names: Iterable[str] = (),
    ) -> None:
        if not isinstance(workspace, Path):
            raise TypeError("MCP Runtime Manager requires a Path workspace")
        self._workspace = workspace
        self._connection_factory = connection_factory or _default_connection_factory
        self._built_in_tools = tuple(built_in_tools)
        self._built_in_names = _validate_built_in_names(
            self._built_in_tools,
            built_in_names,
        )
        self._configuration: dict[str, MCPServerConfiguration] = {}
        self._connections: dict[str, MCPConnectionAdapter] = {}
        self._discovered_tools: dict[str, tuple[MCPTool, ...]] = {}
        self._connected_servers: set[str] = set()
        self._failed_servers: set[str] = set()
        self._snapshot: MCPToolSnapshot = ()
        self._pending_report: MCPSnapshotReport | None = None
        self._started = False

    @property
    def snapshot(self) -> MCPToolSnapshot:
        """Return the active generation's immutable MCP Tool Snapshot."""
        return self._snapshot

    @property
    def mcp_snapshot(self) -> MCPToolSnapshot:
        """Alias for callers that distinguish the MCP snapshot from a full catalog."""
        return self._snapshot

    @property
    def catalog(self) -> tuple[BaseTool, ...]:
        """Return Built-in Tools followed by the active MCP Tool Snapshot."""
        return self._built_in_tools + self._snapshot

    @property
    def configuration(self) -> Mapping[str, MCPServerConfiguration]:
        """Return the configured Server table without exposing mutable manager state."""
        return MappingProxyType(dict(self._configuration))

    @property
    def connections(self) -> Mapping[str, MCPConnectionAdapter]:
        """Return the Runtime-Lifetime connection table as a read-only mapping."""
        return MappingProxyType(dict(self._connections))

    @property
    def failed_servers(self) -> tuple[str, ...]:
        """Return Servers selected for retry, in deterministic order."""
        return tuple(sorted(self._failed_servers))

    @property
    def healthy_servers(self) -> tuple[str, ...]:
        """Return connected Servers not selected for retry, in deterministic order."""
        return tuple(sorted(self._connected_servers - self._failed_servers))

    @property
    def started(self) -> bool:
        """Whether the Manager owns a started Runtime Lifetime state."""
        return self._started

    async def start(
        self,
        configuration: Mapping[str, MCPServerConfiguration],
    ) -> MCPStartupReport:
        """Connect enabled Servers and activate the initial generation snapshot."""
        normalized = _normalize_configuration(configuration)
        if self._started or self._connections:
            await self.close()
        self._pending_report = None

        connections: dict[str, MCPConnectionAdapter] = {}
        failed: set[str] = set()
        for mcp_name, server_configuration in normalized.items():
            if not server_configuration.enabled:
                continue
            try:
                connections[mcp_name] = self._connection_factory(
                    server_configuration,
                    self._workspace,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                failed.add(mcp_name)

        try:
            attempts = await self._connect_many(connections)
        except BaseException:
            await _close_connections(connections.values())
            raise

        discovered: dict[str, tuple[MCPTool, ...]] = {}
        connected: set[str] = set()
        for mcp_name, tools in attempts.items():
            if tools is None:
                failed.add(mcp_name)
                continue
            discovered[mcp_name] = tools
            connected.add(mcp_name)

        self._configuration = normalized
        self._connections = connections
        self._discovered_tools = discovered
        self._connected_servers = connected
        self._failed_servers = failed
        self._started = True
        self._snapshot = self._build_snapshot()
        return MCPStartupReport(
            snapshot=self._snapshot,
            connected_servers=tuple(sorted(connected)),
            failed_servers=self.failed_servers,
        )

    async def prepare_generation(self) -> MCPSnapshotReport:
        """Prepare a candidate snapshot while retaining the active snapshot.

        Healthy connections and their discovered Tool definitions are reused. Only
        Servers marked unavailable, or whose previous connection attempt failed,
        are connected again. The caller chooses when to activate the returned
        candidate through :meth:`activate_generation`.
        """
        if not self._started:
            raise RuntimeError("MCP Runtime Manager has not been started")
        self._pending_report = None

        self._sync_unavailable_servers()
        retry_names = tuple(sorted(self._failed_servers))
        retry_connections: dict[str, MCPConnectionAdapter] = {}
        for mcp_name in retry_names:
            connection = self._connections.get(mcp_name)
            if connection is not None:
                retry_connections[mcp_name] = connection
                continue
            server_configuration = self._configuration[mcp_name]
            try:
                connection = self._connection_factory(server_configuration, self._workspace)
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            self._connections[mcp_name] = connection
            retry_connections[mcp_name] = connection

        attempts = await self._connect_many(retry_connections)

        for mcp_name in retry_names:
            tools = attempts.get(mcp_name)
            if tools is None:
                self._failed_servers.add(mcp_name)
                continue
            self._failed_servers.discard(mcp_name)
            self._connected_servers.add(mcp_name)
            self._discovered_tools[mcp_name] = tools

        candidate = self._build_snapshot()
        reused = tuple(
            sorted(
                mcp_name
                for mcp_name in self._connected_servers - self._failed_servers
                if mcp_name not in retry_names
            )
        )
        report = MCPSnapshotReport(
            snapshot=candidate,
            reused_servers=reused,
            retried_servers=retry_names,
            failed_servers=self.failed_servers,
        )
        self._pending_report = report
        return report

    def activate_generation(
        self,
        report: MCPSnapshotReport,
    ) -> MCPToolSnapshot:
        """Publish a previously prepared candidate as the active snapshot."""
        if not isinstance(report, MCPSnapshotReport):
            raise TypeError("MCP generation activation requires an MCP snapshot report")
        if not self._started:
            raise RuntimeError("MCP Runtime Manager has not been started")
        if report is not self._pending_report:
            raise ValueError("MCP snapshot report is not the current prepared candidate")
        self._snapshot = report.snapshot
        self._pending_report = None
        return self._snapshot

    commit_generation = activate_generation

    async def close(self) -> None:
        """Close all Runtime-Lifetime MCP connections and clear Manager state."""
        connections = tuple(self._connections.values())
        self._configuration = {}
        self._connections = {}
        self._discovered_tools = {}
        self._connected_servers = set()
        self._failed_servers = set()
        self._snapshot = ()
        self._pending_report = None
        self._started = False
        await _close_connections(connections)

    async def _connect_many(
        self,
        connections: Mapping[str, MCPConnectionAdapter],
    ) -> dict[str, tuple[MCPTool, ...] | None]:
        async def connect_one(
            mcp_name: str,
            connection: MCPConnectionAdapter,
        ) -> tuple[str, tuple[MCPTool, ...] | None]:
            try:
                tools = await connection.connect()
                if not isinstance(tools, (tuple, list)):
                    raise TypeError("MCP Server connection returned an invalid Tool collection")
                if connection.unavailable:
                    raise RuntimeError("MCP Server connection became unavailable during connect")
                return mcp_name, tuple(tools)
            except asyncio.CancelledError:
                raise
            except Exception:
                return mcp_name, None

        tasks = tuple(
            asyncio.create_task(connect_one(mcp_name, connections[mcp_name]))
            for mcp_name in sorted(connections)
        )
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return dict(results)

    def _sync_unavailable_servers(self) -> None:
        for mcp_name, connection in self._connections.items():
            if connection.unavailable:
                self._failed_servers.add(mcp_name)

    def _build_snapshot(self) -> MCPToolSnapshot:
        used_names = set(self._built_in_names)
        snapshot: list[MCPTool] = []
        for mcp_name in sorted(self._configuration):
            configuration = self._configuration[mcp_name]
            if not configuration.enabled or mcp_name in self._failed_servers:
                continue
            connection = self._connections.get(mcp_name)
            if connection is None or connection.unavailable:
                continue
            named_tools: list[MCPTool] = []
            for tool in _sorted_tools(self._discovered_tools.get(mcp_name, ())):
                if tool.unavailable:
                    continue
                model_name = allocate_mcp_tool_name(mcp_name, tool.remote_name)
                if model_name is None:
                    continue
                named_tool = tool if tool.name == model_name else tool.with_model_name(model_name)
                named_tools.append(named_tool)
                if model_name in used_names:
                    continue
                used_names.add(model_name)
                snapshot.append(named_tool)
            self._discovered_tools[mcp_name] = tuple(named_tools)
        return tuple(snapshot)


def _default_connection_factory(
    configuration: MCPServerConfiguration,
    workspace: Path,
) -> MCPServerConnection:
    return MCPServerConnection(
        configuration,
        workspace,
        model_name_for=lambda remote_name: (
            allocate_mcp_tool_name(
                configuration.mcp_name,
                remote_name,
            )
            or ""
        ),
    )


def _normalize_configuration(
    configuration: Mapping[str, MCPServerConfiguration],
) -> dict[str, MCPServerConfiguration]:
    if not isinstance(configuration, Mapping):
        raise TypeError("MCP Runtime Manager configuration must be a mapping")
    normalized: dict[str, MCPServerConfiguration] = {}
    for server_configuration in configuration.values():
        if not isinstance(server_configuration, MCPServerConfiguration):
            raise TypeError(
                "MCP Runtime Manager configuration values must be MCPServerConfiguration"
            )
        mcp_name = server_configuration.mcp_name
        if mcp_name in normalized:
            raise ValueError(f"Duplicate MCP Server name: {mcp_name}")
        normalized[mcp_name] = server_configuration
    return normalized


def _validate_built_in_names(
    built_in_tools: tuple[BaseTool, ...],
    built_in_names: Iterable[str],
) -> frozenset[str]:
    names = list(built_in_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Built-in Tool names must be non-empty strings")
    for tool in built_in_tools:
        if not isinstance(tool, BaseTool):
            raise TypeError("MCP Runtime Manager built-in Tools must be BaseTool instances")
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Built-in Tool names must be non-empty strings")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Built-in Tool names must be unique")
    return frozenset(names)


def _sorted_tools(tools: Sequence[MCPTool]) -> tuple[MCPTool, ...]:
    valid_tools = tuple(tool for tool in tools if isinstance(tool, MCPTool))
    return tuple(
        sorted(
            valid_tools,
            key=lambda tool: (
                tool.remote_name,
                tool.description,
                tool.name,
                json.dumps(tool.parameters, sort_keys=True, separators=(",", ":")),
            ),
        )
    )


async def _close_connections(connections: Iterable[MCPConnectionAdapter]) -> None:
    async def close_all() -> None:
        results = await asyncio.gather(
            *(connection.close() for connection in connections),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("MCP connection cleanup failed", errors)

    cleanup_task = asyncio.create_task(close_all())
    await await_task_preserving_cancellation(cleanup_task)


__all__ = [
    "MCPConnectionAdapter",
    "MCPConnectionFactory",
    "MCPRuntimeManager",
    "MCPSnapshotReport",
    "MCPStartupReport",
    "MCPToolSnapshot",
    "allocate_mcp_tool_name",
]
