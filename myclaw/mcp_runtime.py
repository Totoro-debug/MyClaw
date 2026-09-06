"""Composition surface for the CLI-owned MCP Runtime Lifetime component."""

from myclaw.tools.mcp_runtime import (
    MCPConnectionAdapter,
    MCPConnectionFactory,
    MCPRuntimeManager,
    MCPServerFailure,
    MCPSnapshotReport,
    MCPStartupReport,
    MCPToolSnapshot,
    allocate_mcp_tool_name,
)
from myclaw.tools.tool_gateway import BUILT_IN_TOOL_NAMES

__all__ = [
    "BUILT_IN_TOOL_NAMES",
    "MCPConnectionAdapter",
    "MCPConnectionFactory",
    "MCPRuntimeManager",
    "MCPServerFailure",
    "MCPSnapshotReport",
    "MCPStartupReport",
    "MCPToolSnapshot",
    "allocate_mcp_tool_name",
]
