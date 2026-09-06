"""MCP SDK adapter for one configured Server and its discovered Tools."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

import mcp.types as types
from loguru import logger
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import (  # type: ignore[attr-defined]
    create_mcp_http_client,
    streamable_http_client,
)
from mcp.shared.exceptions import MCPError

from myclaw.config.config import MCPServerConfiguration
from myclaw.tools.base import BaseTool, ToolError

_MISSING = object()
_NO_OUTPUT = "(no output)"
_CONNECTION_UNAVAILABLE = "MCP Server connection is unavailable."
_TIMEOUT_TEMPLATE = "MCP Tool call timed out after {timeout:g} seconds."


class MCPToolSession(Protocol):
    """The call surface required by one MCPTool."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object: ...


class MCPDiscoverySession(Protocol):
    """The discovery surface required by the paginated tools/list adapter."""

    async def list_tools(self, *, params: types.PaginatedRequestParams | None = None) -> object: ...


class MCPClientSession(MCPToolSession, MCPDiscoverySession, Protocol):
    """The session surface required to connect and discover one MCP Server."""

    async def initialize(self) -> object: ...


class MCPToolSchemaError(ValueError):
    """Raised when a discovered MCP Tool cannot enter the local Tool contract."""


@dataclass(frozen=True, slots=True)
class MCPToolSpec:
    """The validated, model-facing projection of one remote MCP Tool."""

    server_name: str
    remote_name: str
    model_name: str
    description: str
    parameters: dict[str, Any]

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.server_name, "server_name"),
            (self.remote_name, "remote_name"),
            (self.model_name, "model_name"),
            (self.description, "description"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"MCP Tool {field_name} must be a non-empty string")
        _validate_input_schema(self.parameters)
        object.__setattr__(self, "parameters", deepcopy(self.parameters))


class MCPTool(BaseTool):
    """Expose one discovered MCP Tool through the common BaseTool boundary."""

    parameters: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        spec: MCPToolSpec,
        session: MCPToolSession,
        *,
        call_timeout: float = 60,
        on_closed: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(spec, MCPToolSpec):
            raise TypeError("MCPTool requires an MCPToolSpec")
        if session is None:
            raise TypeError("MCPTool requires an MCP client session")
        _validate_timeout(call_timeout, field="call_timeout")
        self.server_name = spec.server_name
        self.remote_name = spec.remote_name
        self.name = spec.model_name
        self.description = spec.description
        object.__setattr__(self, "parameters", deepcopy(spec.parameters))
        self._session = session
        self._call_timeout = float(call_timeout)
        self._on_closed = on_closed
        self._unavailable = False

    @classmethod
    def from_remote_tool(
        cls,
        remote_tool: object,
        *,
        server_name: str,
        model_name: str,
        session: MCPToolSession,
        call_timeout: float = 60,
        on_closed: Callable[[], None] | None = None,
    ) -> MCPTool:
        """Build a Tool from one SDK Tool after validating its input schema."""
        spec = mcp_tool_spec_from_remote(
            remote_tool,
            server_name=server_name,
            model_name=model_name,
        )
        return cls(spec, session, call_timeout=call_timeout, on_closed=on_closed)

    @property
    def unavailable(self) -> bool:
        """Whether a closed MCP session was observed during a Tool call."""
        return self._unavailable

    def with_model_name(self, model_name: str) -> MCPTool:
        """Return the same remote Tool with a different model-facing name."""
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("MCP Tool model name must be a non-empty string")
        return type(self)(
            MCPToolSpec(
                server_name=self.server_name,
                remote_name=self.remote_name,
                model_name=model_name,
                description=self.description,
                parameters=self.parameters,
            ),
            self._session,
            call_timeout=self._call_timeout,
            on_closed=self._on_closed,
        )

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Forward the complete argument object without local schema processing."""
        return deepcopy(arguments)

    async def execute_prepared(self, arguments: dict[str, Any]) -> str:
        """Call the remote Tool and project its result to text."""
        try:
            async with asyncio.timeout(self._call_timeout):
                result = await self._session.call_tool(self.remote_name, deepcopy(arguments))
        except asyncio.CancelledError:
            raise
        except TimeoutError as error:
            raise ToolError(_TIMEOUT_TEMPLATE.format(timeout=self._call_timeout)) from error
        except Exception as error:
            if _is_closed_session_error(error):
                self._mark_unavailable()
                raise ToolError(_CONNECTION_UNAVAILABLE) from error
            raise

        content = _content_text(result)
        if _result_is_error(result):
            raise ToolError(content)
        return content

    def _mark_unavailable(self) -> None:
        if self._unavailable:
            return
        self._unavailable = True
        if self._on_closed is not None:
            self._on_closed()


def normalize_nullable(value: Any) -> Any:
    """Normalize the MCP ``nullable`` extension recursively for model schemas."""
    if isinstance(value, dict):
        nullable = value.get("nullable") is True
        normalized: dict[Any, Any] = {
            key: normalize_nullable(item) for key, item in value.items() if key != "nullable"
        }
        if not nullable:
            return normalized

        type_value = normalized.get("type", _MISSING)
        if isinstance(type_value, str):
            normalized["type"] = [type_value] if type_value == "null" else [type_value, "null"]
            return normalized
        if isinstance(type_value, list):
            if "null" not in type_value:
                normalized["type"] = [*type_value, "null"]
            return normalized
        if type_value is _MISSING:
            return {"anyOf": [normalized, {"type": "null"}]}
        return normalized
    if isinstance(value, list):
        return [normalize_nullable(item) for item in value]
    return deepcopy(value)


def mcp_tool_spec_from_remote(
    remote_tool: object,
    *,
    server_name: str,
    model_name: str,
) -> MCPToolSpec:
    """Validate and project one SDK or fake Tool object into an ``MCPToolSpec``."""
    remote_name = _field(remote_tool, "name")
    if not isinstance(remote_name, str) or not remote_name:
        raise MCPToolSchemaError("MCP Tool name must be a non-empty string")

    input_schema = _field(remote_tool, "input_schema", "inputSchema")
    if not isinstance(input_schema, dict):
        raise MCPToolSchemaError("MCP Tool inputSchema must be a dictionary")
    _validate_json_serializable(input_schema)
    if input_schema.get("type") != "object":
        raise MCPToolSchemaError("MCP Tool inputSchema must use an object root")

    normalized = normalize_nullable(input_schema)
    if not isinstance(normalized, dict) or normalized.get("type") != "object":
        raise MCPToolSchemaError("MCP Tool inputSchema must use an object root")
    _validate_json_serializable(normalized)

    description = _field(remote_tool, "description")
    if description is _MISSING or description is None or description == "":
        description = remote_name
    if not isinstance(description, str):
        raise MCPToolSchemaError("MCP Tool description must be a string")

    return MCPToolSpec(
        server_name=server_name,
        remote_name=remote_name,
        model_name=model_name,
        description=description,
        parameters=cast(dict[str, Any], normalized),
    )


async def discover_mcp_tool_specs(
    session: MCPDiscoverySession,
    *,
    server_name: str,
    model_name_for: Callable[[str], str] | None = None,
    timeout: float | None = None,
    on_tool_skipped: Callable[[], None] | None = None,
) -> tuple[MCPToolSpec, ...]:
    """Discover every page of Tools, stopping safely on a repeated cursor."""
    if not isinstance(server_name, str) or not server_name:
        raise ValueError("MCP server_name must be a non-empty string")
    if timeout is not None:
        _validate_timeout(timeout, field="timeout")

    async def collect() -> tuple[MCPToolSpec, ...]:
        cursor: str | None = None
        seen_cursors: set[str] = set()
        specs: list[MCPToolSpec] = []
        while True:
            page = await _list_tools_page(session, cursor)
            tools = _field(page, "tools")
            if not isinstance(tools, (list, tuple)):
                raise TypeError("MCP tools/list returned an invalid tools collection")
            for remote_tool in tools:
                try:
                    remote_name = _field(remote_tool, "name")
                    if not isinstance(remote_name, str) or not remote_name:
                        raise MCPToolSchemaError("MCP Tool name must be a non-empty string")
                    allocator = model_name_for or (lambda name: name)
                    model_name = allocator(remote_name)
                    if not isinstance(model_name, str) or not model_name:
                        raise MCPToolSchemaError("MCP Tool model name must be a non-empty string")
                    specs.append(
                        mcp_tool_spec_from_remote(
                            remote_tool,
                            server_name=server_name,
                            model_name=model_name,
                        )
                    )
                except MCPToolSchemaError:
                    if on_tool_skipped is not None:
                        on_tool_skipped()
                    continue

            next_cursor = _field(page, "next_cursor", "nextCursor")
            if next_cursor is _MISSING or next_cursor is None:
                break
            if not isinstance(next_cursor, str):
                raise TypeError("MCP tools/list returned an invalid cursor")
            if next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        specs.sort(key=lambda item: item.remote_name)
        return tuple(specs)

    if timeout is None:
        return await collect()
    async with asyncio.timeout(float(timeout)):
        return await collect()


async def discover_mcp_tools(
    session: MCPClientSession,
    *,
    server_name: str,
    model_name_for: Callable[[str], str] | None = None,
    call_timeout: float = 60,
    timeout: float | None = None,
    on_closed: Callable[[], None] | None = None,
    on_tool_skipped: Callable[[], None] | None = None,
) -> tuple[MCPTool, ...]:
    """Discover and wrap one Server's valid Tools."""
    specs = await discover_mcp_tool_specs(
        session,
        server_name=server_name,
        model_name_for=model_name_for,
        timeout=timeout,
        on_tool_skipped=on_tool_skipped,
    )
    return tuple(
        MCPTool(spec, session, call_timeout=call_timeout, on_closed=on_closed) for spec in specs
    )


class MCPServerConnection:
    """Own one SDK transport/session and expose its current discovered Tool set."""

    def __init__(
        self,
        configuration: MCPServerConfiguration,
        workspace: Path,
        *,
        transport_factory: Callable[
            [MCPServerConfiguration, Path], AbstractAsyncContextManager[object]
        ]
        | None = None,
        session_factory: Callable[[object, object], MCPClientSession] | None = None,
        model_name_for: Callable[[str], str] | None = None,
    ) -> None:
        if not isinstance(configuration, MCPServerConfiguration):
            raise TypeError("MCP Server connection requires MCPServerConfiguration")
        if not isinstance(workspace, Path):
            raise TypeError("MCP Server connection requires a Path workspace")
        self.configuration = configuration
        self.workspace = workspace
        self._transport_factory = transport_factory or _transport_for
        self._session_factory = session_factory or _new_client_session
        self._model_name_for = model_name_for
        self._stack: AsyncExitStack | None = None
        self._session: MCPClientSession | None = None
        self._tools: tuple[MCPTool, ...] = ()
        self._unavailable = False
        self._skipped_tool_count = 0

    @property
    def session(self) -> MCPClientSession | None:
        return self._session

    @property
    def tools(self) -> tuple[MCPTool, ...]:
        return self._tools

    @property
    def unavailable(self) -> bool:
        return self._unavailable

    @property
    def skipped_tool_count(self) -> int:
        """Return the number of invalid remote Tools skipped during discovery."""
        return self._skipped_tool_count

    async def connect(self) -> tuple[MCPTool, ...]:
        """Open transport/session, initialize, and discover a deterministic Tool set."""
        if not self.configuration.enabled:
            await self.close()
            return ()
        await self.close()
        self._unavailable = False
        self._skipped_tool_count = 0
        stack = AsyncExitStack()
        self._stack = stack
        try:
            async with asyncio.timeout(float(self.configuration.connect_timeout)):
                transport = self._transport_factory(self.configuration, self.workspace)
                streams = await stack.enter_async_context(transport)
                read_stream, write_stream = _transport_streams(streams)
                session = self._session_factory(read_stream, write_stream)
                entered_session = await stack.enter_async_context(cast(Any, session))
                self._session = cast(MCPClientSession, entered_session)
                await self._session.initialize()
                self._tools = await discover_mcp_tools(
                    self._session,
                    server_name=self.configuration.mcp_name,
                    model_name_for=self._model_name_for,
                    call_timeout=self.configuration.call_timeout,
                    on_closed=self._mark_unavailable,
                    on_tool_skipped=self._record_tool_skip,
                )
        except BaseException:
            self._session = None
            self._tools = ()
            await stack.aclose()
            self._stack = None
            raise
        return self._tools

    async def close(self) -> None:
        """Close the session and transport, releasing all SDK resources."""
        stack = self._stack
        self._stack = None
        self._session = None
        self._tools = ()
        if stack is not None:
            await stack.aclose()

    def _mark_unavailable(self) -> None:
        self._unavailable = True

    def _record_tool_skip(self) -> None:
        self._skipped_tool_count += 1
        logger.error(
            "MCP Tool skipped mcp_name={} phase=discovery type=MCPToolSchemaError",
            self.configuration.mcp_name,
        )


MCPServerAdapter = MCPServerConnection


@asynccontextmanager
async def stdio_transport(
    configuration: MCPServerConfiguration,
    workspace: Path,
) -> AsyncIterator[object]:
    """Create the SDK stdio transport for one configured Server."""
    command = configuration.command
    if command is None:
        raise ValueError("stdio MCP Server requires a command")
    parameters = StdioServerParameters(
        command=command,
        args=list(configuration.args),
        env=dict(os.environ),
        cwd=configuration.resolve_cwd(workspace),
    )
    async with stdio_client(parameters) as streams:
        yield streams


@asynccontextmanager
async def streamable_http_transport(
    configuration: MCPServerConfiguration,
    workspace: Path,
) -> AsyncIterator[object]:
    """Create the SDK Streamable HTTP transport with configured static headers."""
    del workspace
    url = configuration.url
    if url is None:
        raise ValueError("streamable-http MCP Server requires a URL")
    http_client = create_mcp_http_client(headers=dict(configuration.headers))
    try:
        async with streamable_http_client(url, http_client=http_client) as streams:
            yield streams
    finally:
        await http_client.aclose()


def _transport_for(
    configuration: MCPServerConfiguration,
    workspace: Path,
) -> AbstractAsyncContextManager[object]:
    if configuration.transport == "stdio":
        return stdio_transport(configuration, workspace)
    if configuration.transport == "streamable-http":
        return streamable_http_transport(configuration, workspace)
    raise ValueError(f"Unsupported MCP transport: {configuration.transport}")


def _new_client_session(read_stream: object, write_stream: object) -> MCPClientSession:
    client_session = cast(Any, ClientSession)
    return cast(MCPClientSession, client_session(read_stream, write_stream))


async def _list_tools_page(session: MCPDiscoverySession, cursor: str | None) -> object:
    params = None if cursor is None else types.PaginatedRequestParams(cursor=cursor)
    return await session.list_tools(params=params)


def _transport_streams(streams: object) -> tuple[object, object]:
    if not isinstance(streams, (tuple, list)) or len(streams) < 2:
        raise TypeError("MCP transport factory must yield read and write streams")
    return streams[0], streams[1]


def _field(value: object, *names: str) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return _MISSING
    for name in names:
        candidate = getattr(value, name, _MISSING)
        if candidate is not _MISSING:
            return candidate
    return _MISSING


def _validate_input_schema(schema: object) -> None:
    if not isinstance(schema, dict):
        raise MCPToolSchemaError("MCP Tool inputSchema must be a dictionary")
    _validate_json_serializable(schema)
    if schema.get("type") != "object":
        raise MCPToolSchemaError("MCP Tool inputSchema must use an object root")


def _validate_json_serializable(value: object) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise MCPToolSchemaError("MCP Tool inputSchema must be JSON serializable") from error


def _validate_timeout(value: float, *, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        or not isfinite(float(value))
    ):
        raise ValueError(f"MCP {field} must be a positive number")


def _content_text(result: object) -> str:
    content = _field(result, "content")
    if not isinstance(content, (list, tuple)):
        raise TypeError("MCP Tool result content must be a sequence")
    parts = [
        block.text if isinstance(block, types.TextContent) else str(block) for block in content
    ]
    return "\n".join(parts) or _NO_OUTPUT


def _result_is_error(result: object) -> bool:
    value = _field(result, "is_error", "isError")
    return value is True


def _is_closed_session_error(error: Exception) -> bool:
    return isinstance(error, MCPError) and error.code == types.CONNECTION_CLOSED


__all__ = [
    "MCPClientSession",
    "MCPDiscoverySession",
    "MCPServerAdapter",
    "MCPServerConnection",
    "MCPTool",
    "MCPToolSchemaError",
    "MCPToolSession",
    "MCPToolSpec",
    "discover_mcp_tool_specs",
    "discover_mcp_tools",
    "mcp_tool_spec_from_remote",
    "normalize_nullable",
    "stdio_transport",
    "streamable_http_transport",
]
