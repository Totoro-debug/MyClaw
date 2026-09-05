from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mcp.types as types
import pytest
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

import myclaw.tools.mcp as mcp_adapter
from myclaw.config.config import MCPServerConfiguration
from myclaw.tools.base import ToolError
from myclaw.tools.mcp import (
    MCPServerConnection,
    MCPTool,
    MCPToolSchemaError,
    MCPToolSpec,
    discover_mcp_tool_specs,
    mcp_tool_spec_from_remote,
    normalize_nullable,
)
from myclaw.tools.tool_gateway import ModelToolCall, ToolGateway


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        return CallToolResult(content=[TextContent(text="ok")])


@pytest.mark.asyncio
async def test_mcp_tool_forwards_the_complete_argument_object_through_gateway() -> None:
    session = _Session()
    tool = MCPTool(
        MCPToolSpec(
            server_name="local",
            remote_name="echo",
            model_name="mcp_local_echo",
            description="Echo values.",
            parameters={"type": "object", "properties": {}},
        ),
        session,
    )

    arguments = {
        "nested": {"value": "text"},
        "items": [1, None, {"extra": True}],
        "undeclared": "preserved",
    }
    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-1", name="mcp_local_echo", arguments=json.dumps(arguments))
    )

    assert result.status == "success"
    assert result.content == "ok"
    assert session.calls == [("echo", arguments)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"nested": {"key": "value"}},
        {"items": [1, 2, {"nested": True}]},
        {"nullable": None},
        {"additional": {"kept": "as-is"}},
        {"undeclared": "forwarded"},
    ],
)
async def test_mcp_tool_forwards_each_complete_argument_shape(
    arguments: dict[str, Any],
) -> None:
    session = _Session()
    tool = MCPTool(
        MCPToolSpec(
            server_name="local",
            remote_name="echo",
            model_name="mcp_local_echo",
            description="Echo values.",
            parameters={"type": "object", "properties": {}},
        ),
        session,
    )

    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-forward", name=tool.name, arguments=json.dumps(arguments))
    )

    assert result.status == "success"
    assert session.calls == [("echo", arguments)]


def test_mcp_tool_schema_uses_remote_name_when_description_is_missing() -> None:
    spec = mcp_tool_spec_from_remote(
        Tool(name="search", description=None, input_schema={"type": "object"}),
        server_name="remote",
        model_name="mcp_remote_search",
    )

    assert spec.description == "search"
    assert spec.parameters == {"type": "object"}


@pytest.mark.parametrize(
    "schema",
    [
        ["not", "a", "dictionary"],
        None,
        "object",
    ],
)
def test_mcp_tool_loading_rejects_non_dictionary_input_schema(schema: object) -> None:
    with pytest.raises(MCPToolSchemaError, match="dictionary"):
        mcp_tool_spec_from_remote(
            {"name": "invalid", "inputSchema": schema},
            server_name="remote",
            model_name="mcp_remote_invalid",
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"value": object()}},
        {"type": "object", "properties": {"value": {"default": {1, 2}}}},
    ],
)
def test_mcp_tool_loading_rejects_non_json_serializable_input_schema(schema: object) -> None:
    with pytest.raises(MCPToolSchemaError, match="JSON serializable"):
        mcp_tool_spec_from_remote(
            {"name": "invalid", "inputSchema": schema},
            server_name="remote",
            model_name="mcp_remote_invalid",
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
        {"type": ["object", "null"]},
        {"properties": {}},
    ],
)
def test_mcp_tool_loading_rejects_non_object_input_schema(schema: object) -> None:
    with pytest.raises(MCPToolSchemaError, match="object root"):
        mcp_tool_spec_from_remote(
            {"name": "invalid", "inputSchema": schema},
            server_name="remote",
            model_name="mcp_remote_invalid",
        )


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "string", "nullable": True}, {"type": ["string", "null"]}),
        ({"type": "null", "nullable": True}, {"type": ["null"]}),
        (
            {"type": ["string", "integer"], "nullable": True},
            {"type": ["string", "integer", "null"]},
        ),
        (
            {"type": ["string", "null"], "nullable": True},
            {"type": ["string", "null"]},
        ),
        (
            {"nullable": True, "description": "any value"},
            {"anyOf": [{"description": "any value"}, {"type": "null"}]},
        ),
        (
            {"type": "object", "properties": {"value": {"type": "boolean", "nullable": True}}},
            {"type": "object", "properties": {"value": {"type": ["boolean", "null"]}}},
        ),
        (
            {"type": "array", "items": [{"type": "string", "nullable": True}]},
            {"type": "array", "items": [{"type": ["string", "null"]}]},
        ),
        (
            {"type": "object", "properties": {"items": [{"nullable": True}]}},
            {"type": "object", "properties": {"items": [{"anyOf": [{}, {"type": "null"}]}]}},
        ),
        (
            {"anyOf": [{"type": "string"}, {"type": "integer"}], "nullable": True},
            {
                "anyOf": [
                    {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                    {"type": "null"},
                ]
            },
        ),
        (
            {"oneOf": [{"$ref": "#/definitions/value"}], "nullable": True},
            {"anyOf": [{"oneOf": [{"$ref": "#/definitions/value"}]}, {"type": "null"}]},
        ),
        (
            {"$ref": "#/definitions/value", "nullable": True},
            {"anyOf": [{"$ref": "#/definitions/value"}, {"type": "null"}]},
        ),
    ],
)
def test_normalize_nullable_golden_cases(schema: dict[str, Any], expected: dict[str, Any]) -> None:
    assert normalize_nullable(schema) == expected
    assert "nullable" not in json.dumps(normalize_nullable(schema))


class _PageSession:
    def __init__(self, pages: dict[str | None, tuple[list[object], str | None]]) -> None:
        self.pages = pages
        self.cursors: list[str | None] = []

    async def list_tools(self, *, params: types.PaginatedRequestParams | None = None) -> object:
        cursor = None if params is None else params.cursor
        self.cursors.append(cursor)
        tools, next_cursor = self.pages[cursor]
        return {"tools": tools, "nextCursor": next_cursor}


def _remote(name: str) -> Tool:
    return Tool(name=name, input_schema={"type": "object"}, description=name)


@pytest.mark.asyncio
async def test_discovery_collects_three_pages_and_sorts_remote_tools() -> None:
    session = _PageSession(
        {
            None: ([_remote("third")], "page-2"),
            "page-2": ([_remote("first")], "page-3"),
            "page-3": ([_remote("second")], None),
        }
    )

    specs = await discover_mcp_tool_specs(
        session,
        server_name="remote",
        model_name_for=lambda name: f"model_{name}",
    )

    assert session.cursors == [None, "page-2", "page-3"]
    assert [(spec.remote_name, spec.model_name) for spec in specs] == [
        ("first", "model_first"),
        ("second", "model_second"),
        ("third", "model_third"),
    ]


@pytest.mark.asyncio
async def test_discovery_handles_empty_list_without_tools() -> None:
    specs = await discover_mcp_tool_specs(
        _PageSession({None: ([], None)}),
        server_name="remote",
    )

    assert specs == ()


@pytest.mark.asyncio
async def test_discovery_stops_on_repeated_cursor_after_processing_page() -> None:
    session = _PageSession(
        {
            None: ([_remote("first")], "same"),
            "same": ([_remote("second")], "same"),
        }
    )

    specs = await discover_mcp_tool_specs(session, server_name="remote")

    assert [spec.remote_name for spec in specs] == ["first", "second"]
    assert session.cursors == [None, "same"]


@pytest.mark.asyncio
async def test_discovery_skips_invalid_tools_without_dropping_valid_tools() -> None:
    session = _PageSession(
        {
            None: (
                [
                    _remote("valid-first"),
                    {"name": "invalid", "inputSchema": {"type": "string"}},
                ],
                None,
            )
        }
    )

    specs = await discover_mcp_tool_specs(session, server_name="remote")

    assert [spec.remote_name for spec in specs] == ["valid-first"]


def _tool_for_session(
    session: Any,
    *,
    call_timeout: float = 60,
    on_closed: Any = None,
) -> MCPTool:
    return MCPTool(
        MCPToolSpec(
            server_name="remote",
            remote_name="remote_echo",
            model_name="mcp_remote_echo",
            description="Remote echo.",
            parameters={"type": "object"},
        ),
        session,
        call_timeout=call_timeout,
        on_closed=on_closed,
    )


class _ResultSession:
    def __init__(self, result: object) -> None:
        self.result = result

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        del name, arguments
        return self.result


class _StructuredContentMustNotBeRead:
    is_error = False

    def __init__(self, content: list[object]) -> None:
        self.content = content

    @property
    def structured_content(self) -> object:
        raise AssertionError("structured content is outside the text boundary")


@pytest.mark.asyncio
async def test_mcp_result_joins_text_content_in_order() -> None:
    tool = _tool_for_session(
        _ResultSession(
            CallToolResult(content=[TextContent(text="first"), TextContent(text="second")])
        )
    )

    assert await tool.execute_prepared({}) == "first\nsecond"


@pytest.mark.asyncio
async def test_mcp_result_converts_non_text_blocks_and_ignores_structured_content() -> None:
    non_text = ImageContent(data="ZmFrZQ==", mime_type="image/png")
    result = _StructuredContentMustNotBeRead([non_text, TextContent(text="text")])

    class _MixedResultSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            return result

    tool = _tool_for_session(_MixedResultSession())

    assert await tool.execute_prepared({}) == f"{non_text}\ntext"


@pytest.mark.asyncio
async def test_mcp_empty_result_has_explicit_output_marker() -> None:
    tool = _tool_for_session(_ResultSession(CallToolResult(content=[])))

    assert await tool.execute_prepared({}) == "(no output)"


@pytest.mark.asyncio
async def test_mcp_error_result_preserves_server_content_as_tool_error() -> None:
    tool = _tool_for_session(
        _ResultSession(
            CallToolResult(content=[TextContent(text="server rejected it")], is_error=True)
        )
    )

    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-error-result", name=tool.name, arguments="{}")
    )

    assert (result.status, result.content) == ("error", "server rejected it")
    assert tool.unavailable is False


@pytest.mark.asyncio
async def test_mcp_error_result_with_multiple_blocks_preserves_order() -> None:
    tool = _tool_for_session(
        _ResultSession(
            CallToolResult(
                content=[TextContent(text="first"), TextContent(text="second")],
                is_error=True,
            )
        )
    )

    with pytest.raises(ToolError, match="first\\nsecond"):
        await tool.execute_prepared({})


@pytest.mark.asyncio
async def test_mcp_call_timeout_is_a_tool_error_without_marking_connection_unavailable() -> None:
    class _ImmediateTimeoutSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise TimeoutError("transport timeout")

    tool = _tool_for_session(_ImmediateTimeoutSession(), call_timeout=0.01)
    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-timeout", name=tool.name, arguments="{}")
    )

    assert result.status == "error"
    assert "timed out" in result.content
    assert tool.unavailable is False


@pytest.mark.asyncio
async def test_hanging_mcp_call_is_bounded_by_tool_timeout() -> None:
    never = asyncio.Event()

    class _HangingSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            await never.wait()
            return CallToolResult(content=[])

    tool = _tool_for_session(_HangingSession(), call_timeout=0.01)
    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-hanging-timeout", name=tool.name, arguments="{}")
    )

    assert result.status == "error"
    assert result.content == "MCP Tool call timed out after 0.01 seconds."
    assert tool.unavailable is False


@pytest.mark.asyncio
async def test_closed_mcp_session_returns_safe_error_and_marks_connection() -> None:
    class _ClosedSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise MCPError(types.CONNECTION_CLOSED, "private transport detail")

    marked: list[bool] = []
    tool = _tool_for_session(_ClosedSession(), on_closed=lambda: marked.append(True))
    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-closed", name=tool.name, arguments="{}")
    )

    assert (result.status, result.content) == (
        "error",
        "MCP Server connection is unavailable.",
    )
    assert tool.unavailable is True
    assert marked == [True]


@pytest.mark.asyncio
async def test_closed_mcp_session_raises_tool_error_from_direct_execution() -> None:
    class _ClosedSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise MCPError(types.CONNECTION_CLOSED, "private transport detail")

    tool = _tool_for_session(_ClosedSession())

    with pytest.raises(ToolError, match="connection is unavailable"):
        await tool.execute_prepared({})

    assert tool.unavailable is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("private detail"),
        ValueError("private value"),
        MCPError(-32603, "protocol failure"),
        type(
            "ClosedLookalikeError",
            (RuntimeError,),
            {"code": types.CONNECTION_CLOSED},
        )("not an SDK error"),
    ],
)
async def test_other_mcp_exceptions_use_gateway_generic_failure(error: Exception) -> None:
    class _FailingSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise error

    tool = _tool_for_session(_FailingSession())
    result = await ToolGateway._for_memory((tool,)).call(
        ModelToolCall(id="call-generic", name=tool.name, arguments="{}")
    )

    assert result.content == "mcp_remote_echo could not complete the request."
    assert "private" not in result.content
    assert tool.unavailable is False


@pytest.mark.asyncio
async def test_mcp_cancellation_propagates_from_gateway() -> None:
    class _CancelledSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise asyncio.CancelledError

    tool = _tool_for_session(_CancelledSession())

    with pytest.raises(asyncio.CancelledError):
        await ToolGateway._for_memory((tool,)).call(
            ModelToolCall(id="call-cancelled", name=tool.name, arguments="{}")
        )


@pytest.mark.asyncio
async def test_mcp_cancellation_propagates_from_direct_execution() -> None:
    class _CancelledSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
            del name, arguments
            raise asyncio.CancelledError

    tool = _tool_for_session(_CancelledSession())

    with pytest.raises(asyncio.CancelledError):
        await tool.execute_prepared({})


def _server_configuration(
    *,
    transport: str = "stdio",
    command: str = "python",
    args: tuple[str, ...] = (),
    cwd: Any = None,
    url: str | None = None,
    headers: Any = None,
    connect_timeout: float = 1,
    call_timeout: float = 1,
) -> MCPServerConfiguration:
    return MCPServerConfiguration(
        mcp_name="remote",
        enabled=True,
        transport=transport,  # type: ignore[arg-type]
        command=command if transport == "stdio" else None,
        args=args,
        cwd=cwd,
        url=url,
        headers={} if headers is None else headers,
        connect_timeout=connect_timeout,  # type: ignore[arg-type]
        call_timeout=call_timeout,  # type: ignore[arg-type]
    )


class _AsyncTransport:
    def __init__(self) -> None:
        self.entered = False
        self.closed = False

    async def __aenter__(self) -> tuple[str, str]:
        self.entered = True
        return ("read", "write")

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.closed = True


class _ConnectionSession:
    def __init__(self) -> None:
        self.entered = False
        self.closed = False
        self.initialized = False

    async def __aenter__(self) -> _ConnectionSession:
        self.entered = True
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.closed = True

    async def initialize(self) -> object:
        self.initialized = True
        return object()

    async def list_tools(self, *, params: types.PaginatedRequestParams | None = None) -> object:
        del params
        return {"tools": [_remote("echo")], "nextCursor": None}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        del name, arguments
        return CallToolResult(content=[TextContent(text="connected")])


@pytest.mark.asyncio
async def test_server_connection_initializes_discovers_and_closes_one_session() -> None:
    transport = _AsyncTransport()
    session = _ConnectionSession()
    connection = MCPServerConnection(
        _server_configuration(),
        Path("."),
        transport_factory=lambda configuration, workspace: transport,
        session_factory=lambda read_stream, write_stream: session,
        model_name_for=lambda remote_name: f"mcp_remote_{remote_name}",
    )

    tools = await connection.connect()

    assert transport.entered is True
    assert session.initialized is True
    assert [tool.name for tool in tools] == ["mcp_remote_echo"]
    assert connection.tools == tools
    assert connection.session is session

    await connection.close()

    assert session.closed is True
    assert transport.closed is True
    assert connection.session is None
    assert connection.tools == ()


@pytest.mark.asyncio
async def test_stdio_transport_resolves_cwd_and_inherits_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("MYCLAW_MCP_TEST_ENV", "inherited")

    @asynccontextmanager
    async def fake_stdio(parameters: object) -> Any:
        captured["parameters"] = parameters
        yield ("read", "write")

    monkeypatch.setattr(mcp_adapter, "stdio_client", fake_stdio)
    workspace = Path("C:/workspace")
    configuration = _server_configuration(
        args=("-c", "pass"),
        cwd=Path("nested"),
    )
    async with mcp_adapter.stdio_transport(configuration, workspace) as streams:
        assert streams == ("read", "write")

    parameters = captured["parameters"]
    assert isinstance(parameters, StdioServerParameters)
    assert parameters.command == "python"
    assert parameters.args == ["-c", "pass"]
    assert parameters.cwd == workspace / "nested"
    assert parameters.env == dict(os.environ)


@pytest.mark.asyncio
async def test_http_transport_passes_headers_and_closes_created_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _HTTPClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    client = _HTTPClient()

    def fake_create(*, headers: dict[str, str]) -> _HTTPClient:
        captured["headers"] = headers
        return client

    @asynccontextmanager
    async def fake_http(url: str, *, http_client: _HTTPClient) -> Any:
        captured["url"] = url
        captured["client"] = http_client
        yield ("read", "write")

    monkeypatch.setattr(mcp_adapter, "create_mcp_http_client", fake_create)
    monkeypatch.setattr(mcp_adapter, "streamable_http_client", fake_http)
    configuration = _server_configuration(
        transport="streamable-http",
        url="http://127.0.0.1:8765/mcp",
        headers={"Authorization": "Bearer local"},
    )
    async with mcp_adapter.streamable_http_transport(configuration, Path(".")) as streams:
        assert streams == ("read", "write")

    assert captured == {
        "headers": {"Authorization": "Bearer local"},
        "url": "http://127.0.0.1:8765/mcp",
        "client": client,
    }
    assert client.closed is True


@pytest.mark.asyncio
async def test_stdio_transport_connects_to_a_local_real_mcp_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYCLAW_MCP_TEST_ENV", "inherited")
    server_script = "\n".join(
        (
            "import os",
            "from mcp.server.mcpserver import MCPServer",
            "server = MCPServer('local-stdio')",
            "@server.tool()",
            "def echo(value: str) -> str:",
            "    return f\"{value}:{os.environ['MYCLAW_MCP_TEST_ENV']}\"",
            "server.run()",
        )
    )
    connection = MCPServerConnection(
        _server_configuration(
            command=sys.executable,
            args=("-c", server_script),
            cwd=tmp_path,
            connect_timeout=10,
            call_timeout=5,
        ),
        tmp_path,
        model_name_for=lambda remote_name: f"mcp_remote_{remote_name}",
    )

    try:
        tools = await connection.connect()
        assert [tool.remote_name for tool in tools] == ["echo"]
        result = await ToolGateway._for_memory(tools).call(
            ModelToolCall(
                id="call-stdio",
                name="mcp_remote_echo",
                arguments=json.dumps({"value": "stdio result"}),
            )
        )
        assert (result.status, result.content) == ("success", "stdio result:inherited")
    finally:
        await connection.close()


async def _read_local_server_port(process: asyncio.subprocess.Process) -> int:
    if process.stdout is None:
        raise AssertionError("local MCP HTTP server stdout is unavailable")
    async with asyncio.timeout(10):
        while line := await process.stdout.readline():
            text = line.decode().strip()
            if text.startswith("MCP_TEST_READY:"):
                return int(text.partition(":")[2])
    raise AssertionError(f"local MCP HTTP server exited with {process.returncode}")


@pytest.mark.asyncio
async def test_streamable_http_transport_connects_to_a_local_real_mcp_server() -> None:
    server_script = "\n".join(
        (
            "import asyncio",
            "import socket",
            "import sys",
            "import uvicorn",
            "from mcp.server.mcpserver import MCPServer",
            "server = MCPServer('local-http')",
            "@server.tool()",
            "def echo(value: str) -> str:",
            "    return value",
            "async def run():",
            "    listener = socket.socket()",
            "    listener.bind(('127.0.0.1', 0))",
            "    listener.listen()",
            "    port = listener.getsockname()[1]",
            "    config = uvicorn.Config(",
            "        server.streamable_http_app(),",
            "        log_level='critical',",
            "    )",
            "    http_server = uvicorn.Server(config)",
            "    task = asyncio.create_task(http_server.serve(sockets=[listener]))",
            "    while not http_server.started:",
            "        if task.done():",
            "            await task",
            "        await asyncio.sleep(0.01)",
            "    print(f'MCP_TEST_READY:{port}', flush=True)",
            "    await task",
            "asyncio.run(run())",
        )
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        server_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    connection: MCPServerConnection | None = None
    try:
        port = await _read_local_server_port(process)
        connection = MCPServerConnection(
            _server_configuration(
                transport="streamable-http",
                url=f"http://127.0.0.1:{port}/mcp",
                headers={"X-Test-Header": "local"},
                connect_timeout=10,
                call_timeout=5,
            ),
            Path.cwd(),
            model_name_for=lambda remote_name: f"mcp_remote_{remote_name}",
        )
        tools = await connection.connect()
        assert [tool.remote_name for tool in tools] == ["echo"]
        result = await ToolGateway._for_memory(tools).call(
            ModelToolCall(
                id="call-http",
                name="mcp_remote_echo",
                arguments=json.dumps({"value": "http result"}),
            )
        )
        assert (result.status, result.content) == ("success", "http result")
    finally:
        if connection is not None:
            await connection.close()
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()
