from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import mcp.types as types
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent

from myclaw.config.config import MCPServerConfiguration
from myclaw.tools.base import BaseTool
from myclaw.tools.mcp import MCPTool, MCPToolSpec
from myclaw.tools.mcp_runtime import MCPRuntimeManager, allocate_mcp_tool_name
from myclaw.tools.tool_gateway import ModelToolCall, ToolGateway


class _ResultSession:
    def __init__(self, result: object | None = None, error: BaseException | None = None) -> None:
        self.result = (
            result if result is not None else CallToolResult(content=[TextContent(text="ok")])
        )
        self.error = error

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        del name, arguments
        if self.error is not None:
            raise self.error
        return self.result


class _FakeConnection:
    def __init__(
        self,
        configuration: MCPServerConfiguration,
        tools: tuple[MCPTool, ...] = (),
        *,
        tools_factory: Callable[[], tuple[MCPTool, ...]] | None = None,
    ) -> None:
        self.configuration = configuration
        self._tools = tools
        self._tools_factory = tools_factory
        self.connect_calls = 0
        self.close_calls = 0
        self.unavailable = False
        self.skipped_tool_count = 0
        self.fail_connect: BaseException | None = None
        self.fail_close: BaseException | None = None

    @property
    def tools(self) -> tuple[MCPTool, ...]:
        return self._tools

    async def connect(self) -> tuple[MCPTool, ...]:
        self.connect_calls += 1
        if self.fail_connect is not None:
            raise self.fail_connect
        self.unavailable = False
        if self._tools_factory is not None:
            self._tools = self._tools_factory()
        return self._tools

    async def close(self) -> None:
        self.close_calls += 1
        if self.fail_close is not None:
            raise self.fail_close

    def mark_unavailable(self) -> None:
        self.unavailable = True


class _BlockingConnection(_FakeConnection):
    def __init__(self, configuration: MCPServerConfiguration) -> None:
        super().__init__(configuration)
        self.connect_started = asyncio.Event()
        self.connect_cancelled = False

    async def connect(self) -> tuple[MCPTool, ...]:
        self.connect_calls += 1
        self.connect_started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("blocking connection unexpectedly resumed")
        except asyncio.CancelledError:
            self.connect_cancelled = True
            raise


class _ConcurrentConnection(_FakeConnection):
    def __init__(
        self,
        configuration: MCPServerConfiguration,
        started: asyncio.Event,
        release: asyncio.Event,
        *,
        completed: asyncio.Event | None = None,
        failure: BaseException | None = None,
    ) -> None:
        super().__init__(configuration)
        self._started = started
        self._release = release
        self._completed = completed
        self._failure = failure

    async def connect(self) -> tuple[MCPTool, ...]:
        self.connect_calls += 1
        self._started.set()
        await self._release.wait()
        if self._failure is not None:
            raise self._failure
        self.unavailable = False
        if self._completed is not None:
            self._completed.set()
        return self._tools


class _DelayedCloseConnection(_FakeConnection):
    def __init__(
        self,
        configuration: MCPServerConfiguration,
        release_close: asyncio.Event,
    ) -> None:
        super().__init__(configuration)
        self._release_close = release_close
        self.close_started = asyncio.Event()
        self.close_completed = False
        self.close_cancelled = False

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        try:
            await self._release_close.wait()
        except asyncio.CancelledError:
            self.close_cancelled = True
            raise
        self.close_completed = True


class _BuiltInTool(BaseTool):
    name = "builtin_read"
    description = "Read a local file."
    parameters: ClassVar[dict[str, Any]] = {"type": "object"}

    async def execute(self) -> str:
        return "built-in"


def _configuration(name: str) -> MCPServerConfiguration:
    return MCPServerConfiguration(
        mcp_name=name,
        enabled=True,
        transport="stdio",
        command="mcp-server",
    )


def _tool(
    server_name: str,
    remote_name: str,
    *,
    description: str | None = None,
    session: _ResultSession | None = None,
    on_closed: Callable[[], None] | None = None,
    model_name: str | None = None,
) -> MCPTool:
    return MCPTool(
        MCPToolSpec(
            server_name=server_name,
            remote_name=remote_name,
            model_name=model_name or remote_name,
            description=description or remote_name,
            parameters={"type": "object"},
        ),
        session or _ResultSession(),
        on_closed=on_closed,
    )


def _manager(
    connections: Mapping[str, _FakeConnection],
    *,
    built_in_tools: tuple[BaseTool, ...] = (),
) -> MCPRuntimeManager:
    return MCPRuntimeManager(
        Path("."),
        built_in_tools=built_in_tools,
        connection_factory=lambda configuration, workspace: connections[configuration.mcp_name],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_order", "alpha_order", "zulu_order"),
    [
        (("zulu", "alpha"), ("b-tool", "a-tool"), ("z-tool", "a-tool")),
        (("alpha", "zulu"), ("a-tool", "b-tool"), ("a-tool", "z-tool")),
        (("zulu", "alpha"), ("a-tool", "b-tool"), ("z-tool", "a-tool")),
    ],
)
async def test_runtime_snapshot_orders_builtins_servers_and_remote_tools(
    server_order: tuple[str, str],
    alpha_order: tuple[str, str],
    zulu_order: tuple[str, str],
) -> None:
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    connections = {
        "zulu": _FakeConnection(
            zulu,
            tuple(_tool("zulu", remote_name) for remote_name in zulu_order),
        ),
        "alpha": _FakeConnection(
            alpha,
            tuple(_tool("alpha", remote_name) for remote_name in alpha_order),
        ),
    }
    builtin = _BuiltInTool()

    manager = _manager(connections, built_in_tools=(builtin,))
    configured = {name: {"alpha": alpha, "zulu": zulu}[name] for name in server_order}
    report = await manager.start(configured)

    assert isinstance(report.snapshot, tuple)
    assert [tool.name for tool in manager.catalog] == [
        "builtin_read",
        "mcp_alpha_a-tool",
        "mcp_alpha_b-tool",
        "mcp_zulu_a-tool",
        "mcp_zulu_z-tool",
    ]


@pytest.mark.asyncio
async def test_runtime_connects_two_servers_without_waiting_for_a_third_server_timeout() -> None:
    success_release = asyncio.Event()
    timeout_release = asyncio.Event()
    alpha_started = asyncio.Event()
    beta_started = asyncio.Event()
    zulu_started = asyncio.Event()
    alpha_completed = asyncio.Event()
    beta_completed = asyncio.Event()
    configurations = {
        name: _configuration(name) for name in ("alpha", "beta", "zulu")
    }
    connections = {
        "alpha": _ConcurrentConnection(
            configurations["alpha"],
            alpha_started,
            success_release,
            completed=alpha_completed,
        ),
        "beta": _ConcurrentConnection(
            configurations["beta"],
            beta_started,
            success_release,
            completed=beta_completed,
        ),
        "zulu": _ConcurrentConnection(
            configurations["zulu"],
            zulu_started,
            timeout_release,
            failure=TimeoutError("sensitive timeout detail"),
        ),
    }
    manager = _manager(connections)

    start_task = asyncio.create_task(manager.start(configurations))
    await asyncio.gather(
        alpha_started.wait(),
        beta_started.wait(),
        zulu_started.wait(),
    )

    assert start_task.done() is False
    success_release.set()
    await asyncio.wait_for(
        asyncio.gather(alpha_completed.wait(), beta_completed.wait()),
        timeout=1,
    )
    assert start_task.done() is False

    timeout_release.set()
    report = await start_task

    assert report.connected_servers == ("alpha", "beta")
    assert report.failed_servers == ("zulu",)
    assert all(connection.connect_calls == 1 for connection in connections.values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remote_name", "expected_name"),
    [
        ("echo", "mcp_alpha_echo"),
        ("r" * 54, "mcp_alpha_" + "r" * 54),
        ("r" * 55, "mcp_" + "r" * 55),
        ("r" * 61, None),
        ("bad.tool", None),
    ],
)
async def test_runtime_snapshot_allocates_provider_safe_names(
    remote_name: str,
    expected_name: str | None,
) -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", remote_name),))
    report = await _manager({"alpha": connection}).start({"alpha": configuration})

    assert [tool.name for tool in report.snapshot] == (
        [] if expected_name is None else [expected_name]
    )


@pytest.mark.parametrize(
    ("mcp_name", "remote_name", "expected_name"),
    [
        ("alpha", "echo", "mcp_alpha_echo"),
        ("alpha", "r" * 54, "mcp_alpha_" + "r" * 54),
        ("alpha", "r" * 55, "mcp_" + "r" * 55),
        ("alpha", "r" * 61, None),
        ("alpha", "bad.tool", None),
    ],
)
def test_allocate_mcp_tool_name_matches_provider_contract(
    mcp_name: str,
    remote_name: str,
    expected_name: str | None,
) -> None:
    assert allocate_mcp_tool_name(mcp_name, remote_name) == expected_name


@pytest.mark.asyncio
async def test_runtime_snapshot_skips_builtin_and_fallback_collisions_deterministically() -> None:
    long_remote_name = "r" * 59
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    alpha_connection = _FakeConnection(alpha, (_tool("alpha", long_remote_name),))
    zulu_connection = _FakeConnection(zulu, (_tool("zulu", long_remote_name),))
    builtin = _BuiltInTool()
    builtin.name = "mcp_alpha_echo"

    projections: list[tuple[tuple[str, str], ...]] = []
    for _ in range(3):
        alpha_connection._tools = (_tool("alpha", long_remote_name),)
        zulu_connection._tools = (_tool("zulu", long_remote_name),)
        manager = _manager(
            {"alpha": alpha_connection, "zulu": zulu_connection},
            built_in_tools=(builtin,),
        )
        report = await manager.start({"zulu": zulu, "alpha": alpha})
        projections.append(tuple((tool.name, tool.description) for tool in report.snapshot))

    assert projections == [(("mcp_" + long_remote_name, long_remote_name),)] * 3
    assert [tool.name for tool in manager.catalog] == [
        "mcp_alpha_echo",
        "mcp_" + long_remote_name,
    ]


@pytest.mark.asyncio
async def test_runtime_snapshot_skips_collision_with_a_builtin_tool() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "echo"),))
    builtin = _BuiltInTool()
    builtin.name = "mcp_alpha_echo"

    manager = _manager({"alpha": connection}, built_in_tools=(builtin,))
    report = await manager.start({"alpha": configuration})

    assert report.snapshot == ()
    assert manager.catalog == (builtin,)


@pytest.mark.asyncio
async def test_runtime_reports_skipped_tool_counts() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "echo"),))
    connection.skipped_tool_count = 2
    manager = _manager({"alpha": connection})

    report = await manager.start({"alpha": configuration})

    assert report.skipped_tool_counts == (("alpha", 2),)


@pytest.mark.asyncio
async def test_rebuilding_same_configuration_keeps_snapshot_schema_deterministic() -> None:
    configuration = _configuration("alpha")
    projections: list[tuple[tuple[str, object], ...]] = []

    for _ in range(100):
        connection = _FakeConnection(
            configuration,
            (
                _tool("alpha", "second", description="Second"),
                _tool("alpha", "first", description="First"),
            ),
        )
        report = await _manager({"alpha": connection}).start({"alpha": configuration})
        projections.append(tuple((tool.name, tool.to_schema()) for tool in report.snapshot))

    assert all(projection == projections[0] for projection in projections)


@pytest.mark.asyncio
async def test_prepare_generation_reuses_healthy_connection_and_discovered_tools() -> None:
    configuration = _configuration("alpha")
    tool = _tool("alpha", "echo")
    connection = _FakeConnection(configuration, (tool,))
    manager = _manager({"alpha": connection})

    initial = await manager.start({"alpha": configuration})
    candidate = await manager.prepare_generation()

    assert connection.connect_calls == 1
    assert candidate.snapshot == initial.snapshot
    assert candidate.snapshot[0] is initial.snapshot[0]
    assert manager.snapshot == initial.snapshot
    assert candidate.reused_servers == ("alpha",)
    assert candidate.retried_servers == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        CallToolResult(content=[TextContent(text="server error")], is_error=True),
        CallToolResult(content=[TextContent(text="another error")], is_error=True),
    ],
)
async def test_timeout_and_is_error_results_do_not_enter_failed_server_set(result: object) -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration)
    connection._tools = (_tool("alpha", "echo", session=_ResultSession(result)),)
    manager = _manager({"alpha": connection})
    await manager.start({"alpha": configuration})

    outcome = await ToolGateway._for_memory(connection.tools).call(
        ModelToolCall(id="call", name=connection.tools[0].name, arguments="{}")
    )
    candidate = await manager.prepare_generation()

    assert outcome.status == "error"
    assert connection.connect_calls == 1
    assert manager.failed_servers == ()
    assert candidate.failed_servers == ()


@pytest.mark.asyncio
async def test_timeout_does_not_enter_failed_server_set() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration)
    connection._tools = (
        _tool("alpha", "echo", session=_ResultSession(error=TimeoutError("timed out"))),
    )
    manager = _manager({"alpha": connection})
    await manager.start({"alpha": configuration})

    outcome = await ToolGateway._for_memory(connection.tools).call(
        ModelToolCall(id="call-timeout", name=connection.tools[0].name, arguments="{}")
    )
    await manager.prepare_generation()

    assert outcome.status == "error"
    assert connection.connect_calls == 1
    assert manager.failed_servers == ()


@pytest.mark.asyncio
async def test_closed_session_enters_failed_set_and_is_retried_for_next_generation() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration)

    def replacement_tool() -> tuple[MCPTool, ...]:
        return (
            _tool(
                "alpha",
                "echo",
                session=_ResultSession(error=MCPError(types.CONNECTION_CLOSED, "closed")),
                on_closed=connection.mark_unavailable,
            ),
        )

    connection._tools_factory = replacement_tool
    connection._tools = replacement_tool()
    manager = _manager({"alpha": connection})
    await manager.start({"alpha": configuration})

    outcome = await ToolGateway._for_memory(connection.tools).call(
        ModelToolCall(id="call-closed", name=connection.tools[0].name, arguments="{}")
    )
    candidate = await manager.prepare_generation()

    assert outcome.status == "error"
    assert connection.connect_calls == 2
    assert manager.failed_servers == ()
    assert candidate.retried_servers == ("alpha",)
    assert [tool.name for tool in candidate.snapshot] == ["mcp_alpha_echo"]


@pytest.mark.asyncio
async def test_failed_candidate_keeps_previous_snapshot_unchanged() -> None:
    alpha = _configuration("alpha")
    beta = _configuration("beta")
    alpha_connection = _FakeConnection(alpha, (_tool("alpha", "alpha-tool"),))
    beta_connection = _FakeConnection(beta, (_tool("beta", "beta-tool"),))
    manager = _manager({"alpha": alpha_connection, "beta": beta_connection})

    initial = await manager.start({"alpha": alpha, "beta": beta})
    beta_connection.mark_unavailable()
    beta_connection.fail_connect = RuntimeError("connection failed")

    candidate = await manager.prepare_generation()

    assert initial.snapshot == (
        initial.snapshot[0],
        initial.snapshot[1],
    )
    assert manager.snapshot == initial.snapshot
    assert [tool.name for tool in candidate.snapshot] == ["mcp_alpha_alpha-tool"]
    assert candidate.failed_servers == ("beta",)
    assert manager.failed_servers == ("beta",)


@pytest.mark.asyncio
async def test_healthy_collision_loser_can_win_after_previous_server_fails() -> None:
    long_remote_name = "r" * 59
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    alpha_connection = _FakeConnection(alpha, (_tool("alpha", long_remote_name),))
    zulu_connection = _FakeConnection(zulu, (_tool("zulu", long_remote_name),))
    manager = _manager({"alpha": alpha_connection, "zulu": zulu_connection})

    initial = await manager.start({"alpha": alpha, "zulu": zulu})
    assert [tool.name for tool in initial.snapshot] == ["mcp_" + long_remote_name]

    alpha_connection.mark_unavailable()
    alpha_connection.fail_connect = RuntimeError("alpha is still unavailable")
    candidate = await manager.prepare_generation()

    assert manager.snapshot == initial.snapshot
    assert [tool.name for tool in candidate.snapshot] == ["mcp_" + long_remote_name]
    assert alpha_connection.connect_calls == 2
    assert zulu_connection.connect_calls == 1
    assert candidate.failed_servers == ("alpha",)


@pytest.mark.asyncio
async def test_activate_generation_replaces_snapshot_only_after_candidate_is_selected() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "first"),))
    manager = _manager({"alpha": connection})

    initial = await manager.start({"alpha": configuration})
    connection.mark_unavailable()
    connection._tools = (_tool("alpha", "second"),)
    candidate = await manager.prepare_generation()

    assert manager.snapshot == initial.snapshot
    assert [tool.name for tool in candidate.snapshot] == ["mcp_alpha_second"]

    activated = manager.activate_generation(candidate)

    assert activated == candidate.snapshot
    assert manager.snapshot == candidate.snapshot


@pytest.mark.asyncio
async def test_cancelled_connection_cancels_and_settles_sibling_connections() -> None:
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    blocking = _BlockingConnection(alpha)
    cancelled = _FakeConnection(zulu)
    cancelled.fail_connect = asyncio.CancelledError()
    manager = _manager({"alpha": blocking, "zulu": cancelled})

    with pytest.raises(asyncio.CancelledError):
        await manager.start({"alpha": alpha, "zulu": zulu})

    assert blocking.connect_started.is_set()
    assert blocking.connect_cancelled is True
    assert blocking.close_calls == 1
    assert cancelled.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, RuntimeError])
async def test_close_attempts_every_connection_before_propagating_failure(
    error_type: type[BaseException],
) -> None:
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    first = _FakeConnection(alpha)
    second = _FakeConnection(zulu)
    manager = _manager({"alpha": first, "zulu": second})
    await manager.start({"alpha": alpha, "zulu": zulu})
    first.fail_close = error_type("close failed")

    with pytest.raises(error_type):
        await manager.close()

    assert first.close_calls == 1
    assert second.close_calls == 1


@pytest.mark.asyncio
async def test_cancelling_manager_close_waits_for_every_connection_to_finish() -> None:
    alpha = _configuration("alpha")
    zulu = _configuration("zulu")
    release_close = asyncio.Event()
    first = _DelayedCloseConnection(alpha, release_close)
    second = _DelayedCloseConnection(zulu, release_close)
    manager = _manager({"alpha": first, "zulu": second})
    await manager.start({"alpha": alpha, "zulu": zulu})
    close_task = asyncio.create_task(manager.close())
    await asyncio.gather(first.close_started.wait(), second.close_started.wait())

    close_task.cancel()
    await asyncio.sleep(0)
    close_task.cancel()
    await asyncio.sleep(0)

    assert close_task.done() is False
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert first.close_completed is True
    assert second.close_completed is True
    assert first.close_cancelled is False
    assert second.close_cancelled is False


@pytest.mark.asyncio
async def test_activate_generation_rejects_a_candidate_from_another_manager() -> None:
    configuration = _configuration("alpha")
    first_connection = _FakeConnection(configuration, (_tool("alpha", "first"),))
    second_connection = _FakeConnection(configuration, (_tool("alpha", "second"),))
    first_manager = _manager({"alpha": first_connection})
    second_manager = _manager({"alpha": second_connection})
    await first_manager.start({"alpha": configuration})
    await second_manager.start({"alpha": configuration})
    foreign_candidate = await first_manager.prepare_generation()
    active_snapshot = second_manager.snapshot

    with pytest.raises(ValueError, match="current prepared candidate"):
        second_manager.activate_generation(foreign_candidate)

    assert second_manager.snapshot == active_snapshot


@pytest.mark.asyncio
async def test_activate_generation_rejects_a_superseded_candidate() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "echo"),))
    manager = _manager({"alpha": connection})
    await manager.start({"alpha": configuration})
    stale_candidate = await manager.prepare_generation()
    current_candidate = await manager.prepare_generation()
    active_snapshot = manager.snapshot

    with pytest.raises(ValueError, match="current prepared candidate"):
        manager.activate_generation(stale_candidate)

    assert manager.snapshot == active_snapshot
    assert manager.activate_generation(current_candidate) == current_candidate.snapshot


@pytest.mark.asyncio
async def test_activate_generation_rejects_a_candidate_after_close() -> None:
    configuration = _configuration("alpha")
    connection = _FakeConnection(configuration, (_tool("alpha", "echo"),))
    manager = _manager({"alpha": connection})
    await manager.start({"alpha": configuration})
    candidate = await manager.prepare_generation()
    await manager.close()

    with pytest.raises(RuntimeError, match="has not been started"):
        manager.activate_generation(candidate)

    assert manager.snapshot == ()
