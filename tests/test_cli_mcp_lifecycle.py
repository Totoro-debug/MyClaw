from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import myclaw.terminal.cli as cli
from myclaw.errors import ErrorInfo
from myclaw.management.service import ManagementError
from myclaw.tools.mcp_runtime import MCPServerFailure, MCPStartupReport


def _configuration() -> Any:
    return SimpleNamespace(
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(max_iterations=50),
        mcp={},
    )


def test_cli_reports_one_safe_notice_for_each_failure_and_one_aggregate_skip_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[str] = []
    monkeypatch.setattr(cli, "_print_mcp_notice", notices.append)

    cli._report_mcp_generation(
        MCPStartupReport(
            snapshot=(),
            connected_servers=(),
            failed_servers=("fallback", "broken"),
            failures=(
                MCPServerFailure(
                    mcp_name="broken",
                    phase="connect",
                    exception_type="TimeoutError",
                ),
            ),
            skipped_tool_counts=(("partial", 2),),
        )
    )

    assert notices == [
        "MCP Server 'broken' unavailable during connect (TimeoutError).",
        "MCP Server 'fallback' unavailable during connect (MCPConnectionError).",
        "MCP Server 'partial' skipped 2 invalid MCP Tools.",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_name", ["broken-one", "broken-two", "broken-three"])
async def test_cli_starts_mcp_before_initial_loop_and_closes_it_after_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failed_name: str,
) -> None:
    events: list[str] = []
    notices: list[str] = []
    remote_tool = object()

    class FakeMCPRuntimeManager:
        def __init__(self, workspace: Path, **kwargs: object) -> None:
            assert workspace == (tmp_path / "workspace").resolve()
            assert kwargs["built_in_names"]
            events.append("mcp_init")

        async def start(self, configuration: object) -> object:
            assert configuration == {}
            events.append("mcp_start")
            return SimpleNamespace(
                snapshot=(remote_tool,),
                failed_servers=(failed_name,),
                failures=(
                    MCPServerFailure(
                        mcp_name=failed_name,
                        phase="connect",
                        exception_type="TimeoutError",
                    ),
                ),
                skipped_tool_counts=(),
            )

        async def close(self) -> None:
            events.append("mcp_close")

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeBus:
        pass

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: object) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run during startup")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            pass

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            events.append("schedule_start")

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        async def close(self) -> None:
            events.append("schedule_close")

        def resume(self) -> None:
            pass

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["mcp_tools"] == (remote_tool,)
            events.append("loop_init")
            self.control = object()
            self.skill_metadata = ()

        def preflight(self) -> None:
            events.append("loop_preflight")

        async def start(self) -> None:
            events.append("loop_start")

        async def close(self) -> None:
            events.append("loop_close")

        async def abort(self) -> None:
            events.append("loop_abort")

        def project_foreground_conversation(self) -> object:
            return object()

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def deactivate(self) -> None:
            pass

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            events.append("app_run")

    monkeypatch.setattr(cli, "_print_mcp_notice", notices.append)
    monkeypatch.setattr(cli, "MCPRuntimeManager", FakeMCPRuntimeManager)
    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    from myclaw.config.agent_home import AgentHome

    await cli._run_cli_conversation(
        agent_home=AgentHome(tmp_path / "agent-home"),
        workspace=tmp_path / "workspace",
        configuration=_configuration(),
    )

    assert events.index("mcp_start") < events.index("loop_init")
    assert events.index("loop_close") < events.index("mcp_close")
    assert events.index("mcp_close") < events.index("dream_close")
    assert events.index("dream_close") < events.index("router_close")
    assert events.index("schedule_pause") < events.index("loop_close")
    assert notices == [f"MCP Server '{failed_name}' unavailable during connect (TimeoutError)."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preparation_error",
    [
        RuntimeError("sensitive transport endpoint"),
        TimeoutError("sensitive timeout detail"),
        OSError("sensitive protocol detail"),
    ],
    ids=("transport", "timeout", "protocol"),
)
async def test_cli_keeps_old_generation_when_mcp_candidate_preparation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preparation_error: Exception,
) -> None:
    events: list[str] = []
    current_callback: Callable[[], object] | None = None
    replace_callback: Callable[[str, bool], Any] | None = None
    old_loop: object | None = None

    class FakeMCPRuntimeManager:
        def __init__(self, workspace: Path, **kwargs: object) -> None:
            del workspace, kwargs

        async def start(self, configuration: object) -> object:
            del configuration
            return SimpleNamespace(snapshot=(), failed_servers=(), failures=())

        async def prepare_generation(self) -> object:
            events.append("mcp_prepare")
            raise preparation_error

        async def close(self) -> None:
            events.append("mcp_close")

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def close(self) -> None:
            pass

    class FakeMemoryManager:
        def __init__(self, workspace_state: object) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run")

        async def close(self) -> None:
            pass

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            pass

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            pass

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        async def close(self) -> None:
            pass

        def resume(self) -> None:
            pass

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            nonlocal old_loop
            assert kwargs["session_id"] is None
            old_loop = self
            self.control = SimpleNamespace(has_active_run=False)
            self.skill_metadata = ()
            self.session = SimpleNamespace(
                session_id="old-session",
                wait_for_pending_persist=lambda: None,
            )

        def preflight(self) -> None:
            pass

        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass

        async def abort(self) -> None:
            events.append("old_abort")

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            nonlocal current_callback, replace_callback
            current_callback = cast(Callable[[], object], kwargs["current_agent_loop"])
            replace_callback = cast(Callable[[str, bool], Any], kwargs["replace_agent_loop"])

        def deactivate(self) -> None:
            pass

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            assert replace_callback is not None
            with pytest.raises(ManagementError) as raised:
                await replace_callback("target-session", False)
            assert raised.value.error == ErrorInfo(
                "persistence_error",
                "MCP Tool Generation could not be prepared.",
            )
            assert "schedule_pause" not in events
            assert "old_abort" not in events
            assert current_callback is not None
            assert current_callback() is old_loop

    monkeypatch.setattr(cli, "MCPRuntimeManager", FakeMCPRuntimeManager)
    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", lambda: object())
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    from myclaw.config.agent_home import AgentHome

    await cli._run_cli_conversation(
        agent_home=AgentHome(tmp_path / "agent-home"),
        workspace=tmp_path / "workspace",
        configuration=_configuration(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_name", ["broken-one", "broken-two", "broken-three"])
async def test_cli_uses_failed_mcp_candidate_without_mutating_old_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failed_name: str,
) -> None:
    events: list[str] = []
    notices: list[str] = []
    old_loop: object | None = None
    target_loop: object | None = None
    replace_callback: Callable[[str, bool], Any] | None = None
    initial_tool = object()

    class FakeMCPRuntimeManager:
        def __init__(self, workspace: Path, **kwargs: object) -> None:
            assert workspace == (tmp_path / "workspace").resolve()
            assert kwargs["built_in_names"]

        async def start(self, configuration: object) -> object:
            del configuration
            return SimpleNamespace(
                snapshot=(initial_tool,),
                failed_servers=(),
                failures=(),
                skipped_tool_counts=(),
            )

        async def prepare_generation(self) -> object:
            events.append("mcp_prepare")
            return SimpleNamespace(
                snapshot=(),
                reused_servers=("healthy",),
                retried_servers=(failed_name,),
                failed_servers=(failed_name,),
                failures=(
                    MCPServerFailure(
                        mcp_name=failed_name,
                        phase="connect",
                        exception_type="TimeoutError",
                    ),
                ),
                skipped_tool_counts=(),
            )

        def activate_generation(self, report: object) -> tuple[object, ...]:
            assert cast(Any, report).failed_servers == (failed_name,)
            events.append("mcp_activate")
            return ()

        async def close(self) -> None:
            events.append("mcp_close")

    class FakeWorkspaceState:
        def __init__(self, workspace_path: Path) -> None:
            self.workspace_path = workspace_path

        def initialize(self, *, agent_home_root: Path) -> None:
            del agent_home_root

    class FakeBus:
        async def reset(self) -> None:
            events.append("bus_reset")

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemoryManager:
        def __init__(self, workspace_state: object) -> None:
            del workspace_state

    class FakeDream:
        def __init__(self, **kwargs: object) -> None:
            assert "mcp_tools" not in kwargs

        async def run(self) -> object:
            raise AssertionError("Dream must not run during startup")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeScheduleService:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def context_timezone_name(self) -> str:
            return "Asia/Shanghai"

        def _prepare_start(self) -> None:
            pass

        async def register_dream_job(self, **kwargs: object) -> None:
            del kwargs

        def start(self) -> None:
            pass

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        async def close(self) -> None:
            events.append("schedule_close")

        def resume(self) -> None:
            events.append("schedule_resume")

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeAgentLoop:
        def __init__(self, **kwargs: object) -> None:
            nonlocal old_loop, target_loop
            self.mcp_tools = tuple(kwargs["mcp_tools"])
            self.session = SimpleNamespace(session_id=kwargs["session_id"] or "old")
            self.control = SimpleNamespace(has_active_run=False)
            self.skill_metadata = ()
            if kwargs["session_id"] is None:
                old_loop = self
                events.append("old_init")
            else:
                target_loop = self
                events.append("target_init")

        def preflight(self) -> None:
            pass

        async def start(self) -> None:
            events.append("target_start" if self is target_loop else "old_start")

        async def close(self) -> None:
            events.append("target_close" if self is target_loop else "old_close")

        async def abort(self) -> None:
            events.append("target_abort" if self is target_loop else "old_abort")

        async def _pause_for_replacement(self) -> None:
            events.append("replacement_pause")

        async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
            del resume_inbound

        def project_foreground_conversation(self) -> object:
            return SimpleNamespace(session_id=self.session.session_id, messages=())

    class FakeManagementService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            nonlocal replace_callback
            replace_callback = cast(Callable[[str, bool], Any], kwargs["replace_agent_loop"])

        def deactivate(self) -> None:
            pass

    class FakeDispatcher:
        def __init__(self, management: object) -> None:
            del management

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            assert replace_callback is not None
            await replace_callback("target", False)
            assert old_loop is not None
            assert target_loop is not None
            assert old_loop.mcp_tools == (initial_tool,)
            assert target_loop.mcp_tools == ()

        async def quiesce_for_rebind(self) -> None:
            events.append("quiesce")

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs
            events.append("rebind")

    monkeypatch.setattr(cli, "_print_mcp_notice", notices.append)
    monkeypatch.setattr(cli, "MCPRuntimeManager", FakeMCPRuntimeManager)
    monkeypatch.setattr(cli, "WorkspaceState", FakeWorkspaceState)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemoryManager)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeScheduleService)
    monkeypatch.setattr(cli, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(cli, "ManagementViewService", FakeManagementService)
    monkeypatch.setattr(cli, "ManagementCommandDispatcher", FakeDispatcher)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    from myclaw.config.agent_home import AgentHome

    await cli._run_cli_conversation(
        agent_home=AgentHome(tmp_path / "agent-home"),
        workspace=tmp_path / "workspace",
        configuration=_configuration(),
    )

    assert old_loop is not None
    assert target_loop is not None
    assert old_loop.mcp_tools == (initial_tool,)
    assert target_loop.mcp_tools == ()
    assert notices == [f"MCP Server '{failed_name}' unavailable during connect (TimeoutError)."]
    assert events.index("mcp_prepare") < events.index("replacement_pause")
    assert events.index("mcp_activate") < events.index("schedule_resume")
    assert events[-5:] == [
        "schedule_close",
        "target_close",
        "mcp_close",
        "dream_close",
        "router_close",
    ]
