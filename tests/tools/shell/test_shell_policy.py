from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.contracts import (
    AssistantModelMessage,
    JsonObject,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    PermissionDecision,
    ToolExecutionContext,
    ToolExecutionLane,
)
from myclaw.tools.shell.shell_policy import ShellRequest
from myclaw.tools.tool_gateway import ToolGateway
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript

SESSION_ID = "20260712-120000-000000_550e8400-e29b-41d4-a716-446655440000"
NOW = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
SESSION_UUIDS = (
    "550e8400-e29b-41d4-a716-446655440000",
    "0f8fad5b-d9cb-469f-a165-70867728950e",
    "7c9e6679-7425-40de-944b-e07fc1f90ae7",
    "9b2c3a42-1d2e-4a1e-a827-61f36dc54713",
    "a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34",
)


class FakeShellBoundary:
    def __init__(self, outcomes: tuple[str, ...]) -> None:
        self._outcomes = iter(outcomes)
        self.requests: list[ShellRequest] = []

    async def execute(self, request: ShellRequest) -> str:
        self.requests.append(request)
        return next(self._outcomes)


def gateway_context(
    agent_home: Path,
    workspace: Path,
    *,
    lane: ToolExecutionLane = "foreground",
) -> ToolExecutionContext:
    return ToolExecutionContext(
        lane=lane,
        workspace=workspace,
        agent_home=agent_home,
        session_id=SESSION_ID,
    )


@pytest.mark.parametrize(
    "command",
    (
        "pwd",
        "git status",
        "git status --short",
        "git diff --stat",
        "git diff --name-only",
    ),
)
@pytest.mark.asyncio
async def test_frozen_read_only_shell_commands_execute_without_confirmation(
    command: str,
    agent_home: Path,
    workspace: Path,
) -> None:
    shell = FakeShellBoundary((f"completed {command}",))
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        shell=shell,
    )
    tool_call = ModelToolCall(
        id="call_shell",
        name="shell",
        arguments={"command": command, "timeout": 60},
    )

    assert gateway.permission_request(tool_call) is None
    result = await gateway.execute(tool_call)

    assert result.status == "success"
    assert result.content == f"completed {command}"
    assert shell.requests == [ShellRequest(command=command, cwd=workspace.resolve(), timeout=60)]


@pytest.mark.parametrize(
    "command",
    (
        "pwd ",
        "git  status",
        "git status --short --branch",
        "git status | more",
        "git status > status.txt",
        "echo $(git status)",
        "git status &",
        "git status; pwd",
        "git status && pwd",
    ),
)
@pytest.mark.asyncio
async def test_other_shell_commands_ask_in_foreground_and_are_refused_in_background(
    command: str,
    agent_home: Path,
    workspace: Path,
) -> None:
    tool_call = ModelToolCall(
        id="call_shell",
        name="shell",
        arguments={"command": command, "timeout": 90},
    )
    foreground_shell = FakeShellBoundary(("approved output",))
    foreground = ToolGateway(
        context=gateway_context(agent_home, workspace),
        shell=foreground_shell,
    )

    permission = foreground.permission_request(tool_call)

    assert permission is not None
    assert permission.decision is PermissionDecision.ASK
    refused = await foreground.execute(tool_call)
    assert refused.status == "refused"
    assert foreground_shell.requests == []

    approved = await foreground.execute(tool_call, approved=True)
    assert approved.status == "success"
    assert foreground_shell.requests == [
        ShellRequest(command=command, cwd=workspace.resolve(), timeout=90)
    ]

    background_shell = FakeShellBoundary(("must not execute",))
    background = ToolGateway(
        context=gateway_context(agent_home, workspace, lane="scheduled_work"),
        shell=background_shell,
    )

    assert background.permission_request(tool_call) is None
    background_result = await background.execute(tool_call)
    assert background_result.status == "refused"
    assert background_result.error is not None
    assert background_result.error.code == "tool_refused"
    assert background_shell.requests == []


@pytest.mark.asyncio
async def test_invalid_shell_command_or_cwd_is_denied_before_confirmation(
    agent_home: Path,
    workspace: Path,
) -> None:
    (workspace / "not-a-directory.txt").write_text("content", encoding="utf-8")
    shell = FakeShellBoundary(("must not execute",))
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        shell=shell,
    )
    invalid_arguments: tuple[JsonObject, ...] = (
        {"command": "pwd", "cwd": "..", "timeout": 60},
        {"command": "pwd", "cwd": "not-a-directory.txt", "timeout": 60},
        {"command": "pwd\x00whoami", "timeout": 60},
        {"command": "pwd\nwhoami", "timeout": 60},
        {"command": "pwd\twhoami", "timeout": 60},
    )

    for index, arguments in enumerate(invalid_arguments):
        tool_call = ModelToolCall(
            id=f"call_invalid_shell_{index}",
            name="shell",
            arguments=arguments,
        )

        assert gateway.permission_request(tool_call) is None
        result = await gateway.execute(tool_call, approved=True)

        assert result.status == "error"
        assert result.error is not None
        assert result.error.code == "tool_denied"

    assert shell.requests == []


@pytest.mark.asyncio
async def test_shell_timeout_accepts_only_integer_seconds_from_60_through_600(
    agent_home: Path,
    workspace: Path,
) -> None:
    shell = FakeShellBoundary(("minimum", "maximum"))
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        shell=shell,
    )

    for timeout, expected in ((60, "minimum"), (600, "maximum")):
        result = await gateway.execute(
            ModelToolCall(
                id=f"call_timeout_{timeout}",
                name="shell",
                arguments={"command": "pwd", "timeout": timeout},
            )
        )
        assert result.status == "success"
        assert result.content == expected

    for index, invalid_timeout in enumerate((None, 59, 601, True, 60.0)):
        arguments: JsonObject = {"command": "pwd"}
        if invalid_timeout is not None:
            arguments["timeout"] = invalid_timeout
        result = await gateway.execute(
            ModelToolCall(
                id=f"call_invalid_timeout_{index}",
                name="shell",
                arguments=arguments,
            ),
            approved=True,
        )
        assert result.status == "error"
        assert result.error is not None

    assert shell.requests == [
        ShellRequest(command="pwd", cwd=workspace.resolve(), timeout=60),
        ShellRequest(command="pwd", cwd=workspace.resolve(), timeout=600),
    ]


@pytest.mark.asyncio
async def test_approved_shell_command_receives_resolved_workspace_cwd(
    agent_home: Path,
    workspace: Path,
) -> None:
    nested = workspace / "src" / "package"
    nested.mkdir(parents=True)
    shell = FakeShellBoundary(("listed",))
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        shell=shell,
    )
    tool_call = ModelToolCall(
        id="call_nested_cwd",
        name="shell",
        arguments={"command": "dir", "cwd": "src/package", "timeout": 120},
    )

    assert gateway.permission_request(tool_call) is not None
    result = await gateway.execute(tool_call, approved=True)

    assert result.status == "success"
    assert shell.requests == [ShellRequest(command="dir", cwd=nested.resolve(), timeout=120)]


@pytest.mark.parametrize("enabled", (True, False))
@pytest.mark.asyncio
async def test_runtime_shell_enablement_controls_catalog_and_system_guidance(
    enabled: bool,
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_content = VALID_CONFIG
    if not enabled:
        config_content = config_content.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        )
    (agent_home / "config.toml").write_text(config_content, encoding="utf-8")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Catalog inspected.",
                            ),
                            usage=ModelUsage(
                                input_tokens=4,
                                output_tokens=2,
                                total_tokens=6,
                            ),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    shell = FakeShellBoundary(())
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=FakeClock(NOW).now,
        new_uuid=iter(map(UUID, SESSION_UUIDS)).__next__,
        shell=None if enabled else shell,
    )

    events = [event async for event in runtime.conversation.submit("Inspect the catalog.")]

    assert events[-1].type == "turn_completed"
    request = provider.stream_requests[0]
    assert isinstance(request, ModelRequest)
    names = [definition.name for definition in request.tools]
    guidance = request.system_prompt.split("<tool_guidance>\n", 1)[1].split("</tool_guidance>", 1)[
        0
    ]
    assert ("shell" in names) is enabled
    assert ("- shell:" in guidance) is enabled
    if enabled:
        assert "not OS filesystem or network sandboxed" in guidance
        assert "confined to the Workspace" not in guidance
    assert shell.requests == []
