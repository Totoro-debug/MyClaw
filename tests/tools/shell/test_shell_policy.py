import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from myclaw.runtime_log import install_runtime_logging
from myclaw.tools.models import ModelToolCall
from myclaw.tools.shell import shell_policy
from myclaw.tools.shell.shell_policy import ShellRequest
from myclaw.tools.shell.shell_tool import ShellTool
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.utils.host_filesystem import (
    POSIX_HOST_FILESYSTEM,
    WINDOWS_HOST_FILESYSTEM,
    HostFilesystem,
)
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


@pytest.mark.parametrize(
    ("host_filesystem", "filename", "accepted"),
    (
        (POSIX_HOST_FILESYSTEM, "git", True),
        (POSIX_HOST_FILESYSTEM, "git.exe", False),
        (WINDOWS_HOST_FILESYSTEM, "git.exe", True),
        (WINDOWS_HOST_FILESYSTEM, "git.EXE", True),
        (WINDOWS_HOST_FILESYSTEM, "git", False),
    ),
)
def test_git_capture_accepts_only_the_host_native_executable_name(
    tmp_path: Path,
    host_filesystem: HostFilesystem,
    filename: str,
    accepted: bool,
) -> None:
    executable = tmp_path / filename
    executable.write_bytes(b"trusted executable")
    captured = shell_policy._capture_git_executable(
        discover=lambda _: str(executable),
        host_filesystem=host_filesystem,
    )

    assert (captured is not None) is accepted
    if captured is not None:
        assert captured.path == executable.resolve(strict=True)


class FakeShellBoundary:
    def __init__(self, outcomes: tuple[str, ...]) -> None:
        self._outcomes = iter(outcomes)
        self.requests: list[ShellRequest] = []

    async def execute(self, request: ShellRequest) -> str:
        self.requests.append(request)
        return next(self._outcomes)


class FailingShellBoundary:
    def __init__(self) -> None:
        self.requests: list[ShellRequest] = []

    async def execute(self, request: ShellRequest) -> str:
        self.requests.append(request)
        raise OSError(f"RAW_PROCESS_BODY_51 command={request.command}")


def _gateway(
    *,
    agent_home: Path,
    workspace: Path,
    shell: FakeShellBoundary,
) -> ToolGateway:
    del agent_home
    gateway = ToolGateway()
    gateway.register_tools((ShellTool(workspace=workspace, boundary=shell),))
    return gateway


def test_shell_exports_accurate_schema_and_zero_retries(workspace: Path) -> None:
    tool = ShellTool(workspace=workspace, boundary=FakeShellBoundary(()))

    assert tool.to_schema() == {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run one of five exact read-only commands from a Workspace directory; this is "
                "not an operating-system filesystem or network sandbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Exact Shell command.",
                        "minLength": 1,
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Workspace-relative working directory.",
                        "default": ".",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds.",
                        "minimum": 60,
                        "maximum": 600,
                    },
                },
                "required": ["command", "timeout"],
            },
        },
    }
    assert tool.max_retries == 0


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
async def test_frozen_read_only_shell_commands_execute_through_gateway(
    command: str,
    agent_home: Path,
    workspace: Path,
) -> None:
    shell = FakeShellBoundary((f"completed {command}",))
    gateway = _gateway(agent_home=agent_home, workspace=workspace, shell=shell)

    result = await gateway.call(
        ModelToolCall(
            id="call_shell",
            name="shell",
            arguments=json.dumps({"command": command, "timeout": 60}),
        )
    )

    assert result.status == "success"
    assert result.content == f"completed {command}"
    assert shell.requests == [ShellRequest(command=command, cwd=workspace.resolve(), timeout=60)]


@pytest.mark.asyncio
async def test_shell_failure_log_excludes_command_and_process_output(
    agent_home: Path,
    workspace: Path,
) -> None:
    shell = FailingShellBoundary()
    gateway = ToolGateway()
    gateway.register_tools((ShellTool(workspace=workspace, boundary=shell),))
    lifetime = install_runtime_logging(AgentHome(agent_home))

    with lifetime.session("foreground-shell-session-51"):
        result = await gateway.call(
            ModelToolCall(
                id="call_shell_failure",
                name="shell",
                arguments='{"command":"git status","timeout":60}',
            )
        )
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert (result.status, result.content) == (
        "error",
        "shell could not complete the request.",
    )
    assert shell.requests == [
        ShellRequest(command="git status", cwd=workspace.resolve(), timeout=60)
    ]
    assert content.count(" ERROR ") == 1
    assert "name=shell attempt=1/1 type=OSError" in content
    assert "session=foreground-shell-session-51" in content
    assert "git status" not in content
    assert "RAW_PROCESS_BODY_51" not in content


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
        "dir",
    ),
)
@pytest.mark.asyncio
async def test_every_other_valid_command_is_refused_without_process_execution(
    command: str,
    agent_home: Path,
    workspace: Path,
) -> None:
    shell = FakeShellBoundary(("must not execute",))
    gateway = _gateway(
        agent_home=agent_home,
        workspace=workspace,
        shell=shell,
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_shell_refused",
            name="shell",
            arguments=json.dumps({"command": command, "timeout": 90}),
        )
    )

    assert result.status == "refused"
    assert result.content == (
        "Shell command refused because it is not in the safe read-only allowlist."
    )
    assert result.artifact is None
    assert shell.requests == []


@pytest.mark.asyncio
async def test_invalid_shell_command_or_cwd_is_rejected_before_process_execution(
    agent_home: Path,
    workspace: Path,
) -> None:
    (workspace / "not-a-directory.txt").write_text("content", encoding="utf-8")
    shell = FakeShellBoundary(("must not execute",))
    gateway = _gateway(agent_home=agent_home, workspace=workspace, shell=shell)
    invalid_arguments = (
        {"command": "pwd", "cwd": "..", "timeout": 60},
        {"command": "pwd", "cwd": "not-a-directory.txt", "timeout": 60},
        {"command": "pwd\x00whoami", "timeout": 60},
        {"command": "pwd\nwhoami", "timeout": 60},
        {"command": "pwd\twhoami", "timeout": 60},
        {"command": "pwd", "timeout": 59},
        {"command": "pwd", "timeout": 601},
    )

    for index, arguments in enumerate(invalid_arguments):
        result = await gateway.call(
            ModelToolCall(
                id=f"call_invalid_shell_{index}",
                name="shell",
                arguments=json.dumps(arguments),
            )
        )
        assert result.status == "error"

    assert shell.requests == []


@pytest.mark.asyncio
async def test_shell_accepts_timeout_bounds_and_resolves_nested_workspace_cwd(
    agent_home: Path,
    workspace: Path,
) -> None:
    nested = workspace / "src" / "package"
    nested.mkdir(parents=True)
    shell = FakeShellBoundary(("minimum", "maximum"))
    gateway = _gateway(agent_home=agent_home, workspace=workspace, shell=shell)

    minimum = await gateway.call(
        ModelToolCall(
            id="call_timeout_minimum",
            name="shell",
            arguments='{"command":"pwd","cwd":"src/package","timeout":60}',
        )
    )
    maximum = await gateway.call(
        ModelToolCall(
            id="call_timeout_maximum",
            name="shell",
            arguments='{"command":"pwd","cwd":"src/package","timeout":600}',
        )
    )

    assert (minimum.status, minimum.content) == ("success", "minimum")
    assert (maximum.status, maximum.content) == ("success", "maximum")
    assert shell.requests == [
        ShellRequest(command="pwd", cwd=nested.resolve(), timeout=60),
        ShellRequest(command="pwd", cwd=nested.resolve(), timeout=600),
    ]


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
                            message=AssistantModelMessage(content="Catalog inspected."),
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
    names = [schema["function"]["name"] for schema in request.tools]
    guidance = request.system_prompt.split("<tool_guidance>\n", 1)[1].split("</tool_guidance>", 1)[
        0
    ]
    assert ("shell" in names) is enabled
    assert ("- shell:" in guidance) is enabled
    if enabled:
        assert "not an operating-system filesystem or network sandbox" in guidance
        assert "confined to the Workspace" not in guidance
    assert shell.requests == []
