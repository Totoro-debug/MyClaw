import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from myclaw.tools.models import ModelToolCall
from myclaw.tools.shell.shell_policy import ShellRequest
from myclaw.tools.shell.shell_process import SubprocessShellBoundary
from myclaw.tools.shell.shell_tool import ShellBoundary, ShellTool
from myclaw.tools.tool_gateway import ToolGateway


def _platform_shell_command(arguments: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def _write_git_shadow(directory: Path, marker: Path) -> None:
    if os.name == "nt":
        (directory / "git.cmd").write_text(
            (
                "@echo off\n"
                'echo %* | %SystemRoot%\\System32\\findstr.exe /C:" config " >nul\n'
                "if not errorlevel 1 exit /b 1\n"
                f'> "{marker}" echo shadow\n'
                "exit /b 0\n"
            ),
            encoding="utf-8",
        )
        return
    shadow = directory / "git"
    shadow.write_text(
        (
            "#!/bin/sh\n"
            'for argument in "$@"; do\n'
            '  test "$argument" = config && exit 1\n'
            "done\n"
            f"printf shadow > {shlex.quote(str(marker))}\n"
        ),
        encoding="utf-8",
    )
    shadow.chmod(0o755)


class RecordingShellBoundary:
    def __init__(self) -> None:
        self.requests: list[ShellRequest] = []

    async def execute(self, request: ShellRequest) -> str:
        self.requests.append(request)
        return "must not execute"


def _shell_gateway(*, workspace: Path, shell: ShellBoundary) -> ToolGateway:
    gateway = ToolGateway()
    gateway.register_tools((ShellTool(workspace=workspace, boundary=shell),))
    return gateway


@pytest.mark.asyncio
async def test_automatic_git_command_does_not_execute_a_workspace_path_shadow(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    marker = workspace / "git-shadow-ran"
    _write_git_shadow(workspace, marker)
    monkeypatch.setenv("PATH", f"{workspace}{os.pathsep}{os.environ['PATH']}")
    shell = SubprocessShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_git_shadow",
        name="shell",
        arguments='{"command":"git status","timeout":60}',
    )

    try:
        result = await gateway.call(tool_call)
    finally:
        await shell.close()

    assert result.status == "success"
    assert not marker.exists()


def test_untrusted_startup_git_path_fails_closed_before_shell_execution(
    agent_home: Path,
    workspace: Path,
) -> None:
    marker = workspace / "startup-git-shadow-ran"
    _write_git_shadow(workspace, marker)
    probe = (
        "import asyncio\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from myclaw.tools.models import ModelToolCall\n"
        "from myclaw.tools.shell.shell_tool import ShellTool\n"
        "from myclaw.tools.tool_gateway import ToolGateway\n"
        "class Shell:\n"
        "    def __init__(self): self.calls = 0\n"
        "    async def execute(self, request): self.calls += 1; return 'ran'\n"
        "async def main():\n"
        "    workspace, _agent_home = map(Path, sys.argv[1:])\n"
        "    call = ModelToolCall(id='call_git', name='shell', "
        "arguments='{\"command\":\"git status\",\"timeout\":60}')\n"
        "    foreground_shell = Shell()\n"
        "    foreground = ToolGateway()\n"
        "    foreground.register_tools((ShellTool(workspace=workspace, "
        "boundary=foreground_shell),))\n"
        "    assert (await foreground.call(call)).status == 'refused'\n"
        "    assert foreground_shell.calls == 0\n"
        "    background_shell = Shell()\n"
        "    background = ToolGateway()\n"
        "    background.register_tools((ShellTool(workspace=workspace, "
        "boundary=background_shell),))\n"
        "    assert (await background.call(call)).status == 'refused'\n"
        "    assert background_shell.calls == 0\n"
        "asyncio.run(main())\n"
    )
    environment = os.environ.copy()
    environment["PATH"] = str(workspace)

    completed = subprocess.run(
        [sys.executable, "-c", probe, str(workspace), str(agent_home)],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows executable lookup only")
@pytest.mark.asyncio
async def test_automatic_git_command_does_not_execute_a_workspace_git_exe(
    agent_home: Path,
    workspace: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    system_root = Path(os.environ["SystemRoot"])
    shutil.copyfile(system_root / "System32" / "whoami.exe", workspace / "git.exe")
    shell = SubprocessShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)

    try:
        result = await gateway.call(
            ModelToolCall(
                id="call_git_exe_shadow",
                name="shell",
                arguments='{"command":"git status","timeout":60}',
            )
        )
    finally:
        await shell.close()

    assert result.status == "success"
    assert "On branch" in result.content


@pytest.mark.asyncio
async def test_automatic_git_command_ignores_a_later_untrusted_path_prefix(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    untrusted_bin = workspace.parent / "untrusted-bin"
    untrusted_bin.mkdir()
    marker = workspace / "path-prefix-shadow-ran"
    _write_git_shadow(untrusted_bin, marker)
    monkeypatch.setenv("PATH", f"{untrusted_bin}{os.pathsep}{os.environ['PATH']}")
    shell = SubprocessShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)

    try:
        result = await gateway.call(
            ModelToolCall(
                id="call_git_path_prefix_shadow",
                name="shell",
                arguments='{"command":"git status --short","timeout":60}',
            )
        )
    finally:
        await shell.close()

    assert result.status == "success"
    assert not marker.exists()


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
async def test_safe_repository_retains_the_five_automatic_shell_commands(
    command: str,
    agent_home: Path,
    workspace: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    shell = RecordingShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_safe_automatic_command",
        name="shell",
        arguments=json.dumps({"command": command, "timeout": 60}),
    )

    result = await gateway.call(tool_call)

    assert result.status == "success"
    assert shell.requests == [ShellRequest(command=command, cwd=workspace.resolve(), timeout=60)]


@pytest.mark.asyncio
async def test_unselected_git_filter_configuration_remains_automatically_allowed(
    agent_home: Path,
    workspace: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "config",
            "filter.secret.clean",
            "./leak-secret",
        ],
        check=True,
    )
    (workspace / ".gitattributes").write_text(
        "*.txt filter=unconfigured\n",
        encoding="utf-8",
    )
    shell = RecordingShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_unselected_git_filter",
        name="shell",
        arguments='{"command":"git status","timeout":60}',
    )

    result = await gateway.call(tool_call)

    assert result.status == "success"
    assert len(shell.requests) == 1


@pytest.mark.asyncio
async def test_configured_global_attributes_filter_is_not_automatically_allowed(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    attributes_file = workspace.parent / "global.attributes"
    attributes_file.write_text("*.txt filter=secret\n", encoding="utf-8")
    global_config = workspace.parent / "global-attributes.gitconfig"
    global_config.write_text(
        (
            '[filter "secret"]\n'
            "\tclean = ./leak-secret\n"
            "[core]\n"
            f"\tattributesFile = {attributes_file.as_posix()}\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    shell = RecordingShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_global_attributes_filter",
        name="shell",
        arguments='{"command":"git status","timeout":60}',
    )

    result = await gateway.call(tool_call)

    assert result.status == "refused"
    assert shell.requests == []


@pytest.mark.asyncio
async def test_git_attribute_source_environment_is_not_automatically_allowed(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "config",
            "filter.secret.clean",
            "./leak-secret",
        ],
        check=True,
    )
    attributes = workspace / ".gitattributes"
    attributes.write_text("*.txt filter=secret\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(workspace), "add", ".gitattributes"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "attributes",
        ],
        check=True,
    )
    attributes.unlink()
    (workspace / "payload.txt").write_text("secret\n", encoding="utf-8")
    monkeypatch.setenv("GIT_ATTR_SOURCE", "HEAD")
    shell = RecordingShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_git_attr_source",
        name="shell",
        arguments='{"command":"git status","timeout":60}',
    )

    result = await gateway.call(tool_call)

    assert result.status == "refused"
    assert shell.requests == []


@pytest.mark.asyncio
async def test_non_allowlisted_secret_command_is_refused_without_execution(
    agent_home: Path,
    workspace: Path,
) -> None:
    secret = "sk-shell-secret"
    script = (
        "import sys; "
        f"print('api-key={secret}'); "
        "print('Traceback (most recent call last):'); "
        "raise SystemExit(7)"
    )
    shell = SubprocessShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)

    try:
        result = await gateway.call(
            ModelToolCall(
                id="call_secret_failure",
                name="shell",
                arguments=json.dumps(
                    {
                        "command": _platform_shell_command([sys.executable, "-c", script]),
                        "timeout": 60,
                    }
                ),
            ),
        )
    finally:
        await shell.close()

    assert result.status == "refused"
    assert result.content == (
        "Shell command refused because it is not in the safe read-only allowlist."
    )
    assert secret not in result.content
    assert "Traceback" not in result.content


@pytest.mark.asyncio
async def test_allowlisted_git_status_does_not_run_a_repository_fsmonitor_hook(
    agent_home: Path,
    workspace: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    marker = workspace / "fsmonitor-ran"
    hook = workspace / "fsmonitor-hook"
    hook.write_bytes(b"#!/bin/sh\nprintf ran > fsmonitor-ran\nexit 1\n")
    hook.chmod(0o755)
    subprocess.run(
        ["git", "-C", str(workspace), "config", "core.fsmonitor", "./fsmonitor-hook"],
        check=True,
    )
    shell = SubprocessShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_git_status",
        name="shell",
        arguments='{"command":"git status","timeout":60}',
    )

    try:
        result = await gateway.call(tool_call)
    finally:
        await shell.close()

    assert result.status == "success"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_git_filter_configuration_refuses_an_allowlisted_command(
    agent_home: Path,
    workspace: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "config",
            "filter.secret.clean",
            "./leak-secret",
        ],
        check=True,
    )
    (workspace / ".gitattributes").write_text(
        "*.txt filter=secret\n",
        encoding="utf-8",
    )
    shell = RecordingShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_git_status",
        name="shell",
        arguments='{"command":"git status","timeout":60}',
    )

    result = await gateway.call(tool_call)

    assert result.status == "refused"
    assert shell.requests == []

    background_shell = RecordingShellBoundary()
    background = _shell_gateway(workspace=workspace, shell=background_shell)

    background_result = await background.call(tool_call)
    assert background_result.status == "refused"
    assert background_shell.requests == []


@pytest.mark.asyncio
async def test_git_info_attributes_filter_refuses_an_allowlisted_command(
    agent_home: Path,
    workspace: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "config",
            "filter.secret.process",
            "./leak-secret",
        ],
        check=True,
    )
    (workspace / ".git" / "info" / "attributes").write_text(
        "*.txt filter=secret\n",
        encoding="utf-8",
    )
    shell = RecordingShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_git_status_info_attributes",
        name="shell",
        arguments='{"command":"git status","timeout":60}',
    )

    result = await gateway.call(tool_call)

    assert result.status == "refused"
    assert shell.requests == []


@pytest.mark.asyncio
async def test_global_git_filter_configuration_is_not_automatically_allowed(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    global_config = workspace.parent / "global.gitconfig"
    global_config.write_text(
        '[filter "secret"]\n\tclean = ./leak-secret\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    (workspace / ".gitattributes").write_text(
        "*.txt filter=secret\n",
        encoding="utf-8",
    )
    shell = RecordingShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_global_git_filter",
        name="shell",
        arguments='{"command":"git status --short","timeout":60}',
    )

    result = await gateway.call(tool_call)

    assert result.status == "refused"
    assert shell.requests == []


@pytest.mark.asyncio
async def test_included_git_filter_configuration_is_not_automatically_allowed(
    agent_home: Path,
    workspace: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    nested_config = workspace / "nested.gitconfig"
    nested_config.write_text(
        '[filter "secret"]\n\tprocess = ./leak-secret\n',
        encoding="utf-8",
    )
    included_config = workspace / "included.gitconfig"
    included_config.write_text(
        f"[include]\n\tpath = {nested_config.as_posix()}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "config",
            "include.path",
            str(included_config),
        ],
        check=True,
    )
    (workspace / ".gitattributes").write_text(
        "*.txt filter=secret\n",
        encoding="utf-8",
    )
    shell = RecordingShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_included_git_filter",
        name="shell",
        arguments='{"command":"git diff --stat","timeout":60}',
    )

    result = await gateway.call(tool_call)

    assert result.status == "refused"
    assert shell.requests == []


@pytest.mark.asyncio
async def test_worktree_git_filter_configuration_is_not_automatically_allowed(
    agent_home: Path,
    workspace: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "config",
            "extensions.worktreeConfig",
            "true",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "config",
            "--worktree",
            "filter.secret.clean",
            "./leak-secret",
        ],
        check=True,
    )
    (workspace / ".gitattributes").write_text(
        "*.txt filter=secret\n",
        encoding="utf-8",
    )
    shell = RecordingShellBoundary()
    gateway = _shell_gateway(workspace=workspace, shell=shell)
    tool_call = ModelToolCall(
        id="call_worktree_git_filter",
        name="shell",
        arguments='{"command":"git diff --name-only","timeout":60}',
    )

    result = await gateway.call(tool_call)

    assert result.status == "refused"
    assert shell.requests == []
