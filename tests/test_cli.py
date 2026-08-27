import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import myclaw.terminal.cli as cli
from myclaw.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.agent.runtime import SkillContextTooLargeError
from myclaw.config.agent_home import AgentHome
from myclaw.errors import ErrorInfo
from tests.configuration.test_config import (
    EXPECTED_DEFAULT_CONFIG,
    EXPECTED_REDACTED_CONFIG,
    EXPECTED_REDACTED_MALFORMED_CONFIG,
    MALFORMED_CONFIG,
    REDACTION_CONFIG,
    VALID_CONFIG,
)


@pytest.mark.asyncio
async def test_composition_driver_owns_runtime_lifecycle_outside_terminal_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("app_init")

        async def quiesce_for_rebind(self) -> None:
            events.append("quiesce")

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs
            events.append("rebind")

        async def run_async(self) -> None:
            events.append("app_run")

    class FakeRuntime:
        bus = object()
        control = object()
        management_dispatcher = object()
        presentation = SimpleNamespace(control=control, skill_metadata=())

        def bind_terminal(self, rebind: object, *, quiesce: object) -> None:
            del rebind, quiesce
            events.append("bind")

        def unbind_terminal(self, rebind: object) -> None:
            del rebind
            events.append("unbind")

        async def start(self) -> None:
            events.append("start")

        async def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    await cli._run_runtime_conversation(FakeRuntime())  # type: ignore[arg-type]

    assert events == ["app_init", "bind", "start", "app_run", "unbind", "close"]


@pytest.mark.asyncio
async def test_composition_driver_closes_runtime_when_initial_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("app_init")

        async def quiesce_for_rebind(self) -> None:
            events.append("quiesce")

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs
            events.append("rebind")

        async def run_async(self) -> None:
            events.append("app_run")

    class FakeRuntime:
        bus = object()
        control = object()
        management_dispatcher = object()
        presentation = SimpleNamespace(control=control, skill_metadata=())

        def bind_terminal(self, rebind: object, *, quiesce: object) -> None:
            del rebind, quiesce
            events.append("bind")

        def unbind_terminal(self, rebind: object) -> None:
            del rebind
            events.append("unbind")

        async def start(self) -> None:
            events.append("start")
            raise RuntimeError("initial runtime startup failed")

        async def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    with pytest.raises(RuntimeError, match="initial runtime startup failed"):
        await cli._run_runtime_conversation(FakeRuntime())  # type: ignore[arg-type]

    assert events == ["app_init", "bind", "start", "unbind", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ("app_init", "bind"))
async def test_composition_driver_closes_runtime_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    events: list[str] = []

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("app_init")
            if failure_point == "app_init":
                raise RuntimeError("application setup failed")

        async def quiesce_for_rebind(self) -> None:
            raise AssertionError("quiesce must not run")

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs
            raise AssertionError("rebind must not run")

        async def run_async(self) -> None:
            raise AssertionError("application must not run")

    class FakeRuntime:
        bus = object()
        control = object()
        management_dispatcher = object()
        presentation = SimpleNamespace(control=control, skill_metadata=())

        def bind_terminal(self, rebind: object, *, quiesce: object) -> None:
            del rebind, quiesce
            events.append("bind")
            raise RuntimeError("terminal binding failed")

        def unbind_terminal(self, rebind: object) -> None:
            del rebind
            events.append("unbind")

        async def start(self) -> None:
            events.append("start")

        async def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    expected = "application setup failed" if failure_point == "app_init" else "terminal binding failed"
    with pytest.raises(RuntimeError, match=expected):
        await cli._run_runtime_conversation(FakeRuntime())  # type: ignore[arg-type]

    prefix = ["app_init"] if failure_point == "app_init" else ["app_init", "bind"]
    assert events == [*prefix, "close"]


@pytest.mark.asyncio
async def test_composition_driver_closes_once_when_application_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            events.append("app_init")

        async def quiesce_for_rebind(self) -> None:
            events.append("quiesce")

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            events.append("app_run")
            raise asyncio.CancelledError()

    class FakeRuntime:
        bus = object()
        control = object()
        management_dispatcher = object()
        presentation = SimpleNamespace(control=control, skill_metadata=())

        def bind_terminal(self, rebind: object, *, quiesce: object) -> None:
            del rebind, quiesce
            events.append("bind")

        def unbind_terminal(self, rebind: object) -> None:
            del rebind
            events.append("unbind")

        async def start(self) -> None:
            events.append("start")

        async def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    with pytest.raises(asyncio.CancelledError):
        await cli._run_runtime_conversation(FakeRuntime())  # type: ignore[arg-type]

    assert events == ["app_init", "bind", "start", "app_run", "unbind", "close"]
    assert events.count("close") == 1


@pytest.mark.asyncio
async def test_composition_driver_preserves_messages_queued_before_app_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    bus = MessageBus()
    inbound = InboundMessage(content="queued before mount")
    outbound = OutboundMessage(type="model_response", content="queued before mount")
    mounted_snapshot: tuple[InboundMessage, ...] | None = None
    mounted_outbound: OutboundMessage | None = None

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["bus"] is bus

        async def quiesce_for_rebind(self) -> None:
            return

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            nonlocal mounted_snapshot, mounted_outbound
            events.append("mount")
            mounted_snapshot = await bus.inbound_snapshot()
            mounted_outbound = await bus.get_outbound()

    class FakeRuntime:
        control = object()
        management_dispatcher = object()
        presentation = SimpleNamespace(control=control, skill_metadata=())

        @property
        def bus(self) -> MessageBus:
            return bus

        def bind_terminal(self, rebind: object, *, quiesce: object) -> None:
            del rebind, quiesce

        def unbind_terminal(self, rebind: object) -> None:
            del rebind

        async def start(self) -> None:
            events.append("start")
            await bus.put_inbound(inbound)
            await bus.put_outbound(outbound)

        async def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    await cli._run_runtime_conversation(FakeRuntime())  # type: ignore[arg-type]

    assert events == ["start", "mount", "close"]
    assert mounted_snapshot == (inbound,)
    assert mounted_outbound is outbound


@pytest.mark.asyncio
async def test_composition_driver_retrieves_close_task_that_emits_after_app_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = MessageBus()
    close_calls = 0

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def quiesce_for_rebind(self) -> None:
            return

        async def rebind_agent_loop(self, **kwargs: object) -> None:
            del kwargs

        async def run_async(self) -> None:
            return

    class FakeRuntime:
        control = object()
        management_dispatcher = object()
        presentation = SimpleNamespace(control=control, skill_metadata=())

        @property
        def bus(self) -> MessageBus:
            return bus

        def bind_terminal(self, rebind: object, *, quiesce: object) -> None:
            del rebind, quiesce

        def unbind_terminal(self, rebind: object) -> None:
            del rebind

        async def start(self) -> None:
            return

        async def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

            async def emit() -> None:
                await bus.put_outbound(
                    OutboundMessage(type="system_control", content="closed")
                )

            await asyncio.create_task(emit(), name="close-outbound")

    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    await cli._run_runtime_conversation(FakeRuntime())  # type: ignore[arg-type]

    assert close_calls == 1
    assert (await bus.get_outbound()).content == "closed"
    assert all(task.get_name() != "close-outbound" for task in asyncio.all_tasks())


@pytest.mark.parametrize(
    ("failure", "expected_code", "secret"),
    (
        (
            SkillContextTooLargeError(
                ErrorInfo(
                    "skill_context_too_large",
                    "Always-loaded Skill content exceeds the foreground chat input budget.",
                )
            ),
            "skill_context_too_large",
            "C:\\sensitive\\skill\\SKILL.md",
        ),
    ),
)
def test_cli_reports_runtime_skill_startup_failures_without_starting_conversation(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_code: str,
    secret: str,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    monkeypatch.setattr(AgentHome, "production", lambda: home)
    monkeypatch.setattr(cli, "is_interactive_terminal", lambda: True)
    conversation_calls: list[object] = []

    def fail_runtime(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise failure

    async def record_conversation(runtime: object) -> None:
        conversation_calls.append(runtime)

    monkeypatch.setattr(cli, "RuntimeHost", fail_runtime)
    monkeypatch.setattr(cli, "_run_runtime_conversation", record_conversation)
    monkeypatch.chdir(workspace)

    result = CliRunner().invoke(cli.app, [])

    assert result.exit_code == 1
    assert result.output.count(f"{expected_code}:") == 1
    assert result.output.count(str(failure)) == 1
    assert secret not in result.output
    assert "Traceback" not in result.output
    assert conversation_calls == []


def run_installed_myclaw(
    agent_home: Path,
    *arguments: str,
    workspace: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("myclaw")
    assert executable is not None
    environment = os.environ.copy()
    environment["HOME"] = str(agent_home.parent)
    environment["USERPROFILE"] = str(agent_home.parent)
    source_root = str(Path(__file__).parent.parent)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else os.pathsep.join((source_root, existing_pythonpath))
    )
    return subprocess.run(
        [executable, *arguments],
        capture_output=True,
        check=False,
        cwd=agent_home.parent if workspace is None else workspace,
        env=environment,
        text=True,
    )


def assert_plaintext_absent(output: str, *plaintext_values: str) -> None:
    if any(value in output for value in plaintext_values):
        pytest.fail("CLI output leaked a plaintext provider API key", pytrace=False)


def legacy_runtime_log_snapshot(agent_home: Path) -> dict[str, bytes]:
    logs = agent_home / "logs"
    return {
        path.name: path.read_bytes()
        for path in logs.iterdir()
        if path.is_file() and path.name.startswith("run.log.")
    }


def test_installed_myclaw_console_entry_starts() -> None:
    executable = shutil.which("myclaw")

    assert executable is not None
    result = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "MyClaw Personal Agent" in result.stdout


@pytest.mark.skipif(
    os.name == "nt",
    reason="The Windows Python runtime has no termios/pty harness; use the Windows Terminal matrix.",
)
def test_installed_wheel_terminal_conversation_pseudo_terminal_smoke(tmp_path: Path) -> None:
    pty = pytest.importorskip("pty")
    termios = pytest.importorskip("termios")
    import select
    import time

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    build_result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
        text=True,
    )
    assert build_result.returncode == 0, build_result.stderr
    wheels = tuple(wheel_dir.glob("myclaw-*.whl"))
    assert len(wheels) == 1

    venv = tmp_path / "venv"
    venv_result = subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert venv_result.returncode == 0, venv_result.stderr
    venv_bin = venv / "bin"
    venv_python = venv_bin / "python"
    install_result = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
        capture_output=True,
        check=False,
        text=True,
    )
    assert install_result.returncode == 0, install_result.stderr

    agent_home = tmp_path / "home" / ".myclaw"
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["HOME"] = str(agent_home.parent)

    master_fd, slave_fd = pty.openpty()
    process: subprocess.Popen[bytes] | None = None
    try:
        original_terminal = termios.tcgetattr(slave_fd)
        process = subprocess.Popen(
            [str(venv_bin / "myclaw")],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=workspace,
            env=environment,
            close_fds=True,
        )
        output = bytearray()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and b"Message MyClaw" not in output:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                output.extend(os.read(master_fd, 4096))
        assert b"Message MyClaw" in output or b"\x1b[?1049h" in output
        assert process.poll() is None

        os.write(master_fd, b"\x03")
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    output.extend(os.read(master_fd, 4096))
                except OSError:
                    break
        assert process.poll() == 0

        while True:
            ready, _, _ = select.select([master_fd], [], [], 0)
            if not ready:
                break
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)

        terminal_output = bytes(output)
        restoration_pairs = (
            (b"\x1b[?2004h", b"\x1b[?2004l"),
            (b"\x1b[?1000h", b"\x1b[?1000l"),
            (b"\x1b[?1003h", b"\x1b[?1003l"),
            (b"\x1b[?1015h", b"\x1b[?1015l"),
            (b"\x1b[?1006h", b"\x1b[?1006l"),
            (b"\x1b[?1004h", b"\x1b[?1004l"),
            (b"\x1b[?1049h", b"\x1b[?1049l"),
            (b"\x1b[?25l", b"\x1b[?25h"),
            (b"\x1b[>1u", b"\x1b[<u"),
        )
        assert b"\x1b[?1049h" in terminal_output
        for enabled, restored in restoration_pairs:
            if enabled in terminal_output:
                assert restored in terminal_output
                assert terminal_output.rfind(restored) > terminal_output.rfind(enabled)
        assert termios.tcgetattr(slave_fd) == original_terminal
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master_fd)
        os.close(slave_fd)


def test_installed_myclaw_generates_missing_configuration_and_stops(
    agent_home: Path,
    workspace: Path,
) -> None:
    result = run_installed_myclaw(agent_home, workspace=workspace)

    assert result.returncode == 2
    assert (agent_home / "config.toml").read_text(encoding="utf-8") == EXPECTED_DEFAULT_CONFIG
    assert result.stdout.count("config_missing") == 1
    assert str(agent_home / "config.toml") in result.stdout
    assert "edit" in result.stdout.lower()
    assert result.stderr == ""
    assert "configuration gate passed" not in result.stdout
    assert not (workspace / ".myclaw").exists()
    assert not (agent_home / "logs").exists()


def test_installed_myclaw_does_not_modify_legacy_runtime_log_data(
    agent_home: Path,
    workspace: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"legacy slot zero\n")
    (logs / "run.log.1").write_bytes(b"legacy slot one\n")
    (logs / "run.log.cursor").write_bytes(b"1\n")
    (logs / "run.log.lock").write_bytes(b"legacy lock\n")
    before = legacy_runtime_log_snapshot(agent_home)

    result = run_installed_myclaw(agent_home, workspace=workspace)
    config_result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 2
    assert config_result.returncode == 0
    assert legacy_runtime_log_snapshot(agent_home) == before


def test_installed_config_command_generates_and_displays_missing_configuration(
    agent_home: Path,
    workspace: Path,
) -> None:
    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 0, result.stderr
    assert f"Path: {agent_home / 'config.toml'}" in result.stdout
    assert EXPECTED_DEFAULT_CONFIG in result.stdout
    assert "configuration gate passed" not in result.stdout
    assert not (agent_home / "logs").exists()
    assert not (workspace / ".myclaw").exists()


def test_installed_config_command_redacts_valid_configuration(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(REDACTION_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 0, result.stderr
    assert EXPECTED_REDACTED_CONFIG in result.stdout
    assert f"Path: {agent_home / 'config.toml'}" in result.stdout
    assert_plaintext_absent(result.stdout + result.stderr, "plaintext-primary-key")
    assert not (agent_home / "logs").exists()
    assert not (workspace / ".myclaw").exists()


def test_installed_config_command_shows_safe_malformed_configuration(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(MALFORMED_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 2
    assert result.stdout.count("config_parse_error") == 1
    assert f"Path: {agent_home / 'config.toml'}" in result.stdout
    assert EXPECTED_REDACTED_MALFORMED_CONFIG in result.stdout
    assert result.stderr == ""
    assert_plaintext_absent(
        result.stdout + result.stderr,
        "first-plaintext-key",
        "second-plaintext-key",
    )
    assert not (agent_home / "logs").exists()
    assert not (workspace / ".myclaw").exists()


def test_installed_config_command_hides_invalid_utf8_and_traceback(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    config_path = agent_home / "config.toml"
    config_path.write_bytes(b'api_key = "sk-invalid-utf8-secret"\ninvalid = "\xff"\n')

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    visible = result.stdout + result.stderr
    assert result.returncode == 1
    assert "persistence_error" in result.stdout
    assert f"Path: {config_path}" in result.stdout
    assert "sk-invalid-utf8-secret" not in visible
    assert "Traceback" not in visible


def test_installed_config_command_keeps_undefined_content_inspectable(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    content = REDACTION_CONFIG.replace(
        "max_tool_result_chars = 50000",
        "max_tool_result_chars = 50000\nmisspelled_setting = true",
    )
    (agent_home / "config.toml").write_text(content, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config", workspace=workspace)

    assert result.returncode == 2
    assert "config_invalid" in result.stdout
    assert "runtime.misspelled_setting" in result.stdout
    assert "misspelled_setting = true" in result.stdout
    assert_plaintext_absent(result.stdout + result.stderr, "plaintext-primary-key")
    assert not (workspace / ".myclaw").exists()


def test_installed_myclaw_rejects_valid_configuration_without_a_tty(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, workspace=workspace)

    assert result.returncode == 2, result.stderr
    assert "interactive_terminal_required" in result.stdout
    assert "configuration gate passed" not in result.stdout
    assert_plaintext_absent(result.stdout + result.stderr, "sk-ant-secret")
    assert not (agent_home / "logs").exists()
    assert not (workspace / ".myclaw").exists()


def test_installed_myclaw_stops_only_on_parse_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    config_path = agent_home / "config.toml"

    config_path.write_text(MALFORMED_CONFIG, encoding="utf-8")
    parse_result = run_installed_myclaw(agent_home, workspace=workspace)

    schema_content = REDACTION_CONFIG.replace(
        "max_tool_result_chars = 50000",
        "max_tool_result_chars = 50000\nmisspelled_setting = true",
    )
    config_path.write_text(schema_content, encoding="utf-8")
    schema_result = run_installed_myclaw(agent_home, workspace=workspace)

    config_path.write_text(EXPECTED_DEFAULT_CONFIG, encoding="utf-8")
    default_result = run_installed_myclaw(agent_home, workspace=workspace)

    assert (parse_result.returncode, schema_result.returncode, default_result.returncode) == (
        2,
        2,
        2,
    )
    assert "config_parse_error" in parse_result.stdout
    assert "config_invalid" not in schema_result.stdout
    assert "configuration gate passed" not in parse_result.stdout
    assert "interactive_terminal_required" in schema_result.stdout
    assert "interactive_terminal_required" in default_result.stdout
    assert not (workspace / ".myclaw").exists()
    combined_output = "".join(
        result.stdout + result.stderr for result in (parse_result, schema_result, default_result)
    )
    assert all(result.stderr == "" for result in (parse_result, schema_result, default_result))
    assert_plaintext_absent(
        combined_output,
        "first-plaintext-key",
        "second-plaintext-key",
        "plaintext-primary-key",
    )
    assert not (agent_home / "logs").exists()


def test_installed_myclaw_rejects_non_tty_before_unsafe_workspace_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state_path = workspace / ".myclaw"
    state_path.write_text("private collision content", encoding="utf-8")

    result = run_installed_myclaw(agent_home, workspace=workspace)

    assert result.returncode == 2
    assert result.stdout.count("interactive_terminal_required") == 1
    assert "Workspace State" not in result.stdout
    assert str(state_path) not in result.stdout
    assert "private collision content" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert result.stderr == ""
    assert state_path.read_text(encoding="utf-8") == "private collision content"
    assert not (agent_home / "logs").exists()


def test_installed_myclaw_rejects_non_tty_before_corrupt_schedule_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state_path = workspace / ".myclaw"
    state_path.mkdir()
    schedule_path = state_path / "schedule.json"
    schedule_path.write_text("{corrupt", encoding="utf-8")

    result = run_installed_myclaw(agent_home, workspace=workspace)

    assert result.returncode == 2
    assert result.stdout.count("interactive_terminal_required") == 1
    assert "schedule_state_error" not in result.stdout
    assert str(schedule_path) not in result.stdout
    assert "{corrupt" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert result.stderr == ""
    assert schedule_path.read_text(encoding="utf-8") == "{corrupt"
    assert not (state_path / "logs").exists()


def test_installed_myclaw_rejects_non_tty_before_user_home_workspace_validation(
    agent_home: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, workspace=agent_home.parent)

    assert result.returncode == 2
    assert result.stdout.count("interactive_terminal_required") == 1
    assert "Workspace State" not in result.stdout
    assert str(agent_home) not in result.stdout
    assert "Traceback" not in result.stdout + result.stderr
    assert result.stderr == ""
    assert not (agent_home / "memory").exists()
    assert not (agent_home / "sessions").exists()
