import asyncio
import os
import shutil
import subprocess
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import typer

import myclaw.terminal.cli as cli
from tests.configuration.test_config import (
    EXPECTED_DEFAULT_CONFIG,
    EXPECTED_REDACTED_CONFIG,
    EXPECTED_REDACTED_MALFORMED_CONFIG,
    MALFORMED_CONFIG,
    REDACTION_CONFIG,
    VALID_CONFIG,
)


def run_installed_myclaw(agent_home: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("myclaw")
    assert executable is not None
    environment = os.environ.copy()
    environment["HOME"] = str(agent_home.parent)
    environment["USERPROFILE"] = str(agent_home.parent)
    source_root = str(Path(__file__).parent.parent)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing_pythonpath else os.pathsep.join((source_root, existing_pythonpath))
    )
    return subprocess.run(
        [executable, *arguments],
        capture_output=True,
        check=False,
        cwd=agent_home.parent,
        env=environment,
        text=True,
    )


def assert_plaintext_absent(output: str, *plaintext_values: str) -> None:
    if any(value in output for value in plaintext_values):
        pytest.fail("CLI output leaked a plaintext provider API key", pytrace=False)


def test_cli_drains_interrupts_before_restoring_handler_and_preserves_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeLoader:
        def __init__(self, agent_home: object) -> None:
            self.agent_home = agent_home
            self.path = Path("config.toml")

        def load_for_startup(self) -> object:
            return object()

    class FakeConversation:
        async def cancel_active_turn(self) -> None:
            return None

    class FailingRuntime:
        conversation = FakeConversation()

        async def run(self, *, input_reader: object, writer: object) -> None:
            del input_reader, writer
            events.append("runtime")
            raise LookupError("runtime failed")

    class FailingInterruptController:
        def __init__(
            self,
            *,
            loop: asyncio.AbstractEventLoop,
            cancel_foreground: Callable[[], Awaitable[None]],
        ) -> None:
            del loop, cancel_foreground
            self.installed = False

        def install(self) -> None:
            self.installed = True
            events.append("install")

        async def close(self) -> None:
            events.append("interrupt-close")
            if not self.installed:
                raise AssertionError("SIGINT handler restored before interrupt drain")
            raise RuntimeError("interrupt cleanup failed")

        def restore(self) -> None:
            self.installed = False
            events.append("restore")

    class RecordingRunner:
        def __init__(self) -> None:
            self._loop = asyncio.new_event_loop()

        def __enter__(self) -> "RecordingRunner":
            return self

        def __exit__(self, *errors: object) -> None:
            del errors
            self._loop.close()

        def get_loop(self) -> asyncio.AbstractEventLoop:
            return self._loop

        def run(self, awaitable: Coroutine[object, object, object]) -> object:
            return self._loop.run_until_complete(awaitable)

    runtime = FailingRuntime()
    monkeypatch.setattr(cli, "ConfigLoader", FakeLoader)
    monkeypatch.setattr(cli, "prepare_repl_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(cli, "ForegroundInterruptController", FailingInterruptController)
    monkeypatch.setattr(asyncio, "Runner", RecordingRunner)
    context = cast(typer.Context, SimpleNamespace(invoked_subcommand=None))

    with pytest.raises(LookupError, match="runtime failed") as raised:
        cli.main(context)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "interrupt cleanup failed"
    assert events == ["install", "runtime", "interrupt-close", "restore"]


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


def test_installed_myclaw_generates_missing_configuration_and_stops(agent_home: Path) -> None:
    result = run_installed_myclaw(agent_home)

    assert result.returncode == 2
    assert (agent_home / "config.toml").read_text(encoding="utf-8") == EXPECTED_DEFAULT_CONFIG
    assert "config_missing" in result.stdout
    assert str(agent_home / "config.toml") in result.stdout
    assert "edit" in result.stdout.lower()
    assert "configuration gate passed" not in result.stdout
    runtime_log = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert "ERROR" in runtime_log
    assert "session=- myclaw.terminal.cli: Startup failed code=config_missing" in runtime_log


def test_installed_config_command_generates_and_displays_missing_configuration(
    agent_home: Path,
) -> None:
    result = run_installed_myclaw(agent_home, "config")

    assert result.returncode == 0, result.stderr
    assert f"Path: {agent_home / 'config.toml'}" in result.stdout
    assert EXPECTED_DEFAULT_CONFIG in result.stdout
    assert "configuration gate passed" not in result.stdout
    assert not (agent_home / "logs").exists()


def test_installed_config_command_redacts_valid_configuration(agent_home: Path) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(REDACTION_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config")

    assert result.returncode == 0, result.stderr
    assert EXPECTED_REDACTED_CONFIG in result.stdout
    assert f"Path: {agent_home / 'config.toml'}" in result.stdout
    assert_plaintext_absent(result.stdout + result.stderr, "plaintext-primary-key")


def test_installed_config_command_shows_safe_malformed_configuration(
    agent_home: Path,
) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(MALFORMED_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config")

    assert result.returncode == 2
    assert "config_parse_error" in result.stdout
    assert f"Path: {agent_home / 'config.toml'}" in result.stdout
    assert EXPECTED_REDACTED_MALFORMED_CONFIG in result.stdout
    assert_plaintext_absent(
        result.stdout + result.stderr,
        "first-plaintext-key",
        "second-plaintext-key",
    )
    runtime_log = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert "ERROR" in runtime_log
    assert (
        "session=- myclaw.terminal.cli: Configuration command failed code=config_parse_error"
        in runtime_log
    )
    assert_plaintext_absent(runtime_log, "first-plaintext-key", "second-plaintext-key")
    assert "Traceback (most recent call last)" not in runtime_log
    assert "ConfigError" not in runtime_log


def test_installed_config_command_keeps_schema_invalid_content_inspectable(
    agent_home: Path,
) -> None:
    agent_home.mkdir(parents=True)
    content = REDACTION_CONFIG.replace(
        "max_tool_result_chars = 50000",
        "max_tool_result_chars = 50000\nmisspelled_setting = true",
    )
    (agent_home / "config.toml").write_text(content, encoding="utf-8")

    result = run_installed_myclaw(agent_home, "config")

    assert result.returncode == 2
    assert "config_invalid" in result.stdout
    assert "runtime.misspelled_setting" in result.stdout
    assert "misspelled_setting = true" in result.stdout
    assert_plaintext_absent(result.stdout + result.stderr, "plaintext-primary-key")


def test_installed_myclaw_passes_valid_configuration_gate(agent_home: Path) -> None:
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")

    result = run_installed_myclaw(agent_home)

    assert result.returncode == 0, result.stderr
    assert "configuration gate passed" in result.stdout
    assert_plaintext_absent(result.stdout + result.stderr, "sk-ant-secret")


def test_installed_myclaw_stops_on_parse_schema_and_default_failures(agent_home: Path) -> None:
    agent_home.mkdir(parents=True)
    config_path = agent_home / "config.toml"

    config_path.write_text(MALFORMED_CONFIG, encoding="utf-8")
    parse_result = run_installed_myclaw(agent_home)

    schema_content = REDACTION_CONFIG.replace(
        "max_tool_result_chars = 50000",
        "max_tool_result_chars = 50000\nmisspelled_setting = true",
    )
    config_path.write_text(schema_content, encoding="utf-8")
    schema_result = run_installed_myclaw(agent_home)

    config_path.write_text(EXPECTED_DEFAULT_CONFIG, encoding="utf-8")
    default_result = run_installed_myclaw(agent_home)

    assert (parse_result.returncode, schema_result.returncode, default_result.returncode) == (
        2,
        2,
        2,
    )
    assert "config_parse_error" in parse_result.stdout
    assert "config_invalid" in schema_result.stdout
    assert "route_unavailable" in default_result.stdout
    assert all(
        "configuration gate passed" not in result.stdout
        for result in (parse_result, schema_result, default_result)
    )
    combined_output = "".join(
        result.stdout + result.stderr for result in (parse_result, schema_result, default_result)
    )
    assert_plaintext_absent(
        combined_output,
        "first-plaintext-key",
        "second-plaintext-key",
        "plaintext-primary-key",
    )
    runtime_log = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert (
        "myclaw.config.config.ConfigError: User Configuration TOML could not be parsed."
        in runtime_log
    )
    assert "Traceback (most recent call last)" in runtime_log
    assert "[models.providers.primary]" not in runtime_log
    assert "api_key =" not in runtime_log
    assert_plaintext_absent(
        runtime_log,
        "first-plaintext-key",
        "second-plaintext-key",
        "plaintext-primary-key",
    )
