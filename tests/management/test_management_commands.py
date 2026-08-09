import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import (
    ManagementViewService,
    ResolvedChatStatus,
    RuntimeStatusInput,
    RuntimeStatusService,
)
from myclaw.memory.memory_task import WorkspaceFileMemoryStore
from myclaw.session.session import Session
from tests.fixtures.diagnostic_capture import configured_process_logging

CONFIG_CONTENT = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "command-secret"
models = ["model-id"]

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

REDACTED_CONFIG_CONTENT = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "***REDACTED***"
models = ["model-id"]

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

MALFORMED_CONFIG_CONTENT = """[runtime
api_key = "first-command-secret"
  API-Key = 'second-command-secret' # this value is not safe
not_api_key = "diagnostic-content"
broken = [
"""

REDACTED_MALFORMED_CONFIG_CONTENT = """[runtime
api_key = "***REDACTED***"
  API-Key = "***REDACTED***"
not_api_key = "diagnostic-content"
broken = [
"""

SCHEMA_INVALID_CONFIG_CONTENT = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "schema-command-secret"
models = ["model-id"]
unexpected = true

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

REDACTED_SCHEMA_INVALID_CONFIG_CONTENT = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "***REDACTED***"
models = ["model-id"]
unexpected = true

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

DEFAULT_CONFIG_CONTENT = """[runtime]
max_tool_result_chars = 4096

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[tools.web]
enabled = true

[tools.shell]
enabled = true

[models.providers.openai-local]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []

# Replace provider_id, model, and model limits with values supported by your provider.
# Remove any purpose-specific route to fall back to default.
[models.routes.default]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.chat]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.memory]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.schedule]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120
"""

LOCAL_OFFSET = timezone(timedelta(hours=8))
STATUS_CREATED_AT = datetime(2026, 7, 11, 15, 30, tzinfo=LOCAL_OFFSET)
STATUS_SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")


class ConversationProviderSpy:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.provider_calls = 0

    async def converse(self, text: str) -> None:
        self.messages.append(text)
        self.provider_calls += 1


@pytest.mark.asyncio
async def test_config_command_returns_renderable_complete_redacted_text(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(CONFIG_CONTENT, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))

    result = await dispatcher.dispatch("/config")

    assert result.handled is True
    assert result.output == f"Path: {config_path}\n{REDACTED_CONFIG_CONTENT}"
    assert "command-secret" not in result.output


@pytest.mark.asyncio
async def test_config_command_renders_safe_parse_error_and_redacted_source(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(MALFORMED_CONFIG_CONTENT, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))
    with configured_process_logging():
        result = await dispatcher.dispatch("/config")

    assert result.handled is True
    assert result.output == (
        "config_parse_error: User Configuration TOML could not be parsed.\n"
        f"Path: {config_path}\n"
        f"{REDACTED_MALFORMED_CONFIG_CONTENT}"
    )
    assert "first-command-secret" not in result.output
    assert "second-command-secret" not in result.output
    assert capsys.readouterr().err == ""
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_config_command_renders_safe_persistence_failure(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_bytes(b'api_key = "raw-command-secret"\xff')
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))
    with configured_process_logging():
        result = await dispatcher.dispatch("/config")

    assert (result.handled, result.output) == (
        True,
        "persistence_error: User Configuration could not be read or written.",
    )
    assert result.output is not None
    assert "raw-command-secret" not in result.output
    assert capsys.readouterr().err == ""
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_config_command_keeps_undefined_source_inspectable(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(SCHEMA_INVALID_CONFIG_CONTENT, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))

    result = await dispatcher.dispatch("/config")

    assert result.output == (
        "config_invalid: Configuration field "
        "'models.providers.primary.unexpected' is not recognized.\n"
        f"Path: {config_path}\n"
        f"{REDACTED_SCHEMA_INVALID_CONFIG_CONTENT}"
    )
    assert "schema-command-secret" not in result.output


@pytest.mark.asyncio
async def test_config_command_generates_and_displays_missing_configuration(
    agent_home: Path,
) -> None:
    config_path = agent_home / "config.toml"
    dispatcher = ManagementCommandDispatcher(ManagementViewService(AgentHome(agent_home)))

    result = await dispatcher.dispatch("/config")

    assert result.output == f"Path: {config_path}\n{DEFAULT_CONFIG_CONTENT}"
    assert config_path.read_text(encoding="utf-8") == DEFAULT_CONFIG_CONTENT


@pytest.mark.asyncio
async def test_memory_command_returns_renderable_complete_disk_text(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    content = "# Long-term Memory\n\n## Lesson\n\u5b8c\u6574\u5185\u5bb9\n" + (
        "memory-line\n" * 8_000
    )
    state.long_term_memory_path.write_text(content, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_store=WorkspaceFileMemoryStore(state))
    )

    result = await dispatcher.dispatch("/memory")

    assert result.handled is True
    assert result.output == content


@pytest.mark.asyncio
async def test_memory_command_renders_safe_persistence_failure(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    state.long_term_memory_path.unlink()
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_store=WorkspaceFileMemoryStore(state))
    )
    with configured_process_logging():
        result = await dispatcher.dispatch("/memory")

    assert (result.handled, result.output) == (
        True,
        "persistence_error: Long-term Memory could not be read.",
    )
    assert capsys.readouterr().err == ""
    assert not state.logs_directory.exists()


@pytest.mark.asyncio
async def test_status_command_renders_safe_persistence_failure(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_session() -> Session:
        raise OSError("PRIVATE_STATUS_PERSISTENCE_BODY_52")

    home = AgentHome(agent_home)
    home.initialize()
    status_service = RuntimeStatusService(
        session=failing_session,
        resolved_chat=lambda: ResolvedChatStatus(
            provider_id="provider",
            model="model",
            context_window=8,
        ),
        next_input=lambda _session: RuntimeStatusInput(
            system_prompt="",
            retained_messages=(),
            tool_definitions=(),
            runtime_context="",
        ),
        monotonic=lambda: 0.0,
    )
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, status_service=status_service)
    )
    with configured_process_logging():
        result = await dispatcher.dispatch("/status")

    assert result.output == "persistence_error: Runtime status could not be read."
    assert capsys.readouterr().err == ""
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_status_command_renders_actual_runtime_and_session_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(
        state,
        now=lambda: STATUS_CREATED_AT,
        new_uuid=iter((STATUS_SESSION_UUID,)).__next__,
    )
    session.add_message("user", "Session state.")
    session.add_message(
        "assistant",
        "Visible.",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
        },
    )
    session.last_consolidated = 1
    status_service = RuntimeStatusService(
        session=session,
        resolved_chat=lambda: ResolvedChatStatus(
            provider_id="fallback",
            model="chat-model",
            context_window=8,
        ),
        next_input=lambda _session: RuntimeStatusInput(
            system_prompt="abcd",
            retained_messages=(),
            tool_definitions=(),
            runtime_context="",
        ),
        monotonic=iter((10.0, 75.8)).__next__,
    )
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, status_service=status_service)
    )

    result = await dispatcher.dispatch("/status")

    assert result.handled is True
    assert json.loads(result.output or "") == {
        "version": "0.1.0",
        "chat_model": "fallback/chat-model",
        "uptime_seconds": 65,
        "estimated_input_tokens": 1,
        "context_window": 8,
        "context_used_percent": 12.5,
        "session_message_count": 2,
        "last_consolidated": 1,
        "cumulative_usage": {
            "model_calls": 1,
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/unknown", "/config extra", "/memory ", "/CONFIG"])
async def test_unknown_or_inexact_slash_command_is_left_unhandled(
    agent_home: Path,
    command: str,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))

    result = await dispatcher.dispatch(command)

    assert (result.handled, result.output) == (False, None)


@pytest.mark.asyncio
async def test_config_and_memory_commands_bypass_conversation_and_provider(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(CONFIG_CONTENT, encoding="utf-8")
    state.long_term_memory_path.write_text("current memory\n", encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_store=WorkspaceFileMemoryStore(state))
    )
    conversation = ConversationProviderSpy()

    async def dispatch_or_converse(text: str) -> bool:
        result = await dispatcher.dispatch(text)
        if not result.handled:
            await conversation.converse(text)
        return result.handled

    config_handled = await dispatch_or_converse("/config")
    memory_handled = await dispatch_or_converse("/memory")

    assert (config_handled, memory_handled) == (True, True)
    assert conversation.messages == []
    assert conversation.provider_calls == 0

    unknown_handled = await dispatch_or_converse("/ordinary-slash")

    assert unknown_handled is False
    assert conversation.messages == ["/ordinary-slash"]
    assert conversation.provider_calls == 1
