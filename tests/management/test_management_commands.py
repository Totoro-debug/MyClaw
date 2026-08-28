import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.errors import ErrorInfo
from myclaw.management.commands import (
    MANAGEMENT_COMMANDS,
    RESUME_MANAGEMENT_COMMAND,
    ManagementCommandDispatcher,
)
from myclaw.management.service import RuntimeStatusInput
from myclaw.memory.dream import DreamResult
from myclaw.memory.manager import MemoryManager
from myclaw.session.session import Session
from tests.fixtures.diagnostic_capture import configured_process_logging
from tests.management.factories import management_service

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


class _ResultDream:
    def __init__(self, result: DreamResult) -> None:
        self.result = result
        self.calls = 0

    async def run(self) -> DreamResult:
        self.calls += 1
        return self.result


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


def test_management_command_catalog_owns_ordered_tokens_and_descriptions() -> None:
    assert tuple((command.token, command.description) for command in MANAGEMENT_COMMANDS) == (
        ("/config", "View User Configuration"),
        ("/status", "View Runtime Status"),
        ("/resume", "Resume a Conversation Session"),
        ("/memory", "View Long-term Memory"),
        ("/dream", "Process pending Conversation Summaries"),
    )
    assert all(command.token and command.description for command in MANAGEMENT_COMMANDS)
    assert any(command is RESUME_MANAGEMENT_COMMAND for command in MANAGEMENT_COMMANDS)


@pytest.mark.parametrize(
    ("dream_result", "expected"),
    (
        (
            DreamResult(
                status="No pending summaries",
                processed_count=0,
                memory_updated=False,
                cursor=0,
            ),
            "No pending summaries",
        ),
        (
            DreamResult(
                status="Memory Task failed.",
                processed_count=0,
                memory_updated=False,
                cursor=1,
                error=ErrorInfo("model_failed", "Memory model failed."),
            ),
            (
                "model_failed: Memory model failed.\n"
                "processed_count: 0\n"
                "memory_updated: false\n"
                "cursor: 1"
            ),
        ),
        (
            DreamResult(
                status="Memory Task failed.",
                processed_count=0,
                memory_updated=False,
                cursor=0,
                error=ErrorInfo(
                    "persistence_error",
                    "Summary Cursor could not be updated.",
                ),
            ),
            (
                "persistence_error: Summary Cursor could not be updated.\n"
                "processed_count: 0\n"
                "memory_updated: false\n"
                "cursor: 0"
            ),
        ),
    ),
    ids=("no-pending", "model-failure", "cursor-publication-failure"),
)
@pytest.mark.asyncio
async def test_dream_command_projects_the_complete_final_result(
    agent_home: Path,
    dream_result: DreamResult,
    expected: str,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    dream = _ResultDream(dream_result)
    dispatcher = ManagementCommandDispatcher(management_service(home, dream=dream))

    result = await dispatcher.dispatch("/dream")

    assert result.handled is True
    assert result.output == expected
    assert dream.calls == 1


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


class _StatusProjectionLoop:
    def __init__(self, projection: RuntimeStatusInput) -> None:
        self._projection = projection

    def runtime_status_input(self) -> RuntimeStatusInput:
        return self._projection


DEFAULT_CONFIG_CONTENT = """[runtime]
max_tool_result_chars = 4096
max_iterations = 50
enable_skill_always_load = false

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

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
    dispatcher = ManagementCommandDispatcher(management_service(home))

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
    dispatcher = ManagementCommandDispatcher(management_service(home))
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
    dispatcher = ManagementCommandDispatcher(management_service(home))
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
    dispatcher = ManagementCommandDispatcher(management_service(home))

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
    dispatcher = ManagementCommandDispatcher(management_service(AgentHome(agent_home)))

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
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    content = "# Long-term Memory\n\n## Lesson\n\u5b8c\u6574\u5185\u5bb9\n" + (
        "memory-line\n" * 8_000
    )
    state.long_term_memory_path.write_text(content, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            workspace_state=state,
            memory_manager=MemoryManager(state),
        )
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
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    memory_manager = MemoryManager(state)
    state.long_term_memory_path.unlink()
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            workspace_state=state,
            memory_manager=memory_manager,
        )
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
    class FailingStatusProjection:
        def runtime_status_input(self) -> RuntimeStatusInput:
            raise OSError("PRIVATE_STATUS_PERSISTENCE_BODY_52")

    def failing_loop() -> FailingStatusProjection:
        return FailingStatusProjection()

    home = AgentHome(agent_home)
    home.initialize()
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            current_agent_loop=failing_loop,
            monotonic=lambda: 0.0,
        )
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
    state = WorkspaceState(workspace)
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
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            workspace_state=state,
            current_agent_loop=lambda: _StatusProjectionLoop(
                RuntimeStatusInput(
                    system_prompt="abcd",
                    retained_messages=(),
                    tool_definitions=(),
                    runtime_context="",
                    session_id=session.session_id,
                    session_title="New Conversation",
                    session_message_count=len(session.messages),
                    last_consolidated=session.last_consolidated,
                    cumulative_usage=(
                        ("model_calls", 1),
                        ("input_tokens", 10),
                        ("output_tokens", 3),
                        ("total_tokens", 13),
                    ),
                    chat_model="fallback/chat-model",
                    context_window=8,
                    generation_started_at=10.0,
                )
            ),
            monotonic=lambda: 75.8,
        )
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
        "schedule": {"status": "available", "active_job_count": 0},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/unknown", "/config extra", "/memory ", "/CONFIG"])
async def test_unknown_or_inexact_slash_command_is_left_unhandled(
    agent_home: Path,
    command: str,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    dispatcher = ManagementCommandDispatcher(management_service(home))

    result = await dispatcher.dispatch(command)

    assert (result.handled, result.output) == (False, None)


@pytest.mark.asyncio
async def test_config_and_memory_commands_bypass_conversation_and_provider(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(CONFIG_CONTENT, encoding="utf-8")
    state.long_term_memory_path.write_text("current memory\n", encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            workspace_state=state,
            memory_manager=MemoryManager(state),
        )
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
