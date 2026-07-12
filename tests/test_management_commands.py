from pathlib import Path

import pytest

from myclaw.agent_home import AgentHome
from myclaw.management import ManagementViewService
from myclaw.management_commands import ManagementCommandDispatcher

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
max_tool_result_chars = 50000

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[tools.web]
enabled = true

[tools.shell]
enabled = true

[models.providers.anthropic-default]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = ""
models = []

[models.providers.openai-local]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []
"""


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
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(MALFORMED_CONFIG_CONTENT, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))

    result = await dispatcher.dispatch("/config")

    assert result.handled is True
    assert result.output == (
        "config_parse_error: User Configuration TOML could not be parsed.\n"
        f"Path: {config_path}\n"
        f"{REDACTED_MALFORMED_CONFIG_CONTENT}"
    )
    assert "first-command-secret" not in result.output
    assert "second-command-secret" not in result.output


@pytest.mark.asyncio
async def test_config_command_renders_safe_persistence_failure(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_bytes(b'api_key = "raw-command-secret"\xff')
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))

    result = await dispatcher.dispatch("/config")

    assert (result.handled, result.output) == (
        True,
        "persistence_error: User Configuration could not be read or written.",
    )
    assert result.output is not None
    assert "raw-command-secret" not in result.output


@pytest.mark.asyncio
async def test_config_command_keeps_schema_invalid_source_inspectable(
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
async def test_memory_command_returns_renderable_complete_disk_text(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    content = "# Long-term Memory\n\n## Lesson\n\u5b8c\u6574\u5185\u5bb9\n" + (
        "memory-line\n" * 8_000
    )
    (agent_home / "memory" / "memory.md").write_text(content, encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))

    result = await dispatcher.dispatch("/memory")

    assert result.handled is True
    assert result.output == content


@pytest.mark.asyncio
async def test_memory_command_renders_safe_persistence_failure(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "memory" / "memory.md").unlink()
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))

    result = await dispatcher.dispatch("/memory")

    assert (result.handled, result.output) == (
        True,
        "persistence_error: Long-term Memory could not be read.",
    )


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
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(CONFIG_CONTENT, encoding="utf-8")
    (agent_home / "memory" / "memory.md").write_text("current memory\n", encoding="utf-8")
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home))
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
