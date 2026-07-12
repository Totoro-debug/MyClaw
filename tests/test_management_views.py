from pathlib import Path

import pytest

from myclaw.agent_home import AgentHome
from myclaw.management import ManagementError, ManagementViewService

CONFIG_WITH_PLAINTEXT_KEYS = """# User Configuration remains source-preserved.
[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "first-plaintext-key"
models = ["model-id"]

[models.providers.empty-template]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

EXPECTED_REDACTED_CONFIG = """# User Configuration remains source-preserved.
[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "***REDACTED***"
models = ["model-id"]

[models.providers.empty-template]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""


@pytest.mark.asyncio
async def test_config_view_returns_complete_source_with_plaintext_keys_redacted(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(CONFIG_WITH_PLAINTEXT_KEYS, encoding="utf-8")

    view = await ManagementViewService(home).config_view()

    assert (view.path, view.redacted_content, view.error) == (
        config_path,
        EXPECTED_REDACTED_CONFIG,
        None,
    )


@pytest.mark.asyncio
async def test_config_view_converts_decode_failure_to_safe_persistence_error(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_bytes(b'api_key = "raw-secret"\xff')

    with pytest.raises(ManagementError) as raised:
        await ManagementViewService(home).config_view()

    assert (raised.value.error.code, raised.value.error.message) == (
        "persistence_error",
        "User Configuration could not be read or written.",
    )
    assert "raw-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_memory_view_reads_complete_latest_utf8_content_on_every_call(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    memory_path = agent_home / "memory" / "memory.md"
    memory_path.write_text("initial memory\n", encoding="utf-8")
    service = ManagementViewService(home)

    initial = await service.memory_view()
    updated_content = "# Long-term Memory\n\n\u7528\u6237\u504f\u597d\n" + (
        "complete-content\n" * 8_000
    )
    memory_path.write_text(updated_content, encoding="utf-8")
    updated = await service.memory_view()

    assert initial == "initial memory\n"
    assert updated == updated_content


@pytest.mark.asyncio
async def test_memory_view_converts_decode_failure_to_safe_persistence_error(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "memory" / "memory.md").write_bytes(b"raw-secret\xff")

    with pytest.raises(ManagementError) as raised:
        await ManagementViewService(home).memory_view()

    assert (raised.value.error.code, raised.value.error.message) == (
        "persistence_error",
        "Long-term Memory could not be read.",
    )
    assert str(raised.value) == "Long-term Memory could not be read."
    assert "raw-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_memory_view_converts_read_failure_to_safe_persistence_error(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "memory" / "memory.md").unlink()

    with pytest.raises(ManagementError) as raised:
        await ManagementViewService(home).memory_view()

    assert (raised.value.error.code, raised.value.error.message) == (
        "persistence_error",
        "Long-term Memory could not be read.",
    )
