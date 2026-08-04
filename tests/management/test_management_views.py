from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.management.service import (
    ManagementError,
    ManagementViewService,
    ResolvedChatStatus,
    RuntimeStatus,
    RuntimeStatusInput,
    RuntimeStatusService,
)
from myclaw.memory.memory_task import WorkspaceFileMemoryStore
from myclaw.session.session import Session

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

LOCAL_OFFSET = timezone(timedelta(hours=8))
CREATED_AT = datetime(2026, 7, 11, 15, 30, 12, 123456, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")


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
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    memory_path = state.long_term_memory_path
    memory_path.write_text("initial memory\n", encoding="utf-8")
    service = ManagementViewService(home, memory_store=WorkspaceFileMemoryStore(state))

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
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    state.long_term_memory_path.write_bytes(b"raw-secret\xff")

    with pytest.raises(ManagementError) as raised:
        await ManagementViewService(
            home, memory_store=WorkspaceFileMemoryStore(state)
        ).memory_view()

    assert (raised.value.error.code, raised.value.error.message) == (
        "persistence_error",
        "Long-term Memory could not be read.",
    )
    assert str(raised.value) == "Long-term Memory could not be read."
    assert "raw-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_memory_view_converts_read_failure_to_safe_persistence_error(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    state.long_term_memory_path.unlink()

    with pytest.raises(ManagementError) as raised:
        await ManagementViewService(
            home, memory_store=WorkspaceFileMemoryStore(state)
        ).memory_view()

    assert (raised.value.error.code, raised.value.error.message) == (
        "persistence_error",
        "Long-term Memory could not be read.",
    )


@pytest.mark.asyncio
async def test_status_reports_prepared_session_and_frozen_utf8_token_estimate(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(
        state,
        now=lambda: CREATED_AT,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    monotonic = iter((100.0, 112.9)).__next__
    status_service = RuntimeStatusService(
        session=session,
        resolved_chat=lambda: ResolvedChatStatus(
            provider_id="primary",
            model="model-id",
            context_window=10,
        ),
        next_input=lambda _session: RuntimeStatusInput(
            system_prompt="abcd",
            retained_messages=("\u00e9",),
            tool_definitions=("tool",),
            runtime_context="\u4f60",
        ),
        monotonic=monotonic,
    )
    management = ManagementViewService(home, status_service=status_service)

    status = await management.status()

    assert status == RuntimeStatus(
        version="0.1.0",
        chat_model="primary/model-id",
        uptime_seconds=12,
        estimated_input_tokens=4,
        context_window=10,
        context_used_percent=40.0,
        session_message_count=0,
        last_consolidated=0,
        cumulative_usage={
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    )
