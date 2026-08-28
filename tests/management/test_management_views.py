from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.errors import ErrorInfo
from myclaw.management.service import (
    ManagementError,
    RuntimeStatus,
    RuntimeStatusInput,
    RuntimeStatusService,
)
from myclaw.memory.dream import DreamResult
from myclaw.memory.manager import MemoryManager
from myclaw.session.session import Session
from tests.management.factories import management_service

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


class _StatusProjectionLoop:
    def __init__(self, projection: RuntimeStatusInput) -> None:
        self._projection = projection
        self.projection_calls = 0

    def runtime_status_input(self) -> RuntimeStatusInput:
        self.projection_calls += 1
        return self._projection


class _MemoryReader:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    async def read_long_term(self) -> str:
        self.calls += 1
        return self._content


class _DreamRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self) -> DreamResult:
        self.calls += 1
        return DreamResult(
            status="Dream complete",
            processed_count=1,
            memory_updated=True,
            cursor=1,
        )


@pytest.mark.asyncio
async def test_config_view_returns_complete_source_with_plaintext_keys_redacted(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_path = agent_home / "config.toml"
    config_path.write_text(CONFIG_WITH_PLAINTEXT_KEYS, encoding="utf-8")

    view = await management_service(home).config_view()

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
        await management_service(home).config_view()

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
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    memory_path = state.long_term_memory_path
    memory_path.write_text("initial memory\n", encoding="utf-8")
    service = management_service(
        home,
        workspace_state=state,
        memory_manager=MemoryManager(state),
    )

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
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    memory_manager = MemoryManager(state)
    state.long_term_memory_path.write_bytes(b"raw-secret\xff")

    with pytest.raises(ManagementError) as raised:
        await management_service(
            home,
            workspace_state=state,
            memory_manager=memory_manager,
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
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    memory_manager = MemoryManager(state)
    state.long_term_memory_path.unlink()

    with pytest.raises(ManagementError) as raised:
        await management_service(
            home,
            workspace_state=state,
            memory_manager=memory_manager,
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
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    session = Session.create(
        state,
        now=lambda: CREATED_AT,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    monotonic = iter((112.9,)).__next__
    management = management_service(
        home,
        workspace_state=state,
        current_agent_loop=lambda: _StatusProjectionLoop(
            RuntimeStatusInput(
                system_prompt="abcd",
                retained_messages=("\u00e9",),
                tool_definitions=("tool",),
                runtime_context="\u4f60",
                session_id=session.session_id,
                session_title="New Conversation",
                context_window=10,
                chat_model="primary/model-id",
                cumulative_usage=(
                    ("model_calls", 0),
                    ("input_tokens", 0),
                    ("output_tokens", 0),
                    ("total_tokens", 0),
                ),
                generation_started_at=100.0,
            )
        ),
        monotonic=monotonic,
    )

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
        schedule={"status": "available", "active_job_count": 0},
    )


@pytest.mark.asyncio
async def test_status_reads_one_current_loop_projection_per_request_and_resets_uptime(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    first = _StatusProjectionLoop(
        RuntimeStatusInput(
            system_prompt="first",
            retained_messages=(),
            tool_definitions=(),
            runtime_context="",
            session_id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
            session_title="First generation",
            session_message_count=2,
            last_consolidated=1,
            cumulative_usage=(
                ("model_calls", 1),
                ("input_tokens", 10),
                ("output_tokens", 3),
                ("total_tokens", 13),
            ),
            chat_model="first-provider/first-model",
            context_window=100,
            generation_started_at=10.0,
        )
    )
    second = _StatusProjectionLoop(
        RuntimeStatusInput(
            system_prompt="second",
            retained_messages=(),
            tool_definitions=(),
            runtime_context="",
            session_id="20260711-153012-123457_6fa459ea-ee8a-4ca4-894e-db77e160355e",
            session_title="Second generation",
            session_message_count=5,
            last_consolidated=4,
            cumulative_usage=(
                ("model_calls", 2),
                ("input_tokens", 20),
                ("output_tokens", 6),
                ("total_tokens", 26),
            ),
            chat_model="second-provider/second-model",
            context_window=200,
            generation_started_at=50.0,
        )
    )
    current = first
    provider_calls = 0

    def current_loop() -> _StatusProjectionLoop:
        nonlocal provider_calls
        provider_calls += 1
        return current

    monotonic = iter((100.0, 60.0)).__next__
    service = RuntimeStatusService(
        current_agent_loop=current_loop,
        monotonic=monotonic,
    )

    first_status = await service.status()
    current = second
    second_status = await service.status()

    assert first_status.session_message_count == 2
    assert first_status.last_consolidated == 1
    assert first_status.chat_model == "first-provider/first-model"
    assert first_status.context_window == 100
    assert first_status.uptime_seconds == 90
    assert first_status.cumulative_usage["total_tokens"] == 13
    assert second_status.session_message_count == 5
    assert second_status.last_consolidated == 4
    assert second_status.chat_model == "second-provider/second-model"
    assert second_status.context_window == 200
    assert second_status.uptime_seconds == 10
    assert second_status.cumulative_usage["total_tokens"] == 26
    assert provider_calls == 2
    assert first.projection_calls == 1
    assert second.projection_calls == 1


@pytest.mark.asyncio
async def test_management_status_builds_once_from_the_current_loop_projection(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    loop = _StatusProjectionLoop(
        RuntimeStatusInput(
            system_prompt="status",
            retained_messages=(),
            tool_definitions=(),
            runtime_context="",
            session_id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
            session_title="Current generation",
            session_message_count=3,
            last_consolidated=2,
            cumulative_usage=(
                ("model_calls", 1),
                ("input_tokens", 2),
                ("output_tokens", 3),
                ("total_tokens", 5),
            ),
            chat_model="provider/model",
            context_window=100,
            generation_started_at=10.0,
        )
    )
    provider_calls = 0
    schedule_calls = 0

    def current_loop() -> _StatusProjectionLoop:
        nonlocal provider_calls
        provider_calls += 1
        return loop

    def schedule_status() -> dict[str, object]:
        nonlocal schedule_calls
        schedule_calls += 1
        return {"status": "available", "active_job_count": 2}

    service = management_service(
        home,
        current_agent_loop=current_loop,
        monotonic=lambda: 15.0,
        schedule_status=schedule_status,
    )

    status = await service.status()

    assert status.session_message_count == 3
    assert status.last_consolidated == 2
    assert status.uptime_seconds == 5
    assert status.cumulative_usage["total_tokens"] == 5
    assert provider_calls == 1
    assert loop.projection_calls == 1
    assert schedule_calls == 1
    assert status.schedule == {"status": "available", "active_job_count": 2}


@pytest.mark.asyncio
async def test_status_uptime_clamps_a_monotonic_clock_rollback_to_zero(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    loop = _StatusProjectionLoop(
        RuntimeStatusInput(
            system_prompt="status",
            retained_messages=(),
            tool_definitions=(),
            runtime_context="",
            session_id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
            session_title="Rollback generation",
            chat_model="provider/model",
            context_window=100,
            generation_started_at=100.0,
        )
    )
    service = management_service(
        home,
        current_agent_loop=lambda: loop,
        monotonic=lambda: 50.0,
    )

    status = await service.status()

    assert status.uptime_seconds == 0


@pytest.mark.asyncio
async def test_generation_sensitive_views_snapshot_each_current_provider_once(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    target = Session.create(
        state,
        now=lambda: CREATED_AT,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    target.add_message("user", "Resume target")
    target.close()
    loop = _StatusProjectionLoop(
        RuntimeStatusInput(
            system_prompt="current",
            retained_messages=(),
            tool_definitions=(),
            runtime_context="",
        )
    )
    memory = _MemoryReader("current memory")
    dream = _DreamRunner()
    current_loop_calls = 0
    replacements: list[tuple[str, bool]] = []

    def current_loop() -> _StatusProjectionLoop:
        nonlocal current_loop_calls
        current_loop_calls += 1
        return loop

    async def replace_agent_loop(session_id: str, force: bool) -> None:
        replacements.append((session_id, force))

    service = management_service(
        home,
        current_agent_loop=current_loop,
        workspace_state=state,
        replace_agent_loop=replace_agent_loop,
        memory_manager=memory,
        dream=dream,
    )

    assert await service.memory_view() == "current memory"
    dream_result = await service.dream()
    resume_result = await service.resume(target.session_id, force=True)

    assert dream_result.status == "Dream complete"
    assert resume_result.session_id == target.session_id
    assert current_loop_calls == 3
    assert memory.calls == 1
    assert dream.calls == 1
    assert replacements == [(target.session_id, True)]

    invalid_id = "20260711-153012-123456_6fa459ea-ee8a-4ca4-894e-db77e160355e"
    with pytest.raises(ManagementError) as raised:
        await service.resume(invalid_id)

    assert raised.value.error.code == "model_invalid_request"
    assert current_loop_calls == 4
    assert replacements == [(target.session_id, True)]


@pytest.mark.asyncio
async def test_resume_prepares_before_loading_resumable_sessions(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    target = Session.create(
        state,
        now=lambda: CREATED_AT,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    target.add_message("user", "Resume target")
    loop = _StatusProjectionLoop(
        RuntimeStatusInput(
            system_prompt="current",
            retained_messages=(),
            tool_definitions=(),
            runtime_context="",
        )
    )
    events: list[str] = []

    async def prepare_session_resume(session_id: str) -> None:
        assert session_id == target.session_id
        events.append("prepare")
        target.close()

    async def replace_agent_loop(session_id: str, force: bool) -> None:
        assert session_id == target.session_id
        assert force is False
        events.append("replace")

    service = management_service(
        home,
        current_agent_loop=lambda: loop,
        workspace_state=state,
        replace_agent_loop=replace_agent_loop,
        prepare_session_resume=prepare_session_resume,
    )

    result = await service.resume(target.session_id)

    assert result.session_id == target.session_id
    assert events == ["prepare", "replace"]


@pytest.mark.asyncio
async def test_generation_sensitive_views_keep_unavailable_error_and_skip_resources(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    memory = _MemoryReader("must not be read")
    dream = _DreamRunner()
    current_calls = 0
    schedule_calls = 0
    prepare_calls = 0
    replacements: list[tuple[str, bool]] = []

    def unavailable_loop() -> _StatusProjectionLoop:
        nonlocal current_calls
        current_calls += 1
        raise ManagementError(ErrorInfo("route_unavailable", "Runtime Generation is unavailable."))

    def schedule_status() -> dict[str, object]:
        nonlocal schedule_calls
        schedule_calls += 1
        return {"status": "available"}

    async def prepare_session_resume(_session_id: str) -> None:
        nonlocal prepare_calls
        prepare_calls += 1

    async def replace_agent_loop(session_id: str, force: bool) -> None:
        replacements.append((session_id, force))

    service = management_service(
        home,
        current_agent_loop=unavailable_loop,
        workspace_state=state,
        replace_agent_loop=replace_agent_loop,
        prepare_session_resume=prepare_session_resume,
        memory_manager=memory,
        dream=dream,
        schedule_status=schedule_status,
    )
    unavailable_id = "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000"

    operations = (
        service.status,
        service.memory_view,
        service.dream,
        service.resumable_listing,
        lambda: service.resume(unavailable_id),
    )
    for operation in operations:
        with pytest.raises(ManagementError) as raised:
            await operation()
        assert (raised.value.error.code, raised.value.error.message) == (
            "route_unavailable",
            "Runtime Generation is unavailable.",
        )

    assert current_calls == 5
    assert schedule_calls == 0
    assert prepare_calls == 0
    assert memory.calls == 0
    assert dream.calls == 0
    assert replacements == []
