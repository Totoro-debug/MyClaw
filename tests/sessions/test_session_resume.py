import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.runtime import prepare_runtime
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.management.service import ManagementViewService
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
)
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.terminal.repl import run_repl
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript

NOW = datetime(2026, 8, 1, 12, 0, 0, 123000, tzinfo=timezone(timedelta(hours=8)))
FIRST_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
SECOND_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    return state


def _session(state: WorkspaceState, session_uuid: UUID, title: str) -> Session:
    session = Session.create(state, now=lambda: NOW, new_uuid=lambda: session_uuid)
    session.update_metadata(title=title)
    session.add_message("user", f"History for {title}.")
    return session


@pytest.mark.asyncio
async def test_resume_listing_returns_current_workspace_sessions_in_update_order(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    older = _session(state, FIRST_UUID, "Older session")
    newer = _session(state, SECOND_UUID, "Newer session")
    older.close()
    newer.update_metadata(title="Newest session")
    newer.close()
    service = ManagementViewService(home, workspace_state=state)

    listing = await service.resumable_listing()

    assert [item.id for item in listing.sessions] == [newer.session_id, older.session_id]
    assert [item.title for item in listing.sessions] == ["Newest session", "Older session"]
    assert [item.message_count for item in listing.sessions] == [1, 1]


@pytest.mark.asyncio
async def test_resume_listing_excludes_schedule_session_partition(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    schedule = Session.create(
        state,
        partition=SessionStoragePartition.SCHEDULE,
        job_id=str(FIRST_UUID),
        now=lambda: NOW,
    )
    schedule.add_message("user", "Background work")
    schedule.close()

    listing = await ManagementViewService(home, workspace_state=state).resumable_listing()

    assert listing.sessions == ()
    assert listing.skipped_count == 0
    assert Session.load(state, schedule.session_id).messages == schedule.messages


@pytest.mark.asyncio
async def test_resume_listing_skips_corrupt_entries_without_mutating_them(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    valid = _session(state, FIRST_UUID, "Valid session")
    valid.close()
    corrupt_path = state.sessions_directory / (
        "20260801-120000-123000_6fa459ea-ee8a-4ca4-894e-db77e160355e.jsonl"
    )
    corrupt_path.write_text("not-json\n", encoding="utf-8")
    before = corrupt_path.read_bytes()

    listing = await ManagementViewService(home, workspace_state=state).resumable_listing()

    assert listing.skipped_count == 1
    assert [item.id for item in listing.sessions] == [valid.session_id]
    assert corrupt_path.read_bytes() == before


@pytest.mark.asyncio
async def test_resume_listing_skips_a_session_with_malformed_field_types(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    valid = _session(state, FIRST_UUID, "Valid session")
    valid.close()
    corrupt_id = "20260801-120000-123000_6fa459ea-ee8a-4ca4-894e-db77e160355e"
    corrupt_path = state.sessions_directory / f"{corrupt_id}.jsonl"
    corrupt_path.write_text(
        json.dumps(
            {
                "session_id": corrupt_id,
                "created_at": "2026-08-01T12:00:00.123+08:00",
                "updated_at": "2026-08-01T12:00:00.123+08:00",
                "last_consolidated": 0,
                "metadata": {
                    "title": "Corrupt session",
                    "token_usage": "not-an-object",
                },
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    listing = await ManagementViewService(home, workspace_state=state).resumable_listing()

    assert listing.skipped_count == 1
    assert [item.id for item in listing.sessions] == [valid.session_id]


@pytest.mark.asyncio
async def test_resume_selects_the_loaded_session_for_the_runtime_owner(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    target = _session(state, SECOND_UUID, "Target session")
    target.close()
    selected: list[tuple[str, bool]] = []

    async def replace_session(session_id: str, force: bool) -> None:
        selected.append((session_id, force))

    service = ManagementViewService(
        home,
        workspace_state=state,
        replace_session=replace_session,
    )

    result = await service.resume(target.session_id)

    assert result.session_id == target.session_id
    assert selected == [(target.session_id, False)]


class _ScriptedInput:
    def __init__(self, values: tuple[str | None, ...]) -> None:
        self._values = iter(values)

    async def read(self) -> str | None:
        return next(self._values)


class _RecordingWriter:
    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []

    async def write_delta(self, delta: str) -> None:
        self.operations.append(("delta", delta))

    async def finish_turn(self) -> None:
        self.operations.append(("finish", ""))

    async def write_line(self, content: str) -> None:
        self.operations.append(("line", content))


@pytest.mark.asyncio
async def test_repl_resume_cannot_bypass_the_runtime_host_generation_owner(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    target = _session(state, SECOND_UUID, "Target session")
    target.close()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Continued answer."),
                            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=lambda: FIRST_UUID,
    )
    initial_session_id = runtime.session_id
    await runtime.start()
    writer = _RecordingWriter()
    await run_repl(
        bus=runtime.bus,
        control=runtime.control,
        input_reader=_ScriptedInput(("/resume", "1", "Continue here", "exit")),
        writer=writer,
        management_dispatcher=runtime.management_dispatcher,
    )
    await runtime.close()

    assert runtime.session_id == initial_session_id
    assert writer.operations[0][1].startswith("Resumable sessions:\n1. Target session |")
    assert writer.operations[1] == (
        "line",
        "route_unavailable: Session resume is unavailable.",
    )
    assert writer.operations[-1] == ("finish", "")
    assert [
        message["content"] for message in runtime.session.messages if message["role"] == "user"
    ][-1] == ("Continue here")
