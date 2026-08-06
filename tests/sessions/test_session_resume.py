import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.events import AgentEvent, TurnCompletedPayload, TurnStartedPayload
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import ManagementViewService
from myclaw.provider.models import ModelUsage
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.session.session_resume import SwitchableConversationPort
from myclaw.terminal.repl import run_repl

NOW = datetime(2026, 8, 1, 12, 0, 0, 123000, tzinfo=timezone(timedelta(hours=8)))
FIRST_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
SECOND_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
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
    selected: list[Session] = []
    service = ManagementViewService(
        home,
        workspace_state=state,
        switch_session=selected.append,
    )

    result = await service.resume(target.session_id)

    assert result.session_id == target.session_id
    assert len(selected) == 1
    assert selected[0].session_id == target.session_id
    assert selected[0].messages == target.messages


class _RecordingConversation:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.cancelled = False
        self.closed = False

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        self.submitted.append(text)
        yield AgentEvent(
            type="turn_started",
            event_id=0,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=TurnStartedPayload(),
        )
        yield AgentEvent(
            type="turn_completed",
            event_id=1,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=TurnCompletedPayload(
                content="done",
                usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            ),
        )

    async def cancel_active_turn(self) -> None:
        self.cancelled = True

    def respond_to_confirmation(self, confirmation_id: UUID, decision: str) -> None:
        del confirmation_id, decision

    async def close(self) -> None:
        self.closed = True


class _BlockingConversation(_RecordingConversation):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        self.submitted.append(text)
        yield AgentEvent(
            type="turn_started",
            event_id=0,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=TurnStartedPayload(),
        )
        await self.release.wait()
        yield AgentEvent(
            type="turn_completed",
            event_id=1,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=TurnCompletedPayload(
                content="done",
                usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            ),
        )

    async def cancel_active_turn(self) -> None:
        self.cancelled = True
        self.release.set()


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
async def test_switchable_port_closes_every_delegate_and_preserves_previous_history(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    first = _session(state, FIRST_UUID, "First session")
    second = _session(state, SECOND_UUID, "Second session")
    delegates = {
        first.session_id: _RecordingConversation(),
        second.session_id: _RecordingConversation(),
    }
    built_for: list[Session] = []

    def build(session: Session) -> _RecordingConversation:
        built_for.append(session)
        return delegates[session.session_id]

    conversation = SwitchableConversationPort(session=first, build_conversation=build)

    _ = [event async for event in conversation.submit("First turn")]
    conversation.switch_session(second)
    _ = [event async for event in conversation.submit("Second turn")]
    await conversation.close()

    assert built_for == [first, second]
    assert delegates[first.session_id].closed
    assert delegates[second.session_id].closed
    assert Session.load(state, first.session_id).messages == first.messages


@pytest.mark.asyncio
async def test_active_turn_rejects_switch_and_keeps_cancellation_on_original_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    first = _session(state, FIRST_UUID, "First session")
    second = _session(state, SECOND_UUID, "Second session")
    first_delegate = _BlockingConversation()
    second_delegate = _RecordingConversation()
    delegates = {
        first.session_id: first_delegate,
        second.session_id: second_delegate,
    }
    conversation = SwitchableConversationPort(
        session=first,
        build_conversation=lambda session: delegates[session.session_id],
    )
    events = conversation.submit("Blocking turn")

    assert (await anext(events)).type == "turn_started"
    with pytest.raises(RuntimeError, match="active foreground turn"):
        conversation.switch_session(second)
    await conversation.cancel_active_turn()
    assert [event.type async for event in events] == ["turn_completed"]

    conversation.switch_session(second)

    assert first_delegate.cancelled
    assert conversation.session is second
    assert second_delegate.cancelled is False
    await conversation.close()


@pytest.mark.asyncio
async def test_repl_resume_routes_the_next_input_through_the_selected_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace, agent_home)
    initial = Session.create(state, now=lambda: NOW, new_uuid=lambda: FIRST_UUID)
    target = _session(state, SECOND_UUID, "Target session")
    target.close()
    delegates: dict[str, _RecordingConversation] = {}

    def build(session: Session) -> _RecordingConversation:
        delegate = _RecordingConversation()
        delegates[session.session_id] = delegate
        return delegate

    conversation = SwitchableConversationPort(session=initial, build_conversation=build)
    management = ManagementCommandDispatcher(
        ManagementViewService(
            home,
            workspace_state=state,
            switch_session=conversation.switch_session,
        )
    )
    writer = _RecordingWriter()

    await run_repl(
        conversation=conversation,
        input_reader=_ScriptedInput(("/resume", "1", "Continue here", "exit")),
        writer=writer,
        management_dispatcher=management,
    )

    assert conversation.session_id == target.session_id
    assert delegates[target.session_id].submitted == ["Continue here"]
    assert writer.operations[0][1].startswith("Resumable sessions:\n1. Target session |")
    assert writer.operations[1] == ("line", f"Resumed session {target.session_id}.")
    assert writer.operations[-1] == ("finish", "")
    assert not (state.sessions_directory / f"{initial.session_id}.jsonl").exists()
    await conversation.close()
