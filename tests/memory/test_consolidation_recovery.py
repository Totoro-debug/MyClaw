import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.memory.conversation_summary import (
    ConsolidatingSummaryStore,
    ConversationSummaryManager,
    SummaryModelSettings,
    WorkspaceJsonlSummaryStore,
)
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
)
from myclaw.runtime_log import install_runtime_logging
from myclaw.session.records import (
    AssistantSessionMessage,
    UserSessionMessage,
)
from myclaw.session.session_store import JsonlSessionStore
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 16, 0, 0, tzinfo=LOCAL_OFFSET)


def _state(workspace: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _usage() -> ModelUsage:
    return ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6)


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=_usage(),
        finish_reason="stop",
    )


async def _session_ready_for_consolidation(
    home: AgentHome,
    workspace: Path,
) -> tuple[JsonlSessionStore, str]:
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    session_id = sessions.prepare().id
    for role, content in (
        ("user", "First question."),
        ("assistant", "First answer."),
        ("user", "Second question."),
        ("assistant", "Second answer."),
        ("user", "Current question."),
    ):
        if role == "user":
            await sessions.append_message(
                session_id,
                UserSessionMessage(id=str(uuid4()), created_at=NOW, content=content),
            )
        else:
            await sessions.append_message(
                session_id,
                AssistantSessionMessage(
                    id=str(uuid4()),
                    created_at=NOW,
                    content=content,
                    tool_calls=(),
                    status="completed",
                    error=None,
                    usage=_usage(),
                ),
            )
    return sessions, session_id


def _manager(
    *,
    sessions: JsonlSessionStore,
    summaries: ConsolidatingSummaryStore,
) -> ConversationSummaryManager:
    return ConversationSummaryManager(
        provider=ScriptedFakeProvider(completions=(_response("First turn summary."),)),
        sessions=sessions,
        summaries=summaries,
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=10_000,
        chat_max_output=1_000,
        consolidation_message_threshold=4,
        chat_system_prompt="CHAT SYSTEM",
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )


@pytest.mark.asyncio
async def test_pending_consolidation_recovery_failure_is_recorded_at_its_boundary(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions, _session_id = await _session_ready_for_consolidation(home, workspace)
    summaries = WorkspaceJsonlSummaryStore(_state(workspace))
    summaries.pending_directory.mkdir(parents=True)
    (summaries.pending_directory / "invalid.json").write_text(
        "PRIVATE CONSOLIDATION JOURNAL CONTENT",
        encoding="utf-8",
    )
    lifetime = install_runtime_logging(home)

    with pytest.raises(ModelCallError) as raised:
        await _manager(sessions=sessions, summaries=summaries).recover_pending()
    lifetime.close()

    assert raised.value.error.code == "persistence_error"
    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert content.count(" ERROR ") == 1
    assert (
        "session=- myclaw.memory.conversation_summary: "
        "Pending Conversation Summary recovery failed code=persistence_error"
    ) in content
    assert "JSONDecodeError" in content
    assert "PRIVATE CONSOLIDATION JOURNAL CONTENT" not in content


@pytest.mark.asyncio
async def test_summary_write_failure_leaves_recovery_journal_and_old_cursor(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions, session_id = await _session_ready_for_consolidation(home, workspace)
    replace_calls = 0

    def fail_summary_replace(target: Path, content: bytes) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected summary replacement failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    summaries = WorkspaceJsonlSummaryStore(_state(workspace), replace_bytes=fail_summary_replace)

    with pytest.raises(ModelCallError) as raised:
        await _manager(sessions=sessions, summaries=summaries).prepare(
            await sessions.load(session_id)
        )

    assert raised.value.error.code == "persistence_error"
    assert not summaries.path.exists()
    assert (await sessions.load(session_id)).metadata.consolidation_cursor == 0
    journals = list((_state(workspace).memory_directory / "pending-consolidations").glob("*.json"))
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert journal == {
        "session_id": session_id,
        "old_cursor": 0,
        "new_cursor": 2,
        "summary_index": 1,
        "timestamp": "2026-07-11T16:00:00.000+08:00",
        "content": "First turn summary.",
    }


@pytest.mark.asyncio
async def test_pending_journal_recovers_exact_summary_and_cursor_once(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions, session_id = await _session_ready_for_consolidation(home, workspace)
    replace_calls = 0

    def fail_summary_replace(target: Path, content: bytes) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected summary replacement failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    failing = WorkspaceJsonlSummaryStore(_state(workspace), replace_bytes=fail_summary_replace)
    with pytest.raises(ModelCallError):
        await _manager(sessions=sessions, summaries=failing).prepare(
            await sessions.load(session_id)
        )

    recovered = WorkspaceJsonlSummaryStore(_state(workspace))
    await recovered.recover_pending(sessions)
    await recovered.recover_pending(sessions)

    records = [json.loads(line) for line in recovered.path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "index": 1,
            "timestamp": "2026-07-11T16:00:00.000+08:00",
            "content": "First turn summary.",
        }
    ]
    assert (await sessions.load(session_id)).metadata.consolidation_cursor == 2
    assert list(recovered.pending_directory.glob("*.json")) == []


@pytest.mark.asyncio
async def test_pending_consolidation_recovery_records_one_degradation_warning(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions, session_id = await _session_ready_for_consolidation(home, workspace)
    replace_calls = 0

    def fail_summary_replace(target: Path, content: bytes) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected summary replacement failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    with pytest.raises(ModelCallError):
        await _manager(
            sessions=sessions,
            summaries=WorkspaceJsonlSummaryStore(
                _state(workspace), replace_bytes=fail_summary_replace
            ),
        ).prepare(await sessions.load(session_id))
    manager = _manager(sessions=sessions, summaries=WorkspaceJsonlSummaryStore(_state(workspace)))
    lifetime = install_runtime_logging(home)

    await manager.recover_pending()
    await manager.recover_pending()
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert content.count(" WARNING ") == 1
    assert " ERROR " not in content
    assert (
        "session=- myclaw.memory.conversation_summary: "
        "Pending Conversation Summary recovery completed count=1"
    ) in content
    assert "First turn summary." not in content


@pytest.mark.asyncio
async def test_runtime_recovers_all_pending_consolidations_before_first_turn_event(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    sessions, old_session_id = await _session_ready_for_consolidation(home, workspace)
    replace_calls = 0

    def fail_summary_replace(target: Path, content: bytes) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected summary replacement failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    with pytest.raises(ModelCallError):
        await _manager(
            sessions=sessions,
            summaries=WorkspaceJsonlSummaryStore(state, replace_bytes=fail_summary_replace),
        ).prepare(await sessions.load(old_session_id))
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(ModelCompleted(response=_response("Chat answer.")),)),)
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    events = runtime.conversation.submit("New session input.")
    first_event = await anext(events)

    assert first_event.type == "turn_started"
    assert (await runtime.sessions.load(old_session_id)).metadata.consolidation_cursor == 2
    assert list((state.memory_directory / "pending-consolidations").glob("*.json")) == []
    _ = [event async for event in events]
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_ignores_pending_consolidation_from_another_workspace(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    old_workspace = workspace / "old"
    new_workspace = workspace / "new"
    old_workspace.mkdir()
    new_workspace.mkdir()
    old_state = WorkspaceState(Workspace.from_path(old_workspace))
    old_state.initialize(agent_home_root=Path.home() / ".myclaw")
    old_sessions, old_session_id = await _session_ready_for_consolidation(home, old_workspace)
    replace_calls = 0

    def fail_summary_replace(target: Path, content: bytes) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected summary replacement failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    with pytest.raises(ModelCallError):
        await _manager(
            sessions=old_sessions,
            summaries=WorkspaceJsonlSummaryStore(old_state, replace_bytes=fail_summary_replace),
        ).prepare(await old_sessions.load(old_session_id))
    old_session_bytes = old_sessions.path_for(old_session_id).read_bytes()
    old_pending = old_state.memory_directory / "pending-consolidations"
    old_journal_bytes = {path: path.read_bytes() for path in old_pending.glob("*.json")}
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(ModelCompleted(response=_response("Chat answer.")),)),)
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=new_workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    events = runtime.conversation.submit("New workspace input.")
    first_event = await anext(events)
    _ = [event async for event in events]
    await runtime.close()

    assert first_event.type == "turn_started"
    assert (await old_sessions.load(old_session_id)).metadata.consolidation_cursor == 0
    assert old_sessions.path_for(old_session_id).read_bytes() == old_session_bytes
    assert {path: path.read_bytes() for path in old_pending.glob("*.json")} == old_journal_bytes
    assert len(provider.stream_requests) == 1


@pytest.mark.asyncio
async def test_runtime_reports_safe_error_for_conflicting_reserved_summary_index(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    sessions, session_id = await _session_ready_for_consolidation(home, workspace)
    replace_calls = 0

    def fail_summary_replace(target: Path, content: bytes) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected summary replacement failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    with pytest.raises(ModelCallError):
        await _manager(
            sessions=sessions,
            summaries=WorkspaceJsonlSummaryStore(state, replace_bytes=fail_summary_replace),
        ).prepare(await sessions.load(session_id))
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("A different summary.", NOW)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    with pytest.raises(ModelCallError) as raised:
        await anext(runtime.conversation.submit("Do not accept this turn."))
    await runtime.close()

    assert raised.value.error.code == "persistence_error"
    assert "different summary" not in raised.value.error.message.lower()
    assert (await sessions.load(session_id)).metadata.consolidation_cursor == 0
    assert len(list(summaries.pending_directory.glob("*.json"))) == 1


@pytest.mark.parametrize("failed_step", ("journal", "summary", "cursor", "delete"))
@pytest.mark.asyncio
async def test_each_consolidation_commit_crash_window_is_recoverable(
    agent_home: Path,
    workspace: Path,
    failed_step: str,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions, session_id = await _session_ready_for_consolidation(home, workspace)
    operations: list[str] = []

    def replace_summary_state(target: Path, content: bytes) -> None:
        step = "journal" if target.suffix == ".json" else "summary"
        operations.append(step)
        if step == failed_step:
            raise OSError(f"injected {step} failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    def replace_session_cursor(target: Path, content: bytes) -> None:
        operations.append("cursor")
        if failed_step == "cursor":
            raise OSError("injected cursor failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    def unlink_journal(target: Path) -> None:
        operations.append("delete")
        if failed_step == "delete":
            raise OSError("injected journal deletion failure")
        target.unlink()

    transaction_sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
        replace_bytes=replace_session_cursor,
    )
    transaction_summaries = WorkspaceJsonlSummaryStore(
        _state(workspace),
        replace_bytes=replace_summary_state,
        unlink_file=unlink_journal,
    )

    with pytest.raises(OSError):
        await transaction_summaries.commit_consolidation(
            sessions=transaction_sessions,
            session_id=session_id,
            old_cursor=0,
            new_cursor=2,
            content="First turn summary.",
            timestamp=NOW,
        )

    expected_prefix = ("journal", "summary", "cursor", "delete")
    assert operations == list(expected_prefix[: expected_prefix.index(failed_step) + 1])
    healthy = WorkspaceJsonlSummaryStore(_state(workspace))
    if failed_step == "journal":
        await healthy.commit_consolidation(
            sessions=sessions,
            session_id=session_id,
            old_cursor=0,
            new_cursor=2,
            content="First turn summary.",
            timestamp=NOW,
        )
    else:
        await healthy.recover_pending(sessions)

    assert (await sessions.load(session_id)).metadata.consolidation_cursor == 2
    assert [json.loads(line) for line in healthy.path.read_text(encoding="utf-8").splitlines()] == [
        {
            "index": 1,
            "timestamp": "2026-07-11T16:00:00.000+08:00",
            "content": "First turn summary.",
        }
    ]
    assert list(healthy.pending_directory.glob("*.json")) == []
