import asyncio
import json
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from myclaw.agent.events import AgentEvent
from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, ProviderConfiguration
from myclaw.logging.session import session_log
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import ManagementViewService
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.records import (
    UserSessionMessage,
)
from myclaw.session.session import Session
from myclaw.session.session_resume import SwitchableConversationPort
from myclaw.session.session_store import JsonlSessionStore
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    FakeClock,
    ScriptedFakeProvider,
    StreamScript,
    unexpected_provider_factory,
)
from tests.fixtures.diagnostic_capture import capture_diagnostics, configured_process_logging

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
OLDER_SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
NEWER_SESSION_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
OTHER_WORKSPACE_SESSION_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
FIRST_USER_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
SECOND_USER_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
THIRD_USER_UUID = UUID("a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34")
CORRUPT_METADATA_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")
CORRUPT_MIDDLE_UUID = UUID("886313e1-3b8a-4a2d-9f7f-77611a4b6f4e")
CORRUPT_TAIL_UUID = UUID("b3f37212-6f3a-4a1b-8d2e-78ab3f9c4567")
TURN_UUID = UUID("a8098c1a-f86e-4f33-8a28-25f602f8e603")
REQUEST_UUID = UUID("67e55044-10b1-426f-9247-bb680e5fe0c8")
NEW_ASSISTANT_UUID = UUID("11111111-1111-4111-8111-111111111111")
SECOND_TURN_UUID = UUID("22222222-2222-4222-8222-222222222222")
SECOND_REQUEST_UUID = UUID("33333333-3333-4333-8333-333333333333")
SECOND_ASSISTANT_UUID = UUID("44444444-4444-4444-8444-444444444444")


def _current_session(
    state: WorkspaceState,
    session_uuid: UUID,
    *,
    now: datetime = NOW,
    title: str | None = None,
) -> Session:
    session = Session.create(
        state,
        now=lambda: now,
        new_uuid=lambda: session_uuid,
    )
    if title is not None:
        session.update_metadata(title=title)
    return session


def _persist_session(session: Session) -> Session:
    session.close()
    return session


@pytest.mark.asyncio
async def test_session_picker_lists_only_current_workspace_with_stable_summaries(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    state = WorkspaceState(Workspace.from_path(workspace))
    older = _current_session(state, OLDER_SESSION_UUID, title="Older title")
    older.add_message("user", "Older workspace conversation.")
    _persist_session(older)

    clock.advance(30)
    newer = _current_session(state, NEWER_SESSION_UUID, now=clock.now(), title="Newer title")
    newer.add_message("user", "Newer workspace conversation.")
    newer.add_message("user", "One more message.")
    _persist_session(newer)

    other_workspace = workspace.parent / "other-workspace"
    other_state = WorkspaceState(Workspace.from_path(other_workspace))
    other = _current_session(other_state, OTHER_WORKSPACE_SESSION_UUID, now=clock.now())
    other.add_message("user", "Must not leak into the current Workspace.")
    _persist_session(other)

    management = ManagementViewService(home, workspace_state=state)
    summaries = (await management.resumable_listing()).sessions

    assert [summary.to_dict() for summary in summaries] == [
        {
            "id": newer.session_id,
            "title": "Newer title",
            "created_at": "2026-07-11T15:30:42.123+08:00",
            "updated_at": "2026-07-11T15:30:42.123+08:00",
            "message_count": 2,
        },
        {
            "id": older.session_id,
            "title": "Older title",
            "created_at": "2026-07-11T15:30:12.123+08:00",
            "updated_at": "2026-07-11T15:30:12.123+08:00",
            "message_count": 1,
        },
    ]


@pytest.mark.asyncio
async def test_session_picker_skips_corrupt_sessions_without_modifying_them(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    prepared = [
        _current_session(state, session_uuid, title="Valid title" if index == 0 else None)
        for index, session_uuid in enumerate(
            (
                OLDER_SESSION_UUID,
                CORRUPT_METADATA_UUID,
                CORRUPT_MIDDLE_UUID,
                CORRUPT_TAIL_UUID,
            )
        )
    ]
    for index, session in enumerate(prepared):
        session.add_message("user", f"Session {index}.")
        _persist_session(session)

    metadata_path = state.sessions_directory / f"{prepared[1].session_id}.jsonl"
    metadata_path.write_bytes(b'{"session_id":"broken"}\n')
    middle_path = state.sessions_directory / f"{prepared[2].session_id}.jsonl"
    middle_lines = middle_path.read_bytes().splitlines(keepends=True)
    middle_path.write_bytes(middle_lines[0] + b"not-json\n" + b"".join(middle_lines[1:]))
    tail_path = state.sessions_directory / f"{prepared[3].session_id}.jsonl"
    tail_path.write_bytes(tail_path.read_bytes() + b'{"role":"user"\n')
    corrupt_snapshots = {
        path: path.read_bytes() for path in (metadata_path, middle_path, tail_path)
    }

    summaries = (
        await ManagementViewService(home, workspace_state=state).resumable_listing()
    ).sessions

    assert [(summary.id, summary.title) for summary in summaries] == [
        (prepared[0].session_id, "Valid title")
    ]
    assert {path: path.read_bytes() for path in corrupt_snapshots} == corrupt_snapshots


@pytest.mark.asyncio
async def test_management_listing_warns_once_for_each_skipped_session_entry(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    valid = _current_session(state, OLDER_SESSION_UUID)
    valid.add_message("user", "Valid history.")
    _persist_session(valid)
    corrupt_path = state.sessions_directory / f"{CORRUPT_METADATA_UUID}.jsonl"
    corrupt_path.write_text("persisted content must stay private\n", encoding="utf-8")
    unreadable_path = state.sessions_directory / f"{CORRUPT_MIDDLE_UUID}.jsonl"
    unreadable_path.mkdir()
    management = ManagementViewService(home, workspace_state=state)

    log_capture = capture_diagnostics()
    try:
        listing = await management.resumable_listing()
    finally:
        log_capture.close()

    assert [summary.id for summary in listing.sessions] == [valid.session_id]
    assert listing.skipped_count == 2
    content = log_capture.text
    assert content.count(" WARNING ") == 2
    marker = "Skipped corrupt or unreadable Conversation Session entry"
    assert content.count(marker) == 2
    assert str(corrupt_path) in content
    assert str(unreadable_path) in content
    assert "persisted content must stay private" not in content


@pytest.mark.asyncio
async def test_unavailable_session_listing_renders_error_without_terminal_duplicate(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()

    state = WorkspaceState(Workspace.from_path(workspace))
    state.sessions_directory.mkdir(parents=True)
    (state.sessions_directory / f"{OLDER_SESSION_UUID}.jsonl").touch()

    def fail_load(
        _cls: type[Session],
        /,
        _state: WorkspaceState,
        _session_id: str,
        **_kwargs: object,
    ) -> Session:
        raise OSError("session directory unavailable")

    monkeypatch.setattr(Session, "load", classmethod(fail_load))

    dispatcher = ManagementCommandDispatcher(ManagementViewService(home, workspace_state=state))
    with configured_process_logging():
        result = await dispatcher.dispatch("/resume")

    assert result.output == (
        "Warning: Skipped 1 corrupt Conversation Session.\nNo resumable Conversation Sessions."
    )
    assert capsys.readouterr().err == ""
    assert not (workspace / ".myclaw" / "logs").exists()


@pytest.mark.asyncio
async def test_resume_load_failure_renders_safe_output_without_terminal_duplicate(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    target = _current_session(state, OLDER_SESSION_UUID)
    target.add_message("user", "Persisted resume content must stay private.")
    _persist_session(target)
    original_load = Session.load
    load_calls = 0

    def fail_second_load(
        cls: type[Session],
        workspace_state: WorkspaceState,
        session_id: str,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> Session:
        nonlocal load_calls
        load_calls += 1
        if load_calls == 2:
            raise OSError("resume target disappeared")
        if now is None:
            return original_load(workspace_state, session_id)
        return original_load(workspace_state, session_id, now=now)

    monkeypatch.setattr(Session, "load", classmethod(fail_second_load))

    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            home,
            workspace_state=state,
            switch_session=lambda _session: None,
        )
    )
    with configured_process_logging():
        result = await dispatcher.resume(target.session_id)

    assert (
        result.output == "persistence_error: The selected Conversation Session could not be loaded."
    )
    assert capsys.readouterr().err == ""
    assert not (workspace / ".myclaw" / "logs").exists()


@pytest.mark.asyncio
async def test_session_switch_failure_is_terminal_only_without_a_session_log(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    target = _current_session(state, OLDER_SESSION_UUID)
    target.add_message("user", "Private history.")
    _persist_session(target)

    def fail_switch(_session: Session) -> None:
        raise RuntimeError("session switch failed")

    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            home,
            workspace_state=state,
            switch_session=fail_switch,
        )
    )
    with configured_process_logging(), session_log(state, target.session_id):
        with pytest.raises(RuntimeError, match="session switch failed"):
            await dispatcher.resume(target.session_id)

    assert capsys.readouterr().err == (
        "Management command failed command=/resume type=RuntimeError\n"
    )
    assert not (state.logs_directory / f"{target.session_id}.log").exists()


@pytest.mark.asyncio
async def test_incomplete_final_append_is_recovered_on_the_next_successful_write(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    metadata = sessions.prepare()
    await sessions.append_message(
        metadata.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=NOW,
            content="Complete history.",
        ),
    )
    path = sessions.path_for(metadata.id)
    path.write_bytes(
        path.read_bytes() + b'{"record_type":"message","id":"a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34"'
    )
    crashed_bytes = path.read_bytes()
    restarted = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter(()).__next__,
    )

    recovered = await restarted.load(metadata.id)

    assert [(message.role, message.content) for message in recovered.messages] == [
        ("user", "Complete history.")
    ]
    assert path.read_bytes() == crashed_bytes

    await restarted.append_message(
        metadata.id,
        UserSessionMessage(
            id=str(SECOND_USER_UUID),
            created_at=NOW + timedelta(seconds=30),
            content="Continue after recovery.",
        ),
    )

    reloaded = await restarted.load(metadata.id)
    assert [(message.role, message.content) for message in reloaded.messages] == [
        ("user", "Complete history."),
        ("user", "Continue after recovery."),
    ]
    repaired_bytes = path.read_bytes()
    assert repaired_bytes.endswith(b"\n")
    assert b'a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34"' not in repaired_bytes


class ScriptedInput:
    def __init__(self, values: tuple[str | None, ...]) -> None:
        self._values = iter(values)

    async def read(self) -> str | None:
        return next(self._values)


class RecordingWriter:
    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []

    async def write_delta(self, delta: str) -> None:
        self.operations.append(("delta", delta))

    async def finish_turn(self) -> None:
        self.operations.append(("finish", ""))

    async def write_line(self, content: str) -> None:
        self.operations.append(("line", content))


class DelayedResumeTitleProvider:
    def __init__(self) -> None:
        self.title_started = asyncio.Event()
        self.release_title = asyncio.Event()
        self.title_request_count = 0
        self.title_stopped_count = 0
        self._chat_contents = iter(("First answer.", "Target answer."))

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if request.system_prompt == session_title_prompt():
            self.title_request_count += 1
            self.title_started.set()
            try:
                await self.release_title.wait()
                yield ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(content="Late original title"),
                        usage=ModelUsage(input_tokens=5, output_tokens=3, total_tokens=8),
                        finish_reason="stop",
                    )
                )
            finally:
                self.title_stopped_count += 1
            return
        content = next(self._chat_contents)
        yield TextDelta(delta=content)
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content=content),
                usage=ModelUsage(input_tokens=7, output_tokens=3, total_tokens=10),
                finish_reason="stop",
            )
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

    async def close(self) -> None:
        return None


class BlockingTurnProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.started.set()
        await self.release.wait()
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content=self.content),
                usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                finish_reason="stop",
            )
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_switchable_conversation_close_stops_delegates_from_every_selected_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    first = _current_session(state, OLDER_SESSION_UUID)
    second = _current_session(state, NEWER_SESSION_UUID)
    provider = DelayedResumeTitleProvider()

    def conversation_for(active_session: Session) -> StreamingConversationPort:
        return StreamingConversationPort(
            provider=provider,
            session=active_session,
            settings=ChatModelSettings(
                model="test-model",
                max_output=1024,
                temperature=0.2,
                reasoning_effort=None,
                timeout_seconds=30,
            ),
            now=lambda: NOW,
            new_uuid=lambda: REQUEST_UUID,
            title_prompt=session_title_prompt(),
            workspace_state=state,
        )

    conversation = SwitchableConversationPort(
        session=first,
        build_conversation=conversation_for,
    )
    _ = [event async for event in conversation.submit("Start the first title.")]
    conversation.switch_session(second)
    _ = [event async for event in conversation.submit("Start the second title.")]
    for _ in range(100):
        if provider.title_request_count == 2:
            break
        await asyncio.sleep(0)

    await conversation.close()

    assert provider.title_stopped_count == 2


@pytest.mark.asyncio
async def test_switchable_conversation_close_settles_every_selected_adapter(tmp_path: Path) -> None:
    class ClosingConversation:
        def __init__(self, *, fail: bool = False) -> None:
            self._fail = fail
            self.closed = False

        async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
            del text
            events: tuple[AgentEvent, ...] = ()
            for event in events:
                yield event

        async def cancel_active_turn(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True
            if self._fail:
                raise RuntimeError("first adapter close failed")

    first = ClosingConversation(fail=True)
    second = ClosingConversation()
    state = WorkspaceState(Workspace.from_path(tmp_path))
    first_session = _current_session(state, OLDER_SESSION_UUID)
    second_session = _current_session(state, NEWER_SESSION_UUID)
    delegates = {first_session.session_id: first, second_session.session_id: second}
    conversation = SwitchableConversationPort(
        session=first_session,
        build_conversation=lambda session: delegates[session.session_id],
    )
    _ = [event async for event in conversation.submit("first")]
    conversation.switch_session(second_session)
    _ = [event async for event in conversation.submit("second")]

    with pytest.raises(RuntimeError, match="first adapter close failed"):
        await conversation.close()

    assert first.closed
    assert second.closed


@pytest.mark.asyncio
async def test_repl_resume_continues_target_history_and_switches_runtime_status(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    clock = FakeClock(NOW)
    state = WorkspaceState(Workspace.from_path(workspace))
    historical = _current_session(state, OLDER_SESSION_UUID, title="Friday deployment")
    historical.add_message("user", "Remember the deployment decision.")
    historical.add_message(
        "assistant",
        "Deploy on Friday.",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 8,
            "output_tokens": 4,
            "total_tokens": 12,
        },
    )
    _persist_session(historical)
    response = ModelResponse(
        message=AssistantModelMessage(content="The Friday decision is still current."),
        usage=ModelUsage(input_tokens=15, output_tokens=7, total_tokens=22),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="The Friday decision is still current."),
                    ModelCompleted(response=response),
                )
            ),
        )
    )

    def provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
        del configuration
        return provider

    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=provider_factory,
        now=clock.now,
        new_uuid=iter(
            (
                NEWER_SESSION_UUID,
                TURN_UUID,
                THIRD_USER_UUID,
                REQUEST_UUID,
                NEW_ASSISTANT_UUID,
            )
        ).__next__,
        monotonic_now=clock.monotonic,
    )
    startup_session_id = runtime.session_id
    writer = RecordingWriter()

    await runtime.run(
        input_reader=ScriptedInput(
            ("/resume", "1", "Continue from that decision.", "/status", None)
        ),
        writer=writer,
    )

    assert writer.operations[0] == (
        "line",
        "Resumable sessions:\n1. Friday deployment | 2026-07-11T15:30:12.123+08:00 | 2 messages",
    )
    assert writer.operations[1] == ("line", f"Resumed session {historical.session_id}.")
    assert writer.operations[2:4] == [
        ("delta", "The Friday decision is still current."),
        ("finish", ""),
    ]
    status = json.loads(writer.operations[4][1])
    assert status["session_message_count"] == 4
    assert status["cumulative_usage"] == {
        "model_calls": 2,
        "input_tokens": 23,
        "output_tokens": 11,
        "total_tokens": 34,
    }

    request = provider.stream_requests[0]
    assert len(provider.stream_requests) == 1
    assert isinstance(request, ModelRequest)
    assert [message.to_dict() for message in request.messages] == [
        {"role": "user", "content": "Remember the deployment decision."},
        {"role": "assistant", "content": "Deploy on Friday.", "tool_calls": []},
        {
            "role": "user",
            "content": (
                "<runtime_context>\n"
                "current_time: 2026-07-11T15:30:12.123+08:00\n"
                f"session_id: {historical.session_id}\n"
                "</runtime_context>\n\n"
                "<user_input>\n"
                "Continue from that decision.\n"
                "</user_input>"
            ),
        },
    ]
    target = Session.load(state, historical.session_id)
    assert runtime.session_id == historical.session_id
    assert [(message["role"], message["content"]) for message in target.messages] == [
        ("user", "Remember the deployment decision."),
        ("assistant", "Deploy on Friday."),
        ("user", "Continue from that decision."),
        ("assistant", "The Friday decision is still current."),
    ]
    assert not (state.sessions_directory / f"{startup_session_id}.jsonl").exists()


@pytest.mark.asyncio
async def test_resuming_current_session_keeps_status_on_the_live_authority(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state = WorkspaceState(Workspace.from_path(workspace))
    response = ModelResponse(
        message=AssistantModelMessage(content="Current authority updated."),
        usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(ModelCompleted(response=response),)),)
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    authority = runtime.session
    authority.add_message("user", "Existing current history.")
    authority.persist()
    path = state.sessions_directory / f"{authority.session_id}.jsonl"
    for _ in range(100):
        if path.exists():
            break
        await asyncio.sleep(0)

    resumed = await runtime.management_dispatcher.resume(authority.session_id)
    events = [event async for event in runtime.conversation.submit("Continue current history.")]
    status_result = await runtime.management_dispatcher.dispatch("/status")

    assert resumed.output == f"Resumed session {authority.session_id}."
    assert runtime.session is authority
    assert events[-1].type == "turn_completed"
    status = json.loads(status_result.output or "")
    assert status["session_message_count"] == 3
    assert status["cumulative_usage"] == {
        "model_calls": 1,
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    await runtime.close()


@pytest.mark.asyncio
async def test_picker_rejects_unsupported_metadata_and_message_record_types(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    prepared = [
        _current_session(state, session_uuid)
        for session_uuid in (OLDER_SESSION_UUID, CORRUPT_METADATA_UUID, CORRUPT_MIDDLE_UUID)
    ]
    for index, session in enumerate(prepared):
        session.add_message("user", f"Record contract {index}.")
        _persist_session(session)

    unsupported_metadata_path = state.sessions_directory / f"{prepared[1].session_id}.jsonl"
    unsupported_lines = unsupported_metadata_path.read_text(encoding="utf-8").splitlines()
    unsupported_metadata = json.loads(unsupported_lines[0])
    unsupported_metadata["schema_version"] = 2
    unsupported_metadata_path.write_text(
        json.dumps(unsupported_metadata, separators=(",", ":"))
        + "\n"
        + unsupported_lines[1]
        + "\n",
        encoding="utf-8",
    )
    wrong_message_path = state.sessions_directory / f"{prepared[2].session_id}.jsonl"
    wrong_message_lines = wrong_message_path.read_text(encoding="utf-8").splitlines()
    wrong_message = json.loads(wrong_message_lines[1])
    wrong_message["record_type"] = "metadata"
    wrong_message_path.write_text(
        wrong_message_lines[0] + "\n" + json.dumps(wrong_message, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    corrupt_snapshots = {
        path: path.read_bytes() for path in (unsupported_metadata_path, wrong_message_path)
    }

    summaries = (
        await ManagementViewService(home, workspace_state=state).resumable_listing()
    ).sessions

    assert [summary.id for summary in summaries] == [prepared[0].session_id]
    assert {path: path.read_bytes() for path in corrupt_snapshots} == corrupt_snapshots


@pytest.mark.asyncio
async def test_resume_command_aggregates_corrupt_session_warning(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    prepared = [
        _current_session(
            state,
            session_uuid,
            title="Valid picker item" if index == 0 else None,
        )
        for index, session_uuid in enumerate(
            (OLDER_SESSION_UUID, CORRUPT_METADATA_UUID, CORRUPT_MIDDLE_UUID)
        )
    ]
    for index, session in enumerate(prepared):
        session.add_message("user", f"Picker item {index}.")
        _persist_session(session)
    (state.sessions_directory / f"{prepared[1].session_id}.jsonl").write_bytes(b"not metadata\n")
    (state.sessions_directory / f"{prepared[2].session_id}.jsonl").write_bytes(b'{"broken":\n')
    dispatcher = ManagementCommandDispatcher(ManagementViewService(home, workspace_state=state))
    with configured_process_logging(), session_log(state, prepared[0].session_id):
        result = await dispatcher.dispatch("/resume")

    assert result.output == (
        "Warning: Skipped 2 corrupt Conversation Sessions.\n"
        "Resumable sessions:\n"
        "1. Valid picker item | 2026-07-11T15:30:12.123+08:00 | 1 message"
    )
    assert result.resume_sessions is not None
    assert [session.id for session in result.resume_sessions] == [prepared[0].session_id]
    assert capsys.readouterr().err == ""
    assert not (state.logs_directory / f"{prepared[0].session_id}.log").exists()


@pytest.mark.asyncio
async def test_concurrent_resume_listings_keep_their_own_corruption_diagnostics(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    other_workspace = workspace.parent / "other-workspace"
    other_workspace.mkdir()
    state = WorkspaceState(Workspace.from_path(workspace))
    valid = _current_session(state, OLDER_SESSION_UUID, title="Concurrent picker item")
    valid.add_message("user", "Valid concurrent picker item.")
    _persist_session(valid)
    corrupt = _current_session(state, CORRUPT_METADATA_UUID)
    corrupt.add_message("user", "Will become corrupt.")
    _persist_session(corrupt)
    (state.sessions_directory / f"{corrupt.session_id}.jsonl").write_bytes(b"corrupt metadata\n")
    current_dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, workspace_state=state)
    )
    other_dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            home,
            workspace_state=WorkspaceState(Workspace.from_path(other_workspace)),
        )
    )

    current_result = await current_dispatcher.dispatch("/resume")
    other_result = await other_dispatcher.dispatch("/resume")

    assert current_result.output == (
        "Warning: Skipped 1 corrupt Conversation Session.\n"
        "Resumable sessions:\n"
        "1. Concurrent picker item | 2026-07-11T15:30:12.123+08:00 | 1 message"
    )
    assert other_result.output == "No resumable Conversation Sessions."


@pytest.mark.asyncio
async def test_management_resume_revalidates_workspace_and_keeps_current_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    other_workspace = workspace.parent / "other-workspace"
    other_workspace.mkdir()
    other_store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(other_workspace)),
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    other = other_store.prepare()
    await other_store.append_message(
        other.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=NOW,
            content="Other Workspace history.",
        ),
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=unexpected_provider_factory,
        now=lambda: NOW,
        new_uuid=iter((NEWER_SESSION_UUID,)).__next__,
    )

    other_result = await runtime.management_dispatcher.resume(other.id)
    invalid_result = await runtime.management_dispatcher.resume("../not-a-session")
    status_result = await runtime.management_dispatcher.dispatch("/status")

    assert other_result.output == (
        "model_invalid_request: The selected Conversation Session is not resumable."
    )
    assert invalid_result.output == other_result.output
    assert json.loads(status_result.output or "")["session_message_count"] == 0
    assert not runtime.sessions.path_for(runtime.session_id).exists()
    assert HOST_FILESYSTEM.path_for_io(other_store.path_for(other.id)).exists()


@pytest.mark.asyncio
async def test_legacy_agent_home_session_is_not_resumable(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    metadata = sessions.prepare()
    message = UserSessionMessage(
        id=str(FIRST_USER_UUID),
        created_at=NOW,
        content="Legacy history must remain ignored.",
    )
    legacy_path = agent_home / "sessions" / "legacy-workspace-slug" / f"{metadata.id}.jsonl"
    legacy_io_path = HOST_FILESYSTEM.path_for_io(legacy_path)
    legacy_io_path.parent.mkdir(parents=True)
    legacy_io_path.write_text(metadata.to_json_line() + message.to_json_line(), encoding="utf-8")
    legacy_bytes = legacy_io_path.read_bytes()
    switched: list[Session] = []
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            home,
            workspace_state=state,
            switch_session=switched.append,
        )
    )

    listing = await dispatcher.dispatch("/resume")
    resumed = await dispatcher.resume(metadata.id)

    assert listing.output == "No resumable Conversation Sessions."
    assert resumed.output == (
        "model_invalid_request: The selected Conversation Session is not resumable."
    )
    assert switched == []
    assert legacy_io_path.read_bytes() == legacy_bytes


@pytest.mark.asyncio
async def test_resuming_from_a_nonempty_session_preserves_its_complete_history(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state = WorkspaceState(Workspace.from_path(workspace))
    target = _current_session(
        state,
        OLDER_SESSION_UUID,
        now=NOW - timedelta(minutes=5),
    )
    target.add_message("user", "Target history.")
    _persist_session(target)
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=unexpected_provider_factory,
        now=lambda: NOW,
        new_uuid=iter((NEWER_SESSION_UUID,)).__next__,
    )
    original_session_id = runtime.session_id
    original_session = runtime.session
    original_session.add_message("user", "Original nonempty history.")
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    write_attempts: list[Path] = []

    def fail_first_snapshot(target_path: Path, _content: bytes) -> None:
        write_attempts.append(target_path)
        monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", replace)
        raise OSError("injected ordinary persistence failure")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_first_snapshot)
    original_session.persist()
    for _ in range(100):
        if write_attempts:
            break
        await asyncio.sleep(0)
    original_path = state.sessions_directory / f"{original_session_id}.jsonl"
    assert write_attempts == [original_path]
    assert not original_path.exists()

    result = await runtime.management_dispatcher.resume(target.session_id)

    assert result.output == f"Resumed session {target.session_id}."
    assert runtime.session.session_id == target.session_id
    assert [message["content"] for message in runtime.session.messages] == ["Target history."]
    await runtime.close()
    preserved = Session.load(state, original_session_id)
    assert [message["content"] for message in preserved.messages] == ["Original nonempty history."]


@pytest.mark.asyncio
async def test_late_title_stays_with_original_and_resumed_session_is_not_retitled(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state = WorkspaceState(Workspace.from_path(workspace))
    target = _current_session(
        state,
        OLDER_SESSION_UUID,
        now=NOW - timedelta(minutes=5),
        title="Existing target title",
    )
    target.add_message("user", "Existing target history.")
    _persist_session(target)
    provider = DelayedResumeTitleProvider()

    def provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
        del configuration
        return provider

    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=provider_factory,
        now=lambda: NOW,
        new_uuid=iter(
            (
                NEWER_SESSION_UUID,
                TURN_UUID,
                THIRD_USER_UUID,
                REQUEST_UUID,
                NEW_ASSISTANT_UUID,
                SECOND_TURN_UUID,
                CORRUPT_METADATA_UUID,
                SECOND_REQUEST_UUID,
                SECOND_ASSISTANT_UUID,
            )
        ).__next__,
    )
    _ = [event async for event in runtime.conversation.submit("Create the original title.")]
    await asyncio.wait_for(provider.title_started.wait(), timeout=1)
    original_session = runtime.session

    resume_result = await runtime.management_dispatcher.resume(target.session_id)
    provider.release_title.set()
    for _ in range(100):
        if original_session.metadata["title"] == "Late original title":
            break
        await asyncio.sleep(0)

    _ = [event async for event in runtime.conversation.submit("Continue the existing target.")]

    assert resume_result.output == f"Resumed session {target.session_id}."
    assert original_session.metadata["title"] == "Late original title"
    assert runtime.session.metadata["title"] == "Existing target title"
    assert provider.title_request_count == 1
    await runtime.close()
    resumed = Session.load(state, target.session_id)
    assert [(message["role"], message["content"]) for message in resumed.messages] == [
        ("user", "Existing target history."),
        ("user", "Continue the existing target."),
        ("assistant", "Target answer."),
    ]


@pytest.mark.asyncio
async def test_complete_json_without_a_jsonl_newline_is_rejected_without_repair(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    metadata = sessions.prepare()
    await sessions.append_message(
        metadata.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=NOW,
            content="Complete record without final newline.",
        ),
    )
    path = sessions.path_for(metadata.id)
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))
    snapshot = path.read_bytes()

    with pytest.raises(ValueError, match="must end with a newline"):
        await sessions.load(metadata.id)

    assert path.read_bytes() == snapshot


@pytest.mark.asyncio
async def test_failed_atomic_tail_repair_preserves_the_crashed_session_bytes(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    metadata = sessions.prepare()
    await sessions.append_message(
        metadata.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=NOW,
            content="History before interrupted append.",
        ),
    )
    path = sessions.path_for(metadata.id)
    path.write_bytes(path.read_bytes() + b'{"record_type":"message"')
    crashed_bytes = path.read_bytes()

    def fail_atomic_replace(_path: Path, _content: bytes) -> None:
        raise OSError("injected atomic repair failure")

    restarted = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter(()).__next__,
        replace_bytes=fail_atomic_replace,
    )

    with pytest.raises(OSError, match="injected atomic repair failure"):
        await restarted.append_message(
            metadata.id,
            UserSessionMessage(
                id=str(SECOND_USER_UUID),
                created_at=NOW + timedelta(seconds=1),
                content="Write after restart.",
            ),
        )

    assert path.read_bytes() == crashed_bytes


@pytest.mark.asyncio
async def test_active_turn_rejects_session_switch_and_keeps_cancellation_on_original(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    session_a = _current_session(state, OLDER_SESSION_UUID)
    session_b = _current_session(state, NEWER_SESSION_UUID)
    provider_a = BlockingTurnProvider("A should be cancelled.")
    provider_b = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="B completed."),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    uuids = {
        session_a.session_id: iter(
            (TURN_UUID, FIRST_USER_UUID, REQUEST_UUID, NEW_ASSISTANT_UUID)
        ).__next__,
        session_b.session_id: iter(
            (
                SECOND_TURN_UUID,
                SECOND_USER_UUID,
                SECOND_REQUEST_UUID,
                SECOND_ASSISTANT_UUID,
            )
        ).__next__,
    }

    def build_port(active_session: Session) -> StreamingConversationPort:
        provider: ModelProvider = (
            provider_a if active_session.session_id == session_a.session_id else provider_b
        )
        return StreamingConversationPort(
            provider=provider,
            session=active_session,
            settings=ChatModelSettings(
                model="test-model",
                max_output=1024,
                temperature=0.2,
                reasoning_effort=None,
                timeout_seconds=30,
            ),
            now=lambda: NOW,
            new_uuid=uuids[active_session.session_id],
            workspace_state=state,
        )

    conversation = SwitchableConversationPort(
        session=session_a,
        build_conversation=build_port,
    )

    async def collect(text: str) -> list[str]:
        return [event.type async for event in conversation.submit(text)]

    active_a = asyncio.create_task(collect("Start A."))
    await asyncio.wait_for(provider_a.started.wait(), timeout=1)
    try:
        with pytest.raises(RuntimeError, match="active foreground turn"):
            conversation.switch_session(session_b)

        assert conversation.session_id == session_a.session_id
        await conversation.cancel_active_turn()
        assert await asyncio.wait_for(active_a, timeout=1) == [
            "turn_started",
            "turn_cancelled",
        ]

        conversation.switch_session(session_b)
        assert await collect("Start B.") == ["turn_started", "turn_completed"]
        assert conversation.session_id == session_b.session_id
    finally:
        provider_a.release.set()
        if not active_a.done():
            await asyncio.gather(active_a, return_exceptions=True)
