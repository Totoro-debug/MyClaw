import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

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
    AssistantSessionMessage,
    ConversationSession,
    MetadataUpdate,
    UserSessionMessage,
)
from myclaw.session.session_resume import SwitchableConversationPort
from myclaw.session.session_store import JsonlSessionStore, SessionListingReport
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    FakeClock,
    ScriptedFakeProvider,
    StreamScript,
    unexpected_provider_factory,
)
from tests.fixtures.log_capture import configured_process_logging, install_log_capture

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


@pytest.mark.asyncio
async def test_session_picker_lists_only_current_workspace_with_stable_summaries(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((OLDER_SESSION_UUID, NEWER_SESSION_UUID)).__next__,
    )
    older = sessions.prepare()
    await sessions.append_message(
        older.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=clock.now(),
            content="Older workspace conversation.",
        ),
    )
    await sessions.update_metadata(older.id, MetadataUpdate(title="Older title"))

    clock.advance(30)
    newer = sessions.prepare()
    await sessions.append_message(
        newer.id,
        UserSessionMessage(
            id=str(SECOND_USER_UUID),
            created_at=clock.now(),
            content="Newer workspace conversation.",
        ),
    )
    await sessions.append_message(
        newer.id,
        UserSessionMessage(
            id=str(THIRD_USER_UUID),
            created_at=clock.now(),
            content="One more message.",
        ),
    )
    await sessions.update_metadata(newer.id, MetadataUpdate(title="Newer title"))

    other_workspace = workspace.parent / "other-workspace"
    other_sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(other_workspace)),
        now=clock.now,
        new_uuid=iter((OTHER_WORKSPACE_SESSION_UUID,)).__next__,
    )
    other = other_sessions.prepare()
    await other_sessions.append_message(
        other.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=clock.now(),
            content="Must not leak into the current Workspace.",
        ),
    )

    summaries = (await sessions.scan_for_workspace(workspace)).sessions

    assert [summary.to_dict() for summary in summaries] == [
        {
            "id": newer.id,
            "title": "Newer title",
            "created_at": "2026-07-11T15:30:42.123+08:00",
            "updated_at": "2026-07-11T15:30:42.123+08:00",
            "message_count": 2,
        },
        {
            "id": older.id,
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
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter(
            (
                OLDER_SESSION_UUID,
                CORRUPT_METADATA_UUID,
                CORRUPT_MIDDLE_UUID,
                CORRUPT_TAIL_UUID,
            )
        ).__next__,
    )

    prepared = [sessions.prepare() for _ in range(4)]
    for index, metadata in enumerate(prepared):
        await sessions.append_message(
            metadata.id,
            UserSessionMessage(
                id=str(
                    (
                        FIRST_USER_UUID,
                        SECOND_USER_UUID,
                        THIRD_USER_UUID,
                        OTHER_WORKSPACE_SESSION_UUID,
                    )[index]
                ),
                created_at=NOW,
                content=f"Session {index}.",
            ),
        )
    await sessions.update_metadata(prepared[0].id, MetadataUpdate(title="Valid title"))

    metadata_path = sessions.path_for(prepared[1].id)
    metadata_path.write_bytes(b'{"record_type":"metadata"}\n')
    middle_path = sessions.path_for(prepared[2].id)
    middle_lines = middle_path.read_bytes().splitlines(keepends=True)
    middle_path.write_bytes(middle_lines[0] + b"not-json\n" + b"".join(middle_lines[1:]))
    tail_path = sessions.path_for(prepared[3].id)
    tail_path.write_bytes(tail_path.read_bytes() + b'{"record_type":"message"\n')
    corrupt_snapshots = {
        path: path.read_bytes() for path in (metadata_path, middle_path, tail_path)
    }

    summaries = (await sessions.scan_for_workspace(workspace)).sessions

    assert [(summary.id, summary.title) for summary in summaries] == [
        (prepared[0].id, "Valid title")
    ]
    assert {path: path.read_bytes() for path in corrupt_snapshots} == corrupt_snapshots


@pytest.mark.asyncio
async def test_management_listing_warns_once_for_each_skipped_session_entry(
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
    valid = sessions.prepare()
    await sessions.append_message(
        valid.id,
        UserSessionMessage(id=str(FIRST_USER_UUID), created_at=NOW, content="Valid history."),
    )
    corrupt_path = sessions.directory / f"{CORRUPT_METADATA_UUID}.jsonl"
    corrupt_path.write_text("persisted content must stay private\n", encoding="utf-8")
    unreadable_path = sessions.directory / f"{CORRUPT_MIDDLE_UUID}.jsonl"
    unreadable_path.mkdir()
    management = ManagementViewService(home, sessions=sessions, workspace=workspace)

    log_capture = install_log_capture(home)
    try:
        listing = await management.resumable_listing()
    finally:
        log_capture.close()

    assert [summary.id for summary in listing.sessions] == [valid.id]
    assert listing.skipped_count == 2
    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert content.count("WARNING pid=") == 2
    marker = "session=- myclaw.session.session_store: Skipped corrupt or unreadable"
    assert content.count(marker) == 2
    assert content.count("Skipped corrupt or unreadable Conversation Session entry") == 2
    assert str(corrupt_path) in content
    assert str(unreadable_path) in content
    assert "persisted content must stay private" not in content


@pytest.mark.asyncio
async def test_unavailable_session_listing_renders_error_without_terminal_duplicate(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()

    class FailingListingStore:
        async def scan_for_workspace(self, candidate: Path) -> SessionListingReport:
            del candidate
            raise OSError("session directory unavailable")

        async def load(self, session_id: str) -> ConversationSession:
            raise AssertionError(f"Unexpected Session load: {session_id}")

        async def current_session(self, session_id: str) -> ConversationSession:
            raise AssertionError(f"Unexpected current Session read: {session_id}")

    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, sessions=FailingListingStore(), workspace=workspace)
    )
    with configured_process_logging():
        result = await dispatcher.dispatch("/resume")

    assert result.output == "persistence_error: Conversation Sessions could not be listed."
    assert capsys.readouterr().err == ""
    assert not (workspace / ".myclaw" / "logs").exists()


@pytest.mark.asyncio
async def test_resume_load_failure_renders_safe_output_without_terminal_duplicate(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    persisted = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    target = persisted.prepare()
    await persisted.append_message(
        target.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=NOW,
            content="Persisted resume content must stay private.",
        ),
    )

    class FailingResumeLoadStore:
        async def scan_for_workspace(self, candidate: Path) -> SessionListingReport:
            return await persisted.scan_for_workspace(candidate)

        async def load(self, session_id: str) -> ConversationSession:
            del session_id
            raise OSError("resume target disappeared")

        async def current_session(self, session_id: str) -> ConversationSession:
            return await persisted.current_session(session_id)

    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            home,
            sessions=FailingResumeLoadStore(),
            workspace=workspace,
            switch_session=lambda _session_id: None,
        )
    )
    with configured_process_logging():
        result = await dispatcher.resume(target.id)

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
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    target = sessions.prepare()
    await sessions.append_message(
        target.id,
        UserSessionMessage(id=str(FIRST_USER_UUID), created_at=NOW, content="Private history."),
    )

    def fail_switch(_session_id: str) -> None:
        raise RuntimeError("session switch failed")

    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            home,
            sessions=sessions,
            workspace=workspace,
            switch_session=fail_switch,
        )
    )
    with configured_process_logging(), session_log(state, target.id):
        with pytest.raises(RuntimeError, match="session switch failed"):
            await dispatcher.resume(target.id)

    assert capsys.readouterr().err == (
        "Management command failed command=/resume type=RuntimeError\n"
    )
    assert not (state.logs_directory / f"{target.id}.log").exists()


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


class CoordinatedListingStore:
    def __init__(self, delegate: JsonlSessionStore) -> None:
        self._delegate = delegate
        self.current_listing_ready = asyncio.Event()
        self.release_current_listing = asyncio.Event()

    async def current_session(self, session_id: str) -> ConversationSession:
        return await self._delegate.current_session(session_id)

    async def load(self, session_id: str) -> ConversationSession:
        return await self._delegate.load(session_id)

    async def scan_for_workspace(self, workspace: Path) -> SessionListingReport:
        listing = await self._delegate.scan_for_workspace(workspace)
        if Workspace.from_path(workspace) == self._delegate.workspace:
            self.current_listing_ready.set()
            await self.release_current_listing.wait()
        return listing


@pytest.mark.asyncio
async def test_switchable_conversation_close_stops_delegates_from_every_selected_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID, NEWER_SESSION_UUID)).__next__,
    )
    first = sessions.prepare()
    second = sessions.prepare()
    provider = DelayedResumeTitleProvider()

    def conversation_for(session_id: str) -> StreamingConversationPort:
        return StreamingConversationPort(
            provider=provider,
            sessions=sessions,
            session_id=session_id,
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
        )

    conversation = SwitchableConversationPort(
        session_id=first.id,
        build_conversation=conversation_for,
    )
    _ = [event async for event in conversation.submit("Start the first title.")]
    conversation.switch_session(second.id)
    _ = [event async for event in conversation.submit("Start the second title.")]
    for _ in range(100):
        if provider.title_request_count == 2:
            break
        await asyncio.sleep(0)

    await conversation.close()

    assert provider.title_stopped_count == 2


@pytest.mark.asyncio
async def test_switchable_conversation_close_settles_every_selected_adapter() -> None:
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
    delegates = {"first": first, "second": second}
    conversation = SwitchableConversationPort(
        session_id="first",
        build_conversation=delegates.__getitem__,
    )
    _ = [event async for event in conversation.submit("first")]
    conversation.switch_session("second")
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
    historical_store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    historical = historical_store.prepare()
    await historical_store.append_message(
        historical.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=clock.now(),
            content="Remember the deployment decision.",
        ),
    )
    await historical_store.append_message(
        historical.id,
        AssistantSessionMessage(
            id=str(SECOND_USER_UUID),
            created_at=clock.now(),
            content="Deploy on Friday.",
            tool_calls=(),
            status="completed",
            error=None,
            usage=ModelUsage(input_tokens=8, output_tokens=4, total_tokens=12),
        ),
    )
    await historical_store.update_metadata(
        historical.id,
        MetadataUpdate(title="Friday deployment"),
    )
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
    assert writer.operations[1] == ("line", f"Resumed session {historical.id}.")
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
                f"session_id: {historical.id}\n"
                "</runtime_context>\n\n"
                "<user_input>\n"
                "Continue from that decision.\n"
                "</user_input>"
            ),
        },
    ]
    target = await runtime.sessions.load(historical.id)
    assert runtime.session_id == historical.id
    assert [(message.role, message.content) for message in target.messages] == [
        ("user", "Remember the deployment decision."),
        ("assistant", "Deploy on Friday."),
        ("user", "Continue from that decision."),
        ("assistant", "The Friday decision is still current."),
    ]
    assert not runtime.sessions.path_for(startup_session_id).exists()


@pytest.mark.asyncio
async def test_picker_rejects_unsupported_metadata_and_message_record_types(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID, CORRUPT_METADATA_UUID, CORRUPT_MIDDLE_UUID)).__next__,
    )
    prepared = [sessions.prepare() for _ in range(3)]
    for index, metadata in enumerate(prepared):
        await sessions.append_message(
            metadata.id,
            UserSessionMessage(
                id=str((FIRST_USER_UUID, SECOND_USER_UUID, THIRD_USER_UUID)[index]),
                created_at=NOW,
                content=f"Record contract {index}.",
            ),
        )

    unsupported_metadata_path = sessions.path_for(prepared[1].id)
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
    wrong_message_path = sessions.path_for(prepared[2].id)
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

    summaries = (await sessions.scan_for_workspace(workspace)).sessions

    assert [summary.id for summary in summaries] == [prepared[0].id]
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
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID, CORRUPT_METADATA_UUID, CORRUPT_MIDDLE_UUID)).__next__,
    )
    prepared = [sessions.prepare() for _ in range(3)]
    for index, metadata in enumerate(prepared):
        await sessions.append_message(
            metadata.id,
            UserSessionMessage(
                id=str((FIRST_USER_UUID, SECOND_USER_UUID, THIRD_USER_UUID)[index]),
                created_at=NOW,
                content=f"Picker item {index}.",
            ),
        )
    await sessions.update_metadata(prepared[0].id, MetadataUpdate(title="Valid picker item"))
    sessions.path_for(prepared[1].id).write_bytes(b"not metadata\n")
    sessions.path_for(prepared[2].id).write_bytes(b'{"broken":\n')
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, sessions=sessions, workspace=workspace)
    )
    with configured_process_logging(), session_log(state, prepared[0].id):
        result = await dispatcher.dispatch("/resume")

    assert result.output == (
        "Warning: Skipped 2 corrupt Conversation Sessions.\n"
        "Resumable sessions:\n"
        "1. Valid picker item | 2026-07-11T15:30:12.123+08:00 | 1 message"
    )
    assert result.resume_sessions is not None
    assert [session.id for session in result.resume_sessions] == [prepared[0].id]
    assert capsys.readouterr().err == ""
    assert not (state.logs_directory / f"{prepared[0].id}.log").exists()


@pytest.mark.asyncio
async def test_concurrent_resume_listings_keep_their_own_corruption_diagnostics(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    other_workspace = workspace.parent / "other-workspace"
    other_workspace.mkdir()
    persisted = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID, CORRUPT_METADATA_UUID)).__next__,
    )
    valid = persisted.prepare()
    corrupt = persisted.prepare()
    await persisted.append_message(
        valid.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=NOW,
            content="Valid concurrent picker item.",
        ),
    )
    await persisted.update_metadata(valid.id, MetadataUpdate(title="Concurrent picker item"))
    await persisted.append_message(
        corrupt.id,
        UserSessionMessage(
            id=str(SECOND_USER_UUID),
            created_at=NOW,
            content="Will become corrupt.",
        ),
    )
    persisted.path_for(corrupt.id).write_bytes(b"corrupt metadata\n")
    coordinated = CoordinatedListingStore(persisted)
    current_dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, sessions=coordinated, workspace=workspace)
    )
    other_dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, sessions=coordinated, workspace=other_workspace)
    )

    current_listing = asyncio.create_task(current_dispatcher.dispatch("/resume"))
    await asyncio.wait_for(coordinated.current_listing_ready.wait(), timeout=1)
    other_result = await other_dispatcher.dispatch("/resume")
    coordinated.release_current_listing.set()
    current_result = await asyncio.wait_for(current_listing, timeout=1)

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
    switched: list[str] = []
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            home,
            sessions=sessions,
            workspace=workspace,
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
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    target_store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW - timedelta(minutes=5),
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    target = target_store.prepare()
    await target_store.append_message(
        target.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=NOW - timedelta(minutes=5),
            content="Target history.",
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
    original_session_id = runtime.session_id
    await runtime.sessions.append_message(
        original_session_id,
        UserSessionMessage(
            id=str(SECOND_USER_UUID),
            created_at=NOW,
            content="Original nonempty history.",
        ),
    )
    original_path = HOST_FILESYSTEM.path_for_io(runtime.sessions.path_for(original_session_id))
    original_bytes = original_path.read_bytes()

    result = await runtime.management_dispatcher.resume(target.id)

    assert result.output == f"Resumed session {target.id}."
    assert original_path.read_bytes() == original_bytes
    original = await runtime.sessions.load(original_session_id)
    assert [(message.role, message.content) for message in original.messages] == [
        ("user", "Original nonempty history.")
    ]


@pytest.mark.asyncio
async def test_late_title_stays_with_original_and_resumed_session_is_not_retitled(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    target_store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW - timedelta(minutes=5),
        new_uuid=iter((OLDER_SESSION_UUID,)).__next__,
    )
    target = target_store.prepare()
    await target_store.append_message(
        target.id,
        UserSessionMessage(
            id=str(FIRST_USER_UUID),
            created_at=NOW - timedelta(minutes=5),
            content="Existing target history.",
        ),
    )
    await target_store.update_metadata(
        target.id,
        MetadataUpdate(title="Existing target title"),
    )
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
    original_session_id = runtime.session_id

    _ = [event async for event in runtime.conversation.submit("Create the original title.")]
    await asyncio.wait_for(provider.title_started.wait(), timeout=1)

    resume_result = await runtime.management_dispatcher.resume(target.id)
    provider.release_title.set()
    for _ in range(100):
        original = await runtime.sessions.load(original_session_id)
        if original.metadata.title == "Late original title":
            break
        await asyncio.sleep(0)

    _ = [event async for event in runtime.conversation.submit("Continue the existing target.")]

    resumed = await runtime.sessions.load(target.id)
    assert resume_result.output == f"Resumed session {target.id}."
    assert original.metadata.title == "Late original title"
    assert resumed.metadata.title == "Existing target title"
    assert provider.title_request_count == 1
    assert [(message.role, message.content) for message in resumed.messages] == [
        ("user", "Existing target history."),
        ("user", "Continue the existing target."),
        ("assistant", "Target answer."),
    ]
    await runtime.close()


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
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=iter((OLDER_SESSION_UUID, NEWER_SESSION_UUID)).__next__,
    )
    session_a = sessions.prepare()
    session_b = sessions.prepare()
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
        session_a.id: iter((TURN_UUID, FIRST_USER_UUID, REQUEST_UUID, NEW_ASSISTANT_UUID)).__next__,
        session_b.id: iter(
            (
                SECOND_TURN_UUID,
                SECOND_USER_UUID,
                SECOND_REQUEST_UUID,
                SECOND_ASSISTANT_UUID,
            )
        ).__next__,
    }

    def build_port(session_id: str) -> StreamingConversationPort:
        provider: ModelProvider = provider_a if session_id == session_a.id else provider_b
        return StreamingConversationPort(
            provider=provider,
            sessions=sessions,
            session_id=session_id,
            settings=ChatModelSettings(
                model="test-model",
                max_output=1024,
                temperature=0.2,
                reasoning_effort=None,
                timeout_seconds=30,
            ),
            now=lambda: NOW,
            new_uuid=uuids[session_id],
        )

    conversation = SwitchableConversationPort(
        session_id=session_a.id,
        build_conversation=build_port,
    )

    async def collect(text: str) -> list[str]:
        return [event.type async for event in conversation.submit(text)]

    active_a = asyncio.create_task(collect("Start A."))
    await asyncio.wait_for(provider_a.started.wait(), timeout=1)
    try:
        with pytest.raises(RuntimeError, match="active foreground turn"):
            conversation.switch_session(session_b.id)

        assert conversation.session_id == session_a.id
        await conversation.cancel_active_turn()
        assert await asyncio.wait_for(active_a, timeout=1) == [
            "turn_started",
            "turn_cancelled",
        ]

        conversation.switch_session(session_b.id)
        assert await collect("Start B.") == ["turn_started", "turn_completed"]
        assert conversation.session_id == session_b.id
    finally:
        provider_a.release.set()
        if not active_a.done():
            await asyncio.gather(active_a, return_exceptions=True)
