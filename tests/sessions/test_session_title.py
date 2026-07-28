import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.ports import ConversationPort
from myclaw.agent.workspace import Workspace
from myclaw.config.agent_home import AgentHome
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.runtime_log import install_runtime_logging
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.records import (
    AssistantSessionMessage,
    ConversationSession,
    CumulativeUsage,
    MetadataUpdate,
    SessionMessage,
    SessionSummary,
    UserSessionMessage,
)
from myclaw.session.session_store import JsonlSessionStore
from myclaw.tools.models import ModelToolCall
from tests.fixtures import FakeClock

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
SESSION_TWO_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
REQUEST_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
TITLE_PROMPT = "Return a short title for this Conversation Session."


def _runtime_log_text(agent_home: Path) -> str:
    logs = agent_home / "logs"
    return "".join(
        path.read_text(encoding="utf-8")
        for name in ("run.log.0", "run.log.1")
        if (path := logs / name).exists()
    )


class DelayedTitleProvider:
    def __init__(
        self, title_content: str = "\n  \u300c  Project\t review  \u300d\nIgnore this line"
    ) -> None:
        self.title_content = title_content
        self.title_started = asyncio.Event()
        self.title_stopped = asyncio.Event()
        self.release_title = asyncio.Event()
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if request.system_prompt == TITLE_PROMPT:
            self.title_started.set()
            try:
                await self.release_title.wait()
                yield ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(content=self.title_content),
                        usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                        finish_reason="stop",
                    )
                )
            finally:
                self.title_stopped.set()
            return

        await self.title_started.wait()
        yield TextDelta(delta="First answer.")
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content="First answer."),
                usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
                finish_reason="stop",
            )
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

    async def close(self) -> None:
        return None


class FailedTitleProvider:
    def __init__(self) -> None:
        self.title_started = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if request.system_prompt == TITLE_PROMPT:
            self.title_started.set()
            raise ModelCallError(
                ErrorInfo(code="model_failed", message="Title generation failed.")
            ) from RuntimeError("private-title-provider-body")

        await self.title_started.wait()
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content="First answer."),
                usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
                finish_reason="stop",
            )
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

    async def close(self) -> None:
        return None


class FailingTitleStreamCloseProvider:
    def __init__(self) -> None:
        self.title_started = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if request.system_prompt == TITLE_PROMPT:
            self.title_started.set()
            try:
                yield ModelCompleted(
                    response=ModelResponse(
                        message=AssistantModelMessage(content="Generated title"),
                        usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                        finish_reason="stop",
                    )
                )
            finally:
                raise OSError("PRIVATE_TITLE_STREAM_BODY_52")
            return

        await self.title_started.wait()
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content="First answer."),
                usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
                finish_reason="stop",
            )
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

    async def close(self) -> None:
        return None


class InvalidTitleProvider:
    def __init__(self) -> None:
        self.title_started = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if request.system_prompt == TITLE_PROMPT:
            self.title_started.set()
            yield ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(
                        content="Disallowed generated title",
                        tool_calls=(
                            ModelToolCall(
                                id="call-title",
                                name="read_file",
                                arguments='{"path":"README.md"}',
                            ),
                        ),
                    ),
                    usage=ModelUsage(input_tokens=9, output_tokens=4, total_tokens=13),
                    finish_reason="tool_calls",
                )
            )
            return

        await self.title_started.wait()
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content="First answer."),
                usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
                finish_reason="stop",
            )
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

    async def close(self) -> None:
        return None


class BlockingLoadSessionStore:
    def __init__(self, delegate: JsonlSessionStore) -> None:
        self._delegate = delegate
        self.load_started = asyncio.Event()
        self.release_load = asyncio.Event()

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        await self._delegate.append_message(session_id, message)

    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None:
        await self._delegate.update_metadata(session_id, update)

    async def load(self, session_id: str) -> ConversationSession:
        self.load_started.set()
        await self.release_load.wait()
        return await self._delegate.load(session_id)

    async def list_for_workspace(self, workspace: Path) -> tuple[SessionSummary, ...]:
        return await self._delegate.list_for_workspace(workspace)


class MetadataUpdateFailingStore(JsonlSessionStore):
    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None:
        del session_id, update
        try:
            raise RuntimeError("private-title-metadata-cause")
        except RuntimeError as cause:
            raise OSError("private-title-metadata-detail") from cause


class FailingMetadataSessionStore:
    def __init__(self, delegate: JsonlSessionStore) -> None:
        self._delegate = delegate

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        await self._delegate.append_message(session_id, message)

    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None:
        del session_id, update
        raise OSError("title metadata write failed")

    async def load(self, session_id: str) -> ConversationSession:
        return await self._delegate.load(session_id)

    async def list_for_workspace(self, workspace: Path) -> tuple[SessionSummary, ...]:
        return await self._delegate.list_for_workspace(workspace)


async def _collect_events(conversation: ConversationPort, text: str) -> list[str]:
    return [event.type async for event in conversation.submit(text)]


@pytest.mark.asyncio
async def test_first_turn_finishes_while_title_updates_only_session_metadata_later(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = DelayedTitleProvider()
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="chat system prompt",
        title_prompt=TITLE_PROMPT,
    )

    event_types = await asyncio.wait_for(
        _collect_events(conversation, "Review this project."),
        timeout=1,
    )
    before_title = await store.load(session.id)

    assert event_types == ["turn_started", "text_delta", "turn_completed"]
    assert before_title.metadata.title == "Untitled session"
    assert [message.role for message in before_title.messages] == ["user", "assistant"]

    provider.release_title.set()
    for _ in range(100):
        after_title = await store.load(session.id)
        if after_title.metadata.title == "Project review":
            break
        await asyncio.sleep(0)

    assert after_title.metadata.title == "Project review"
    assert after_title.metadata.cumulative_usage == CumulativeUsage(
        model_calls=2,
        input_tokens=20,
        output_tokens=5,
        total_tokens=25,
    )
    assert after_title.messages == before_title.messages


@pytest.mark.asyncio
async def test_close_waits_for_the_detached_title_task_to_stop(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = DelayedTitleProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="chat system prompt",
        title_prompt=TITLE_PROMPT,
    )
    runtime_log = install_runtime_logging(home)

    try:
        with runtime_log.session(session.id):
            await _collect_events(conversation, "Start a title that will still be running.")
            await provider.title_started.wait()
            await conversation.close()
    finally:
        runtime_log.close()

    assert provider.title_stopped.is_set()
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_cancelling_after_first_user_persistence_does_not_cancel_title_generation(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    persisted = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = persisted.prepare()
    sessions = BlockingLoadSessionStore(persisted)
    provider = DelayedTitleProvider("Cancellation-safe title")
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="chat system prompt",
        title_prompt=TITLE_PROMPT,
    )
    collector = asyncio.create_task(_collect_events(conversation, "Persist this first input."))

    try:
        await asyncio.wait_for(sessions.load_started.wait(), timeout=1)
        await conversation.cancel_active_turn()
        event_types = await asyncio.wait_for(collector, timeout=1)
    finally:
        sessions.release_load.set()
        provider.release_title.set()
        if not collector.done():
            collector.cancel()
            await asyncio.gather(collector, return_exceptions=True)

    for _ in range(100):
        reloaded = await persisted.load(session.id)
        if reloaded.metadata.title == "Cancellation-safe title":
            break
        await asyncio.sleep(0)

    assert event_types == ["turn_started", "turn_cancelled"]
    assert reloaded.metadata.title == "Cancellation-safe title"
    assert [message.role for message in reloaded.messages] == ["user"]
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_failed_title_call_uses_normalized_unicode_bounded_input_fallback(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = FailedTitleProvider()
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="chat system prompt",
        title_prompt=TITLE_PROMPT,
    )
    first_input = "  Plan\t" + "\u754c" * 70
    runtime_log = install_runtime_logging(home)

    try:
        with runtime_log.session(session.id):
            event_types = await _collect_events(conversation, first_input)
            for _ in range(100):
                reloaded = await store.load(session.id)
                if reloaded.metadata.title != "Untitled session":
                    break
                await asyncio.sleep(0)
    finally:
        runtime_log.close()

    assert event_types == ["turn_started", "turn_completed"]
    assert reloaded.metadata.title == "Plan " + "\u754c" * 55
    assert len(reloaded.metadata.title) == 60
    content = _runtime_log_text(agent_home)
    records = [
        line for line in content.splitlines() if "myclaw.session.conversation:" in line
    ]
    assert len(records) == 1
    assert " WARNING " in records[0]
    assert "Session title fallback selected code=model_failed type=ModelCallError" in records[0]
    assert f"session={session.id}" in records[0]
    assert content.count("Traceback (most recent call last)") == 1
    assert "ModelCallError: Session title generation failed." in content
    assert first_input not in content
    assert TITLE_PROMPT not in content
    assert "private-title-provider-body" not in content


@pytest.mark.asyncio
async def test_terminal_title_metadata_failure_is_logged_without_repeating_provider_cause(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = MetadataUpdateFailingStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = FailedTitleProvider()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="chat system prompt",
        title_prompt=TITLE_PROMPT,
    )
    lifetime = install_runtime_logging(home)

    with lifetime.session(session.id):
        event_types = await _collect_events(conversation, "Fallback title input")
        await conversation.close()
    lifetime.close()

    assert event_types == ["turn_started", "turn_completed"]
    content = _runtime_log_text(agent_home)
    records = [
        line for line in content.splitlines() if "myclaw.session.conversation:" in line
    ]
    assert len(records) == 2
    assert " WARNING " in records[0]
    assert "Session title fallback selected code=model_failed type=ModelCallError" in records[0]
    assert " ERROR " in records[1]
    assert (
        "Session title failed code=persistence_error operation=metadata_update type=OSError"
        in records[1]
    )
    assert all(f"session={session.id}" in record for record in records)
    error_record = content[content.index(records[1]) :]
    assert "Traceback (most recent call last)" in error_record
    assert "OSError" in error_record
    assert "RuntimeError" in error_record
    assert "ModelCallError" not in error_record
    assert "private-title-provider-body" not in content
    assert "private-title-metadata-detail" not in content
    assert "private-title-metadata-cause" not in content


@pytest.mark.asyncio
async def test_title_metadata_failure_records_one_error_without_changing_foreground_events(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    persisted = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = persisted.prepare()
    provider = DelayedTitleProvider("Generated title")
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=FailingMetadataSessionStore(persisted),
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="chat system prompt",
        title_prompt=TITLE_PROMPT,
    )
    first_input = "This content must not enter the Runtime Log."
    runtime_log = install_runtime_logging(home)

    try:
        with runtime_log.session(session.id):
            event_types = await _collect_events(conversation, first_input)
            provider.release_title.set()
            await conversation.close()
    finally:
        runtime_log.close()

    assert event_types == ["turn_started", "text_delta", "turn_completed"]
    content = _runtime_log_text(agent_home)
    records = [
        line for line in content.splitlines() if "myclaw.session.conversation:" in line
    ]
    assert len(records) == 1
    assert " ERROR " in records[0]
    assert (
        "Session title failed code=persistence_error operation=metadata_update type=OSError"
        in records[0]
    )
    assert f"session={session.id}" in records[0]
    assert "OSError: Session title persistence failed." in content
    assert "title metadata write failed" not in content
    assert first_input not in content


@pytest.mark.asyncio
async def test_title_stream_cleanup_failure_warns_and_keeps_the_generated_title(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    conversation = StreamingConversationPort(
        provider=FailingTitleStreamCloseProvider(),
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="chat system prompt",
        title_prompt=TITLE_PROMPT,
    )
    runtime_log = install_runtime_logging(home)

    try:
        with runtime_log.session(session.id):
            events = await _collect_events(conversation, "Name this safely.")
            await conversation.close()
    finally:
        runtime_log.close()

    assert events == ["turn_started", "turn_completed"]
    assert (await store.load(session.id)).metadata.title == "Generated title"
    content = _runtime_log_text(agent_home)
    marker = (
        f"session={session.id} myclaw.session.conversation: "
        "Session title stream cleanup failed type=OSError"
    )
    assert content.count("WARNING pid=") == 1
    assert content.count("ERROR pid=") == 0
    assert content.count(marker) == 1
    assert "OSError: [REDACTED]" in content
    assert "PRIVATE_TITLE_STREAM_BODY_52" not in content


@pytest.mark.asyncio
async def test_metadata_update_atomically_adds_auxiliary_model_usage(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    await store.append_message(
        session.id,
        UserSessionMessage(
            id=str(REQUEST_UUID),
            created_at=NOW,
            content="Review this project.",
        ),
    )
    await store.append_message(
        session.id,
        AssistantSessionMessage(
            id=str(REQUEST_UUID),
            created_at=NOW,
            content="First answer.",
            tool_calls=(),
            status="completed",
            error=None,
            usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
        ),
    )

    await store.update_metadata(
        session.id,
        MetadataUpdate(
            title="Project review",
            usage_delta=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
        ),
    )

    reloaded = await store.load(session.id)
    assert reloaded.metadata.title == "Project review"
    assert reloaded.metadata.cumulative_usage == CumulativeUsage(
        model_calls=2,
        input_tokens=20,
        output_tokens=5,
        total_tokens=25,
    )


@pytest.mark.asyncio
async def test_title_completion_with_tool_calls_uses_fallback_but_counts_usage(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = InvalidTitleProvider()
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="chat system prompt",
        title_prompt=TITLE_PROMPT,
    )
    lifetime = install_runtime_logging(home)

    with lifetime.session(session.id):
        await _collect_events(conversation, "Fallback project title")
    for _ in range(100):
        reloaded = await store.load(session.id)
        if reloaded.metadata.cumulative_usage.model_calls == 2:
            break
        await asyncio.sleep(0)
    lifetime.close()

    assert reloaded.metadata.title == "Fallback project title"
    assert reloaded.metadata.cumulative_usage == CumulativeUsage(
        model_calls=2,
        input_tokens=21,
        output_tokens=7,
        total_tokens=28,
    )
    assert [message.role for message in reloaded.messages] == ["user", "assistant"]
    content = _runtime_log_text(agent_home)
    records = [
        line for line in content.splitlines() if "myclaw.session.conversation:" in line
    ]
    assert len(records) == 1
    assert " WARNING " in records[0]
    assert "Session title fallback selected code=model_failed" in records[0]
    assert f"session={session.id}" in records[0]
    assert "Disallowed generated title" not in content
    assert '"path":"README.md"' not in content
    assert "Fallback project title" not in content


@pytest.mark.asyncio
async def test_empty_title_and_empty_normalized_input_keep_untitled_with_usage(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    provider = DelayedTitleProvider(title_content="\n\t  ")
    conversation: ConversationPort = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=lambda: REQUEST_UUID,
        system_prompt="chat system prompt",
        title_prompt=TITLE_PROMPT,
    )
    lifetime = install_runtime_logging(home)

    with lifetime.session(session.id):
        await _collect_events(conversation, '""')
    provider.release_title.set()
    for _ in range(100):
        reloaded = await store.load(session.id)
        if reloaded.metadata.cumulative_usage.model_calls == 2:
            break
        await asyncio.sleep(0)
    lifetime.close()

    assert reloaded.metadata.title == "Untitled session"
    assert reloaded.metadata.cumulative_usage == CumulativeUsage(
        model_calls=2,
        input_tokens=20,
        output_tokens=5,
        total_tokens=25,
    )
    assert [message.role for message in reloaded.messages] == ["user", "assistant"]
    content = _runtime_log_text(agent_home)
    records = [
        line for line in content.splitlines() if "myclaw.session.conversation:" in line
    ]
    assert len(records) == 1
    assert " WARNING " in records[0]
    assert "Session title fallback selected code=model_failed" in records[0]
    assert f"session={session.id}" in records[0]
    assert '""' not in content


@pytest.mark.asyncio
async def test_late_titles_update_the_session_that_started_each_call(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID, SESSION_TWO_UUID)).__next__,
    )
    first_session = store.prepare()
    second_session = store.prepare()
    first_provider = DelayedTitleProvider(title_content="First session title")
    second_provider = DelayedTitleProvider(title_content="Second session title")

    def conversation_for(
        session_id: str,
        provider: DelayedTitleProvider,
    ) -> ConversationPort:
        return StreamingConversationPort(
            provider=provider,
            sessions=store,
            session_id=session_id,
            settings=ChatModelSettings(
                model="test-model",
                max_output=1024,
                temperature=0.2,
                reasoning_effort=None,
                timeout_seconds=30,
            ),
            now=clock.now,
            new_uuid=lambda: REQUEST_UUID,
            system_prompt="chat system prompt",
            title_prompt=TITLE_PROMPT,
        )

    await _collect_events(
        conversation_for(first_session.id, first_provider),
        "First session input",
    )
    await _collect_events(
        conversation_for(second_session.id, second_provider),
        "Second session input",
    )

    second_provider.release_title.set()
    for _ in range(100):
        first_before_late = await store.load(first_session.id)
        second_after_title = await store.load(second_session.id)
        if second_after_title.metadata.title == "Second session title":
            break
        await asyncio.sleep(0)
    assert first_before_late.metadata.title == "Untitled session"

    first_provider.release_title.set()
    for _ in range(100):
        first_after_title = await store.load(first_session.id)
        second_after_late = await store.load(second_session.id)
        if first_after_title.metadata.title == "First session title":
            break
        await asyncio.sleep(0)

    assert first_after_title.metadata.title == "First session title"
    assert second_after_late.metadata.title == "Second session title"
