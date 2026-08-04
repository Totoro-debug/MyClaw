import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session import Session
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
REQUEST_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")


class RuntimeTitleProvider:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if "<long_term_memory>" not in request.system_prompt:
            yield ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content='"Runtime   project"'),
                    usage=ModelUsage(input_tokens=4, output_tokens=1, total_tokens=5),
                    finish_reason="stop",
                )
            )
            return
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content="Main answer."),
                usage=ModelUsage(input_tokens=7, output_tokens=2, total_tokens=9),
                finish_reason="stop",
            )
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

    async def close(self) -> None:
        return None


class TitleFirstProvider:
    def __init__(self) -> None:
        self.chat_started = asyncio.Event()
        self.release_chat = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if request.system_prompt == "Generate a title.":
            yield ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content='"Generated before chat"'),
                    usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                    finish_reason="stop",
                )
            )
            return
        self.chat_started.set()
        await self.release_chat.wait()
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content="Main answer."),
                usage=ModelUsage(input_tokens=7, output_tokens=2, total_tokens=9),
                finish_reason="stop",
            )
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_title_finishing_before_chat_does_not_publish_an_intermediate_snapshot(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW, new_uuid=lambda: SESSION_UUID)
    provider = TitleFirstProvider()
    replacements: list[bytes] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(path: Path, content: bytes) -> None:
        replacements.append(content)
        replace(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replace)
    conversation = StreamingConversationPort(
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=iter((TURN_UUID, REQUEST_UUID)).__next__,
        title_prompt="Generate a title.",
    )
    events = conversation.submit("First input.")

    assert (await anext(events)).type == "turn_started"
    terminal = asyncio.create_task(anext(events))
    await provider.chat_started.wait()
    for _ in range(100):
        if session.metadata["title"] == "Generated before chat":
            break
        await asyncio.sleep(0)
    await asyncio.sleep(0)

    session_path = state.sessions_directory / f"{session.session_id}.jsonl"
    assert session.metadata["title"] == "Generated before chat"
    assert not session_path.exists()
    assert replacements == []

    provider.release_chat.set()
    assert (await terminal).type == "turn_completed"
    with pytest.raises(StopAsyncIteration):
        await anext(events)
    await asyncio.sleep(0)

    assert len(replacements) == 1
    reloaded = Session.load(state, session.session_id)
    assert reloaded.metadata["title"] == "Generated before chat"
    assert [message["role"] for message in reloaded.messages] == ["user", "assistant"]
    await conversation.close()


@pytest.mark.asyncio
async def test_prepared_runtime_uses_an_isolated_chat_stream_for_session_title(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    clock = FakeClock(NOW)
    provider = RuntimeTitleProvider()
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID, REQUEST_UUID)).__next__,
        retry_clock=clock,
    )

    event_types = [
        event.type async for event in runtime.conversation.submit("  First\t runtime input.  ")
    ]
    for _ in range(100):
        if len(provider.requests) == 2 and runtime.session.metadata["title"] == "Runtime project":
            break
        await asyncio.sleep(0)

    assert event_types == ["turn_started", "turn_completed"]
    assert len(provider.requests) == 2
    title_request = next(
        request
        for request in provider.requests
        if "<long_term_memory>" not in request.system_prompt
    )
    assert title_request.route == "chat"
    assert title_request.stream is True
    assert title_request.tools == ()
    assert [message.to_dict() for message in title_request.messages] == [
        {"role": "user", "content": "First runtime input."}
    ]
    assert "<runtime_context>" not in title_request.system_prompt
    assert "<long_term_memory>" not in title_request.system_prompt
    assert runtime.session.metadata["title"] == "Runtime project"
    assert runtime.session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 11,
        "output_tokens": 3,
        "total_tokens": 14,
    }
    assert [message["role"] for message in runtime.session.messages] == ["user", "assistant"]
