import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from loguru import logger

from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import PreparedReplRuntime, prepare_repl_runtime
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
from myclaw.session.session import Session
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 4, 10, 20, 30, 123000, tzinfo=LOCAL_OFFSET)


class RuntimeProvider:
    def __init__(
        self,
        chat_responses: tuple[ModelResponse, ...],
        *,
        title_response: ModelResponse | None = None,
    ) -> None:
        self._chat_responses = list(chat_responses)
        self._title_response = title_response or ModelResponse(
            message=AssistantModelMessage(content='"Generated runtime title"'),
            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
            finish_reason="stop",
        )
        self.requests: list[ModelRequest] = []
        self.title_started = asyncio.Event()
        self.release_title = asyncio.Event()
        self.delay_title = False
        self.log_marker: str | None = None
        self.closed = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if request.system_prompt == session_title_prompt():
            self.title_started.set()
            if self.delay_title:
                await self.release_title.wait()
            yield ModelCompleted(response=self._title_response)
            return
        if not self._chat_responses:
            raise AssertionError("No chat response was scripted")
        if self.log_marker is not None:
            logger.warning(self.log_marker)
        yield ModelCompleted(response=self._chat_responses.pop(0))

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.route != "memory":
            raise AssertionError(f"Unexpected completion request: {request!r}")
        return ModelResponse(
            message=AssistantModelMessage(content="Summary of the earlier turn."),
            usage=ModelUsage(input_tokens=3, output_tokens=1, total_tokens=4),
            finish_reason="stop",
        )

    async def close(self) -> None:
        self.closed = True


def _runtime(
    agent_home: Path,
    workspace: Path,
    provider: RuntimeProvider,
    *,
    config: str = VALID_CONFIG,
) -> PreparedReplRuntime:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(config, encoding="utf-8")
    clock = FakeClock(NOW)
    return prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=uuid4,
        retry_clock=clock,
    )


@pytest.mark.asyncio
async def test_runtime_routes_turn_title_status_and_close_through_one_active_session(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RuntimeProvider(
        (
            ModelResponse(
                message=AssistantModelMessage(content="First response."),
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                finish_reason="stop",
            ),
        )
    )
    provider.delay_title = True
    runtime = _runtime(agent_home, workspace, provider)

    assert isinstance(runtime.session, Session)
    session = runtime.session
    assert runtime.session_id == session.session_id
    assert not (workspace / ".myclaw" / "sessions" / f"{session.session_id}.jsonl").exists()

    events = [event async for event in runtime.conversation.submit("First input.")]
    await provider.title_started.wait()

    assert [event.type for event in events] == ["turn_started", "turn_completed"]
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.metadata["title"] == "Untitled session"

    status_result = await runtime.management_dispatcher.dispatch("/status")
    assert status_result.output is not None
    status = json.loads(status_result.output)
    assert status["session_message_count"] == 2
    assert status["cumulative_usage"] == {
        "model_calls": 1,
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }

    provider.release_title.set()
    for _ in range(100):
        if session.metadata["title"] == "Generated runtime title":
            break
        await asyncio.sleep(0)
    assert session.metadata["title"] == "Generated runtime title"
    assert session.metadata["token_usage"]["model_calls"] == 2

    close_calls: list[bool] = []
    close = session.close

    def record_close() -> None:
        close_calls.append(True)
        close()

    monkeypatch.setattr(session, "close", record_close)
    await runtime.close()

    assert close_calls == [True]
    assert provider.closed
    assert (workspace / ".myclaw" / "sessions" / f"{session.session_id}.jsonl").exists()
    assert Session.load(session.workspace_state, session.session_id).metadata["title"] == (
        "Generated runtime title"
    )


@pytest.mark.asyncio
async def test_late_title_is_saved_by_the_next_complete_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = RuntimeProvider(
        (
            ModelResponse(
                message=AssistantModelMessage(content="First response."),
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                finish_reason="stop",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Second response."),
                usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                finish_reason="stop",
            ),
        )
    )
    provider.delay_title = True
    runtime = _runtime(agent_home, workspace, provider)
    session = runtime.session

    _ = [event async for event in runtime.conversation.submit("First input.")]
    await provider.title_started.wait()
    await asyncio.sleep(0)
    assert Session.load(session.workspace_state, session.session_id).metadata["title"] == (
        "Untitled session"
    )

    provider.release_title.set()
    for _ in range(100):
        if session.metadata["title"] == "Generated runtime title":
            break
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert session.metadata["title"] == "Generated runtime title"
    assert Session.load(session.workspace_state, session.session_id).metadata["title"] == (
        "Untitled session"
    )

    _ = [event async for event in runtime.conversation.submit("Second input.")]
    await asyncio.sleep(0)

    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == "Generated runtime title"
    assert [message["role"] for message in reloaded.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_rejects_tool_call_title_and_counts_its_usage(
    agent_home: Path,
    workspace: Path,
) -> None:
    title_tool_call = ModelToolCall(
        id="call_invalid_title",
        name="read_file",
        arguments='{"path":"README.md"}',
    )
    provider = RuntimeProvider(
        (
            ModelResponse(
                message=AssistantModelMessage(content="First response."),
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                finish_reason="stop",
            ),
        ),
        title_response=ModelResponse(
            message=AssistantModelMessage(
                content="Do not use this title",
                tool_calls=(title_tool_call,),
            ),
            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            finish_reason="tool_calls",
        ),
    )
    runtime = _runtime(agent_home, workspace, provider)

    events = [event async for event in runtime.conversation.submit("  First fallback title.  ")]
    for _ in range(100):
        if runtime.session.metadata["token_usage"]["model_calls"] == 2:
            break
        await asyncio.sleep(0)

    assert events[-1].type == "turn_completed"
    assert runtime.session.metadata["title"] == "First fallback title."
    assert runtime.session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 8,
        "output_tokens": 4,
        "total_tokens": 12,
    }
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_empty_generated_title_uses_the_first_user_fallback(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = RuntimeProvider(
        (
            ModelResponse(
                message=AssistantModelMessage(content="First response."),
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                finish_reason="stop",
            ),
        ),
        title_response=ModelResponse(
            message=AssistantModelMessage(content="\n\t  "),
            usage=ModelUsage(input_tokens=3, output_tokens=1, total_tokens=4),
            finish_reason="stop",
        ),
    )
    runtime = _runtime(agent_home, workspace, provider)

    events = [
        event async for event in runtime.conversation.submit("  Meaningful first question.  ")
    ]
    for _ in range(100):
        if runtime.session.metadata["token_usage"]["model_calls"] == 2:
            break
        await asyncio.sleep(0)

    assert events[-1].type == "turn_completed"
    assert runtime.session.metadata["title"] == "Meaningful first question."
    assert runtime.session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 8,
        "output_tokens": 3,
        "total_tokens": 11,
    }
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_shutdown_applies_first_user_title_fallback_before_final_save(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = RuntimeProvider(
        (
            ModelResponse(
                message=AssistantModelMessage(content="First response."),
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                finish_reason="stop",
            ),
        )
    )
    provider.delay_title = True
    runtime = _runtime(agent_home, workspace, provider)
    session = runtime.session

    _ = [event async for event in runtime.conversation.submit("  Shutdown fallback title.  ")]
    await provider.title_started.wait()
    await runtime.close()

    assert session.metadata["title"] == "Shutdown fallback title."
    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == "Shutdown fallback title."
    assert reloaded.metadata["token_usage"]["model_calls"] == 1


@pytest.mark.asyncio
async def test_immediate_turn_cancellation_keeps_the_first_user_title_lifecycle(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = RuntimeProvider(
        (
            ModelResponse(
                message=AssistantModelMessage(content="Unused response."),
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                finish_reason="stop",
            ),
        )
    )
    provider.delay_title = True
    runtime = _runtime(agent_home, workspace, provider)
    session = runtime.session
    events = runtime.conversation.submit("  Cancelled first title.  ")

    assert (await anext(events)).type == "turn_started"
    await runtime.conversation.cancel_active_turn()
    assert [event.type async for event in events] == ["turn_cancelled"]
    await asyncio.wait_for(provider.title_started.wait(), timeout=1)
    await runtime.close()

    assert session.metadata["title"] == "Cancelled first title."
    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == "Cancelled first title."


@pytest.mark.asyncio
async def test_runtime_shutdown_keeps_an_empty_session_memory_only(
    agent_home: Path,
    workspace: Path,
) -> None:
    runtime = _runtime(agent_home, workspace, RuntimeProvider(()))
    session_path = workspace / ".myclaw" / "sessions" / f"{runtime.session.session_id}.jsonl"

    await runtime.close()

    assert not session_path.exists()


@pytest.mark.asyncio
async def test_runtime_shutdown_swallows_final_session_close_fault_after_router_close(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RuntimeProvider(
        (
            ModelResponse(
                message=AssistantModelMessage(content="First response."),
                usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                finish_reason="stop",
            ),
        )
    )
    runtime = _runtime(agent_home, workspace, provider)
    _ = [event async for event in runtime.conversation.submit("Trigger provider construction.")]
    close_order: list[str] = []
    provider_close = provider.close

    async def record_provider_close() -> None:
        close_order.append("provider")
        await provider_close()

    def fail_session_close() -> None:
        close_order.append("session")
        raise OSError("injected final Session close failure")

    monkeypatch.setattr(provider, "close", record_provider_close)
    monkeypatch.setattr(runtime.session, "close", fail_session_close)

    await runtime.close()

    assert provider.closed
    assert close_order == ["provider", "session"]


@pytest.mark.asyncio
async def test_runtime_active_session_keeps_artifact_and_log_correlation_when_persist_fails(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_tool_result = "oversized tool result " * 100
    (workspace / "large.txt").write_text(raw_tool_result, encoding="utf-8")
    tool_call = ModelToolCall(
        id="call_active_artifact",
        name="read_file",
        arguments='{"path":"large.txt"}',
    )
    provider = RuntimeProvider(
        (
            ModelResponse(
                message=AssistantModelMessage(content="", tool_calls=(tool_call,)),
                usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Artifact recorded."),
                usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                finish_reason="stop",
            ),
        )
    )
    config = VALID_CONFIG.replace("max_tool_result_chars = 60000", "max_tool_result_chars = 1000")
    runtime = _runtime(agent_home, workspace, provider, config=config)
    session = runtime.session
    provider.log_marker = "active Session correlation marker"

    def fail_persist() -> None:
        raise OSError("ordinary snapshot failure")

    monkeypatch.setattr(session, "persist", fail_persist)
    events = [event async for event in runtime.conversation.submit("Inspect large.txt.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    tool_message = session.messages[2]
    assert tool_message["role"] == "tool"
    artifact = tool_message["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["path"] == f"artifacts/{session.session_id}/call_active_artifact.txt"
    artifact_path = workspace / ".myclaw" / "sessions" / artifact["path"]
    assert artifact_path.read_text(encoding="utf-8") == raw_tool_result

    await runtime.close()
    log_path = workspace / ".myclaw" / "logs" / f"{session.session_id}.log"
    assert session.session_id in log_path.name
    assert "active Session correlation marker" in log_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_runtime_summary_advances_the_same_active_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = RuntimeProvider(
        (
            ModelResponse(
                message=AssistantModelMessage(content="First response."),
                usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                finish_reason="stop",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Second response."),
                usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                finish_reason="stop",
            ),
            ModelResponse(
                message=AssistantModelMessage(content="Third response."),
                usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                finish_reason="stop",
            ),
        )
    )
    config = VALID_CONFIG.replace(
        "consolidation_message_threshold = 50", "consolidation_message_threshold = 4"
    )
    runtime = _runtime(agent_home, workspace, provider, config=config)
    session = runtime.session

    events = [event async for event in runtime.conversation.submit("First input.")]
    assert events[-1].type == "turn_completed"
    events = [event async for event in runtime.conversation.submit("Second input.")]
    assert events[-1].type == "turn_completed"
    events = [event async for event in runtime.conversation.submit("Third input.")]
    assert events[-1].type == "turn_completed"
    assert session.last_consolidated == 2
    assert (workspace / ".myclaw" / "memory" / "summary.jsonl").exists()
    await runtime.close()
