from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.events import AgentEvent, ConfirmationRequestedPayload
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
from myclaw.tools.core.web_fetch import JinaReaderClient
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


class _RuntimeProvider:
    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = deque(responses)
        self.stream_requests: list[ModelRequest] = []
        self.closed = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if request.system_prompt == session_title_prompt():
            yield ModelCompleted(
                response=ModelResponse(
                    message=AssistantModelMessage(content="Read external file"),
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    finish_reason="stop",
                )
            )
            return
        self.stream_requests.append(request)
        if not self._responses:
            raise AssertionError("No scripted Runtime response remains")
        yield ModelCompleted(response=self._responses.popleft())

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected non-chat request: {request!r}")

    async def close(self) -> None:
        self.closed = True


class _BlockingClock:
    def __init__(self) -> None:
        self._wake = asyncio.Event()

    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds
        await self._wake.wait()


def _response(*, content: str, tool_call: ModelToolCall | None = None) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(
            content=content,
            tool_calls=() if tool_call is None else (tool_call,),
        ),
        usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
        finish_reason="tool_calls" if tool_call is not None else "stop",
    )


def _runtime(
    agent_home: Path,
    workspace: Path,
    provider: _RuntimeProvider,
) -> PreparedReplRuntime:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    return prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        memory_scheduler_clock=_BlockingClock(),
        schedule_scheduler_clock=_BlockingClock(),
    )


async def _drain_until_terminal(
    events: AsyncIterator[AgentEvent],
    runtime: PreparedReplRuntime,
) -> list[AgentEvent]:
    observed: list[AgentEvent] = []
    async for event in events:
        observed.append(event)
        if event.type == "confirmation_requested":
            assert isinstance(event.payload, ConfirmationRequestedPayload)
            runtime.conversation.respond_to_confirmation(
                event.payload.confirmation_id,
                "approved",
            )
    return observed


@pytest.mark.asyncio
async def test_runtime_uses_fixed_catalog_for_provider_confirmation_and_persistence(
    agent_home: Path,
    workspace: Path,
) -> None:
    outside = (workspace.parent / "external-note.txt").resolve()
    outside.write_text("outside content", encoding="utf-8")
    provider = _RuntimeProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_external_read",
                    name="read_file",
                    arguments=json.dumps({"path": str(outside)}),
                ),
            ),
            _response(content="Done."),
        )
    )
    runtime = _runtime(agent_home, workspace, provider)
    try:
        events = await _drain_until_terminal(runtime.conversation.submit("Read the file."), runtime)
    finally:
        await runtime.close()

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "confirmation_requested",
        "tool_completed",
        "turn_completed",
    ]
    assert isinstance(events[2].payload, ConfirmationRequestedPayload)
    assert events[2].payload.turn_id == events[0].turn_id
    assert events[2].payload.details["path"] != str(outside) or len(str(outside)) <= 256
    assert provider.stream_requests
    assert [definition["function"]["name"] for definition in provider.stream_requests[0].tools] == [
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "glob",
        "grep",
        "exec",
        "web_search",
        "web_fetch",
        "schedule",
    ]
    tool_messages = [message for message in runtime.session.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "outside content"
    assert tool_messages[0]["status"] == "success"


@pytest.mark.asyncio
async def test_runtime_cancellation_reaches_an_active_fixed_catalog_tool(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def block_fetch(
        self: JinaReaderClient,
        url: str,
        *,
        output_format: str,
    ) -> str:
        del self, url, output_format
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "unexpected"

    monkeypatch.setattr(JinaReaderClient, "fetch", block_fetch)
    provider = _RuntimeProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_blocking_fetch",
                    name="web_fetch",
                    arguments='{"url":"https://8.8.8.8/"}',
                ),
            ),
        )
    )
    runtime = _runtime(agent_home, workspace, provider)
    turn = asyncio.create_task(
        _drain_until_terminal(runtime.conversation.submit("Fetch the URL."), runtime)
    )
    try:
        await started.wait()
        await runtime.conversation.cancel_active_turn()
        events = await asyncio.wait_for(turn, timeout=1)
    finally:
        release.set()
        if not turn.done():
            turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await runtime.close()

    assert cancelled.is_set()
    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "turn_cancelled",
    ]
    tool_messages = [message for message in runtime.session.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["status"] == "error"
    assert tool_messages[0]["content"] == ("Tool call interrupted because the turn was cancelled.")
