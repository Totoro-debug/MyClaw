from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.loop import ConfirmationRequestView
from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import PreparedRuntime, prepare_runtime
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.core.web_fetch import JinaReaderClient
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import DeterministicTaskFramingEvaluator
from tests.fixtures.provider import ProviderCall
from tests.runtime_bus import collect_foreground_outbound

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


class _RuntimeProvider:
    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = deque(responses)
        self.stream_requests: list[ProviderCall] = []
        self.closed = False

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        request = ProviderCall(
            messages=list(messages),
            tools=tuple(tools),
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        if request.messages and request.messages[0] == {
            "role": "system",
            "content": session_title_prompt(),
        }:
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

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        raise AssertionError(
            "Unexpected non-chat request: "
            f"{messages=}, {tools=}, {model=}, {max_output=}, {temperature=}, "
            f"{reasoning_effort=}, {timeout=}"
        )

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
) -> PreparedRuntime:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    return prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        memory_scheduler_clock=_BlockingClock(),
        schedule_scheduler_clock=_BlockingClock(),
        task_framer=DeterministicTaskFramingEvaluator(),
    )


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
    confirmations: list[ConfirmationRequestView] = []
    runtime.control.bind_confirmation_callback(confirmations.append)
    try:
        await runtime.start()
        turn = asyncio.create_task(collect_foreground_outbound(runtime, "Read the file."))
        while not confirmations:
            await asyncio.sleep(0)
        confirmation = confirmations[0]
        runtime.control.respond_to_confirmation(confirmation.confirmation_id, "approved")
        messages = await turn
    finally:
        await runtime.close()

    assert any(message.type == "tool_call" for message in messages)
    assert messages[-1].metadata == {"_streamed": True}
    assert confirmation.details["path"] != str(outside) or len(str(outside)) <= 256
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
async def test_runtime_reads_known_skill_path_without_confirmation(
    agent_home: Path,
    workspace: Path,
) -> None:
    skill_file = agent_home / "skills" / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_bytes(b"---\nname: review\n---\nbody\n")
    provider = _RuntimeProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_skill_read",
                    name="read_file",
                    arguments=json.dumps({"path": str(skill_file)}),
                ),
            ),
            _response(content="Done."),
        )
    )
    runtime = _runtime(agent_home, workspace, provider)
    confirmations: list[ConfirmationRequestView] = []

    def approve(request: ConfirmationRequestView) -> None:
        confirmations.append(request)
        runtime.control.respond_to_confirmation(request.confirmation_id, "approved")

    runtime.control.bind_confirmation_callback(approve)
    try:
        await runtime.start()
        await collect_foreground_outbound(runtime, "Read the skill.")
    finally:
        await runtime.close()

    assert confirmations == []
    tool_messages = [message for message in runtime.session.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "---\nname: review\n---\nbody\n"
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
    await runtime.start()
    turn = asyncio.create_task(collect_foreground_outbound(runtime, "Fetch the URL."))
    try:
        await started.wait()
        await runtime.control.cancel_active_run()
        messages = await asyncio.wait_for(turn, timeout=1)
    finally:
        release.set()
        if not turn.done():
            turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await runtime.close()

    assert cancelled.is_set()
    assert any(message.type == "tool_call" for message in messages)
    assert messages[-1].type == "system_control"
    assert messages[-1].metadata["finish_reason"] == "cancelled"
    tool_messages = [message for message in runtime.session.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["status"] == "error"
    assert tool_messages[0]["content"] == ("Tool call interrupted because the turn was cancelled.")
