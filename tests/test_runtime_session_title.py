import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
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
from myclaw.session.session import Session
from myclaw.tools.base import OpenAIToolSchema
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock, ProviderCall
from tests.runtime_bus import collect_foreground_outbound

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")


class RuntimeTitleProvider:
    def __init__(self) -> None:
        self.requests: list[ProviderCall] = []

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
        self.requests.append(
            ProviderCall(
                messages=list(messages),
                tools=tuple(tools),
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
        )
        if "<long_term_memory>" not in cast(str, messages[0]["content"]):
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
            "Unexpected complete request: "
            f"{messages=}, {tools=}, {model=}, {max_output=}, {temperature=}, "
            f"{reasoning_effort=}, {timeout=}"
        )

    async def close(self) -> None:
        return None


class TitleFirstProvider:
    def __init__(self) -> None:
        self.chat_started = asyncio.Event()
        self.release_chat = asyncio.Event()

    def route_status(self, route: str) -> None:
        """Expose the former structural marker without becoming a Router."""
        raise AssertionError(f"Direct Provider received Model Route {route!r}")

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
        del tools, model, max_output, temperature, reasoning_effort, timeout
        if messages and messages[0] == {
            "role": "system",
            "content": session_title_prompt(),
        }:
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
        raise AssertionError(f"Unexpected complete request: {messages!r}")

    async def close(self) -> None:
        return None


class ExistingSessionProvider:
    def __init__(self) -> None:
        self.requests: list[ProviderCall] = []

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
        self.requests.append(
            ProviderCall(
                messages=list(messages),
                tools=tuple(tools),
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
        )
        content = (
            "Unexpected regenerated title"
            if messages
            and messages[0]
            == {
                "role": "system",
                "content": "Generate a title.",
            }
            else "Continued answer."
        )
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content=content),
                usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                finish_reason="stop",
            )
        )

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
        raise AssertionError(f"Unexpected complete request: {messages!r}")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_title_finishes_before_chat_when_direct_provider_exposes_route_status(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TitleFirstProvider()
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=iter((SESSION_UUID, TURN_UUID)).__next__,
    )
    session = runtime.session
    replacements: list[bytes] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(path: Path, content: bytes) -> None:
        replacements.append(content)
        replace(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replace)
    await runtime.start()
    terminal = asyncio.create_task(collect_foreground_outbound(runtime, "First input."))
    await provider.chat_started.wait()
    for _ in range(100):
        if session.metadata["title"] == "Generated before chat":
            break
        await asyncio.sleep(0)
    await asyncio.sleep(0)

    session_path = workspace / ".myclaw" / "sessions" / f"{session.session_id}.jsonl"
    assert session.metadata["title"] == "Generated before chat"
    assert not session_path.exists()
    assert replacements == []

    provider.release_chat.set()
    messages = await terminal
    assert messages[-1].type == "model_response"
    assert messages[-1].metadata == {"_streamed": True}
    await asyncio.sleep(0)

    assert len(replacements) == 1
    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == "Generated before chat"
    assert [message["role"] for message in reloaded.messages] == ["user", "assistant"]
    await runtime.close()


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
        new_uuid=iter((SESSION_UUID, TURN_UUID)).__next__,
        retry_clock=clock,
    )

    await runtime.start()
    messages = await collect_foreground_outbound(runtime, "  First\t runtime input.  ")
    for _ in range(100):
        if len(provider.requests) == 2 and runtime.session.metadata["title"] == "Runtime project":
            break
        await asyncio.sleep(0)

    assert messages[-1].metadata == {"_streamed": True}
    assert len(provider.requests) == 2
    title_request = next(
        request
        for request in provider.requests
        if "<long_term_memory>" not in cast(str, request.messages[0]["content"])
    )
    assert title_request.tools == ()
    assert title_request.messages == [
        {"role": "system", "content": session_title_prompt()},
        {"role": "user", "content": "First runtime input."},
    ]
    assert "<runtime_context>" not in cast(str, title_request.messages[0]["content"])
    assert "<long_term_memory>" not in cast(str, title_request.messages[0]["content"])
    assert runtime.session.metadata["title"] == "Runtime project"
    assert runtime.session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 11,
        "output_tokens": 3,
        "total_tokens": 14,
    }
    assert [message["role"] for message in runtime.session.messages] == ["user", "assistant"]
    await runtime.close()


@pytest.mark.asyncio
async def test_existing_session_turn_does_not_regenerate_its_title(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW, new_uuid=lambda: SESSION_UUID)
    session.update_metadata(title="Existing title")
    session.add_message("user", "Earlier input.")
    session.add_message(
        "assistant",
        "Earlier answer.",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )
    session.close()
    provider = ExistingSessionProvider()
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=lambda: TURN_UUID,
    )
    await runtime.start()
    result = await runtime.management_dispatcher.resume(session.session_id)
    assert result.output == f"Resumed session {session.session_id}."
    observed = await collect_foreground_outbound(runtime, "Continue this Session.")
    await asyncio.sleep(0)

    assert observed[-1].metadata == {"_streamed": True}
    assert len(provider.requests) == 1
    assert "You are the MyClaw Personal Agent." in cast(
        str, provider.requests[0].messages[0]["content"]
    )
    assert "<tool_guidance>" in cast(str, provider.requests[0].messages[0]["content"])
    assert runtime.session.metadata["title"] == "Existing title"
    assert runtime.session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    await runtime.close()
