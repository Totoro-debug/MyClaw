import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.contracts import (
    AssistantModelMessage,
    CumulativeUsage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
USER_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
REQUEST_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
ASSISTANT_UUID = UUID("a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34")


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
        new_uuid=iter((SESSION_UUID, TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
        retry_clock=clock,
    )

    event_types = [
        event.type async for event in runtime.conversation.submit("  First\t runtime input.  ")
    ]
    for _ in range(100):
        reloaded = await runtime.sessions.load(runtime.session_id)
        if len(provider.requests) == 2 and reloaded.metadata.title == "Runtime project":
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
    assert reloaded.metadata.title == "Runtime project"
    assert reloaded.metadata.cumulative_usage == CumulativeUsage(
        model_calls=2,
        input_tokens=11,
        output_tokens=3,
        total_tokens=14,
    )
    assert [message.role for message in reloaded.messages] == ["user", "assistant"]
