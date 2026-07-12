from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent_home import AgentHome
from myclaw.config import ConfigLoader, ProviderConfiguration
from myclaw.contracts import (
    AssistantModelMessage,
    ModelCompleted,
    ModelProvider,
    ModelResponse,
    ModelUsage,
    TextDelta,
)
from myclaw.runtime import (
    ProviderAdapterUnavailable,
    prepare_repl_runtime,
    unavailable_provider_factory,
)
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript
from tests.test_config import VALID_CONFIG

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
USER_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
REQUEST_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
ASSISTANT_UUID = UUID("a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34")


class ScriptedInput:
    def __init__(self, values: Iterable[str | None]) -> None:
        self._values = deque(values)

    async def read(self) -> str | None:
        return self._values.popleft()


class RecordingWriter:
    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []

    async def write_delta(self, delta: str) -> None:
        self.operations.append(("delta", delta))

    async def finish_turn(self) -> None:
        self.operations.append(("finish", ""))

    async def write_line(self, content: str) -> None:
        self.operations.append(("line", content))


def test_production_provider_factory_fails_closed_until_adapters_are_installed() -> None:
    configuration = ProviderConfiguration(
        provider_id="anthropic-default",
        protocol="anthropic",
        base_url="https://api.anthropic.com",
        api_key="secret",
        models=("test-model",),
    )

    with pytest.raises(ProviderAdapterUnavailable, match="not available"):
        unavailable_provider_factory(configuration)


@pytest.mark.asyncio
async def test_prepared_repl_defers_injected_provider_factory_until_first_nonblank_input(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    clock = FakeClock(NOW)
    response = ModelResponse(
        message=AssistantModelMessage(content="Hello from the configured provider."),
        usage=ModelUsage(input_tokens=8, output_tokens=6, total_tokens=14),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        streams=[
            StreamScript(
                events=(
                    TextDelta(delta="Hello from "),
                    TextDelta(delta="the configured provider."),
                    ModelCompleted(response=response),
                )
            )
        ]
    )
    factory_calls: list[ProviderConfiguration] = []

    def provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
        factory_calls.append(configuration)
        return provider

    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=provider_factory,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )

    assert factory_calls == []
    assert not runtime.sessions.path_for(runtime.session_id).parent.exists()

    writer = RecordingWriter()
    await runtime.run(
        input_reader=ScriptedInput(("   ", "Hello", None)),
        writer=writer,
    )

    assert [call.provider_id for call in factory_calls] == ["anthropic-default"]
    assert writer.operations == [
        ("delta", "Hello from "),
        ("delta", "the configured provider."),
        ("finish", ""),
    ]
    assert [
        message.role for message in (await runtime.sessions.load(runtime.session_id)).messages
    ] == [
        "user",
        "assistant",
    ]
