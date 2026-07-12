from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent_home import AgentHome
from myclaw.config import ConfigError, ConfigLoader, ProviderConfiguration
from myclaw.contracts import (
    AssistantModelMessage,
    ModelCompleted,
    ModelProvider,
    ModelRequest,
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
TURN_TWO_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
USER_TWO_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")
REQUEST_TWO_UUID = UUID("886313e1-3b8a-4a2d-9f7f-77611a4b6f4e")
ASSISTANT_TWO_UUID = UUID("b3f37212-6f3a-4a1b-8d2e-78ab3f9c4567")

CHAT_ROUTE_CONFIG = (
    VALID_CONFIG
    + """

[models.providers.chat-provider]
protocol = "openai-compatible"
base_url = "https://chat.example/v1"
api_key = "chat-secret"
models = ["chat-model"]

[models.routes.chat]
provider_id = "chat-provider"
model = "chat-model"
context_window = 100000
max_output = 4096
temperature = 0.1
reasoning_effort = "high"
timeout = 90
"""
)


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


@pytest.mark.asyncio
async def test_prepared_repl_uses_the_chat_model_route(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(CHAT_ROUTE_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Routed response."),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
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
        now=FakeClock(NOW).now,
        new_uuid=iter((SESSION_UUID, TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID)).__next__,
    )

    await runtime.run(
        input_reader=ScriptedInput(("Use the chat route.", None)),
        writer=RecordingWriter(),
    )

    assert [call.provider_id for call in factory_calls] == ["chat-provider"]
    request = provider.stream_requests[0]
    assert isinstance(request, ModelRequest)
    assert (
        request.route,
        request.model,
        request.max_output,
        request.temperature,
        request.reasoning_effort,
        request.timeout_seconds,
    ) == ("chat", "chat-model", 4096, 0.1, "high", 90)


def test_prepared_repl_rejects_an_unusable_default_even_when_chat_is_usable(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    content = CHAT_ROUTE_CONFIG.replace(
        'protocol = "anthropic"',
        'protocol = "future-protocol"',
        1,
    )
    (agent_home / "config.toml").write_text(content, encoding="utf-8")
    configuration = ConfigLoader(home).load()

    with pytest.raises(ConfigError) as raised:
        prepare_repl_runtime(
            agent_home=home,
            workspace=workspace,
            configuration=configuration,
            provider_factory=unavailable_provider_factory,
            now=FakeClock(NOW).now,
            new_uuid=iter((SESSION_UUID,)).__next__,
        )

    assert raised.value.error.code == "route_unavailable"
    assert "chat-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_prepared_repl_reuses_one_session_and_its_startup_system_context(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    startup_memory = "# Durable Memory\n\nRemember the startup snapshot exactly.\n"
    (agent_home / "memory" / "memory.md").write_text(startup_memory, encoding="utf-8")
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    clock = FakeClock(NOW)
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="First answer."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="First answer."),
                            usage=ModelUsage(input_tokens=10, output_tokens=3, total_tokens=13),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    TextDelta(delta="Second answer."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Second answer."),
                            usage=ModelUsage(input_tokens=18, output_tokens=3, total_tokens=21),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=iter(
            (
                SESSION_UUID,
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TURN_TWO_UUID,
                USER_TWO_UUID,
                REQUEST_TWO_UUID,
                ASSISTANT_TWO_UUID,
            )
        ).__next__,
    )
    (agent_home / "memory" / "memory.md").write_text(
        "# Changed after startup\n",
        encoding="utf-8",
    )

    await runtime.run(
        input_reader=ScriptedInput(("First raw input.", "Second raw input.", None)),
        writer=RecordingWriter(),
    )

    requests = [
        request for request in provider.stream_requests if isinstance(request, ModelRequest)
    ]
    assert len(requests) == 2
    assert [message.to_dict() for message in requests[1].messages] == [
        {"role": "user", "content": "First raw input."},
        {"role": "assistant", "content": "First answer.", "tool_calls": []},
        {
            "role": "user",
            "content": (
                "<runtime_context>\n"
                "current_time: 2026-07-11T15:30:12.123+08:00\n"
                f"session_id: {runtime.session_id}\n"
                "</runtime_context>\n\n"
                "<user_input>\n"
                "Second raw input.\n"
                "</user_input>"
            ),
        },
    ]
    system_prompt = requests[0].system_prompt
    assert requests[1].system_prompt == system_prompt
    workspace_identity = f"Workspace: {workspace.absolute()}"
    memory_block = f"<long_term_memory>\n{startup_memory}</long_term_memory>"
    assert "MyClaw Personal Agent" in system_prompt
    assert workspace_identity in system_prompt
    assert memory_block in system_prompt
    assert "<tool_guidance>\n</tool_guidance>" in system_prompt
    assert system_prompt.index(workspace_identity) < system_prompt.index(memory_block)
    assert system_prompt.index(memory_block) < system_prompt.index("<tool_guidance>")
    assert "Changed after startup" not in system_prompt
    reloaded = await runtime.sessions.load(runtime.session_id)
    assert reloaded.metadata.id == runtime.session_id
    assert [message.role for message in reloaded.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
