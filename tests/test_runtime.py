import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4
from zoneinfo import ZoneInfoNotFoundError

import pytest
from loguru import logger

import myclaw.agent.runtime as runtime_module
from myclaw.agent.blackboard import Blackboard
from myclaw.agent.context import ContextBuilder
from myclaw.agent.loop import ConfirmationRequestView, ForegroundContextPreparer
from myclaw.agent.prompts import (
    conversation_summary_prompt,
    foreground_chat_system_prompt,
    session_title_prompt,
)
from myclaw.agent.runtime import PreparedRuntime, SkillContextTooLargeError, prepare_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, ProviderConfiguration
from myclaw.errors import ErrorInfo
from myclaw.logging.process import configure_process_logging
from myclaw.management.service import RuntimeStatusInput, estimate_input_tokens
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    TextDelta,
)
from myclaw.session.session import Session
from myclaw.skills.catalog import (
    ManualSkillInvocation,
    SkillCatalog,
    SkillMetadata,
    discover_skills,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    DeterministicTaskFramingEvaluator,
    FakeClock,
    ScriptedFakeProvider,
    StreamScript,
    unexpected_provider_factory,
)
from tests.fixtures.diagnostic_capture import capture_diagnostics
from tests.runtime_bus import collect_foreground_outbound

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
TURN_TWO_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")


def _session_log_text(workspace: Path, session_id: str) -> str:
    path = workspace / ".myclaw" / "logs" / f"{session_id}.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


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


class BlockingSessionLogProvider:
    def __init__(self, *, marker: str, release: asyncio.Event) -> None:
        self._marker = marker
        self._release = release
        self.started = asyncio.Event()

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
                    message=AssistantModelMessage(content=f"{self._marker} title"),
                    usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                    finish_reason="stop",
                )
            )
            return
        logger.warning("Concurrent foreground marker={}", self._marker)
        self.started.set()
        await self._release.wait()
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content=f"{self._marker} response"),
                usage=ModelUsage(input_tokens=3, output_tokens=1, total_tokens=4),
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
async def test_runtime_composition_passes_discovered_iana_name_to_context_builder(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    discovered_names: list[str] = []
    catalog_presence: list[bool] = []
    context_builder = ContextBuilder

    def recording_context_builder(
        runtime_workspace: Workspace,
        timezone_name: str,
        *,
        skill_catalog: SkillCatalog | None = None,
    ) -> ContextBuilder:
        discovered_names.append(timezone_name)
        catalog_presence.append(skill_catalog is not None)
        return context_builder(
            runtime_workspace,
            timezone_name,
            skill_catalog=skill_catalog,
        )

    monkeypatch.setattr(runtime_module, "get_localzone_name", lambda: "Asia/Shanghai")
    monkeypatch.setattr(runtime_module, "ContextBuilder", recording_context_builder)

    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=unexpected_provider_factory,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    await runtime.close()

    assert discovered_names == ["Asia/Shanghai", "Asia/Shanghai"]
    assert catalog_presence == [True, True]


@pytest.mark.asyncio
async def test_injected_skill_catalog_root_is_the_only_confirmation_free_skill_root(
    agent_home: Path,
    workspace: Path,
    tmp_path: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    catalog_root = (tmp_path / "catalog-skills").resolve()
    catalog_file = catalog_root / "review" / "SKILL.md"
    catalog_file.parent.mkdir(parents=True)
    catalog_file.write_text("catalog root", encoding="utf-8")
    agent_home_file = home.skills_directory / "legacy" / "SKILL.md"
    agent_home_file.parent.mkdir(parents=True)
    agent_home_file.write_text("agent home root", encoding="utf-8")
    catalog = SkillCatalog(root=catalog_root, entries=())
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="",
                                tool_calls=(
                                    ModelToolCall(
                                        id="call_catalog_root",
                                        name="read_file",
                                        arguments=json.dumps({"path": str(catalog_file)}),
                                    ),
                                    ModelToolCall(
                                        id="call_agent_home_root",
                                        name="read_file",
                                        arguments=json.dumps({"path": str(agent_home_file)}),
                                    ),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Finished reading."),
                            usage=ModelUsage(input_tokens=8, output_tokens=3, total_tokens=11),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        skill_catalog=catalog,
        task_framer=DeterministicTaskFramingEvaluator(),
    )
    runtime.session.update_metadata(title="Existing title")
    requests: list[ConfirmationRequestView] = []

    def approve(request: ConfirmationRequestView) -> None:
        requests.append(request)
        runtime.agent_loop.respond_to_confirmation(request.confirmation_id, "approved")

    runtime.agent_loop.bind_confirmation_callback(approve)
    await runtime.start()
    try:
        outbound = await collect_foreground_outbound(runtime, "Read both Skill roots.")
    finally:
        await runtime.close()

    assert "Finished reading." in "".join(message.content for message in outbound)
    assert [request.details["path"] for request in requests] == [str(agent_home_file)]
    tool_messages = [message for message in runtime.session.messages if message["role"] == "tool"]
    assert [message["content"] for message in tool_messages] == [
        "catalog root",
        "agent home root",
    ]
    assert len(runtime.agent_loop.tool_schemas) == 10


@pytest.mark.asyncio
async def test_foreground_skill_catalog_is_included_in_the_exact_budget_guard(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_text = VALID_CONFIG.replace("context_window = 200000", "context_window = 4096")
    config_text = config_text.replace("max_output = 8192", "max_output = 1024")
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    for index in range(12):
        instruction = agent_home / "skills" / f"skill-{index:02d}" / "SKILL.md"
        instruction.parent.mkdir(parents=True)
        instruction.write_text(
            f"---\nname: skill-{index:02d}\ndescription: {'x' * 1024}\n---\nbody\n",
            encoding="utf-8",
        )
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=unexpected_provider_factory,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    preparer: ForegroundContextPreparer = runtime.agent_loop._context_preparer

    try:
        with pytest.raises(ModelCallError) as raised:
            await preparer(runtime.session, {"role": "user", "content": "Use a Skill."})
    finally:
        await runtime.close()

    assert raised.value.error.code == "memory_context_too_large"
    assert runtime.session.messages == []


def _always_skill_budget_fixture(
    agent_home: Path,
    workspace: Path,
) -> tuple[AgentHome, Workspace, WorkspaceState, SkillCatalog, int]:
    home = AgentHome(agent_home)
    home.initialize()
    instruction = agent_home / "skills" / "always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(
        b"---\nname: always\ndescription: Always loaded\nalways: true\n---\n"
        + b"Always-loaded body for the foreground budget projection.\n"
    )
    runtime_workspace = Workspace.from_path(workspace)
    workspace_state = WorkspaceState(runtime_workspace)
    workspace_state.initialize(agent_home_root=home.path)
    catalog = discover_skills(
        agent_home=home,
        reserved_names=(),
        enable_always_load=True,
    )
    system_prompt = foreground_chat_system_prompt(
        workspace=runtime_workspace.path,
        long_term_memory=workspace_state.long_term_memory_path.read_text(encoding="utf-8"),
        skill_catalog=catalog,
    )
    estimated = estimate_input_tokens(
        RuntimeStatusInput(
            system_prompt=system_prompt,
            retained_messages=(),
            tool_definitions=(),
            runtime_context="",
        )
    )
    return home, runtime_workspace, workspace_state, catalog, estimated


@pytest.mark.asyncio
async def test_always_skill_budget_allows_exact_foreground_projection(
    agent_home: Path,
    workspace: Path,
) -> None:
    home, runtime_workspace, workspace_state, catalog, estimated = _always_skill_budget_fixture(
        agent_home,
        workspace,
    )
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "context_window = 200000",
            f"context_window = {estimated + 8192}",
        ),
        encoding="utf-8",
    )
    configuration = ConfigLoader(home).load()

    prepared = prepare_runtime(
        agent_home=home,
        workspace=runtime_workspace,
        workspace_state=workspace_state,
        configuration=configuration,
        provider_factory=unexpected_provider_factory,
        now=lambda: NOW,
        new_uuid=uuid4,
        skill_catalog=catalog,
    )
    await prepared.close()


@pytest.mark.asyncio
async def test_always_skill_budget_overflow_fails_before_provider_or_agent_loop(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, runtime_workspace, workspace_state, catalog, estimated = _always_skill_budget_fixture(
        agent_home,
        workspace,
    )
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "context_window = 200000",
            f"context_window = {estimated + 8192 - 1}",
        ),
        encoding="utf-8",
    )
    configuration = ConfigLoader(home).load()
    provider_factory_calls: list[ProviderConfiguration] = []
    agent_loop_constructions: list[object] = []

    def provider_factory(provider_configuration: ProviderConfiguration) -> ModelProvider:
        provider_factory_calls.append(provider_configuration)
        raise AssertionError("Provider factory was called before budget preflight")

    def unexpected_agent_loop(*args: object, **kwargs: object) -> None:
        del args, kwargs
        agent_loop_constructions.append(object())

    monkeypatch.setattr(runtime_module, "AgentLoop", unexpected_agent_loop)
    tasks_before = asyncio.all_tasks()
    diagnostics = capture_diagnostics()

    try:
        with pytest.raises(SkillContextTooLargeError) as raised:
            prepare_runtime(
                agent_home=home,
                workspace=runtime_workspace,
                workspace_state=workspace_state,
                configuration=configuration,
                provider_factory=provider_factory,
                now=lambda: NOW,
                new_uuid=uuid4,
                skill_catalog=catalog,
            )
    finally:
        diagnostics.close()

    assert raised.value.error.code == "skill_context_too_large"
    assert provider_factory_calls == []
    assert agent_loop_constructions == []
    assert asyncio.all_tasks() == tasks_before
    assert "Runtime composition failed" not in diagnostics.event_text
    assert "Traceback" not in diagnostics.text


@pytest.mark.asyncio
async def test_conversation_summary_provider_keeps_skill_metadata_out_of_its_prompt(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_text = VALID_CONFIG.replace(
        "consolidation_message_threshold = 50", "consolidation_message_threshold = 4"
    ).replace("[runtime]\n", "[runtime]\nenable_skill_always_load = true\n")
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Private catalog marker\nalways: true\n---\nprivate body\n",
        encoding="utf-8",
    )
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Earlier turns summarized."),
                usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                finish_reason="stop",
            ),
        )
    )
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    runtime.session.add_message("user", "Question one")
    runtime.session.add_message(
        "assistant",
        "Answer one",
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
    runtime.session.add_message("user", "Question two")
    runtime.session.add_message(
        "assistant",
        "Answer two",
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
    preparer: ForegroundContextPreparer = runtime.agent_loop._context_preparer

    try:
        projected = await preparer(
            runtime.session,
            {"role": "user", "content": "Question three"},
        )
    finally:
        await runtime.close()

    assert len(provider.complete_requests) == 1
    summary_messages = provider.complete_requests[0].messages
    assert summary_messages[0]["content"] == conversation_summary_prompt()
    assert "Private catalog marker" not in json.dumps(summary_messages)
    assert "private body" not in json.dumps(summary_messages)
    assert "<skill_catalog>" in cast(str, projected[0]["content"])
    assert "<skill_always_load>" in cast(str, projected[0]["content"])


@pytest.mark.asyncio
async def test_foreground_context_uses_one_staged_blackboard_for_summary_and_chat_projection(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    blackboard = Blackboard(goal="Current task", completion_boundary="Current boundary")
    observed: list[Blackboard | None] = []
    original_projector = runtime_module._project_foreground_messages

    def recording_projector(
        context: ContextBuilder,
        messages: Sequence[dict[str, Any]],
        *,
        session_id: str,
        long_term_memory: str,
        blackboard: Blackboard | None = None,
    ) -> list[dict[str, Any]]:
        observed.append(blackboard)
        return original_projector(
            context,
            messages,
            session_id=session_id,
            long_term_memory=long_term_memory,
            blackboard=blackboard,
        )

    monkeypatch.setattr(runtime_module, "_project_foreground_messages", recording_projector)
    preparer: ForegroundContextPreparer = runtime.agent_loop._context_preparer

    projected_without_blackboard = await preparer(
        runtime.session,
        {"role": "user", "content": "Raw input"},
    )

    assert observed
    assert all(value is None for value in observed)
    assert "<blackboard>" not in projected_without_blackboard[-1]["content"]
    observed.clear()

    projected = await preparer(
        runtime.session,
        {"role": "user", "content": "Raw input"},
        blackboard,
    )

    assert len(observed) >= 2
    assert all(value is blackboard for value in observed)
    assert "<blackboard>" in projected[-1]["content"]
    assert runtime.session.messages == []
    await runtime.close()


@pytest.mark.asyncio
async def test_manual_body_counts_in_foreground_token_budget_but_not_summary_provider(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_text = VALID_CONFIG.replace(
        "consolidation_message_threshold = 50", "consolidation_message_threshold = 100"
    )
    config_text = config_text.replace("context_window = 200000", "context_window = 4096")
    config_text = config_text.replace("max_output = 8192", "max_output = 512")
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: planner\ndescription: Plan work\n---\nmetadata body\n")
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Earlier turns summarized."),
                usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                finish_reason="stop",
            ),
        )
    )
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    runtime.session.add_message("user", "Question one")
    runtime.session.add_message(
        "assistant",
        "Answer one",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    runtime.session.add_message("user", "Question two")
    runtime.session.add_message(
        "assistant",
        "Answer two",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    invocation = ManualSkillInvocation(
        metadata=SkillMetadata(
            name="planner",
            description="Plan work",
            path=instruction.resolve(),
        ),
        request="REQUEST-MARKER",
        body="MANUAL-BODY-MARKER" + ("b" * 9_000),
    )
    preparer: ForegroundContextPreparer = runtime.agent_loop._context_preparer

    try:
        projected = await preparer(
            runtime.session,
            {"role": "user", "content": "/planner REQUEST-MARKER"},
            manual_invocation=invocation,
        )
    finally:
        await runtime.close()

    assert len(provider.complete_requests) == 1
    assert runtime.session.last_consolidated > 0
    summary_payload = json.dumps(provider.complete_requests[0].messages)
    assert "MANUAL-BODY-MARKER" not in summary_payload
    assert "REQUEST-MARKER" not in summary_payload
    current_content = cast(str, projected[-1]["content"])
    assert (
        "MANUAL-BODY-MARKER"
        in json.loads(
            current_content.split("<skill_instructions>\n", 1)[1].split(
                "\n</skill_instructions>", 1
            )[0]
        )["body"]
    )
    assert "REQUEST-MARKER" in json.loads(
        current_content.split("<user_request>\n", 1)[1].split("\n</user_request>", 1)[0]
    )
    assert runtime.session.messages[0]["content"] == "Question one"


@pytest.mark.asyncio
async def test_oversized_manual_body_returns_context_overflow_without_provider_or_commit(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    config_text = VALID_CONFIG.replace(
        "consolidation_message_threshold = 50", "consolidation_message_threshold = 100"
    )
    config_text = config_text.replace("context_window = 200000", "context_window = 4096")
    config_text = config_text.replace("max_output = 8192", "max_output = 512")
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text(
        "---\nname: planner\ndescription: Plan work\n---\n" + ("b" * 20_000),
        encoding="utf-8",
    )
    provider = ScriptedFakeProvider()
    framer = DeterministicTaskFramingEvaluator()
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
        task_framer=framer,
    )
    runtime.session.add_message("user", "Earlier question")
    runtime.session.add_message(
        "assistant",
        "Earlier answer",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={"model_calls": 1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    before_messages = deepcopy(runtime.session.messages)
    before_metadata = deepcopy(runtime.session.metadata)

    try:
        await runtime.start()
        outbound = await collect_foreground_outbound(runtime, "/planner request")
    finally:
        await runtime.close()

    assert outbound[-1].metadata == {
        "finish_reason": "failed",
        "error_code": "model_context_overflow",
        "_streamed": True,
    }
    assert provider.complete_requests == []
    assert provider.stream_requests == []
    assert framer.calls == 1
    assert runtime.session.messages == before_messages
    assert runtime.session.metadata == before_metadata


@pytest.mark.asyncio
async def test_management_command_performs_zero_task_framing_attempts(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    framer = DeterministicTaskFramingEvaluator()
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=unexpected_provider_factory,
        now=lambda: NOW,
        new_uuid=uuid4,
        task_framer=framer,
    )

    result = await runtime.management_dispatcher.dispatch("/status")
    await runtime.close()

    assert result.output is not None
    assert framer.calls == 0


def test_runtime_composition_rejects_invalid_discovered_iana_before_provider_call(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider_calls: list[ProviderConfiguration] = []

    def provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
        provider_calls.append(configuration)
        return ScriptedFakeProvider()

    monkeypatch.setattr(runtime_module, "get_localzone_name", lambda: "Invalid/MyClaw-Zone")

    with pytest.raises(ZoneInfoNotFoundError):
        prepare_runtime(
            agent_home=home,
            workspace=workspace,
            configuration=ConfigLoader(home).load(),
            provider_factory=provider_factory,
            now=lambda: NOW,
            new_uuid=uuid4,
        )

    assert provider_calls == []


@pytest.mark.asyncio
async def test_runtime_leaves_legacy_schedule_state_untouched(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    (workspace / ".myclaw").mkdir()
    legacy_path = (workspace / ".myclaw" / "scheduled-work.json").resolve()
    legacy_path.write_text("[]", encoding="utf-8")
    reads: list[Path] = []
    original_read_text = Path.read_text

    def observe_legacy_reads(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path.resolve() == legacy_path:
            reads.append(path)
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", observe_legacy_reads)
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    try:
        await runtime.start()
        await asyncio.sleep(0)
    finally:
        await runtime.close()

    assert reads == []
    assert legacy_path.read_text(encoding="utf-8") == "[]"


@pytest.mark.parametrize("legacy_kind", ("file", "directory"))
@pytest.mark.asyncio
async def test_runtime_ignores_legacy_schedule_state_path_types(
    agent_home: Path,
    workspace: Path,
    legacy_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    (workspace / ".myclaw").mkdir()
    legacy_path = (workspace / ".myclaw" / "scheduled-work.json").resolve()
    if legacy_kind == "file":
        legacy_path.write_bytes(b"legacy state")
    else:
        legacy_path.mkdir()
    lstat_calls: list[Path] = []
    original_lstat = Path.lstat

    def observe_legacy_lstat(path: Path) -> object:
        if path == legacy_path:
            lstat_calls.append(path)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", observe_legacy_lstat)
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _: ScriptedFakeProvider(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    try:
        await runtime.start()
        await asyncio.sleep(0)
    finally:
        await runtime.close()

    assert lstat_calls == []
    assert legacy_path.is_file() is (legacy_kind == "file")
    assert legacy_path.is_dir() is (legacy_kind == "directory")
    if legacy_kind == "file":
        assert legacy_path.read_bytes() == b"legacy state"


@pytest.mark.asyncio
async def test_prepared_runtime_correlates_foreground_and_title_work_with_its_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    legacy_files = {
        "run.log.0": b"legacy slot zero\n",
        "run.log.1": b"legacy slot one\n",
        "run.log.cursor": b"1\n",
        "run.log.lock": b"legacy lock\n",
    }
    legacy_logs = agent_home / "logs"
    legacy_logs.mkdir()
    for name, legacy_content in legacy_files.items():
        (legacy_logs / name).write_bytes(legacy_content)
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    clock = FakeClock(NOW)
    failure = ModelCallError(ErrorInfo(code="model_failed", message="The model request failed."))
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(), error=failure),))
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID)).__next__,
        retry_clock=clock,
        task_framer=DeterministicTaskFramingEvaluator(),
    )

    private_input = " ".join(("private", "foreground", "input"))
    await runtime.start()
    messages = await collect_foreground_outbound(runtime, private_input)
    await runtime.close()

    assert [(message.type, message.metadata.get("finish_reason")) for message in messages] == [
        ("system_control", "failed"),
    ]
    content = _session_log_text(workspace, runtime.session_id)
    records = [
        line
        for line in content.splitlines()
        if "myclaw.agent.loop:" in line or "myclaw.agent.runner:" in line
    ]
    assert len(records) == 2
    assert "ModelCallError: The model request failed." in content
    assert "ModelCallError: No title response was scripted." in content
    assert private_input not in repr(messages)
    assert {name: (legacy_logs / name).read_bytes() for name in legacy_files} == legacy_files


@pytest.mark.asyncio
async def test_concurrent_foreground_sessions_write_only_to_their_own_session_logs(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    release = asyncio.Event()
    first_provider = BlockingSessionLogProvider(marker="FIRST_SESSION", release=release)
    second_provider = BlockingSessionLogProvider(marker="SECOND_SESSION", release=release)

    def runtime_for(provider: BlockingSessionLogProvider) -> PreparedRuntime:
        return prepare_runtime(
            agent_home=home,
            workspace=workspace,
            configuration=configuration,
            provider_factory=lambda _: provider,
            now=FakeClock(NOW).now,
            new_uuid=uuid4,
            task_framer=DeterministicTaskFramingEvaluator(),
        )

    first_runtime = runtime_for(first_provider)
    second_runtime = runtime_for(second_provider)
    await asyncio.gather(first_runtime.start(), second_runtime.start())
    first_submit = asyncio.create_task(
        collect_foreground_outbound(first_runtime, "First Session request.")
    )
    second_submit = asyncio.create_task(
        collect_foreground_outbound(second_runtime, "Second Session request.")
    )
    await asyncio.wait_for(
        asyncio.gather(first_provider.started.wait(), second_provider.started.wait()),
        timeout=3,
    )
    release.set()

    first_messages, second_messages = await asyncio.gather(first_submit, second_submit)
    await asyncio.gather(first_runtime.close(), second_runtime.close())

    assert first_messages[-1].type == "model_response"
    assert first_messages[-1].metadata == {"_streamed": True}
    assert second_messages[-1].type == "model_response"
    assert second_messages[-1].metadata == {"_streamed": True}
    assert "FIRST_SESSION response" in "".join(message.content for message in first_messages)
    assert "SECOND_SESSION response" in "".join(message.content for message in second_messages)
    first_log = _session_log_text(workspace, first_runtime.session_id)
    second_log = _session_log_text(workspace, second_runtime.session_id)
    assert "marker=FIRST_SESSION" in first_log
    assert "marker=SECOND_SESSION" not in first_log
    assert "marker=SECOND_SESSION" in second_log
    assert "marker=FIRST_SESSION" not in second_log


@pytest.mark.asyncio
async def test_unavailable_session_log_preserves_events_session_and_tool_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    unavailable_workspace = workspace / "unavailable"
    unavailable_workspace.mkdir()

    def provider() -> ScriptedFakeProvider:
        return ScriptedFakeProvider(
            streams=(
                StreamScript(
                    events=(
                        ModelCompleted(
                            response=ModelResponse(
                                message=AssistantModelMessage(
                                    content="",
                                    tool_calls=(
                                        ModelToolCall(
                                            id="call_unavailable",
                                            name="unavailable_tool",
                                            arguments="{}",
                                        ),
                                    ),
                                ),
                                usage=ModelUsage(
                                    input_tokens=4,
                                    output_tokens=2,
                                    total_tokens=6,
                                ),
                                finish_reason="tool_calls",
                            )
                        ),
                    )
                ),
                StreamScript(
                    events=(),
                    error=ModelCallError(
                        ErrorInfo(code="model_failed", message="The model request failed.")
                    ),
                ),
            )
        )

    unavailable_runtime = prepare_runtime(
        agent_home=home,
        workspace=unavailable_workspace,
        configuration=configuration,
        provider_factory=lambda _: provider(),
        now=FakeClock(NOW).now,
        new_uuid=iter((SESSION_UUID, TURN_UUID)).__next__,
    )
    unavailable_logs = unavailable_workspace / ".myclaw" / "logs"
    unavailable_logs.write_text("Session Log unavailable", encoding="utf-8")

    await unavailable_runtime.start()
    unavailable_messages = await collect_foreground_outbound(
        unavailable_runtime,
        "Fail-open request.",
    )
    await unavailable_runtime.close()
    unavailable_session = unavailable_runtime.session

    assert any(message.type == "tool_call" for message in unavailable_messages)
    terminal = unavailable_messages[-1]
    assert terminal.type == "system_control"
    assert terminal.metadata["finish_reason"] == "failed"
    assert terminal.content == "The model request failed."
    assert [message["role"] for message in unavailable_session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [
        message["status"]
        for message in unavailable_session.messages
        if message["role"] in {"assistant", "tool"}
    ] == ["completed", "error", "error"]
    assert unavailable_session.metadata["title"] == "Fail-open request."
    assert unavailable_session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    assert unavailable_logs.is_file()
    assert _session_log_text(unavailable_workspace, unavailable_runtime.session_id) == ""


@pytest.mark.asyncio
async def test_foreground_tool_diagnostics_preserve_boundary_exception_details(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    clock = FakeClock(NOW)
    private_query = "PRIVATE_CONVERSATION_QUERY"
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="",
                                tool_calls=(
                                    ModelToolCall(
                                        id="call_private_search",
                                        name="web_search",
                                        arguments=json.dumps({"query": private_query}),
                                    ),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Search failed safely."),
                            usage=ModelUsage(input_tokens=8, output_tokens=3, total_tokens=11),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )

    class FailingDDGS:
        def __enter__(self) -> "FailingDDGS":
            return self

        def __exit__(self, *errors: object) -> None:
            del errors

        def text(self, query: str, **arguments: object) -> list[dict[str, object]]:
            del arguments
            raise ExceptionGroup(
                "RAW_PROVIDER_BODY",
                [OSError(f"query={query}"), ValueError("auth=PRIVATE_WEB_CREDENTIAL")],
            )

    monkeypatch.setattr("myclaw.tools.core.web_search.DDGS", FailingDDGS)
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=uuid4,
        retry_clock=clock,
    )

    await runtime.start()
    messages = await collect_foreground_outbound(runtime, private_query)
    await runtime.close()

    tool_call = next(message for message in messages if message.type == "tool_call")
    assert tool_call.metadata["arguments"] == json.dumps({"query": private_query})
    assert all(private_query not in message.content for message in messages)
    assert all("PRIVATE_WEB_CREDENTIAL" not in message.content for message in messages)
    content = _session_log_text(workspace, runtime.session_id)
    assert content.count("Tool execution failed name=web_search") == 1
    assert "Traceback (most recent call last):" in content
    assert content.count("RAW_PROVIDER_BODY") >= 1
    assert content.count(f"OSError: query={private_query}") == 1
    assert content.count("ValueError: auth=PRIVATE_WEB_CREDENTIAL") == 1


@pytest.mark.asyncio
async def test_foreground_model_failure_keeps_event_safe_without_log_redaction(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    clock = FakeClock(NOW)
    private_input = "PRIVATE_FOREGROUND_PROMPT"
    provider_failure = ModelCallError(
        ErrorInfo(code="model_failed", message="The model request failed.")
    )
    provider_failure.__cause__ = RuntimeError("RAW_PROVIDER_BODY auth=PRIVATE_MODEL_CREDENTIAL")
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(), error=provider_failure),))
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=uuid4,
        retry_clock=clock,
    )

    configure_process_logging()
    try:
        await runtime.start()
        messages = await collect_foreground_outbound(runtime, private_input)
        await runtime.close()
        terminal_output = capsys.readouterr().err
    finally:
        logger.remove()

    assert messages[-1].type == "system_control"
    assert messages[-1].metadata["finish_reason"] == "failed"
    assert messages[-1].content == "The model request failed."
    content = _session_log_text(workspace, runtime.session_id)
    assert content.count("Agent Run failed code=model_failed type=ModelCallError") == 1
    assert "Traceback (most recent call last):" in content
    assert "ModelCallError: The model request failed." in content
    assert private_input not in content
    assert "RAW_PROVIDER_BODY auth=PRIVATE_MODEL_CREDENTIAL" in content
    assert terminal_output == ""


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

    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=provider_factory,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID)).__next__,
    )

    assert factory_calls == []
    session_path = workspace / ".myclaw" / "sessions" / f"{runtime.session_id}.jsonl"
    assert not session_path.exists()

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
    assert [message["role"] for message in runtime.session.messages] == [
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

    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=provider_factory,
        now=FakeClock(NOW).now,
        new_uuid=iter((SESSION_UUID, TURN_UUID)).__next__,
    )

    await runtime.run(
        input_reader=ScriptedInput(("Use the chat route.", None)),
        writer=RecordingWriter(),
    )

    assert [call.provider_id for call in factory_calls] == ["chat-provider"]
    request = provider.stream_requests[0]
    assert (
        request.model,
        request.max_output,
        request.temperature,
        request.reasoning_effort,
        request.timeout,
    ) == ("chat-model", 4096, 0.1, "high", 90)


@pytest.mark.asyncio
async def test_default_task_framer_uses_model_router_retry_before_one_foreground_run(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(CHAT_ROUTE_CONFIG, encoding="utf-8")
    clock = FakeClock(NOW)
    retryable_failure = ModelCallError(
        ErrorInfo(
            code="provider_timeout",
            message="The provider timed out.",
            retryable=True,
        )
    )
    framing_response = ModelResponse(
        message=AssistantModelMessage(
            content=(
                '{"action":"replace","goal":"Retried framing goal",'
                '"completion_boundary":"Retried framing boundary"}'
            )
        ),
        usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
        finish_reason="stop",
    )
    main_response = ModelResponse(
        message=AssistantModelMessage(content="Retried framing main response."),
        usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        completions=(retryable_failure, framing_response),
        streams=(StreamScript(events=(ModelCompleted(response=main_response),)),),
    )
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=uuid4,
        retry_clock=clock,
    )
    runtime.session.add_message("user", "Existing turn prevents title work.")

    await runtime.start()
    messages = await collect_foreground_outbound(runtime, "Retry framing once.")
    await runtime.close()

    assert messages[-1].metadata == {"_streamed": True}
    assert len(provider.complete_requests) == 2
    assert clock.sleeps == [0.5]
    assert [request.model for request in provider.complete_requests] == [
        "chat-model",
        "chat-model",
    ]
    assert all(request.tools == () for request in provider.complete_requests)
    assert all(request.continuation is None for request in provider.complete_requests)
    assert provider.complete_requests[0].messages == provider.complete_requests[1].messages
    assert json.loads(cast(str, provider.complete_requests[0].messages[1]["content"])) == {
        "previous_blackboard": None,
        "last_assistant_content": "",
        "current_user_input": "Retry framing once.",
    }
    assert len(provider.stream_requests) == 1
    assert runtime.session.metadata["blackboard"] == {
        "goal": "Retried framing goal",
        "completion_boundary": "Retried framing boundary",
    }
    assert runtime.session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 9,
        "output_tokens": 4,
        "total_tokens": 13,
    }


@pytest.mark.asyncio
async def test_default_task_framer_uses_model_router_default_fallback(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(CHAT_ROUTE_CONFIG, encoding="utf-8")
    fallback_failure = ModelCallError(
        ErrorInfo(
            code="provider_auth_error",
            message="The configured provider rejected authentication.",
        )
    )
    framing_response = ModelResponse(
        message=AssistantModelMessage(
            content=(
                '{"action":"replace","goal":"Fallback framing goal",'
                '"completion_boundary":"Fallback framing boundary"}'
            )
        ),
        usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
        finish_reason="stop",
    )
    main_response = ModelResponse(
        message=AssistantModelMessage(content="Fallback framing main response."),
        usage=ModelUsage(input_tokens=3, output_tokens=1, total_tokens=4),
        finish_reason="stop",
    )
    chat_provider = ScriptedFakeProvider(
        completions=(fallback_failure,),
        streams=(StreamScript(events=(ModelCompleted(response=main_response),)),),
    )
    default_provider = ScriptedFakeProvider(completions=(framing_response,))
    provider_ids: list[str] = []

    def provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
        provider_ids.append(configuration.provider_id)
        if configuration.provider_id == "chat-provider":
            return chat_provider
        return default_provider

    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=provider_factory,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    runtime.session.add_message("user", "Existing turn prevents title work.")

    await runtime.start()
    messages = await collect_foreground_outbound(runtime, "Fallback framing once.")
    await runtime.close()

    assert messages[-1].metadata == {"_streamed": True}
    assert provider_ids == ["chat-provider", "anthropic-default"]
    assert len(chat_provider.complete_requests) == 1
    assert len(default_provider.complete_requests) == 1
    framing_requests = [
        chat_provider.complete_requests[0],
        default_provider.complete_requests[0],
    ]
    assert all(request.tools == () for request in framing_requests)
    assert all(request.continuation is None for request in framing_requests)
    assert framing_requests[0].messages == framing_requests[1].messages
    assert json.loads(cast(str, framing_requests[0].messages[1]["content"])) == {
        "previous_blackboard": None,
        "last_assistant_content": "",
        "current_user_input": "Fallback framing once.",
    }
    assert len(chat_provider.stream_requests) == 1
    assert default_provider.stream_requests == []
    assert runtime.session.metadata["blackboard"] == {
        "goal": "Fallback framing goal",
        "completion_boundary": "Fallback framing boundary",
    }
    assert runtime.session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 9,
        "output_tokens": 3,
        "total_tokens": 12,
    }


@pytest.mark.asyncio
async def test_prepared_repl_routes_transient_provider_failures_through_one_retry_budget(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    clock = FakeClock(NOW)
    transient_failure = ModelCallError(
        ErrorInfo(
            code="provider_timeout",
            message="The provider timed out.",
            retryable=True,
        )
    )
    transient_failure.__cause__ = RuntimeError(
        "RAW_RETRY_PROVIDER_BODY auth=PRIVATE_RETRY_CREDENTIAL"
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(),
                error=transient_failure,
            ),
            StreamScript(
                events=(
                    TextDelta(delta="Recovered response."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Recovered response."),
                            usage=ModelUsage(input_tokens=5, output_tokens=2, total_tokens=7),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID)).__next__,
        retry_clock=clock,
    )
    writer = RecordingWriter()

    await runtime.run(
        input_reader=ScriptedInput(("Retry this turn.", None)),
        writer=writer,
    )

    assert len(provider.stream_requests) == 2
    assert clock.sleeps == [0.5]
    assert writer.operations == [("delta", "Recovered response."), ("finish", "")]
    content = _session_log_text(workspace, runtime.session_id)
    assert content.count("Provider attempt failed; retrying attempt=1/5") == 1
    assert "ModelCallError: The provider timed out." in content
    assert "RuntimeError: RAW_RETRY_PROVIDER_BODY auth=PRIVATE_RETRY_CREDENTIAL" in content


@pytest.mark.asyncio
async def test_prepared_repl_status_reports_the_actual_fallback_route_and_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(CHAT_ROUTE_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    clock = FakeClock(NOW)
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(),
                error=ModelCallError(
                    ErrorInfo(
                        code="provider_auth_error",
                        message="The configured chat provider rejected authentication.",
                        retryable=False,
                    )
                ),
            ),
        )
    )
    fallback_usage = ModelUsage(input_tokens=9, output_tokens=3, total_tokens=12)
    default_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    TextDelta(delta="Fallback response."),
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Fallback response."),
                            usage=fallback_usage,
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    factory_calls: list[str] = []

    def provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
        factory_calls.append(configuration.provider_id)
        if configuration.provider_id == "chat-provider":
            return chat_provider
        return default_provider

    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=provider_factory,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID)).__next__,
        retry_clock=clock,
    )
    writer = RecordingWriter()

    await runtime.run(
        input_reader=ScriptedInput(("Use fallback.", "/status", None)),
        writer=writer,
    )

    assert factory_calls == ["chat-provider", "anthropic-default"]
    assert writer.operations[:2] == [("delta", "Fallback response."), ("finish", "")]
    operation, rendered_status = writer.operations[2]
    assert operation == "line"
    status = json.loads(rendered_status)
    assert status["version"] == "0.1.0"
    assert status["chat_model"] == "anthropic-default/claude-model"
    assert status["context_window"] == 200000
    assert status["session_message_count"] == 2
    assert status["last_consolidated"] == 0
    assert status["cumulative_usage"] == {
        "model_calls": 1,
        "input_tokens": 9,
        "output_tokens": 3,
        "total_tokens": 12,
    }
    assert status["estimated_input_tokens"] > 0
    assert isinstance(status["uptime_seconds"], int)
    assert len(default_provider.stream_requests) == 1


@pytest.mark.asyncio
async def test_runtime_status_estimate_omits_a_pure_error_assistant(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    first = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=unexpected_provider_factory,
        now=FakeClock(NOW).now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    second = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=unexpected_provider_factory,
        now=FakeClock(NOW).now,
        new_uuid=iter((TURN_TWO_UUID,)).__next__,
    )
    first.session.add_message("user", "Keep the next context stable.")
    second.session.add_message("user", "Keep the next context stable.")
    second.session.add_message(
        "assistant",
        "",
        tool_calls=[],
        status="error",
        error={"code": "model_failed", "message": "Safe final failure."},
        token_usage={"model_calls": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    first_writer = RecordingWriter()
    second_writer = RecordingWriter()

    await first.run(input_reader=ScriptedInput(("/status", None)), writer=first_writer)
    await second.run(input_reader=ScriptedInput(("/status", None)), writer=second_writer)

    first_status = json.loads(first_writer.operations[0][1])
    second_status = json.loads(second_writer.operations[0][1])
    assert first_status["session_message_count"] == 1
    assert second_status["session_message_count"] == 2
    assert first_status["estimated_input_tokens"] == second_status["estimated_input_tokens"]


@pytest.mark.asyncio
async def test_runtime_status_estimate_includes_the_interrupted_history_marker(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    interrupted = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=unexpected_provider_factory,
        now=FakeClock(NOW).now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    projected = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=unexpected_provider_factory,
        now=FakeClock(NOW).now,
        new_uuid=iter((TURN_TWO_UUID,)).__next__,
    )
    interrupted.session.add_message("user", "Keep the interrupted context stable.")
    projected.session.add_message("user", "Keep the interrupted context stable.")
    interrupted.session.add_message(
        "assistant",
        "Partial first turn.",
        tool_calls=[],
        status="interrupted",
        error={"code": "turn_cancelled", "message": "Turn interrupted by user."},
        token_usage={"model_calls": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    projected.session.add_message(
        "assistant",
        "Partial first turn.\n\n[Turn interrupted by user.]",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage={"model_calls": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    interrupted_writer = RecordingWriter()
    projected_writer = RecordingWriter()

    await interrupted.run(
        input_reader=ScriptedInput(("/status", None)),
        writer=interrupted_writer,
    )
    await projected.run(
        input_reader=ScriptedInput(("/status", None)),
        writer=projected_writer,
    )

    interrupted_status = json.loads(interrupted_writer.operations[0][1])
    projected_status = json.loads(projected_writer.operations[0][1])
    assert interrupted_status["session_message_count"] == 2
    assert projected_status["session_message_count"] == 2
    assert (
        interrupted_status["estimated_input_tokens"] == projected_status["estimated_input_tokens"]
    )


@pytest.mark.asyncio
async def test_prepared_repl_defers_an_unusable_default_until_route_use(
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

    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=unexpected_provider_factory,
        now=FakeClock(NOW).now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )

    await runtime.close()


@pytest.mark.asyncio
async def test_prepared_repl_uses_the_effective_fallback_route_budget(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    content = CHAT_ROUTE_CONFIG.replace(
        'protocol = "openai-compatible"',
        'protocol = "future-protocol"',
        1,
    ).replace(
        "context_window = 100000\nmax_output = 4096",
        "context_window = 1024\nmax_output = 1023",
    )
    (agent_home / "config.toml").write_text(content, encoding="utf-8")
    response = ModelResponse(
        message=AssistantModelMessage(content="Fallback budget used."),
        usage=ModelUsage(input_tokens=3, output_tokens=3, total_tokens=6),
        finish_reason="stop",
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(ModelCompleted(response=response),)),
            StreamScript(events=(ModelCompleted(response=response),)),
        )
    )
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=FakeClock(NOW).now,
        new_uuid=uuid4,
    )

    await runtime.start()
    messages = await collect_foreground_outbound(runtime, "Use the fallback budget.")
    await runtime.close()

    assert messages[-1].type == "model_response"
    assert messages[-1].metadata == {"_streamed": True}
    assert provider.stream_requests
    for request in provider.stream_requests:
        assert request.model == "claude-model"


@pytest.mark.asyncio
async def test_prepared_repl_reuses_one_session_and_its_startup_system_context(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    startup_memory = "# Durable Memory\n\nRemember the startup snapshot exactly.\n"
    state.long_term_memory_path.write_text(startup_memory, encoding="utf-8")
    legacy_memory = b"# Legacy Agent Home Memory\n"
    (agent_home / "memory").mkdir()
    (agent_home / "memory" / "memory.md").write_bytes(legacy_memory)
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
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID, TURN_TWO_UUID)).__next__,
    )
    state.long_term_memory_path.write_text(
        "# Changed after startup\n",
        encoding="utf-8",
    )

    await runtime.run(
        input_reader=ScriptedInput(("First raw input.", "Second raw input.", None)),
        writer=RecordingWriter(),
    )

    requests = provider.stream_requests
    assert len(requests) == 2
    assert requests[1].messages[1:] == [
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
    system_prompt = cast(str, requests[0].messages[0]["content"])
    assert requests[1].messages[0]["content"] == system_prompt
    workspace_identity = f"Workspace: {workspace.absolute()}"
    memory_block = f"<long_term_memory>\n{startup_memory}</long_term_memory>"
    assert "MyClaw Personal Agent" in system_prompt
    assert workspace_identity in system_prompt
    assert memory_block in system_prompt
    assert "<tool_guidance>\n- read_file:" in system_prompt
    assert "- list_dir:" in system_prompt
    assert "- glob:" in system_prompt
    assert "- grep:" in system_prompt
    assert system_prompt.index(workspace_identity) < system_prompt.index(memory_block)
    assert system_prompt.index(memory_block) < system_prompt.index("<tool_guidance>")
    assert "Changed after startup" not in system_prompt
    assert "Legacy Agent Home Memory" not in system_prompt
    assert (agent_home / "memory" / "memory.md").read_bytes() == legacy_memory
    reloaded = Session.load(runtime.session.workspace_state, runtime.session_id)
    assert reloaded.session_id == runtime.session_id
    assert [message["role"] for message in reloaded.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
