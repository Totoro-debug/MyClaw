import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4
from zoneinfo import ZoneInfoNotFoundError

import pytest
from loguru import logger

import myclaw.agent.runtime as runtime_module
from myclaw.agent.context import ContextBuilder
from myclaw.agent.events import ConversationPort, ToolCompletedPayload, TurnFailedPayload
from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import PreparedReplRuntime, prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, ProviderConfiguration
from myclaw.errors import ErrorInfo
from myclaw.logging.process import configure_process_logging
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.session.session import Session
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    FakeClock,
    ScriptedFakeProvider,
    StreamScript,
    unexpected_provider_factory,
)

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
REQUEST_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
FINAL_REQUEST_UUID = UUID("a8098c1a-f86e-4f33-8a28-25f602f8e603")
TURN_TWO_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
REQUEST_TWO_UUID = UUID("886313e1-3b8a-4a2d-9f7f-77611a4b6f4e")


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

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if request.system_prompt == session_title_prompt():
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

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(f"Unexpected complete request: {request!r}")

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
    context_builder = ContextBuilder

    def recording_context_builder(
        runtime_workspace: Workspace,
        timezone_name: str,
    ) -> ContextBuilder:
        discovered_names.append(timezone_name)
        return context_builder(runtime_workspace, timezone_name)

    monkeypatch.setattr(runtime_module, "get_localzone_name", lambda: "Asia/Shanghai")
    monkeypatch.setattr(runtime_module, "ContextBuilder", recording_context_builder)

    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=unexpected_provider_factory,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    await runtime.close()

    assert discovered_names == ["Asia/Shanghai"]


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
        prepare_repl_runtime(
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
    runtime = prepare_repl_runtime(
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
    runtime = prepare_repl_runtime(
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
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID, REQUEST_UUID)).__next__,
        retry_clock=clock,
    )

    private_input = " ".join(("private", "foreground", "input"))
    events = [event async for event in runtime.conversation.submit(private_input)]
    await runtime.close()

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    content = _session_log_text(workspace, runtime.session_id)
    records = [line for line in content.splitlines() if "myclaw.session.conversation:" in line]
    assert len(records) == 2
    assert "ModelCallError: The model request failed." in content
    assert "ModelCallError: No title response was scripted." in content
    assert private_input not in repr(events)
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

    def runtime_for(provider: BlockingSessionLogProvider) -> PreparedReplRuntime:
        return prepare_repl_runtime(
            agent_home=home,
            workspace=workspace,
            configuration=configuration,
            provider_factory=lambda _: provider,
            now=FakeClock(NOW).now,
            new_uuid=uuid4,
        )

    first_runtime = runtime_for(first_provider)
    second_runtime = runtime_for(second_provider)
    first_submit = asyncio.create_task(
        _collect_event_types(first_runtime.conversation, "First Session request.")
    )
    second_submit = asyncio.create_task(
        _collect_event_types(second_runtime.conversation, "Second Session request.")
    )
    await asyncio.wait_for(
        asyncio.gather(first_provider.started.wait(), second_provider.started.wait()),
        timeout=3,
    )
    release.set()

    first_events, second_events = await asyncio.gather(first_submit, second_submit)
    await asyncio.gather(first_runtime.close(), second_runtime.close())

    assert first_events == ["turn_started", "model_call_completed", "turn_completed"]
    assert second_events == ["turn_started", "model_call_completed", "turn_completed"]
    first_log = _session_log_text(workspace, first_runtime.session_id)
    second_log = _session_log_text(workspace, second_runtime.session_id)
    assert "marker=FIRST_SESSION" in first_log
    assert "marker=SECOND_SESSION" not in first_log
    assert "marker=SECOND_SESSION" in second_log
    assert "marker=FIRST_SESSION" not in second_log


async def _collect_event_types(conversation: ConversationPort, text: str) -> list[str]:
    return [event.type async for event in conversation.submit(text)]


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

    unavailable_runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=unavailable_workspace,
        configuration=configuration,
        provider_factory=lambda _: provider(),
        now=FakeClock(NOW).now,
        new_uuid=iter(
            (
                SESSION_UUID,
                TURN_UUID,
                REQUEST_UUID,
                FINAL_REQUEST_UUID,
            )
        ).__next__,
    )
    unavailable_logs = unavailable_workspace / ".myclaw" / "logs"
    unavailable_logs.write_text("Session Log unavailable", encoding="utf-8")

    unavailable_events = [
        event async for event in unavailable_runtime.conversation.submit("Fail-open request.")
    ]
    await unavailable_runtime.close()
    unavailable_session = unavailable_runtime.session

    assert [event.type for event in unavailable_events] == [
        "turn_started",
        "model_call_completed",
        "tool_started",
        "tool_completed",
        "turn_failed",
    ]
    assert isinstance(unavailable_events[3].payload, ToolCompletedPayload)
    assert unavailable_events[3].payload.status == "error"
    assert unavailable_events[3].payload.summary == "The requested tool is not available."
    assert isinstance(unavailable_events[-1].payload, TurnFailedPayload)
    assert unavailable_events[-1].payload.error == ErrorInfo(
        code="model_failed",
        message="The model request failed.",
    )
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
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=uuid4,
        retry_clock=clock,
    )

    events = [event async for event in runtime.conversation.submit(private_query)]
    await runtime.conversation.close()

    assert [event.type for event in events] == [
        "turn_started",
        "model_call_completed",
        "tool_started",
        "tool_completed",
        "model_call_completed",
        "turn_completed",
    ]
    assert isinstance(events[3].payload, ToolCompletedPayload)
    assert events[3].payload.status == "error"
    assert events[3].payload.summary == "web_search could not complete the request."
    assert private_query not in events[3].payload.summary
    assert "PRIVATE_WEB_CREDENTIAL" not in events[3].payload.summary
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
    runtime = prepare_repl_runtime(
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
        events = [event async for event in runtime.conversation.submit(private_input)]
        await runtime.conversation.close()
        terminal_output = capsys.readouterr().err
    finally:
        logger.remove()

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    assert isinstance(events[1].payload, TurnFailedPayload)
    assert events[1].payload.error == ErrorInfo(
        code="model_failed",
        message="The model request failed.",
    )
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

    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=provider_factory,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID, REQUEST_UUID)).__next__,
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

    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=provider_factory,
        now=FakeClock(NOW).now,
        new_uuid=iter((SESSION_UUID, TURN_UUID, REQUEST_UUID)).__next__,
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
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID, REQUEST_UUID)).__next__,
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

    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=provider_factory,
        now=clock.now,
        new_uuid=iter((SESSION_UUID, TURN_UUID, REQUEST_UUID)).__next__,
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
    first = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=unexpected_provider_factory,
        now=FakeClock(NOW).now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    second = prepare_repl_runtime(
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
    interrupted = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=unexpected_provider_factory,
        now=FakeClock(NOW).now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    projected = prepare_repl_runtime(
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

    runtime = prepare_repl_runtime(
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
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=FakeClock(NOW).now,
        new_uuid=uuid4,
    )

    events = [event async for event in runtime.conversation.submit("Use the fallback budget.")]
    await runtime.close()

    assert [event.type for event in events] == [
        "turn_started",
        "model_call_completed",
        "turn_completed",
    ]
    assert provider.stream_requests
    for request in provider.stream_requests:
        assert isinstance(request, ModelRequest)
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
                REQUEST_UUID,
                TURN_TWO_UUID,
                REQUEST_TWO_UUID,
            )
        ).__next__,
    )
    state.long_term_memory_path.write_text(
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
