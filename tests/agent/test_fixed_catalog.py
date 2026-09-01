from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from loguru import logger

from myclaw.agent.loop import AgentLoop, ConfirmationRequestView
from myclaw.agent.message_bus import MessageBus
from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.memory.manager import MemoryManager
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouter
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
)
from myclaw.schedule.service import ScheduleService
from myclaw.session.session import Session
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.core.web_fetch import JinaReaderClient
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import DeterministicBlackboardGenerator, collect_foreground_outbound
from tests.fixtures.provider import ProviderCall

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


class _FixedCatalogProvider:
    def __init__(self, responses: Iterable[ModelResponse | BaseException]) -> None:
        self._responses = deque(responses)
        self.stream_requests: list[ProviderCall] = []
        self.closed = False
        self.log_marker: str | None = None

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
            raise AssertionError("No scripted Agent Loop response remains")
        if self.log_marker is not None:
            logger.warning(self.log_marker)
        outcome = self._responses.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        yield ModelCompleted(response=outcome)

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


def _agent_loop(
    agent_home: Path,
    workspace: Path,
    provider: _FixedCatalogProvider,
    *,
    config_text: str = VALID_CONFIG,
) -> tuple[AgentLoop, ModelRouter, ScheduleService, MessageBus]:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    router = ModelRouter(
        configuration=configuration,
        provider_factory=lambda _configuration: provider,
    )
    loop: AgentLoop | None = None

    async def execute_user_job(job: object) -> None:
        assert loop is not None
        await loop.run_schedule_job(job)  # type: ignore[arg-type]

    async def execute_dream() -> None:
        return None

    schedule = ScheduleService(
        workspace_state=state,
        clock=_BlockingClock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    bus = MessageBus()
    loop = AgentLoop(
        workspace_path=workspace,
        workspace_state=state,
        agent_home=home,
        configuration=configuration,
        bus=bus,
        schedule_service=schedule,
        model_router=router,
        memory_manager=MemoryManager(state),
        session_id=None,
        now=lambda: NOW,
        new_uuid=uuid4,
        monotonic_now=lambda: 0.0,
    )
    object.__setattr__(loop, "_generate_blackboard", DeterministicBlackboardGenerator().generate)
    return loop, router, schedule, bus


async def _close_loop(
    loop: AgentLoop,
    router: ModelRouter,
    schedule: ScheduleService,
) -> None:
    await loop.close()
    await schedule.close()
    await router.close()


@pytest.mark.asyncio
async def test_agent_loop_uses_fixed_catalog_for_provider_confirmation_and_persistence(
    agent_home: Path,
    workspace: Path,
) -> None:
    outside = (workspace.parent / "external-note.txt").resolve()
    outside.write_text("outside content", encoding="utf-8")
    provider = _FixedCatalogProvider(
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
    loop, router, schedule, bus = _agent_loop(agent_home, workspace, provider)
    confirmations: list[ConfirmationRequestView] = []
    loop.bind_confirmation_callback(confirmations.append)
    try:
        await loop.start()
        turn = asyncio.create_task(collect_foreground_outbound(bus, "Read the file."))
        while not confirmations:
            await asyncio.sleep(0)
        confirmation = confirmations[0]
        loop.respond_to_confirmation(confirmation.confirmation_id, "approved")
        messages = await turn
    finally:
        await _close_loop(loop, router, schedule)

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
    tool_messages = [message for message in loop.session.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "outside content"
    assert tool_messages[0]["status"] == "success"


@pytest.mark.asyncio
async def test_agent_loop_reads_known_skill_path_without_confirmation(
    agent_home: Path,
    workspace: Path,
) -> None:
    skill_file = agent_home / "skills" / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_bytes(b"---\nname: review\n---\nbody\n")
    provider = _FixedCatalogProvider(
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
    loop, router, schedule, bus = _agent_loop(agent_home, workspace, provider)
    confirmations: list[ConfirmationRequestView] = []

    def approve(request: ConfirmationRequestView) -> None:
        confirmations.append(request)
        loop.respond_to_confirmation(request.confirmation_id, "approved")

    loop.bind_confirmation_callback(approve)
    try:
        await loop.start()
        await collect_foreground_outbound(bus, "Read the skill.")
    finally:
        await _close_loop(loop, router, schedule)

    assert confirmations == []
    tool_messages = [message for message in loop.session.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "---\nname: review\n---\nbody\n"
    assert tool_messages[0]["status"] == "success"


@pytest.mark.asyncio
async def test_agent_loop_advertises_and_persists_multiple_autonomous_skill_reads(
    agent_home: Path,
    workspace: Path,
) -> None:
    first = agent_home / "skills" / "first" / "SKILL.md"
    second = agent_home / "skills" / "second" / "SKILL.md"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(
        b"---\nname: first\ndescription: First instructions\n---\nfirst body\nsecond body\n"
    )
    second_body = b"---\nname: second\ndescription: Second instructions\n---\n" + b"x" * 1600
    second.write_bytes(second_body)
    config_text = VALID_CONFIG.replace(
        "max_tool_result_chars = 60000", "max_tool_result_chars = 1000"
    )
    provider = _FixedCatalogProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="first-page-one",
                    name="read_file",
                    arguments=json.dumps({"path": str(first), "offset": 1, "limit": 1}),
                ),
            ),
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="first-page-two",
                    name="read_file",
                    arguments=json.dumps({"path": str(first), "offset": 2, "limit": 2}),
                ),
            ),
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="second-full",
                    name="read_file",
                    arguments=json.dumps({"path": str(second), "offset": 1, "limit": 10000}),
                ),
            ),
            _response(content="Used both Skills."),
        )
    )
    loop, router, schedule, bus = _agent_loop(
        agent_home, workspace, provider, config_text=config_text
    )
    session_id = loop.session.session_id
    try:
        await loop.start()
        messages = await collect_foreground_outbound(bus, "Use both Skills.")
    finally:
        await _close_loop(loop, router, schedule)

    assert messages[-1].metadata == {"_streamed": True}
    assert len(provider.stream_requests) == 4
    system_prompt = provider.stream_requests[0].messages[0]["content"]
    assert isinstance(system_prompt, str)
    assert system_prompt.count("<skill_catalog>") == 1
    metadata_lines = [line for line in system_prompt.splitlines() if line.startswith("{")]
    assert [json.loads(line) for line in metadata_lines] == [
        {
            "name": "first",
            "description": "First instructions",
            "path": str(first.resolve()),
        },
        {
            "name": "second",
            "description": "Second instructions",
            "path": str(second.resolve()),
        },
    ]
    assert all(
        [definition["function"]["name"] for definition in request.tools]
        == [
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
        for request in provider.stream_requests
    )

    tool_messages = [message for message in loop.session.messages if message["role"] == "tool"]
    assert [message["content"] for message in tool_messages[:2]] == [
        "---\n",
        "name: first\ndescription: First instructions\n",
    ]
    assert len(tool_messages) == 3
    artifact = tool_messages[2]["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["total_chars"] == len(second_body.decode("utf-8"))
    artifact_path = workspace / str(artifact["path"])
    assert artifact_path.read_bytes() == second_body

    persisted = Session.load(loop.session.workspace_state, session_id)
    persisted_tools = [message for message in persisted.messages if message["role"] == "tool"]
    assert persisted_tools == tool_messages


@pytest.mark.asyncio
async def test_agent_loop_keeps_artifact_and_log_correlation_when_persist_fails(
    agent_home: Path,
    workspace: Path,
) -> None:
    raw_tool_result = "oversized tool result " * 100
    (workspace / "large.txt").write_text(raw_tool_result, encoding="utf-8")
    provider = _FixedCatalogProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_active_artifact",
                    name="read_file",
                    arguments='{"path":"large.txt"}',
                ),
            ),
            _response(content="Artifact recorded."),
        )
    )
    provider.log_marker = "active Session correlation marker"
    config_text = VALID_CONFIG.replace(
        "max_tool_result_chars = 60000", "max_tool_result_chars = 1000"
    )
    loop, router, schedule, bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        config_text=config_text,
    )

    def fail_persist() -> None:
        raise OSError("ordinary snapshot failure")

    loop.session.persist = fail_persist  # type: ignore[method-assign]
    try:
        await loop.start()
        messages = await collect_foreground_outbound(bus, "Inspect large.txt.")
    finally:
        await _close_loop(loop, router, schedule)

    assert messages[-1].metadata == {"_streamed": True}
    tool_message = next(message for message in loop.session.messages if message["role"] == "tool")
    artifact = tool_message["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["path"] == (
        f".myclaw/artifacts/{loop.session.session_id}/call_active_artifact.txt"
    )
    assert (workspace / artifact["path"]).read_text(encoding="utf-8") == raw_tool_result
    log_path = workspace / ".myclaw" / "logs" / f"{loop.session.session_id}.log"
    assert "active Session correlation marker" in log_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_agent_loop_tool_failure_keeps_private_diagnostics_out_of_public_output(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_query = "PRIVATE_CONVERSATION_QUERY"

    class FailingDDGS:
        def __enter__(self) -> FailingDDGS:
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
    provider = _FixedCatalogProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_private_search",
                    name="web_search",
                    arguments=json.dumps({"query": private_query}),
                ),
            ),
            _response(content="Search failed safely."),
        )
    )
    loop, router, schedule, bus = _agent_loop(agent_home, workspace, provider)
    try:
        await loop.start()
        messages = await collect_foreground_outbound(bus, private_query)
    finally:
        await _close_loop(loop, router, schedule)

    tool_call = next(message for message in messages if message.type == "tool_call")
    assert tool_call.metadata["arguments"] == json.dumps({"query": private_query})
    assert all(private_query not in message.content for message in messages)
    assert all("PRIVATE_WEB_CREDENTIAL" not in message.content for message in messages)
    log_path = workspace / ".myclaw" / "logs" / f"{loop.session.session_id}.log"
    log_content = log_path.read_text(encoding="utf-8")
    assert log_content.count("Tool execution failed name=web_search") == 1
    assert "RAW_PROVIDER_BODY" in log_content
    assert log_content.count(f"OSError: query={private_query}") == 1
    assert log_content.count("ValueError: auth=PRIVATE_WEB_CREDENTIAL") == 1


@pytest.mark.asyncio
async def test_agent_loop_model_failure_logs_private_cause_but_emits_safe_terminal(
    agent_home: Path,
    workspace: Path,
) -> None:
    failures: list[ModelCallError] = []
    for _ in range(5):
        failure = ModelCallError(ErrorInfo("model_failed", "The model request failed."))
        failure.__cause__ = RuntimeError("RAW_PROVIDER_BODY auth=PRIVATE_MODEL_CREDENTIAL")
        failures.append(failure)
    provider = _FixedCatalogProvider(failures)
    loop, router, schedule, bus = _agent_loop(agent_home, workspace, provider)
    try:
        await loop.start()
        messages = await collect_foreground_outbound(bus, "PRIVATE_FOREGROUND_PROMPT")
    finally:
        await _close_loop(loop, router, schedule)

    terminal = messages[-1]
    assert terminal.type == "system_control"
    assert terminal.metadata["finish_reason"] == "failed"
    assert terminal.content == "The model request failed."
    assert "RAW_PROVIDER_BODY" not in terminal.content
    log_path = workspace / ".myclaw" / "logs" / f"{loop.session.session_id}.log"
    log_content = log_path.read_text(encoding="utf-8")
    assert log_content.count("Agent Run failed code=model_failed type=ModelCallError") == 1
    assert "RAW_PROVIDER_BODY auth=PRIVATE_MODEL_CREDENTIAL" in log_content
    assert "PRIVATE_FOREGROUND_PROMPT" not in log_content


@pytest.mark.asyncio
async def test_agent_loop_continues_when_session_log_path_is_unavailable(
    agent_home: Path,
    workspace: Path,
) -> None:
    failures = [
        ModelCallError(ErrorInfo("model_failed", "The model request failed.")) for _ in range(5)
    ]
    provider = _FixedCatalogProvider(
        (
            _response(
                content="",
                tool_call=ModelToolCall(
                    id="call_unavailable",
                    name="unavailable_tool",
                    arguments="{}",
                ),
            ),
            *failures,
        )
    )
    loop, router, schedule, bus = _agent_loop(agent_home, workspace, provider)
    unavailable_logs = workspace / ".myclaw" / "logs"
    unavailable_logs.write_text("Session Log unavailable", encoding="utf-8")
    try:
        await loop.start()
        messages = await collect_foreground_outbound(bus, "Fail-open request.")
    finally:
        await _close_loop(loop, router, schedule)

    assert any(message.type == "tool_call" for message in messages)
    terminal = messages[-1]
    assert terminal.type == "system_control"
    assert terminal.metadata["finish_reason"] == "failed"
    assert terminal.content == "The model request failed."
    assert [message["role"] for message in loop.session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [
        message["status"]
        for message in loop.session.messages
        if message["role"] in {"assistant", "tool"}
    ] == ["completed", "error", "error"]
    assert unavailable_logs.is_file()
    assert unavailable_logs.read_text(encoding="utf-8") == "Session Log unavailable"


@pytest.mark.asyncio
async def test_agent_loop_cancellation_reaches_an_active_fixed_catalog_tool(
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
    provider = _FixedCatalogProvider(
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
    loop, router, schedule, bus = _agent_loop(agent_home, workspace, provider)
    await loop.start()
    turn = asyncio.create_task(collect_foreground_outbound(bus, "Fetch the URL."))
    try:
        await started.wait()
        await loop.cancel_active_run()
        messages = await asyncio.wait_for(turn, timeout=1)
    finally:
        release.set()
        if not turn.done():
            turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await _close_loop(loop, router, schedule)

    assert cancelled.is_set()
    assert any(message.type == "tool_call" for message in messages)
    assert messages[-1].type == "system_control"
    assert messages[-1].metadata["finish_reason"] == "cancelled"
    tool_messages = [message for message in loop.session.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["status"] == "error"
    assert tool_messages[0]["content"] == ("Tool call interrupted because the turn was cancelled.")
