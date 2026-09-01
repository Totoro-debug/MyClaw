"""End-to-end Schedule acceptance through AgentLoop boundaries."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from myclaw.agent.context import ContextBuilder
from myclaw.agent.loop import AgentLoop
from myclaw.agent.message_bus import MessageBus
from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runner import AgentRunner, AgentRunnerResult
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.memory.conversation_summary import ConversationSummaryManager
from myclaw.memory.dream import Dream
from myclaw.memory.manager import MemoryManager
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
from myclaw.schedule.model import JobSchedule, ScheduleJob, ScheduleJobState
from myclaw.schedule.service import ScheduleClock, ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    DeterministicTaskFramingEvaluator,
    FakeClock,
    ProviderCall,
    collect_foreground_outbound,
)
from tests.fixtures.diagnostic_capture import capture_diagnostics
from tests.management.factories import management_service

NOW = datetime(2026, 8, 7, 12, 0, 0, 123000, tzinfo=timezone(timedelta(hours=8)))
JOB_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")


class _BlockingClock:
    """Keep a scheduler asleep while a test exercises another boundary."""

    def __init__(self, current: datetime) -> None:
        self._current = current
        self._sleep_forever = asyncio.Event()

    def now(self) -> datetime:
        return self._current

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds
        await self._sleep_forever.wait()


class _ScheduleProvider:
    """Route-aware provider transcript used by Schedule and AgentLoop tests."""

    def __init__(
        self,
        *,
        chat_responses: Iterable[ModelResponse] = (),
        schedule_responses: Iterable[ModelResponse] = (),
        memory_responses: Iterable[ModelResponse] = (),
        block_chat_call: int | None = None,
        block_schedule_call: int | None = None,
    ) -> None:
        self._responses = {
            "chat": deque(chat_responses),
            "schedule": deque(schedule_responses),
            "memory": deque(memory_responses),
        }
        self._block_chat_call = block_chat_call
        self._block_schedule_call = block_schedule_call
        self._chat_call_count = 0
        self._schedule_call_count = 0
        self.chat_block_started = asyncio.Event()
        self.release_chat = asyncio.Event()
        self.schedule_block_started = asyncio.Event()
        self.release_schedule = asyncio.Event()
        self.stream_requests: list[ProviderCall] = []
        self.complete_requests: list[ProviderCall] = []
        self.direct_complete_messages: list[
            tuple[list[dict[str, object]], list[OpenAIToolSchema]]
        ] = []
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
        if messages and messages[0] == {
            "role": "system",
            "content": session_title_prompt(),
        }:
            yield ModelCompleted(response=_response("Schedule acceptance"))
            return

        self.stream_requests.append(
            ProviderCall(
                messages=deepcopy(list(messages)),
                tools=tuple(tools),
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
        )
        self._chat_call_count += 1
        if self._block_chat_call == self._chat_call_count:
            self.chat_block_started.set()
            await self.release_chat.wait()
        yield ModelCompleted(response=self._take("chat"))

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
        self.direct_complete_messages.append((deepcopy(list(messages)), list(tools)))
        call = ProviderCall(
            messages=deepcopy(list(messages)),
            tools=tuple(tools),
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        self.complete_requests.append(call)
        route = "schedule" if len(tools) == 10 else "memory"
        if route == "schedule":
            self._schedule_call_count += 1
            if self._block_schedule_call == self._schedule_call_count:
                self.schedule_block_started.set()
                await self.release_schedule.wait()
        return self._take(route)

    async def close(self) -> None:
        self.closed = True

    def _take(self, route: str) -> ModelResponse:
        responses = self._responses.get(route)
        if responses is None or not responses:
            raise AssertionError(f"No scripted response remains for {route}.")
        return responses.popleft()


def _is_schedule_call(call: ProviderCall) -> bool:
    return len(call.tools) == 10


def _response(content: str, *, tool_call: ModelToolCall | None = None) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(
            content=content,
            tool_calls=() if tool_call is None else (tool_call,),
        ),
        usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
        finish_reason="tool_calls" if tool_call is not None else "stop",
    )


def _schedule_tool_response(call_id: str, arguments: dict[str, object]) -> ModelResponse:
    return _response(
        "",
        tool_call=ModelToolCall(
            id=call_id,
            name="schedule",
            arguments=json.dumps(arguments, separators=(",", ":")),
        ),
    )


def _due_job(*, message: str = "Run this.") -> ScheduleJob:
    return ScheduleJob(
        job_id=str(JOB_UUID),
        message=message,
        schedule=JobSchedule.at("2026-08-07T03:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )


def _agent_loop(
    agent_home: Path,
    workspace: Path,
    provider: _ScheduleProvider,
    *,
    schedule_clock: ScheduleClock,
    config_text: str = VALID_CONFIG,
) -> tuple[
    AgentLoop,
    ModelRouter,
    ScheduleService,
    Dream,
    ManagementCommandDispatcher,
    MessageBus,
]:
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
    memory_manager = MemoryManager(state)
    dream = Dream(
        memory_manager=memory_manager,
        model_router=router,
        batch_size=configuration.memory.batch_size,
        max_iterations=configuration.runtime.max_iterations,
    )
    loop: AgentLoop | None = None

    async def execute_user_job(job: ScheduleJob) -> None:
        assert loop is not None
        await loop.run_schedule_job(job)

    schedule = ScheduleService(
        workspace_state=state,
        clock=schedule_clock,
        execute_user_job=execute_user_job,
        execute_dream=dream.run,
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
        memory_manager=memory_manager,
        session_id=None,
        now=lambda: NOW,
        new_uuid=lambda: JOB_UUID,
        monotonic_now=schedule_clock.monotonic,
    )
    loop._task_framer = DeterministicTaskFramingEvaluator()
    dispatcher = ManagementCommandDispatcher(
        management_service(
            home,
            current_agent_loop=lambda: loop,
            workspace_state=state,
            memory_manager=memory_manager,
            dream=dream,
            schedule_status=lambda: schedule.status_snapshot().to_dict(),
            now=lambda: NOW,
            monotonic=schedule_clock.monotonic,
        )
    )
    return loop, router, schedule, dream, dispatcher, bus


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for Schedule acceptance state.")
        await asyncio.sleep(0)


def _schedule_state(workspace: Path) -> WorkspaceScheduleStore:
    return WorkspaceScheduleStore(WorkspaceState(workspace))


def _tool_json(loop: AgentLoop) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(cast(str, message["content"])))
        for message in loop.session.messages
        if message.get("role") == "tool" and cast(str, message["content"]).startswith("{")
    ]


async def _close_components(
    loop: AgentLoop,
    router: ModelRouter,
    schedule: ScheduleService,
    dream: Dream,
) -> None:
    await schedule.close()
    await loop.close()
    await dream.close()
    await router.close()


@pytest.mark.asyncio
async def test_agent_loop_manages_schedule_jobs_without_confirmation(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(
            _schedule_tool_response(
                "call_add",
                {"action": "add", "message": "  ship report  ", "every_seconds": 60},
            ),
            _response("Added."),
            _schedule_tool_response("call_list", {"action": "list"}),
            _response("Listed."),
            _schedule_tool_response(
                "call_remove",
                {"action": "remove", "job_id": str(JOB_UUID)},
            ),
            _response("Removed."),
        )
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "Schedule the report.")

        jobs = await _schedule_state(workspace).public_snapshot()
        assert len(jobs) == 1
        job_id = jobs[0].job_id
        assert jobs[0].message == "ship report"

        await collect_foreground_outbound(_bus, "List my Schedule Jobs.")
        provider._responses["chat"][0] = _schedule_tool_response(
            "call_remove",
            {"action": "remove", "job_id": job_id},
        )
        results = _tool_json(loop)
        assert results[0]["action"] == "add"
        assert results[1] == {
            "jobs": [
                {
                    "job_id": job_id,
                    "message": "ship report",
                    "schedule": {"type": "every", "every_seconds": 60},
                }
            ]
        }

        await collect_foreground_outbound(_bus, "Remove that Schedule Job.")
        assert await _schedule_state(workspace).public_snapshot() == ()
        assert _tool_json(loop)[-1]["action"] == "remove"
    finally:
        await _close_components(loop, router, schedule, dream)


@pytest.mark.asyncio
async def test_schedule_uses_its_own_complete_context_projection(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    await WorkspaceScheduleStore(state).add_user_job(
        ScheduleJob(
            job_id=str(JOB_UUID),
            message="Run this.",
            schedule=JobSchedule.at("2026-08-07T03:59:00.000+00:00"),
            created_at_ms=1,
            updated_at_ms=1,
        )
    )
    provider = _ScheduleProvider(schedule_responses=(_response("Background result."),))

    def fail_foreground_context(*args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        raise AssertionError("Schedule must not use ContextBuilder")

    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    await loop.start()
    monkeypatch.setattr(ContextBuilder, "build_messages", fail_foreground_context)
    schedule.start()
    try:
        await _wait_until(
            lambda: (
                len(provider.direct_complete_messages) == 1
                and schedule.status_snapshot().active_job_count == 0
            )
        )
    finally:
        await _close_components(loop, router, schedule, dream)

    assert len(provider.direct_complete_messages) == 1
    messages, tools = provider.direct_complete_messages[0]
    assert messages[0]["role"] == "system"
    system_content = cast(str, messages[0]["content"])
    assert str(workspace) in system_content
    assert "## Task goal" not in system_content
    assert "## Completion boundary" not in system_content
    assert system_content.splitlines().count("# Long-term Memory") == 0
    assert system_content.splitlines().count("## Long-term Memory") == 1
    assert all(
        f"### {section}" in system_content
        for section in ("User Info", "User Preference", "Project Fact", "Lesson")
    )
    assert messages[-1]["role"] == "user"
    assert messages[-1] == {
        "role": "user",
        "content": (
            "<runtime_context>\n"
            f"current_time: {NOW.isoformat(timespec='milliseconds')}\n"
            f"session_id: schedule_{JOB_UUID}\n"
            "</runtime_context>\n\n"
            "<user_input>\n"
            "Run this.\n"
            "</user_input>"
        ),
    }
    assert all("timestamp" not in message for message in messages)
    assert [definition["function"]["name"] for definition in tools] == [
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


@pytest.mark.asyncio
async def test_schedule_tool_loop_persists_each_message_from_awaitable_run(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    await WorkspaceScheduleStore(state).add_user_job(
        ScheduleJob(
            job_id=str(JOB_UUID),
            message="Read the memory template.",
            schedule=JobSchedule.at("2026-08-07T03:59:00.000+00:00"),
            created_at_ms=1,
            updated_at_ms=1,
        )
    )
    provider = _ScheduleProvider(
        schedule_responses=(
            _response(
                "",
                tool_call=ModelToolCall(
                    id="call_read",
                    name="read_file",
                    arguments=json.dumps(
                        {"path": str(state.long_term_memory_path)},
                        separators=(",", ":"),
                    ),
                ),
            ),
            _response("The memory template was read."),
        )
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    await loop.start()
    schedule.start()
    try:
        await _wait_until(
            lambda: (
                len(provider.direct_complete_messages) == 2
                and schedule.status_snapshot().active_job_count == 0
            )
        )
    finally:
        await _close_components(loop, router, schedule, dream)

    second_messages, _ = provider.direct_complete_messages[1]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_messages[2]["tool_calls"] == [
        {
            "id": "call_read",
            "name": "read_file",
            "arguments": json.dumps(
                {"path": str(state.long_term_memory_path)},
                separators=(",", ":"),
            ),
        }
    ]
    assert second_messages[3]["tool_call_id"] == "call_read"
    session = Session.load(
        state,
        f"schedule_{JOB_UUID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.messages[-1]["content"] == "The memory template was read."


@pytest.mark.asyncio
async def test_schedule_tool_loop_does_not_prepare_summary_inside_agent_run(
    agent_home: Path,
    workspace: Path,
) -> None:
    config_text = VALID_CONFIG.replace(
        "consolidation_message_threshold = 50",
        "consolidation_message_threshold = 5",
    )
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    await WorkspaceScheduleStore(state).add_user_job(
        ScheduleJob(
            job_id=str(JOB_UUID),
            message="Continue after reading memory.",
            schedule=JobSchedule.at("2026-08-07T03:59:00.000+00:00"),
            created_at_ms=1,
            updated_at_ms=1,
        )
    )
    schedule_session = Session.create_schedule(state, JOB_UUID, now=lambda: NOW)
    schedule_session.add_message("user", "Old scheduled request.")
    schedule_session.add_message(
        "assistant",
        "Old scheduled answer.",
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
    schedule_session.close()
    provider = _ScheduleProvider(
        schedule_responses=(
            _response(
                "",
                tool_call=ModelToolCall(
                    id="call_read",
                    name="read_file",
                    arguments=json.dumps({"path": str(state.long_term_memory_path)}),
                ),
            ),
            _response("Continuation completed."),
        ),
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
        config_text=config_text,
    )

    await loop.start()
    schedule.start()
    try:
        await _wait_until(
            lambda: (
                [_is_schedule_call(request) for request in provider.complete_requests]
                == [True, True]
                and schedule.status_snapshot().active_job_count == 0
            )
        )
    finally:
        await _close_components(loop, router, schedule, dream)

    schedule_messages = [messages for messages, _tools in provider.direct_complete_messages]
    assert [message["role"] for message in schedule_messages[1]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert "Old scheduled request." in json.dumps(schedule_messages[1])
    assert not (state.memory_directory / "summary.jsonl").exists()
    persisted = Session.load(
        state,
        f"schedule_{JOB_UUID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert persisted.last_consolidated == 0


@pytest.mark.asyncio
async def test_schedule_summary_flows_through_memory_to_a_later_schedule_run(
    agent_home: Path,
    workspace: Path,
) -> None:
    config_text = VALID_CONFIG.replace(
        "consolidation_message_threshold = 50",
        "consolidation_message_threshold = 4",
    )
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    store = WorkspaceScheduleStore(state)
    created_at_ms = int((NOW - timedelta(seconds=60)).timestamp() * 1000)
    await store.add_user_job(
        ScheduleJob(
            job_id=str(JOB_UUID),
            source="user",
            message="Continue the scheduled work.",
            schedule=JobSchedule.every(60),
            state=ScheduleJobState(),
            created_at_ms=created_at_ms,
            updated_at_ms=created_at_ms,
        )
    )
    schedule_session = Session.create(
        state,
        now=lambda: NOW,
        partition=SessionStoragePartition.SCHEDULE,
        job_id=JOB_UUID,
    )
    schedule_session.add_message("user", "Oldest scheduled request.")
    schedule_session.add_message(
        "assistant",
        "Oldest scheduled answer.",
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
    schedule_session.add_message("user", "Earlier scheduled request.")
    schedule_session.add_message(
        "assistant",
        "Earlier scheduled answer.",
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
    schedule_session.close()

    memory_path = state.long_term_memory_path
    old_memory = memory_path.read_text(encoding="utf-8")
    new_user_info = "## User Info\n\nFresh schedule preference.\n"
    new_memory = old_memory.replace(
        "## User Info\n",
        new_user_info,
        1,
    )
    provider = _ScheduleProvider(
        schedule_responses=(
            _response("First scheduled result."),
            _response("Second scheduled result."),
        ),
        memory_responses=(
            _response("Schedule history summary."),
            _response(
                "",
                tool_call=ModelToolCall(
                    id="read-memory",
                    name="read_file",
                    arguments=json.dumps({"path": str(memory_path)}),
                ),
            ),
            _response(
                "",
                tool_call=ModelToolCall(
                    id="edit-memory",
                    name="edit_file",
                    arguments=json.dumps(
                        {
                            "path": str(memory_path),
                            "old_text": "## User Info\n",
                            "new_text": new_user_info,
                            "replace_all": "false",
                        }
                    ),
                ),
            ),
            _response("Long-term Memory updated."),
            _response("Second schedule history summary."),
        ),
    )
    clock = FakeClock(NOW)
    loop, router, schedule, dream_owner, dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=clock,
        config_text=config_text,
    )
    try:
        await loop.start()
        schedule.start()
        await _wait_until(
            lambda: (
                len(
                    [
                        request
                        for request in provider.complete_requests
                        if _is_schedule_call(request)
                    ]
                )
                == 1
                and schedule.status_snapshot().active_job_count == 0
            )
        )
        summary_records = [
            json.loads(line)
            for line in (state.memory_directory / "summary.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [record["content"] for record in summary_records] == ["Schedule history summary."]

        dream_result = await dispatcher.dispatch("/dream")
        assert dream_result.output is not None
        assert (state.memory_directory / ".cursor").read_bytes() == b"1\n"
        assert memory_path.read_text(encoding="utf-8") == new_memory

        clock.advance(60)
        await _wait_until(
            lambda: (
                len(
                    [
                        request
                        for request in provider.complete_requests
                        if _is_schedule_call(request)
                    ]
                )
                == 2
                and schedule.status_snapshot().active_job_count == 0
            )
        )
        schedule_requests = [
            request for request in provider.complete_requests if _is_schedule_call(request)
        ]
        assert "Fresh schedule preference." not in cast(
            str, schedule_requests[0].messages[0]["content"]
        )
        assert "Fresh schedule preference." in cast(
            str, schedule_requests[1].messages[0]["content"]
        )
        summary_records = [
            json.loads(line)
            for line in (state.memory_directory / "summary.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [record["content"] for record in summary_records] == [
            "Schedule history summary.",
            "Second schedule history summary.",
        ]
    finally:
        await _close_components(loop, router, schedule, dream_owner)


@pytest.mark.asyncio
async def test_schedule_dispatcher_wakes_for_due_at_job_and_keeps_schedule_session_out_of_resume(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(
            _schedule_tool_response(
                "call_due_add",
                {
                    "action": "add",
                    "message": "run after the dispatcher wakes",
                    "at_time": NOW.isoformat(timespec="milliseconds"),
                },
            ),
            _response("Scheduled."),
        ),
        schedule_responses=(_response("Background result."),),
    )
    clock = FakeClock(NOW)
    loop, router, schedule, dream, dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=clock,
    )
    await loop.start()
    schedule.start()
    try:
        events = await collect_foreground_outbound(_bus, "Schedule this due task.")
        assert events[-1].metadata == {"_streamed": True}
        await _wait_until(
            lambda: (
                len(provider.complete_requests) == 1
                and schedule.status_snapshot().active_job_count == 0
            )
        )

        assert _is_schedule_call(provider.complete_requests[0])
        assert [
            definition["function"]["name"] for definition in provider.complete_requests[0].tools
        ] == [
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
        assert await _schedule_state(workspace).public_snapshot() == ()
        schedule_session_paths = tuple(
            (workspace / ".myclaw" / "schedule-sessions").glob("schedule_*.jsonl")
        )
        assert len(schedule_session_paths) == 1
        schedule_session_id = schedule_session_paths[0].stem
        schedule_session = Session.load(
            WorkspaceState(workspace),
            schedule_session_id,
            partition=SessionStoragePartition.SCHEDULE,
        )
        assert [message["role"] for message in schedule_session.messages] == [
            "user",
            "assistant",
        ]
        assert schedule_session.messages[-1]["content"] == "Background result."
        assert (
            workspace / ".myclaw" / "schedule-sessions" / f"{schedule_session_id}.jsonl"
        ).exists()

        resume = await dispatcher.dispatch("/resume")
        assert resume.output is not None
        assert schedule_session_id not in resume.output
        assert {message.type for message in events} <= {
            "model_reasoning",
            "model_response",
            "tool_call",
            "system_control",
        }
    finally:
        await _close_components(loop, router, schedule, dream)


def test_schedule_service_user_executor_is_bound_to_agent_loop(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _ScheduleProvider()
    loop, _router, schedule, _dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )

    callback = schedule._execute_user_job
    closure_values = tuple(cell.cell_contents for cell in (callback.__closure__ or ()))
    assert loop in closure_values


@pytest.mark.asyncio
async def test_foreground_and_schedule_share_runner_and_gateway_identity(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ScheduleProvider(
        chat_responses=(_response("Foreground result."),),
        schedule_responses=(_response("Schedule result."),),
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    observed: list[tuple[AgentRunner, object, str, object, object]] = []
    original_run = AgentRunner.run

    async def record_run(
        runner: AgentRunner,
        initial_messages: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> AgentRunnerResult:
        observed.append(
            (
                runner,
                kwargs["tool_gateway"],
                cast(str, kwargs["model"]),
                kwargs["confirmation"],
                kwargs["on_output"],
            )
        )
        return await original_run(runner, initial_messages, **kwargs)

    monkeypatch.setattr(AgentRunner, "run", record_run)
    callback = loop.run_schedule_job
    await loop.start()
    try:
        assert (await collect_foreground_outbound(_bus, "Run foreground."))[-1].metadata == {
            "_streamed": True
        }
        await callback(_due_job(message="Run Schedule."))
    finally:
        await _close_components(loop, router, schedule, dream)

    assert len(observed) == 2
    assert observed[0][0] is observed[1][0]
    assert observed[0][1] is observed[1][1]
    assert [call[2] for call in observed] == ["chat", "schedule"]
    assert observed[0][3] is not None
    assert observed[1][3] is None
    assert observed[0][4] is not observed[1][4]


@pytest.mark.asyncio
async def test_foreground_and_schedule_artifacts_remain_separate(
    agent_home: Path,
    workspace: Path,
) -> None:
    large_file = workspace / "large.txt"
    large_file.parent.mkdir(parents=True, exist_ok=True)
    large_file.write_text("x\n" * 4000, encoding="utf-8")
    provider = _ScheduleProvider(
        chat_responses=(
            _response(
                "",
                tool_call=ModelToolCall(
                    id="foreground_artifact",
                    name="read_file",
                    arguments='{"path":"large.txt","limit":10000}',
                ),
            ),
            _response("Foreground artifact stored."),
        ),
        schedule_responses=(
            _response(
                "",
                tool_call=ModelToolCall(
                    id="schedule_artifact",
                    name="read_file",
                    arguments='{"path":"large.txt","limit":10000}',
                ),
            ),
            _response("Schedule artifact stored."),
        ),
    )
    config_text = VALID_CONFIG.replace(
        "max_tool_result_chars = 60000", "max_tool_result_chars = 1000"
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
        config_text=config_text,
    )
    callback = loop.run_schedule_job
    await loop.start()
    try:
        await collect_foreground_outbound(_bus, "Read the large file in foreground.")
        await callback(_due_job(message="Read the large file in Schedule."))
        foreground_session_id = loop.session.session_id
        foreground_tool = next(
            message for message in loop.session.messages if message["role"] == "tool"
        )
    finally:
        await _close_components(loop, router, schedule, dream)

    state = WorkspaceState(workspace)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_UUID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    schedule_tool = next(
        message for message in schedule_session.messages if message["role"] == "tool"
    )
    foreground_artifact = cast(dict[str, object], foreground_tool["artifact"])
    schedule_artifact = cast(dict[str, object], schedule_tool["artifact"])
    assert set(foreground_artifact) == {"path", "total_chars", "preview_chars"}
    assert set(schedule_artifact) == {"path", "total_chars", "preview_chars"}
    assert foreground_artifact["path"] == (
        f".myclaw/artifacts/{foreground_session_id}/foreground_artifact.txt"
    )
    assert schedule_artifact["path"] == (
        f".myclaw/artifacts/schedule_{JOB_UUID}/schedule_artifact.txt"
    )
    assert (workspace / foreground_artifact["path"]).exists()
    assert (workspace / schedule_artifact["path"]).exists()


@pytest.mark.asyncio
async def test_schedule_session_uses_schedule_clock_for_persisted_timestamps(
    agent_home: Path,
    workspace: Path,
) -> None:
    schedule_now = NOW + timedelta(hours=2)
    provider = _ScheduleProvider(schedule_responses=(_response("Background result."),))
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(schedule_now),
    )
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run at the Schedule clock time.",
        schedule=JobSchedule.at("2026-08-07T05:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )

    callback = loop.run_schedule_job
    try:
        await callback(job)
    finally:
        await _close_components(loop, router, schedule, dream)

    schedule_session = Session.load(
        WorkspaceState(workspace),
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    expected_timestamp = schedule_now.isoformat(timespec="milliseconds")
    assert schedule_session.created_at == schedule_now
    assert schedule_session.updated_at == schedule_now
    assert {message["timestamp"] for message in schedule_session.messages} == {expected_timestamp}


@pytest.mark.asyncio
async def test_schedule_shutdown_during_model_persists_user_and_keeps_job_pending(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    job = _due_job(message="Block in the Schedule model.")
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    provider = _ScheduleProvider(
        schedule_responses=(_response("Unused."),),
        block_schedule_call=1,
    )
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )

    schedule.start()
    await provider.schedule_block_started.wait()
    await _close_components(loop, router, schedule, dream)

    assert await store.snapshot() == (job,)
    schedule_session = Session.load(
        state,
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [(message["role"], message["content"]) for message in schedule_session.messages] == [
        ("user", job.message)
    ]


@pytest.mark.asyncio
async def test_schedule_shutdown_during_preparation_persists_user(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    job = _due_job(message="Block in Schedule context preparation.")
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    provider = _ScheduleProvider(schedule_responses=(_response("Unused."),))
    context_started = asyncio.Event()
    context_never_completes = asyncio.Event()
    original_prepare = ConversationSummaryManager.prepare

    async def block_schedule_preparation(
        manager: ConversationSummaryManager,
        session: Session,
        *,
        current_user: dict[str, Any] | None = None,
        continuation: Sequence[dict[str, Any]] = (),
        project_messages: Callable[[Sequence[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        route_context_window: int | None = None,
        route_max_output: int | None = None,
        tools: Sequence[OpenAIToolSchema] | None = None,
    ) -> Session:
        if session.storage_partition is SessionStoragePartition.SCHEDULE:
            context_started.set()
            await context_never_completes.wait()
        return await original_prepare(
            manager,
            session,
            current_user=current_user,
            continuation=continuation,
            project_messages=project_messages,
            route_context_window=route_context_window,
            route_max_output=route_max_output,
            tools=tools,
        )

    monkeypatch.setattr(ConversationSummaryManager, "prepare", block_schedule_preparation)
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )

    schedule.start()
    await context_started.wait()
    await _close_components(loop, router, schedule, dream)

    assert provider.complete_requests == []
    assert await store.snapshot() == (job,)
    schedule_session = Session.load(
        state,
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [(message["role"], message["content"]) for message in schedule_session.messages] == [
        ("user", job.message)
    ]


@pytest.mark.asyncio
async def test_schedule_failure_logs_one_safe_session_warning(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    job = _due_job(message="Fail safely during Schedule preparation.")
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(job)
    provider = _ScheduleProvider()
    failure_started = asyncio.Event()
    original_prepare = ConversationSummaryManager.prepare

    async def fail_schedule_preparation(
        manager: ConversationSummaryManager,
        session: Session,
        *,
        current_user: dict[str, Any] | None = None,
        continuation: Sequence[dict[str, Any]] = (),
        project_messages: Callable[[Sequence[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        route_context_window: int | None = None,
        route_max_output: int | None = None,
        tools: Sequence[OpenAIToolSchema] | None = None,
    ) -> Session:
        if session.storage_partition is SessionStoragePartition.SCHEDULE:
            failure_started.set()
            raise RuntimeError("PRIVATE_SCHEDULE_PREPARATION_BODY")
        return await original_prepare(
            manager,
            session,
            current_user=current_user,
            continuation=continuation,
            project_messages=project_messages,
            route_context_window=route_context_window,
            route_max_output=route_max_output,
            tools=tools,
        )

    monkeypatch.setattr(ConversationSummaryManager, "prepare", fail_schedule_preparation)
    capture = capture_diagnostics()
    loop, router, schedule, dream, _dispatcher, _bus = _agent_loop(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    try:
        schedule.start()
        await failure_started.wait()
        await _wait_until(lambda: schedule.status_snapshot().active_job_count == 0)
    finally:
        await _close_components(loop, router, schedule, dream)
        capture.close()

    assert await _schedule_state(workspace).public_snapshot() == ()
    assert capture.event_text.count("Schedule Job failed") == 1
    assert "code=model_failed" in capture.event_text
    assert "PRIVATE_SCHEDULE_PREPARATION_BODY" not in capture.text
    session_log_text = (state.logs_directory / f"{job.session_id}.log").read_text(encoding="utf-8")
    assert session_log_text.count("Schedule Job failed") == 1
    assert "code=model_failed" in session_log_text
    assert "PRIVATE_SCHEDULE_PREPARATION_BODY" not in session_log_text
