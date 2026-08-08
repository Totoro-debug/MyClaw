"""End-to-end Schedule acceptance through Runtime composition boundaries."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from myclaw.agent.events import AgentEvent, ConfirmationDecision, ConfirmationRequestedPayload
from myclaw.agent.prompts import session_title_prompt
from myclaw.agent.runtime import PreparedReplRuntime, prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.memory.conversation_summary import WorkspaceJsonlSummaryStore
from myclaw.memory.memory_task import WorkspaceFileMemoryStore
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from myclaw.schedule.model import JobSchedule, ScheduleJob, ScheduleJobState
from myclaw.schedule.service import ScheduleClock
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock

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


class _RuntimeProvider:
    """Route-aware provider transcript used by Runtime composition tests."""

    def __init__(
        self,
        *,
        chat_responses: Iterable[ModelResponse] = (),
        schedule_responses: Iterable[ModelResponse] = (),
        memory_responses: Iterable[ModelResponse] = (),
        block_chat_call: int | None = None,
    ) -> None:
        self._responses = {
            "chat": deque(chat_responses),
            "schedule": deque(schedule_responses),
            "memory": deque(memory_responses),
        }
        self._block_chat_call = block_chat_call
        self._chat_call_count = 0
        self.chat_block_started = asyncio.Event()
        self.release_chat = asyncio.Event()
        self.stream_requests: list[ModelRequest] = []
        self.complete_requests: list[ModelRequest] = []
        self.closed = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if request.system_prompt == session_title_prompt():
            yield ModelCompleted(response=_response("Schedule acceptance"))
            return

        self.stream_requests.append(request)
        self._chat_call_count += 1
        if self._block_chat_call == self._chat_call_count:
            self.chat_block_started.set()
            await self.release_chat.wait()
        yield ModelCompleted(response=self._take("chat"))

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.complete_requests.append(request)
        return self._take(request.route)

    async def close(self) -> None:
        self.closed = True

    def _take(self, route: str) -> ModelResponse:
        responses = self._responses.get(route)
        if responses is None or not responses:
            raise AssertionError(f"No scripted response remains for {route}.")
        return responses.popleft()


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


def _runtime(
    agent_home: Path,
    workspace: Path,
    provider: _RuntimeProvider,
    *,
    schedule_clock: ScheduleClock,
    config_text: str = VALID_CONFIG,
) -> PreparedReplRuntime:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    return prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=lambda: JOB_UUID,
        memory_scheduler_clock=_BlockingClock(NOW),
        schedule_scheduler_clock=schedule_clock,
    )


def _event_stream(runtime: PreparedReplRuntime, text: str) -> AsyncGenerator[AgentEvent, None]:
    return cast(AsyncGenerator[AgentEvent, None], runtime.conversation.submit(text))


async def _submit_turn(
    runtime: PreparedReplRuntime,
    text: str,
    *,
    decision: ConfirmationDecision | None = None,
) -> list[AgentEvent]:
    events = _event_stream(runtime, text)
    observed: list[AgentEvent] = []
    try:
        while True:
            try:
                event = await anext(events)
            except StopAsyncIteration:
                return observed
            observed.append(event)
            if event.type == "confirmation_requested":
                if decision is None:
                    raise AssertionError("The scripted turn unexpectedly needs confirmation.")
                payload = cast(ConfirmationRequestedPayload, event.payload)
                runtime.conversation.respond_to_confirmation(payload.confirmation_id, decision)
    finally:
        await events.aclose()


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for Schedule acceptance state.")
        await asyncio.sleep(0)


async def _next_event(events: AsyncIterator[AgentEvent], stage: str) -> AgentEvent:
    try:
        async with asyncio.timeout(1.0):
            return await anext(events)
    except TimeoutError as error:
        raise AssertionError(f"Timed out at {stage}.") from error


def _schedule_state(workspace: Path) -> WorkspaceScheduleStore:
    return WorkspaceScheduleStore(WorkspaceState(Workspace.from_path(workspace)))


def _tool_json(runtime: PreparedReplRuntime) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(cast(str, message["content"])))
        for message in runtime.session.messages
        if message.get("role") == "tool" and cast(str, message["content"]).startswith("{")
    ]


@pytest.mark.asyncio
async def test_runtime_conversation_manages_schedule_jobs_with_confirmation_and_stable_results(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(
        chat_responses=(
            _schedule_tool_response(
                "call_add",
                {"action": "add", "message": "  ship report  ", "every_seconds": 60},
            ),
            _response("Added."),
            _schedule_tool_response(
                "call_declined",
                {"action": "add", "message": "declined", "every_seconds": 60},
            ),
            _response("Declined."),
            _schedule_tool_response("call_list", {"action": "list"}),
            _response("Listed."),
            _schedule_tool_response(
                "call_remove",
                {"action": "remove", "job_id": str(JOB_UUID)},
            ),
            _response("Removed."),
        )
    )
    runtime = _runtime(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    await runtime.start()
    try:
        added = await _submit_turn(runtime, "Schedule the report.", decision="approved")
        assert any(event.type == "confirmation_requested" for event in added)
        added_confirmation = next(
            event for event in added if event.type == "confirmation_requested"
        )
        added_payload = cast(ConfirmationRequestedPayload, added_confirmation.payload)
        assert added_payload.details["message"] == "ship report"
        assert added_payload.details["schedule"] == {"type": "every", "every_seconds": 60}

        jobs = await _schedule_state(workspace).snapshot()
        assert len(jobs) == 1
        assert jobs[0].job_id == str(JOB_UUID)
        assert jobs[0].message == "ship report"

        declined = await _submit_turn(
            runtime, "Also schedule the declined item.", decision="declined"
        )
        assert any(event.type == "confirmation_requested" for event in declined)
        assert await _schedule_state(workspace).snapshot() == jobs

        listed = await _submit_turn(runtime, "List my Schedule Jobs.")
        assert not any(event.type == "confirmation_requested" for event in listed)
        assert "Tool confirmation was declined." in [
            message["content"]
            for message in runtime.session.messages
            if message.get("role") == "tool"
        ]
        results = _tool_json(runtime)
        assert results[0]["action"] == "add"
        assert results[1] == {
            "jobs": [
                {
                    "job_id": str(JOB_UUID),
                    "message": "ship report",
                    "schedule": {"type": "every", "every_seconds": 60},
                }
            ]
        }

        removed = await _submit_turn(runtime, "Remove that Schedule Job.", decision="approved")
        assert any(event.type == "confirmation_requested" for event in removed)
        assert await _schedule_state(workspace).snapshot() == ()
        assert _tool_json(runtime)[-1]["action"] == "remove"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_schedule_summary_flows_through_memory_to_a_later_schedule_run(
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
    state = WorkspaceState(Workspace.from_path(workspace))
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
    provider = _RuntimeProvider(
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
    runtime = _runtime(
        agent_home,
        workspace,
        provider,
        schedule_clock=clock,
        config_text=config_text,
    )
    try:
        await runtime.start()
        await _wait_until(
            lambda: (
                len(
                    [
                        request
                        for request in provider.complete_requests
                        if request.route == "schedule"
                    ]
                )
                == 1
                and runtime.schedule_service.status_snapshot().active_job_count == 0
            )
        )
        summaries = WorkspaceJsonlSummaryStore(state)
        assert [entry.content for entry in await summaries.after(0, 10)] == [
            "Schedule history summary."
        ]

        dream = await runtime.management_dispatcher.dispatch("/dream")
        assert dream.output is not None
        assert await WorkspaceFileMemoryStore(state).read_summary_cursor() == 1
        assert memory_path.read_text(encoding="utf-8") == new_memory

        clock.advance(60)
        await _wait_until(
            lambda: (
                len(
                    [
                        request
                        for request in provider.complete_requests
                        if request.route == "schedule"
                    ]
                )
                == 2
                and runtime.schedule_service.status_snapshot().active_job_count == 0
            )
        )
        schedule_requests = [
            request for request in provider.complete_requests if request.route == "schedule"
        ]
        assert "Fresh schedule preference." not in schedule_requests[0].system_prompt
        assert "Fresh schedule preference." in schedule_requests[1].system_prompt
        assert [entry.content for entry in await summaries.after(0, 10)] == [
            "Schedule history summary.",
            "Second schedule history summary.",
        ]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_confirmation_cancellation_preserves_declined_and_accepted_boundaries(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(
        chat_responses=(
            _schedule_tool_response(
                "call_waiting",
                {"action": "add", "message": "never write", "every_seconds": 60},
            ),
        )
    )
    runtime = _runtime(
        agent_home,
        workspace,
        provider,
        schedule_clock=_BlockingClock(NOW),
    )
    await runtime.start()
    try:
        events = _event_stream(runtime, "Create a job and wait for me.")
        assert (await _next_event(events, "waiting confirmation")).type == "turn_started"
        assert (await _next_event(events, "starting schedule tool")).type == "tool_started"
        assert (
            await _next_event(events, "confirmation requested")
        ).type == "confirmation_requested"
        await runtime.conversation.cancel_active_turn()
        assert (await _next_event(events, "first cancellation")).type == "turn_cancelled"
        await events.aclose()
        assert await _schedule_state(workspace).snapshot() == ()
    finally:
        await runtime.close()

    blocking_provider = _RuntimeProvider(
        chat_responses=(
            _schedule_tool_response(
                "call_accepted",
                {"action": "add", "message": "accepted before cancel", "every_seconds": 60},
            ),
        ),
        block_chat_call=2,
    )
    accepted_workspace = workspace / "accepted"
    accepted_workspace.mkdir()
    second_runtime = _runtime(
        agent_home,
        accepted_workspace,
        blocking_provider,
        schedule_clock=_BlockingClock(NOW),
    )
    await second_runtime.start()
    try:
        confirmation_seen = asyncio.Event()
        tool_completed = asyncio.Event()
        observed: list[AgentEvent] = []

        async def consume() -> None:
            async for event in _event_stream(second_runtime, "Create then cancel after approval."):
                observed.append(event)
                if event.type == "confirmation_requested":
                    payload = cast(ConfirmationRequestedPayload, event.payload)
                    second_runtime.conversation.respond_to_confirmation(
                        payload.confirmation_id, "approved"
                    )
                    confirmation_seen.set()
                elif event.type == "tool_completed":
                    tool_completed.set()

        consumer = asyncio.create_task(consume())
        await confirmation_seen.wait()
        await tool_completed.wait()
        await blocking_provider.chat_block_started.wait()
        await second_runtime.conversation.cancel_active_turn()
        await consumer
        assert [event.type for event in observed][-1] == "turn_cancelled"
        accepted_jobs = await _schedule_state(accepted_workspace).snapshot()
        assert len(accepted_jobs) == 1
        assert accepted_jobs[0].message == "accepted before cancel"
    finally:
        blocking_provider.release_chat.set()
        await second_runtime.close()


@pytest.mark.asyncio
async def test_runtime_dispatcher_wakes_for_due_at_job_and_keeps_schedule_session_out_of_resume(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider(
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
    runtime = _runtime(agent_home, workspace, provider, schedule_clock=clock)
    await runtime.start()
    try:
        events = await _submit_turn(runtime, "Schedule this due task.", decision="approved")
        assert [event.type for event in events][-1] == "turn_completed"
        await _wait_until(
            lambda: (
                len(provider.complete_requests) == 1
                and runtime.schedule_service.status_snapshot().active_job_count == 0
            )
        )

        assert provider.complete_requests[0].route == "schedule"
        assert provider.complete_requests[0].stream is False
        assert await _schedule_state(workspace).snapshot() == ()
        schedule_session_id = f"schedule_{JOB_UUID}"
        schedule_session = Session.load(
            WorkspaceState(Workspace.from_path(workspace)),
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

        resume = await runtime.management_dispatcher.dispatch("/resume")
        assert resume.output is not None
        assert schedule_session_id not in resume.output
        assert all(cast(str, event.type) != "background_completed" for event in events)
    finally:
        await runtime.close()
