from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import pytest

from myclaw.agent.blackboard import Blackboard, TaskFramingEvaluator
from myclaw.agent.loop import AgentLoop
from myclaw.agent.message_bus import InboundMessage
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleJobExecutionError, ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.tools.base import BaseTool, OpenAIToolSchema
from myclaw.tools.core.schedule import ScheduleTool
from myclaw.tools.tool_gateway import ModelToolCall, ToolResult
from tests.fixtures import DeterministicTaskFramingEvaluator

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
JOB_ID = UUID("550e8400-e29b-41d4-a716-446655440000")


class _ScheduleRouter:
    def __init__(self, *outcomes: ModelResponse | BaseException) -> None:
        self.routes: list[str] = []
        self.requests: list[tuple[list[dict[str, Any]], int]] = []
        self._outcomes = list(outcomes)

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del tools, continuation

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            self.routes.append(route)
            self.requests.append((list(messages), 0))
            yield ModelCompleted(response=self._response())

        return replay()

    async def complete(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del continuation
        self.routes.append(route)
        self.requests.append((list(messages), len(tools)))
        outcome = self._outcomes.pop(0) if self._outcomes else self._response()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @staticmethod
    def _response() -> ModelResponse:
        return ModelResponse(
            message=AssistantModelMessage(content="scheduled result", tool_calls=()),
            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
            finish_reason="stop",
        )


class _MaxScheduleRouter(_ScheduleRouter):
    async def complete(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del continuation
        self.routes.append(route)
        self.requests.append((list(messages), len(tools)))
        return ModelResponse(
            message=AssistantModelMessage(
                content="",
                tool_calls=(
                    ModelToolCall(
                        id=f"max-{len(self.requests)}",
                        name="unknown_tool",
                        arguments="{}",
                    ),
                ),
            ),
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            finish_reason="tool_calls",
        )


class _OverlapRouter(_ScheduleRouter):
    def __init__(self) -> None:
        super().__init__()
        self.foreground_started = asyncio.Event()
        self.foreground_release = asyncio.Event()
        self.schedule_started = asyncio.Event()
        self.schedule_release = asyncio.Event()

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, messages, tools, continuation

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            self.foreground_started.set()
            await self.foreground_release.wait()
            yield ModelCompleted(response=self._response())

        return replay()

    async def complete(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del route, messages, tools, continuation
        self.schedule_started.set()
        await self.schedule_release.wait()
        return self._response()


class _ScheduleToolOverlapRouter(_ScheduleRouter):
    def __init__(self, service: ScheduleService) -> None:
        super().__init__()
        self._service = service
        self._foreground_calls = 0
        self._schedule_calls = 0
        self.schedule_started = asyncio.Event()
        self.schedule_release = asyncio.Event()

    def stream(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, messages, tools, continuation

        async def replay() -> AsyncIterator[ModelStreamEvent]:
            self._foreground_calls += 1
            response = (
                _tool_response(
                    call_id="foreground_add",
                    name="schedule",
                    arguments={
                        "action": "add",
                        "message": "foreground add",
                        "every_seconds": 60,
                    },
                )
                if self._foreground_calls == 1
                else self._response()
            )
            yield ModelCompleted(response=response)

        return replay()

    async def complete(
        self,
        route: Literal["chat", "schedule"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        del route, messages, tools, continuation
        self._schedule_calls += 1
        if self._schedule_calls == 1:
            self.schedule_started.set()
            await self.schedule_release.wait()
            return _tool_response(
                call_id="scheduled_add",
                name="schedule",
                arguments={
                    "action": "add",
                    "message": "recursive add",
                    "every_seconds": 60,
                },
            )
        if self._schedule_calls == 2:
            return _tool_response(
                call_id="scheduled_list",
                name="schedule",
                arguments={"action": "list"},
            )
        if self._schedule_calls == 3:
            jobs = await self._service.public_snapshot()
            assert len(jobs) == 1
            return _tool_response(
                call_id="scheduled_remove",
                name="schedule",
                arguments={"action": "remove", "job_id": jobs[0].job_id},
            )
        return self._response()


async def _schedule_context(
    session: Session,
    current_user: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "schedule system"},
        {"role": "user", "content": current_user["content"]},
    ]


def _job(*, job_id: UUID = JOB_ID) -> ScheduleJob:
    return ScheduleJob(
        job_id=str(job_id),
        message="run the scheduled task",
        schedule=JobSchedule.at("2026-08-21T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )


def _loop(
    tmp_path: Path,
    router: _ScheduleRouter,
    *,
    skill_root: Path | None = None,
    schedule_context_preparer: Callable[
        [Session, dict[str, Any]],
        Awaitable[list[dict[str, Any]]],
    ] = _schedule_context,
    externalize_result_for: Callable[[Session], Callable[[ToolResult], ToolResult]] | None = None,
    task_framer: TaskFramingEvaluator | None = None,
) -> tuple[AgentLoop, WorkspaceState, ScheduleService]:
    workspace = Workspace.from_path(tmp_path / "workspace")
    workspace.path.mkdir(parents=True)
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=tmp_path / "agent-home")
    foreground = Session.create(state, now=lambda: NOW)
    service = ScheduleService(store=WorkspaceScheduleStore(state), clock=_Clock())
    selected_task_framer = (
        DeterministicTaskFramingEvaluator() if task_framer is None else task_framer
    )
    loop = AgentLoop(
        workspace=workspace,
        skill_root=skill_root,
        session=foreground,
        schedule_service=service,
        model_router=router,
        context_preparer=_foreground_context,
        task_framer=selected_task_framer,
        schedule_context_preparer=schedule_context_preparer,
        now=lambda: NOW,
        max_iterations=50,
        externalize_result_for=externalize_result_for,
    )
    return loop, state, service


async def _foreground_context(
    session: Session,
    current_user: dict[str, Any],
    blackboard: Blackboard | None = None,
) -> list[dict[str, Any]]:
    assert blackboard is None
    del session, current_user
    return [{"role": "system", "content": "foreground system"}]


class _Clock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds
        await asyncio.Event().wait()


async def _assert_no_outbound(loop: AgentLoop) -> None:
    outbound = asyncio.create_task(loop.bus.get_outbound())
    done, _ = await asyncio.wait((outbound,), timeout=0)
    assert not done
    outbound.cancel()
    await asyncio.gather(outbound, return_exceptions=True)


@pytest.mark.asyncio
async def test_schedule_run_uses_schedule_session_and_keeps_foreground_bus_empty(
    tmp_path: Path,
) -> None:
    router = _ScheduleRouter()
    framer = DeterministicTaskFramingEvaluator()
    loop, state, _ = _loop(tmp_path, router, task_framer=framer)

    outbound = asyncio.create_task(loop.bus.get_outbound())
    await loop.run_schedule_job(_job())

    done, _ = await asyncio.wait((outbound,), timeout=0)
    assert not done
    outbound.cancel()
    await asyncio.gather(outbound, return_exceptions=True)

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in schedule_session.messages] == [
        "user",
        "assistant",
    ]
    assert schedule_session.messages[-1]["content"] == "scheduled result"
    assert router.routes == ["schedule"]
    assert router.requests[0][1] == 10
    assert framer.calls == 0


@pytest.mark.asyncio
async def test_schedule_run_reloads_canonical_session_and_closes_each_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _ScheduleRouter()
    observed_context: list[tuple[str, int]] = []

    async def prepare_context(
        session: Session,
        current_user: dict[str, Any],
    ) -> list[dict[str, Any]]:
        observed_context.append((session.session_id, len(session.messages)))
        return [
            {"role": "system", "content": "schedule system"},
            {"role": "user", "content": current_user["content"]},
        ]

    loop, state, _ = _loop(
        tmp_path,
        router,
        schedule_context_preparer=prepare_context,
    )
    persisted: list[str] = []
    closed: list[str] = []
    original_persist = Session.persist
    original_close = Session.close

    def record_persist(session: Session) -> None:
        persisted.append(session.session_id)
        original_persist(session)

    def record_close(session: Session) -> None:
        closed.append(session.session_id)
        original_close(session)

    monkeypatch.setattr(Session, "persist", record_persist)
    monkeypatch.setattr(Session, "close", record_close)

    await loop.run_schedule_job(_job())
    await loop.run_schedule_job(_job())

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert observed_context == [(f"schedule_{JOB_ID}", 0), (f"schedule_{JOB_ID}", 2)]
    assert persisted == [f"schedule_{JOB_ID}", f"schedule_{JOB_ID}"]
    assert closed == [f"schedule_{JOB_ID}", f"schedule_{JOB_ID}"]
    assert [message["role"] for message in schedule_session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_schedule_failed_runner_persists_safe_error_and_maps_job_failure(
    tmp_path: Path,
) -> None:
    router = _ScheduleRouter(ModelCallError(ErrorInfo("provider_unavailable", "safe failure")))
    loop, state, _ = _loop(tmp_path, router)

    with pytest.raises(ScheduleJobExecutionError) as raised:
        await loop.run_schedule_job(_job())

    assert raised.value.error == ErrorInfo("provider_unavailable", "safe failure")
    await _assert_no_outbound(loop)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert schedule_session.messages[-1]["role"] == "assistant"
    assert schedule_session.messages[-1]["status"] == "error"
    assert schedule_session.messages[-1]["error"] == {
        "code": "provider_unavailable",
        "message": "safe failure",
    }


@pytest.mark.asyncio
async def test_schedule_max_iterations_finishes_tools_without_a_fifty_first_model_call(
    tmp_path: Path,
) -> None:
    router = _MaxScheduleRouter()
    loop, state, _ = _loop(tmp_path, router)

    with pytest.raises(ScheduleJobExecutionError) as raised:
        await loop.run_schedule_job(_job())

    assert raised.value.error.code == "agent_iteration_limit"
    assert len(router.requests) == 50
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert schedule_session.messages[-1]["error"]["code"] == "agent_iteration_limit"


@pytest.mark.asyncio
async def test_schedule_cancelled_runner_persists_user_and_propagates_cancelled_error(
    tmp_path: Path,
) -> None:
    router = _ScheduleRouter()
    loop, state, service = _loop(tmp_path, router)
    await service.close()

    with pytest.raises(asyncio.CancelledError):
        await loop.run_schedule_job(_job())

    await _assert_no_outbound(loop)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in schedule_session.messages] == ["user"]
    assert router.requests == []


@pytest.mark.asyncio
async def test_schedule_context_preparation_failures_reset_contextvar_and_preserve_cancel(
    tmp_path: Path,
) -> None:
    from myclaw.tools.core.schedule import ScheduleTool

    async def unexpected_context(
        session: Session,
        current_user: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del session, current_user
        raise RuntimeError("unexpected preparation failure")

    async def cancelled_context(
        session: Session,
        current_user: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del session, current_user
        raise asyncio.CancelledError()

    success_loop, _, _ = _loop(tmp_path / "success", _ScheduleRouter())
    await success_loop.run_schedule_job(_job())
    assert ScheduleTool._in_schedule_job.get() is False

    failed_loop, failed_state, _ = _loop(
        tmp_path / "failed",
        _ScheduleRouter(),
        schedule_context_preparer=unexpected_context,
    )
    with pytest.raises(ScheduleJobExecutionError) as failed:
        await failed_loop.run_schedule_job(_job())
    assert failed.value.error.code == "model_failed"
    assert ScheduleTool._in_schedule_job.get() is False

    cancelled_loop, cancelled_state, _ = _loop(
        tmp_path / "cancelled",
        _ScheduleRouter(),
        schedule_context_preparer=cancelled_context,
    )
    with pytest.raises(asyncio.CancelledError):
        await cancelled_loop.run_schedule_job(_job())
    assert ScheduleTool._in_schedule_job.get() is False

    failed_session = Session.load(
        failed_state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    cancelled_session = Session.load(
        cancelled_state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in failed_session.messages] == ["user", "assistant"]
    assert [message["role"] for message in cancelled_session.messages] == ["user"]


def _tool_response(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, object],
) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(
            content="",
            tool_calls=(
                ModelToolCall(
                    id=call_id,
                    name=name,
                    arguments=json.dumps(arguments, separators=(",", ":")),
                ),
            ),
        ),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="tool_calls",
    )


@pytest.mark.asyncio
async def test_schedule_confirmation_required_tool_is_refused_without_foreground_control(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    router = _ScheduleRouter(
        _tool_response(
            call_id="call_external",
            name="read_file",
            arguments={"path": str(outside)},
        ),
        _ScheduleRouter._response(),
    )
    loop, state, _ = _loop(tmp_path, router)
    confirmation_requests: list[object] = []
    loop.bind_confirmation_callback(confirmation_requests.append)

    await loop.run_schedule_job(_job())

    await _assert_no_outbound(loop)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    tool_message = next(
        message for message in schedule_session.messages if message["role"] == "tool"
    )
    assert tool_message["status"] == "refused"
    assert tool_message["confirmation"]["decision"] is None
    assert confirmation_requests == []
    assert loop.has_pending_confirmation is False


@pytest.mark.asyncio
async def test_schedule_agent_reads_known_skill_path_via_shared_gateway(tmp_path: Path) -> None:
    skill_root = tmp_path / "agent-home" / "skills"
    skill_file = skill_root / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_bytes(b"---\nname: review\n---\nbody\n")
    router = _ScheduleRouter(
        _tool_response(
            call_id="scheduled_skill_read",
            name="read_file",
            arguments={"path": str(skill_file)},
        ),
        _ScheduleRouter._response(),
    )
    loop, state, _ = _loop(tmp_path, router, skill_root=skill_root)
    confirmation_requests: list[object] = []
    loop.bind_confirmation_callback(confirmation_requests.append)

    await loop.run_schedule_job(_job())

    await _assert_no_outbound(loop)
    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    tool_messages = [message for message in schedule_session.messages if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["status"] == "success"
    assert tool_messages[0]["content"] == "---\nname: review\n---\nbody\n"
    assert confirmation_requests == []
    assert loop.has_pending_confirmation is False
    assert router.routes == ["schedule", "schedule"]
    assert [tool_count for _, tool_count in router.requests] == [10, 10]


@pytest.mark.asyncio
async def test_schedule_oversized_result_uses_canonical_schedule_artifact_session(
    tmp_path: Path,
) -> None:
    def externalizer_for(session: Session) -> Callable[[ToolResult], ToolResult]:
        def externalize(result: ToolResult) -> ToolResult:
            if result.status != "success":
                return result
            output = BaseTool.handle_result(
                result.content,
                workspace=session.workspace_state.workspace,
                session_id=session.session_id,
                tool_call_id=result.tool_call_id,
                limit=4,
            )
            return replace(result, content=output.content, artifact=output.artifact)

        return externalize

    router = _ScheduleRouter(
        _tool_response(
            call_id="call_artifact",
            name="read_file",
            arguments={"path": "large.txt", "limit": 10000},
        ),
        _ScheduleRouter._response(),
    )
    loop, state, _ = _loop(
        tmp_path,
        router,
        externalize_result_for=externalizer_for,
    )
    (state.workspace.path / "large.txt").write_text("x\n" * 4000, encoding="utf-8")

    await loop.run_schedule_job(_job())

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    tool_message = next(
        message for message in schedule_session.messages if message["role"] == "tool"
    )
    artifact = tool_message["artifact"]
    assert artifact["path"].startswith(f".myclaw/artifacts/schedule_{JOB_ID}/")
    assert (state.workspace.path / artifact["path"]).exists()


@pytest.mark.asyncio
async def test_foreground_cancel_does_not_cancel_overlapping_schedule_run(
    tmp_path: Path,
) -> None:
    router = _OverlapRouter()
    loop, state, _ = _loop(tmp_path, router)
    await loop.start()
    await loop.bus.put_inbound(InboundMessage("foreground input"))
    await router.foreground_started.wait()

    schedule_task = asyncio.create_task(loop.run_schedule_job(_job()))
    await router.schedule_started.wait()
    await loop.cancel_active_run()
    assert not schedule_task.done()

    foreground_terminal = await loop.bus.get_outbound()
    assert foreground_terminal.type == "system_control"
    assert foreground_terminal.metadata["finish_reason"] == "cancelled"

    router.schedule_release.set()
    await schedule_task
    await loop.close()

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert schedule_session.messages[-1]["content"] == "scheduled result"


@pytest.mark.asyncio
async def test_schedule_contextvar_refuses_only_scheduled_add_on_shared_gateway(
    tmp_path: Path,
) -> None:
    workspace = Workspace.from_path(tmp_path / "workspace")
    workspace.path.mkdir(parents=True)
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=tmp_path / "agent-home")
    foreground = Session.create(state, now=lambda: NOW)
    service = ScheduleService(store=WorkspaceScheduleStore(state), clock=_Clock())
    router = _ScheduleToolOverlapRouter(service)
    loop = AgentLoop(
        workspace=workspace,
        session=foreground,
        schedule_service=service,
        model_router=router,
        context_preparer=_foreground_context,
        task_framer=DeterministicTaskFramingEvaluator(),
        schedule_context_preparer=_schedule_context,
        now=lambda: NOW,
        max_iterations=50,
    )
    await loop.start()
    schedule_task = asyncio.create_task(loop.run_schedule_job(_job()))
    try:
        await router.schedule_started.wait()
        assert not schedule_task.done()

        await loop.bus.put_inbound(InboundMessage("create a foreground Schedule Job"))
        while True:
            outbound = await loop.bus.get_outbound()
            if outbound.metadata.get("_streamed") is True:
                break

        foreground_jobs = await service.public_snapshot()
        assert len(foreground_jobs) == 1
        assert foreground_jobs[0].message == "foreground add"
        assert not schedule_task.done()

        schedule_outbound = asyncio.create_task(loop.bus.get_outbound())
        router.schedule_release.set()
        await schedule_task
        done, _ = await asyncio.wait((schedule_outbound,), timeout=0)
        assert not done
        schedule_outbound.cancel()
        await asyncio.gather(schedule_outbound, return_exceptions=True)
    finally:
        router.schedule_release.set()
        await asyncio.gather(schedule_task, return_exceptions=True)
        await loop.close()

    assert await service.public_snapshot() == ()
    assert ScheduleTool._in_schedule_job.get() is False

    schedule_session = Session.load(
        state,
        f"schedule_{JOB_ID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    scheduled_tools = [
        message for message in schedule_session.messages if message["role"] == "tool"
    ]
    assert [message["status"] for message in scheduled_tools] == [
        "refused",
        "success",
        "success",
    ]
    assert json.loads(scheduled_tools[1]["content"])["jobs"][0]["message"] == "foreground add"
    assert json.loads(scheduled_tools[2]["content"])["action"] == "remove"
