"""Serial foreground Agent Runner orchestration over the Runtime Message Bus."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn, Protocol
from uuid import UUID

from loguru import logger

from myclaw.agent.message_bus import (
    InboundMessage,
    MessageBus,
    OutboundMessage,
    OutboundMessageType,
)
from myclaw.agent.runner import (
    AgentRunner,
    AgentRunnerResponseSegmentEnd,
    AgentRunnerResult,
    AgentRunnerRouter,
    AgentRunnerToolCallStarted,
    _build_assistant_repair_message,
)
from myclaw.agent.workspace import Workspace
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouteStatus
from myclaw.provider.models import ModelCompleted, ReasoningDelta, TextDelta
from myclaw.schedule.model import ScheduleJob
from myclaw.schedule.service import ScheduleJobExecutionError, ScheduleService
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.core.schedule import ScheduleTool
from myclaw.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ToolGateway,
    ToolResult,
)

type ForegroundContextPreparer = Callable[
    [Session, dict[str, Any]],
    Awaitable[list[dict[str, Any]]],
]
type ScheduleContextPreparer = Callable[
    [Session, dict[str, Any]],
    Awaitable[list[dict[str, Any]]],
]
type ResultExternalizerFactory = Callable[[Session], Callable[[ToolResult], ToolResult]]


class ConfirmationRequestView(Protocol):
    """Stable confirmation data exposed to a foreground control consumer."""

    @property
    def confirmation_id(self) -> UUID: ...

    @property
    def tool_call_id(self) -> str: ...

    @property
    def tool_name(self) -> str: ...

    @property
    def reason(self) -> str: ...

    @property
    def summary(self) -> str: ...

    @property
    def details(self) -> dict[str, Any]: ...

    @property
    def warnings(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class ForegroundConversationProjection:
    """Presentation-safe snapshot of the active foreground conversation."""

    session_id: str
    messages: tuple[dict[str, Any], ...]


class AgentLoopControl(Protocol):
    """The independent foreground control surface owned by AgentLoop."""

    @property
    def has_active_run(self) -> bool: ...

    @property
    def has_pending_confirmation(self) -> bool: ...

    async def cancel_active_run(self) -> None: ...

    def bind_confirmation_callback(self, callback: ConfirmationCallback) -> None: ...

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None: ...


class TerminalAgentLoopControl(AgentLoopControl, Protocol):
    """Foreground control surface including Terminal history projection."""

    def project_foreground_conversation(self) -> ForegroundConversationProjection: ...


type ConfirmationCallback = Callable[[ConfirmationRequestView], None]

_CANCELLED_MESSAGE = "MyClaw 已取消本轮对话。"


@dataclass(slots=True)
class _PendingConfirmation:
    request: ConfirmationRequest
    future: asyncio.Future[ConfirmationDecision]


@dataclass(slots=True)
class _TitleCoordination:
    preparation_started: asyncio.Event
    prepared: asyncio.Future[bool]
    log_ready: asyncio.Event
    foreground_idle: asyncio.Event
    active_foregrounds: int = 0

    def attach_foreground(self) -> None:
        self.active_foregrounds += 1
        self.foreground_idle.clear()

    def release_foreground(self) -> None:
        self.active_foregrounds -= 1
        if self.active_foregrounds == 0:
            self.foreground_idle.set()

    async def wait_until_foreground_idle(self) -> None:
        while True:
            await self.foreground_idle.wait()
            await asyncio.sleep(0)
            if self.active_foregrounds == 0:
                return


@dataclass(frozen=True, slots=True)
class _TitleWork:
    task: asyncio.Task[None]
    coordination: _TitleCoordination


class AgentLoop:
    """Own the complete serial foreground execution path."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        session: Session,
        schedule_service: ScheduleService,
        model_router: AgentRunnerRouter,
        context_preparer: ForegroundContextPreparer,
        now: Callable[[], datetime],
        max_iterations: int,
        schedule_context_preparer: ScheduleContextPreparer | None = None,
        schedule_now: Callable[[], datetime] | None = None,
        title_prompt: str | None = None,
        externalize_result_for: ResultExternalizerFactory | None = None,
    ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("Agent Loop requires a Workspace")
        if not isinstance(session, Session):
            raise TypeError("Agent Loop requires a foreground Session")
        if not isinstance(schedule_service, ScheduleService):
            raise TypeError("Agent Loop requires a Schedule Service for the foreground catalog")
        if not callable(context_preparer):
            raise TypeError("Agent Loop requires a context preparer")
        if not callable(now):
            raise TypeError("Agent Loop requires a clock")
        if schedule_now is not None and not callable(schedule_now):
            raise TypeError("Agent Loop Schedule clock must be callable")

        self._session = session
        self._schedule_service = schedule_service
        self._context_preparer = context_preparer
        self._schedule_context_preparer = schedule_context_preparer
        self._now = now
        self._schedule_now = now if schedule_now is None else schedule_now
        self._title_prompt = title_prompt
        self._externalize_result_for = externalize_result_for
        self._tool_gateway = ToolGateway(
            workspace=workspace,
            schedule_service=schedule_service,
        )
        self._model_router = model_router
        self._runner = AgentRunner(model_router)
        self._max_iterations = max_iterations
        self._bus = MessageBus()
        self._consumer_task: asyncio.Task[None] | None = None
        self._execution_task: asyncio.Task[None] | None = None
        self._aborted_tasks: set[asyncio.Task[Any]] = set()
        self._execution_ready: asyncio.Event | None = None
        self._title_work: dict[str, _TitleWork] = {}
        self._pending_confirmation: _PendingConfirmation | None = None
        self._confirmation_callback: ConfirmationCallback | None = None
        self._cancel_requested = False
        self._closing = False
        self._closed = False
        self._aborted = False
        self._last_foreground_route_status: ModelRouteStatus | None = None

    @property
    def bus(self) -> MessageBus:
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        return self._bus

    @property
    def control(self) -> TerminalAgentLoopControl:
        return self

    @property
    def last_foreground_route_status(self) -> ModelRouteStatus | None:
        return self._last_foreground_route_status

    @property
    def session(self) -> Session:
        return self._session

    @property
    def tool_schemas(self) -> tuple[OpenAIToolSchema, ...]:
        return tuple(self._tool_gateway.schemas)

    @property
    def has_active_run(self) -> bool:
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        task = self._execution_task
        return task is not None and not task.done()

    @property
    def has_pending_confirmation(self) -> bool:
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        pending = self._pending_confirmation
        return pending is not None and not pending.future.done()

    def project_foreground_conversation(self) -> ForegroundConversationProjection:
        """Return presentation data without exposing the owned Session."""
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        return ForegroundConversationProjection(
            session_id=self._session.session_id,
            messages=tuple(deepcopy(message) for message in self._session.messages),
        )

    def bind_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        """Bind the synchronous foreground confirmation callback exactly once."""
        if self._confirmation_callback is not None:
            raise RuntimeError("Agent Loop confirmation callback is already bound")
        if self._closed or self._aborted:
            raise RuntimeError("Agent Loop is closed")
        if not callable(callback):
            raise TypeError("confirmation callback must be callable")
        self._confirmation_callback = callback

    async def start(self) -> None:
        self._prepare_start()
        self._activate_prepared()

    def _prepare_start(self) -> None:
        """Validate activation without creating the foreground consumer task."""
        if self._closed or self._aborted:
            raise RuntimeError("Agent Loop is closed")

    def _activate_prepared(self) -> None:
        """Activate a preflighted Loop using only infallible task creation."""
        if self._consumer_task is not None:
            return
        self._consumer_task = asyncio.create_task(self._consume_foreground())

    async def close(self) -> None:
        if self._aborted:
            return
        if self._closed:
            return
        self._closing = True
        self._cancel_pending_confirmation()
        active = self._execution_task
        if active is not None and not active.done():
            await self.cancel_active_run()

        consumer = self._consumer_task
        if consumer is not None and not consumer.done():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        self._consumer_task = None

        title_work = tuple(self._title_work.values())
        for work in title_work:
            if not work.task.done():
                work.task.cancel()
        if title_work:
            await asyncio.gather(*(work.task for work in title_work), return_exceptions=True)
        self._title_work.clear()
        self._closed = True

    def abort(self) -> None:
        """Synchronously detach this generation without persistence or repair."""
        if self._aborted:
            return
        self._aborted = True
        self._closing = True
        self._cancel_pending_confirmation()
        self._confirmation_callback = None
        self._bus._detach_inbound()

        for task in (self._consumer_task, self._execution_task):
            self._retain_aborted_task(task)
        for work in tuple(self._title_work.values()):
            self._retain_aborted_task(work.task)
        self._consumer_task = None
        self._execution_task = None
        self._execution_ready = None
        self._title_work.clear()

        try:
            self._session.abandon()
        except BaseException:
            pass
        self._closed = True

    def _retain_aborted_task(self, task: asyncio.Task[Any] | None) -> None:
        if task is None or task.done():
            return
        self._aborted_tasks.add(task)
        task.add_done_callback(self._aborted_task_finished)
        task.cancel()

    def _aborted_task_finished(self, task: asyncio.Task[Any]) -> None:
        self._aborted_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except BaseException as error:
            logger.warning(
                "Aborted Agent Loop task failed type={}",
                type(error).__name__,
            )

    def _close_sessions(self) -> None:
        """Close the foreground Session during normal awaited Runtime shutdown."""
        try:
            self._session.close()
        except BaseException:
            pass

    async def cancel_active_run(self) -> None:
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        active = self._execution_task
        if active is None or active.done():
            return
        self._cancel_requested = True
        self._cancel_pending_confirmation()
        ready = self._execution_ready
        if ready is not None and not ready.is_set():
            await ready.wait()
        await asyncio.sleep(0)
        if active.done():
            return
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)

    async def run_schedule_job(self, job: ScheduleJob) -> None:
        """Execute one Schedule Job without using foreground state or output."""
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        token = ScheduleTool._in_schedule_job.set(True)
        try:
            await self._execute_schedule_job(job)
        finally:
            ScheduleTool._in_schedule_job.reset(token)

    async def _execute_schedule_job(self, job: ScheduleJob) -> None:
        schedule_session: Session | None = None
        workspace_state = self._session.workspace_state
        with session_log(workspace_state, job.session_id):
            try:
                try:
                    schedule_session = Session.load(
                        workspace_state,
                        job.session_id,
                        partition=SessionStoragePartition.SCHEDULE,
                        now=self._schedule_now,
                    )
                except FileNotFoundError:
                    schedule_session = Session.create_schedule(
                        workspace_state,
                        job.job_id,
                        now=self._schedule_now,
                    )
                try:
                    await self._run_schedule_agent(schedule_session, job)
                except ScheduleJobExecutionError as failure:
                    logger.warning(
                        "Schedule Job failed job_id={} kind={} code={}",
                        job.job_id,
                        job.schedule.kind,
                        failure.error.code,
                    )
                    raise
            finally:
                if schedule_session is not None:
                    try:
                        if self._aborted:
                            schedule_session.abandon()
                        else:
                            schedule_session.close()
                    except Exception as error:
                        logger.error(
                            "Schedule Session close failed job_id={} type={}",
                            job.job_id,
                            type(error).__name__,
                        )

    async def _run_schedule_agent(self, session: Session, job: ScheduleJob) -> None:
        current_user = {"role": "user", "content": job.message}
        context_preparer = self._schedule_context_preparer
        if context_preparer is None:
            self._persist_schedule_failure(
                session,
                current_user,
                ErrorInfo("model_failed", "The model request failed."),
            )
        try:
            initial_messages = await context_preparer(session, deepcopy(current_user))
        except asyncio.CancelledError:
            if not self._aborted:
                self._persist_schedule_cancelled_user(session, current_user, job)
            raise
        except ModelCallError as failure:
            self._persist_schedule_failure(session, current_user, failure.error)
        except Exception:
            self._persist_schedule_failure(
                session,
                current_user,
                ErrorInfo("model_failed", "The model request failed."),
            )

        result = await self._runner.run(
            initial_messages,
            model="schedule",
            tool_gateway=self._tool_gateway,
            on_output=_discard_runner_output,
            confirmation=None,
            externalize_result=self._result_externalizer_for(session),
            cancel_requested=self._schedule_service.cancellation_requested,
            max_iterations=self._max_iterations,
        )
        session.append_messages([deepcopy(current_user), *deepcopy(result.messages)])
        session.persist()

        if result.finish_reason == "cancelled":
            raise asyncio.CancelledError()
        if result.finish_reason != "completed":
            error = result.error or ErrorInfo("model_failed", "The model request failed.")
            raise ScheduleJobExecutionError(error)

    @staticmethod
    def _persist_schedule_cancelled_user(
        session: Session,
        current_user: dict[str, Any],
        job: ScheduleJob,
    ) -> None:
        try:
            session.append_messages([current_user])
            session.persist()
        except Exception as error:
            logger.error(
                "Schedule cancellation persistence failed job_id={} type={}",
                job.job_id,
                type(error).__name__,
            )

    @staticmethod
    def _persist_schedule_failure(
        session: Session,
        current_user: dict[str, Any],
        error: ErrorInfo,
    ) -> NoReturn:
        session.append_messages(
            [
                deepcopy(current_user),
                _build_assistant_repair_message(
                    content="",
                    status="error",
                    error=error,
                ),
            ]
        )
        session.persist()
        raise ScheduleJobExecutionError(error)

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        if self._aborted:
            raise ValueError("Confirmation response is late or unknown")
        if decision not in {"approved", "declined"}:
            raise ValueError("confirmation decision must be approved or declined")
        pending = self._pending_confirmation
        if pending is None or pending.request.confirmation_id != confirmation_id:
            raise ValueError("Confirmation response is late or unknown")
        if pending.future.done():
            raise ValueError("Confirmation response is late or unknown")
        pending.future.set_result(decision)

    async def _consume_foreground(self) -> None:
        try:
            while not self._closing:
                inbound = await self._bus.get_inbound()
                if self._closing:
                    break
                execution_ready = asyncio.Event()
                execution = asyncio.create_task(
                    self._execute_foreground(inbound, execution_ready=execution_ready)
                )
                self._execution_task = execution
                self._execution_ready = execution_ready
                try:
                    await execution
                except asyncio.CancelledError:
                    if not self._closing:
                        raise
                finally:
                    if self._execution_task is execution:
                        self._execution_task = None
                    if self._execution_ready is execution_ready:
                        self._execution_ready = None
                    self._cancel_requested = False
        except RuntimeError:
            if not self._closing:
                raise
        except asyncio.CancelledError:
            if not self._closing:
                raise

    async def _execute_foreground(
        self,
        inbound: InboundMessage,
        *,
        execution_ready: asyncio.Event,
    ) -> None:
        active_session = self._session
        start_title = not active_session.messages
        created_title_work = (
            self._start_title_if_needed(active_session, inbound.content) if start_title else None
        )
        title_work = created_title_work or self._title_work.get(active_session.session_id)
        if title_work is not None and title_work.task.done():
            title_work = None
        title_coordination = None if title_work is None else title_work.coordination
        if title_coordination is not None:
            title_coordination.attach_foreground()
        committed = False
        try:
            if title_work is None:
                with session_log(active_session):
                    committed = await self._execute_foreground_logged(
                        active_session,
                        inbound,
                        title_work=None,
                        execution_ready=execution_ready,
                    )
            else:
                assert title_coordination is not None
                await title_coordination.log_ready.wait()
                with logger.contextualize(session_id=active_session.session_id):
                    committed = await self._execute_foreground_logged(
                        active_session,
                        inbound,
                        title_work=title_work,
                        execution_ready=execution_ready,
                    )
        finally:
            execution_ready.set()
            if title_coordination is not None:
                title_coordination.release_foreground()
            if created_title_work is not None:
                created_coordination = created_title_work.coordination
                if not created_coordination.prepared.done():
                    created_coordination.prepared.set_result(False)
                if not committed:
                    if not created_title_work.task.done():
                        created_title_work.task.cancel()
                    await asyncio.gather(created_title_work.task, return_exceptions=True)
                    if self._title_work.get(active_session.session_id) is created_title_work:
                        self._title_work.pop(active_session.session_id)

    async def _execute_foreground_logged(
        self,
        active_session: Session,
        inbound: InboundMessage,
        *,
        title_work: _TitleWork | None,
        execution_ready: asyncio.Event,
    ) -> bool:
        current_user = {"role": "user", "content": inbound.content}
        if title_work is not None:
            title_work.coordination.preparation_started.set()
        execution_ready.set()
        if not inbound.content.strip():
            return False
        try:
            initial_messages = await self._context_preparer(
                active_session,
                deepcopy(current_user),
            )
        except asyncio.CancelledError:
            if not self._cancel_requested:
                raise
            await self._publish_preparation_failure(ErrorInfo("turn_cancelled", _CANCELLED_MESSAGE))
            return False
        except ModelCallError as failure:
            await self._publish_preparation_failure(failure.error)
            return False
        except Exception:
            await self._publish_preparation_failure(
                ErrorInfo("model_failed", "The model request failed.")
            )
            return False

        if title_work is not None and not title_work.coordination.prepared.done():
            title_work.coordination.prepared.set_result(True)
        try:
            result = await self._runner.run(
                initial_messages,
                model="chat",
                tool_gateway=self._tool_gateway,
                on_output=self._publish_runner_output,
                confirmation=self._request_confirmation,
                externalize_result=self._externalize_for_run(active_session),
                cancel_requested=lambda: self._cancel_requested,
                max_iterations=self._max_iterations,
            )
            self._remember_foreground_route_status()
        except asyncio.CancelledError:
            if not self._cancel_requested:
                raise
            await self._publish_preparation_failure(ErrorInfo("turn_cancelled", _CANCELLED_MESSAGE))
            return False

        try:
            active_session.append_messages([deepcopy(current_user), *deepcopy(result.messages)])
        except Exception as failure:
            _runtime_logger().opt(exception=failure).error(
                "Agent Run Session increment failed code=persistence_error type={}",
                type(failure).__name__,
            )
            await self._publish_commit_failure(result)
            return False
        try:
            active_session.persist()
        except Exception:
            pass
        await self._publish_terminal(result)
        return True

    def _remember_foreground_route_status(self) -> None:
        current_call_status = getattr(self._model_router, "current_call_status", None)
        if callable(current_call_status):
            status = current_call_status("chat")
            if isinstance(status, ModelRouteStatus):
                self._last_foreground_route_status = status
                return
        route_status = getattr(self._model_router, "route_status", None)
        if callable(route_status):
            status = route_status("chat")
            if isinstance(status, ModelRouteStatus):
                self._last_foreground_route_status = status

    def _externalize_for_run(
        self,
        active_session: Session,
    ) -> Callable[[ToolResult], ToolResult] | None:
        externalizer = self._result_externalizer_for(active_session)
        if externalizer is None:
            return None

        def observe(result: ToolResult) -> ToolResult:
            projected = result if externalizer is None else externalizer(result)
            return projected

        return observe

    def _result_externalizer_for(
        self,
        active_session: Session,
    ) -> Callable[[ToolResult], ToolResult] | None:
        if self._externalize_result_for is None:
            return None
        return self._externalize_result_for(active_session)

    async def _publish_runner_output(self, event: object) -> None:
        if isinstance(event, ReasoningDelta):
            await self._bus.put_outbound(
                OutboundMessage(
                    "model_reasoning",
                    event.delta,
                    {"_stream_delta": True},
                )
            )
            return
        if isinstance(event, TextDelta):
            await self._bus.put_outbound(
                OutboundMessage(
                    "model_response",
                    event.delta,
                    {"_stream_delta": True},
                )
            )
            return
        if isinstance(event, AgentRunnerResponseSegmentEnd):
            outbound_type: OutboundMessageType = (
                "model_reasoning" if event.segment == "reasoning" else "model_response"
            )
            await self._bus.put_outbound(OutboundMessage(outbound_type, "", {"_stream_end": True}))
            return
        if isinstance(event, AgentRunnerToolCallStarted):
            await self._bus.put_outbound(
                OutboundMessage(
                    "tool_call",
                    event.tool_name,
                    {
                        "tool_call_id": event.tool_call_id,
                        "arguments": event.arguments,
                    },
                )
            )
            return
        raise TypeError(f"Unsupported Agent Runner output: {type(event).__name__}")

    async def _publish_terminal(self, result: AgentRunnerResult) -> None:
        if result.finish_reason == "completed":
            await self._bus.put_outbound(OutboundMessage("model_response", "", {"_streamed": True}))
            return
        error = result.error
        if error is None:
            error = ErrorInfo("model_failed", "The model request failed.")
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                error.message,
                {
                    "finish_reason": result.finish_reason,
                    "error_code": error.code,
                    "_streamed": True,
                },
            )
        )

    async def _publish_preparation_failure(self, error: ErrorInfo) -> None:
        if error.code != "turn_cancelled":
            _log_agent_failure(error)
        finish_reason = "cancelled" if error.code == "turn_cancelled" else "failed"
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                error.message,
                {
                    "finish_reason": finish_reason,
                    "error_code": error.code,
                    "_streamed": True,
                },
            )
        )

    async def _publish_commit_failure(self, result: AgentRunnerResult) -> None:
        error = ErrorInfo(
            "persistence_error",
            "The Conversation Session could not be updated.",
        )
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                error.message,
                {
                    "finish_reason": "failed",
                    "error_code": error.code,
                    "_streamed": True,
                },
            )
        )

    async def _request_confirmation(
        self,
        request: ConfirmationRequest,
    ) -> ConfirmationDecision:
        if self._aborted:
            raise asyncio.CancelledError()
        if self._pending_confirmation is not None:
            raise RuntimeError("A foreground confirmation request is already pending")
        callback = self._confirmation_callback
        if callback is None:
            raise RuntimeError("Agent Loop confirmation callback is not bound")
        future: asyncio.Future[ConfirmationDecision] = asyncio.get_running_loop().create_future()
        pending = _PendingConfirmation(request=request, future=future)
        self._pending_confirmation = pending
        try:
            callback(request)
            return await future
        finally:
            if self._pending_confirmation is pending:
                self._pending_confirmation = None

    def _cancel_pending_confirmation(self) -> None:
        pending = self._pending_confirmation
        if pending is not None and not pending.future.done():
            pending.future.cancel()

    def _start_title_if_needed(
        self,
        session: Session,
        content: str,
    ) -> _TitleWork | None:
        if (
            self._title_prompt is None
            or not content.strip()
            or session.metadata.get("title") != "Untitled session"
            or session.session_id in self._title_work
        ):
            return None
        preparation_started = asyncio.Event()
        prepared: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        log_ready = asyncio.Event()
        foreground_idle = asyncio.Event()
        foreground_idle.set()
        coordination = _TitleCoordination(
            preparation_started=preparation_started,
            prepared=prepared,
            log_ready=log_ready,
            foreground_idle=foreground_idle,
        )
        task = asyncio.create_task(
            self._generate_title(
                session,
                content,
                coordination=coordination,
            )
        )
        work = _TitleWork(
            task=task,
            coordination=coordination,
        )
        self._title_work[session.session_id] = work
        task.add_done_callback(self._title_done)
        return work

    def _title_done(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as error:
            logger.opt(exception=error).error(
                "Session title task failed type={}", type(error).__name__
            )

    async def _generate_title(
        self,
        session: Session,
        content: str,
        *,
        coordination: _TitleCoordination,
    ) -> None:
        with session_log(session):
            coordination.log_ready.set()
            try:
                await coordination.preparation_started.wait()
                title, usage_delta = await self._resolve_title(content)
                if (
                    await coordination.prepared
                    and session.metadata.get("title") == "Untitled session"
                ):
                    session.update_metadata(title=title, usage_delta=usage_delta)
            except asyncio.CancelledError:
                if (
                    not self._aborted
                    and coordination.prepared.done()
                    and coordination.prepared.result()
                    and session.metadata.get("title") == "Untitled session"
                ):
                    session.update_metadata(title=Session._normalize_title(content))
                raise
            finally:
                await coordination.wait_until_foreground_idle()

    async def _resolve_title(self, content: str) -> tuple[str, dict[str, int] | None]:
        title = Session._normalize_title(content)
        usage_delta: dict[str, int] | None = None
        events: Any = None
        try:
            events = self._router_stream_title(content)
            async for event in events:
                if not isinstance(event, ModelCompleted):
                    continue
                response = event.response
                usage_delta = {"model_calls": 1, **response.usage.to_dict()}
                if response.message.tool_calls:
                    continue
                candidate = Session._normalize_title_candidate(response.message.content)
                if candidate:
                    title = candidate
                break
        except Exception as error:
            _runtime_logger().opt(exception=error).warning(
                "Session title fallback selected type={}", type(error).__name__
            )
        finally:
            if events is not None:
                close = getattr(events, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except RuntimeError:
                        pass
        return title, usage_delta

    def _router_stream_title(self, content: str) -> Any:
        return self._model_router.stream(
            "chat",
            messages=[
                {"role": "system", "content": self._title_prompt or ""},
                {"role": "user", "content": Session._normalize_title(content)},
            ],
            tools=(),
            continuation=None,
        )


__all__ = [
    "AgentLoop",
    "AgentLoopControl",
    "ConfirmationCallback",
    "ConfirmationRequestView",
    "ForegroundContextPreparer",
    "ScheduleContextPreparer",
]


def _runtime_logger() -> Any:
    def set_runtime_name(record: Any) -> None:
        record["name"] = "myclaw.agent.loop"

    return logger.patch(set_runtime_name)


def _log_agent_failure(error: ErrorInfo) -> None:
    failure = ModelCallError(error)
    _runtime_logger().opt(exception=failure).error(
        "Agent Run failed code={} type={}",
        error.code,
        type(failure).__name__,
    )


async def _discard_runner_output(event: object) -> None:
    del event
