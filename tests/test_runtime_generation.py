import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

import myclaw.agent.loop as loop_module
import myclaw.agent.runtime as runtime_module
from myclaw.agent.loop import AgentLoop
from myclaw.agent.message_bus import InboundMessage, OutboundMessage
from myclaw.agent.runtime import RuntimeGenerationPresentation, RuntimeHost
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import SessionListingEntry
from myclaw.memory.conversation_summary import WorkspaceJsonlSummaryStore
from myclaw.memory.dream import Dream
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
)
from myclaw.schedule.service import ScheduleService
from myclaw.schedule.store import ScheduleStateError
from myclaw.session.session import Session
from myclaw.skills.catalog import SkillLoader, SkillSnapshot
from myclaw.terminal.conversation import TerminalConversationApp
from myclaw.tools.base import OpenAIToolSchema
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock, ScriptedFakeProvider
from tests.runtime_bus import collect_foreground_outbound
from tests.test_runtime_active_session import RuntimeProvider

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 22, 10, 20, 30, 123000, tzinfo=LOCAL_OFFSET)


def _chat_response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


class GatedProvider(RuntimeProvider):
    def __init__(self) -> None:
        super().__init__(())
        self.stream_started = asyncio.Event()
        self.release_stream = asyncio.Event()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.close_finished = asyncio.Event()

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
        self.stream_started.set()
        await self.release_stream.wait()
        async for event in super().stream(
            messages=messages,
            tools=tools,
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            continuation=continuation,
        ):
            yield event

    async def close(self) -> None:
        self.close_started.set()
        await self.release_close.wait()
        self.close_finished.set()


class GatedMemoryProvider(RuntimeProvider):
    def __init__(self) -> None:
        super().__init__(())
        self.memory_started = asyncio.Event()
        self.release_memory = asyncio.Event()
        self.close_order: list[str] = []

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
        self.memory_started.set()
        await self.release_memory.wait()
        response = await super().complete(
            messages=messages,
            tools=tools,
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            continuation=continuation,
        )
        self.close_order.append("dream finished")
        return response

    async def close(self) -> None:
        self.close_order.append("router closed")
        await super().close()


def _host(
    agent_home: Path,
    workspace: Path,
    provider: RuntimeProvider,
    *,
    config_text: str = VALID_CONFIG,
) -> RuntimeHost:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(config_text, encoding="utf-8")
    clock = FakeClock(NOW)
    return RuntimeHost(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=uuid4,
        retry_clock=clock,
    )


class _RuntimeAppDriver:
    """Test composition root for a Textual app and one RuntimeHost."""

    def __init__(self, host: RuntimeHost) -> None:
        presentation = host.presentation
        self._app = TerminalConversationApp(
            bus=host.bus,
            control=presentation.control,
            management_dispatcher=host.management_dispatcher,
            skill_metadata=presentation.skill_metadata,
        )

        async def rebind(presentation: RuntimeGenerationPresentation) -> None:
            self.rebind_sessions.append(presentation.session_projection.session_id)
            await self._app.rebind_agent_loop(
                control=presentation.control,
                skill_metadata=presentation.skill_metadata,
                session_projection=presentation.session_projection,
            )

        self._rebind = rebind
        self._host = host
        self.rebind_sessions: list[str] = []
        host.bind_terminal(self._rebind, quiesce=self._app.quiesce_for_rebind)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_app", "_rebind", "_host", "rebind_sessions"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._app, name, value)

    @asynccontextmanager
    async def run_test(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        await self._host.start()
        try:
            async with self._app.run_test(*args, **kwargs) as pilot:
                yield pilot
        finally:
            self._host.unbind_terminal(self._rebind)
            await self._host.close()


@pytest.mark.asyncio
async def test_runtime_host_refreshes_skill_snapshot_across_generation_replacement(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_skill = agent_home / "skills" / "first" / "SKILL.md"
    first_skill.parent.mkdir(parents=True)
    first_skill.write_bytes(b"---\nname: first\ndescription: Original first\n---\n")
    snapshot_builds = 0
    original_load = SkillLoader.load

    def recording_load(loader: object) -> SkillSnapshot:
        nonlocal snapshot_builds
        snapshot_builds += 1
        return original_load(loader)  # type: ignore[arg-type]

    monkeypatch.setattr(SkillLoader, "load", recording_load)
    provider = RuntimeProvider((_chat_response("First run."), _chat_response("Second run.")))
    host = _host(agent_home, workspace, provider)
    initial_metadata = host.bindings.skill_metadata
    assert tuple(metadata.name for metadata in initial_metadata) == ("first",)
    assert initial_metadata[0].description == "Original first"
    await host.start()
    try:
        await collect_foreground_outbound(host.generation, "Initial request")

        target = Session.create(
            host.generation.session.workspace_state,
            now=lambda: NOW,
            new_uuid=uuid4,
        )
        target.add_message("user", "Persisted target")
        target.close()
        first_skill.write_bytes(b"---\nname: first\ndescription: Changed first\n---\n")
        second_skill = agent_home / "skills" / "second" / "SKILL.md"
        second_skill.parent.mkdir(parents=True)
        second_skill.write_bytes(b"---\nname: second\ndescription: New second\n---\n")

        result = await host.management_dispatcher.resume(target.session_id)
        assert result.resumed_session_id == target.session_id
        assert tuple(metadata.name for metadata in host.bindings.skill_metadata) == (
            "first",
            "second",
        )
        assert host.bindings.skill_metadata[0].description == "Changed first"
        await collect_foreground_outbound(host.generation, "Replacement request")
    finally:
        await host.close()

    chat_requests = [request for request in provider.requests if len(request.tools) == 10]
    assert len(chat_requests) == 2
    initial_prompt = chat_requests[0].messages[0]["content"]
    replacement_prompt = chat_requests[1].messages[0]["content"]
    assert isinstance(initial_prompt, str)
    assert isinstance(replacement_prompt, str)
    assert "Original first" in initial_prompt
    assert "Changed first" not in initial_prompt
    assert "second" not in initial_prompt
    assert "Changed first" in replacement_prompt
    assert "New second" in replacement_prompt

    fresh_provider = RuntimeProvider((_chat_response("Fresh run."),))
    fresh_host = _host(agent_home, workspace, fresh_provider)
    await fresh_host.start()
    try:
        await collect_foreground_outbound(fresh_host.generation, "Fresh request")
    finally:
        await fresh_host.close()

    fresh_request = next(request for request in fresh_provider.requests if len(request.tools) == 10)
    fresh_system_prompt = fresh_request.messages[0]["content"]
    assert isinstance(fresh_system_prompt, str)
    assert "Changed first" in fresh_system_prompt
    assert "New second" in fresh_system_prompt
    assert snapshot_builds == 3


@pytest.mark.asyncio
async def test_runtime_rejects_conflicting_dream_state_before_agent_loop_construction(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    state.schedule_path.write_text(
        json.dumps(
            [
                {
                    "job_id": "dream",
                    "source": "user",
                    "message": "Internal Dream schedule.",
                    "schedule": {
                        "kind": "every",
                        "at_time": None,
                        "every_seconds": 60,
                        "cron_expr": None,
                        "timezone": None,
                    },
                    "state": {
                        "last_finished_at_ms": None,
                        "last_status": None,
                        "last_error": None,
                    },
                    "created_at_ms": 1,
                    "updated_at_ms": 1,
                }
            ],
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    created_dreams: list[Dream] = []
    real_dream = Dream

    def recording_dream(**kwargs: object) -> Dream:
        dream = real_dream(**kwargs)  # type: ignore[arg-type]
        created_dreams.append(dream)
        return dream

    agent_loop_constructions = 0
    real_agent_loop = AgentLoop

    def recording_agent_loop(*args: object, **kwargs: object) -> object:
        nonlocal agent_loop_constructions
        agent_loop_constructions += 1
        return real_agent_loop(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runtime_module, "Dream", recording_dream)
    monkeypatch.setattr(runtime_module, "AgentLoop", recording_agent_loop)
    baseline = asyncio.all_tasks()

    with pytest.raises(ScheduleStateError):
        runtime_module.prepare_runtime(
            agent_home=home,
            workspace=workspace,
            configuration=ConfigLoader(home).load(),
            provider_factory=lambda _configuration: ScriptedFakeProvider(),
            now=lambda: NOW,
            new_uuid=uuid4,
            workspace_state=state,
        )

    assert agent_loop_constructions == 0
    assert len(created_dreams) == 1
    assert created_dreams[0]._aborted is True
    assert created_dreams[0]._closed is True
    assert created_dreams[0]._task is None
    assert asyncio.all_tasks() == baseline


@pytest.mark.asyncio
async def test_runtime_host_refreshes_skill_snapshot_after_generation_skill_deletion(
    agent_home: Path,
    workspace: Path,
) -> None:
    instruction = agent_home / "skills" / "removed" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: removed\ndescription: To be removed\n---\n")
    host = _host(agent_home, workspace, RuntimeProvider(()))
    assert tuple(metadata.name for metadata in host.bindings.skill_metadata) == ("removed",)

    target = Session.create(
        host.generation.session.workspace_state,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    target.add_message("user", "Persisted target")
    target.close()
    instruction.unlink()

    try:
        result = await host.management_dispatcher.resume(target.session_id)

        assert result.resumed_session_id == target.session_id
        assert host.bindings.skill_metadata == ()
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_runtime_host_refreshes_skill_snapshot_when_resuming_current_session(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = agent_home / "skills" / "current" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: current\ndescription: Original\n---\noriginal\n")
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    old.session.add_message("user", "Persist the current Session")
    old.session.persist()
    pending_persist = old.session._pending_persist
    assert pending_persist is not None
    await pending_persist

    original_persist_after = Session._persist_after
    persist_started = asyncio.Event()
    release_persist = asyncio.Event()

    async def gated_persist_after(
        active: Session,
        previous: asyncio.Task[None] | None,
        content: bytes,
    ) -> None:
        if active is old.session:
            persist_started.set()
            await release_persist.wait()
        await original_persist_after(active, previous, content)

    monkeypatch.setattr(Session, "_persist_after", gated_persist_after)
    old.session.add_message("user", "Persist before rebuilding this Session")
    old.session.persist()
    await persist_started.wait()
    instruction.write_bytes(b"---\nname: current\ndescription: Refreshed\n---\nrefreshed\n")

    await host.start()
    try:
        replacement = asyncio.create_task(host.management_dispatcher.resume(old.session_id))
        await asyncio.sleep(0)
        assert not replacement.done()
        release_persist.set()
        result = await replacement

        assert result.resumed_session_id == old.session_id
        assert host.generation is not old
        assert host.bindings.skill_metadata[0].description == "Refreshed"
        assert [message["content"] for message in host.generation.session.messages] == [
            "Persist the current Session",
            "Persist before rebuilding this Session",
        ]
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_runtime_host_refreshes_frozen_always_body_across_generation_replacement(
    agent_home: Path,
    workspace: Path,
) -> None:
    first_skill = agent_home / "skills" / "first" / "SKILL.md"
    first_skill.parent.mkdir(parents=True)
    first_skill.write_bytes(
        b"---\nname: first\ndescription: Original first\nalways: true\n---\nOriginal always body\n"
    )
    config_text = VALID_CONFIG.replace(
        "[runtime]\n",
        "[runtime]\nenable_skill_always_load = true\n",
    )
    provider = RuntimeProvider((_chat_response("First run."), _chat_response("Second run.")))
    host = _host(agent_home, workspace, provider, config_text=config_text)
    await host.start()
    try:
        await collect_foreground_outbound(host.generation, "Initial request")

        target = Session.create(
            host.generation.session.workspace_state,
            now=lambda: NOW,
            new_uuid=uuid4,
        )
        target.add_message("user", "Persisted target")
        target.close()
        first_skill.write_bytes(
            b"---\nname: first\ndescription: Original first\nalways: false\n---\n"
            b"Changed always body\n"
        )

        result = await host.management_dispatcher.resume(target.session_id)
        assert result.resumed_session_id == target.session_id
        await collect_foreground_outbound(host.generation, "Replacement request")
    finally:
        await host.close()

    chat_requests = [request for request in provider.requests if len(request.tools) == 10]
    assert len(chat_requests) == 2
    initial_prompt = chat_requests[0].messages[0]["content"]
    replacement_prompt = chat_requests[1].messages[0]["content"]
    assert isinstance(initial_prompt, str)
    assert isinstance(replacement_prompt, str)
    assert "Original always body" in initial_prompt
    assert "Original always body" not in replacement_prompt
    assert "Changed always body" not in replacement_prompt

    fresh_provider = RuntimeProvider((_chat_response("Fresh run."),))
    fresh_host = _host(agent_home, workspace, fresh_provider, config_text=config_text)
    await fresh_host.start()
    try:
        await collect_foreground_outbound(fresh_host.generation, "Fresh request")
    finally:
        await fresh_host.close()

    fresh_request = next(request for request in fresh_provider.requests if len(request.tools) == 10)
    fresh_system_prompt = fresh_request.messages[0]["content"]
    assert isinstance(fresh_system_prompt, str)
    assert "Original always body" not in fresh_system_prompt
    assert "Changed always body" not in fresh_system_prompt


@pytest.mark.asyncio
async def test_target_generation_preparation_failure_preserves_old_generation(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    old_session = old.session
    old_bus = old.bus
    target = Session.create(
        old_session.workspace_state,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    target.add_message("user", "Persisted target")
    target.close()

    def fail_target_context(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("target validation failed")

    monkeypatch.setattr(loop_module, "ContextBuilder", fail_target_context)

    result = await host.management_dispatcher.resume(target.session_id)

    assert result.output == ("persistence_error: Conversation Session could not be prepared.")
    assert host.generation is old
    assert host.generation.session is old_session
    assert host.generation.bus is old_bus
    old_session.add_message("user", "Old generation remains usable")

    await host.close()


@pytest.mark.asyncio
async def test_target_schedule_service_preflight_failure_preserves_the_started_old_generation(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    await host.start()
    old = host.generation
    old_bus = old.bus
    old_consumer = old.agent_loop._consumer_task
    target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Target whose scheduler cannot be prepared")
    target.close()

    original_prepare_start = ScheduleService._prepare_start

    def fail_target_schedule_preflight(service: ScheduleService) -> None:
        if service is not old.schedule_service:
            raise RuntimeError("target scheduler validation failed")
        original_prepare_start(service)

    monkeypatch.setattr(
        ScheduleService,
        "_prepare_start",
        fail_target_schedule_preflight,
    )

    result = await host.management_dispatcher.resume(target.session_id)

    assert result.output == "persistence_error: Conversation Session could not be prepared."
    assert host.generation is old
    assert host.bus is old_bus
    assert old.agent_loop._consumer_task is old_consumer
    assert old_consumer is not None and not old_consumer.done()
    await old_bus.put_inbound(InboundMessage(content="old generation remains live"))

    await host.close()


@pytest.mark.asyncio
async def test_abort_interrupts_an_in_progress_normal_close_before_final_session_save(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    runtime = host.generation
    runtime.session.add_message("user", "Must not be saved by an aborted close")
    session_path = (
        runtime.session.workspace_state.sessions_directory / f"{runtime.session.session_id}.jsonl"
    )
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def blocked_schedule_close() -> None:
        close_started.set()
        await release_close.wait()

    monkeypatch.setattr(runtime.schedule_service, "close", blocked_schedule_close)
    closing = asyncio.create_task(runtime.close())
    await close_started.wait()

    await runtime.abort()
    assert runtime._lifetime.close_task is not None
    assert runtime._lifetime.close_task.cancelling()
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await closing

    await runtime.close()
    assert not session_path.exists()


@pytest.mark.asyncio
async def test_normal_close_waits_for_an_in_progress_dream_before_closing_the_router(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = GatedMemoryProvider()
    host = _host(agent_home, workspace, provider)
    runtime = host.generation
    summaries = WorkspaceJsonlSummaryStore(runtime.session.workspace_state)
    await summaries.append("Pending summary for Long-term Memory.", NOW)
    dream = asyncio.create_task(runtime.management_dispatcher.dispatch("/dream"))
    await asyncio.wait_for(provider.memory_started.wait(), timeout=2)

    closing = asyncio.create_task(runtime.close())
    for _ in range(10):
        await asyncio.sleep(0)
    close_returned_early = closing.done()
    provider.release_memory.set()
    result = await asyncio.wait_for(dream, timeout=2)
    await asyncio.wait_for(closing, timeout=2)
    provider.close_order.append("runtime close returned")

    assert not close_returned_early
    assert result.output == (
        "Processed 1 summary; Long-term Memory unchanged.\n"
        "processed_count: 1\n"
        "memory_updated: false\n"
        "cursor: 1"
    )
    assert provider.close_order == [
        "dream finished",
        "router closed",
        "runtime close returned",
    ]


@pytest.mark.asyncio
async def test_generation_replacement_aborts_old_dream_and_schedule_service(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = GatedMemoryProvider()
    host = _host(agent_home, workspace, provider)
    await host.start()
    old = host.generation
    summaries = WorkspaceJsonlSummaryStore(old.session.workspace_state)
    await summaries.append("Pending summary for replacement.", NOW)
    dream_task = asyncio.create_task(host.management_dispatcher.dispatch("/dream"))

    await asyncio.wait_for(provider.memory_started.wait(), timeout=2)
    target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Replacement target")
    target.close()

    result = await host.management_dispatcher.resume(target.session_id, force=True)

    assert result.resumed_session_id == target.session_id
    assert dream_task.done()
    assert old._dream._task is None
    assert old._dream._closed is True
    assert old.schedule_service._loop_task is None
    assert not old.schedule_service._run_tasks
    assert not old.schedule_service._terminal_commit_tasks
    with pytest.raises(asyncio.CancelledError):
        await dream_task

    await host.close()


@pytest.mark.asyncio
async def test_runtime_abort_drains_active_dream_and_schedule_service(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = GatedMemoryProvider()
    host = _host(agent_home, workspace, provider)
    await host.start()
    runtime = host.generation
    summaries = WorkspaceJsonlSummaryStore(runtime.session.workspace_state)
    await summaries.append("Pending summary for direct abort.", NOW)
    dream_task = asyncio.create_task(runtime.management_dispatcher.dispatch("/dream"))
    await asyncio.wait_for(provider.memory_started.wait(), timeout=2)

    await runtime.abort()

    assert dream_task.done()
    assert runtime._dream._task is None
    assert runtime.schedule_service._loop_task is None
    assert not runtime.schedule_service._run_tasks
    assert not runtime.schedule_service._terminal_commit_tasks
    with pytest.raises(asyncio.CancelledError):
        await dream_task


@pytest.mark.asyncio
async def test_runtime_abort_drains_only_inbound_after_unbinding_its_callback(
    agent_home: Path,
    workspace: Path,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    runtime = host.generation
    bus = runtime.bus
    observed: list[tuple[InboundMessage, ...]] = []
    bus.set_inbound_changed_callback(observed.append)
    bus.set_inbound_changed_callback(None)
    pending_inbound = InboundMessage(content="discard pending input")
    completed_outbound = OutboundMessage(
        type="model_response",
        content="preserve completed response",
        metadata={"_streamed": True},
    )
    await bus.put_inbound(pending_inbound)
    await bus.put_outbound(completed_outbound)
    observed.clear()

    await runtime.abort()

    assert await bus.inbound_snapshot() == ()
    assert observed == []
    assert await bus.get_outbound() is completed_outbound
    await host.close()


@pytest.mark.asyncio
async def test_generation_replacement_finishes_atomically_when_waiter_is_cancelled(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Replacement committed after caller cancellation")
    target.close()
    drain_started = asyncio.Event()
    release_drain = asyncio.Event()
    rebind_started = asyncio.Event()
    release_rebind = asyncio.Event()
    events: list[str] = []
    original_drain = runtime_module.PreparedRuntime._drain_aborted_memory
    original_start = runtime_module.PreparedRuntime.start

    async def gated_drain(active: runtime_module.PreparedRuntime) -> None:
        await original_drain(active)
        if active is old:
            drain_started.set()
            await release_drain.wait()

    async def record_start(active: runtime_module.PreparedRuntime) -> None:
        if active is not old:
            events.append("target_start")
        await original_start(active)

    async def quiesce() -> None:
        events.append("quiesce")

    async def rebind(presentation: RuntimeGenerationPresentation) -> None:
        events.append("rebind")
        assert presentation.control is host.control
        rebind_started.set()
        await release_rebind.wait()

    monkeypatch.setattr(runtime_module.PreparedRuntime, "_drain_aborted_memory", gated_drain)
    monkeypatch.setattr(runtime_module.PreparedRuntime, "start", record_start)
    host.bind_terminal(rebind, quiesce=quiesce)
    replacement = asyncio.create_task(
        host.management_dispatcher.resume(target.session_id, force=True)
    )
    await asyncio.wait_for(drain_started.wait(), timeout=2)

    replacement.cancel()
    assert not replacement.done()
    release_drain.set()
    await asyncio.wait_for(rebind_started.wait(), timeout=2)
    release_rebind.set()
    with pytest.raises(asyncio.CancelledError):
        await replacement

    assert events == ["quiesce", "rebind", "target_start"]
    assert host.generation is not old
    assert host.generation.session_id == target.session_id
    assert not host.generation._lifetime.aborted
    await host.close()


def test_runtime_composition_failure_closes_a_constructed_dream(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []
    original_dream = Dream

    def recording_dream(*args: object, **kwargs: object) -> Dream:
        dream = original_dream(*args, **kwargs)  # type: ignore[arg-type]
        created.append(dream)
        return dream

    def fail_wiring(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("post-Dream wiring failed")

    monkeypatch.setitem(runtime_module.__dict__, "Dream", recording_dream)
    monkeypatch.setattr(AgentLoop, "preflight", fail_wiring)

    with pytest.raises(RuntimeError, match="post-Dream wiring failed"):
        _host(agent_home, workspace, RuntimeProvider(()))

    assert len(created) == 1
    dream = cast(Dream, created[0])
    assert dream._closed is True
    assert dream._task is None


@pytest.mark.asyncio
async def test_pending_only_resume_replaces_every_generation_owned_component(
    agent_home: Path,
    workspace: Path,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    old_bus = old.bus
    old_components = (
        old.agent_loop,
        old.schedule_service,
        old._router,
        old._memory_manager,
        old.agent_loop._tool_gateway,
        old.agent_loop._runner,
        old._management_service,
        tuple(old.agent_loop._tool_gateway._tools.values()),
    )
    await old_bus.put_inbound(InboundMessage(content="discard this pending input"))

    target = Session.create(
        old.session.workspace_state,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    target.add_message("user", "Persisted target")
    target.close()

    result = await host.management_dispatcher.resume(target.session_id)
    assert result.resumed_session_id == target.session_id
    replacement = host.generation

    assert replacement is not old
    assert replacement.session_id == target.session_id
    replacement_components = (
        replacement.agent_loop,
        replacement.schedule_service,
        replacement._router,
        replacement._memory_manager,
        replacement.agent_loop._tool_gateway,
        replacement.agent_loop._runner,
        replacement._management_service,
        tuple(replacement.agent_loop._tool_gateway._tools.values()),
    )
    assert replacement.bus is old_bus is host.bus
    assert replacement.management_dispatcher is old.management_dispatcher is host.management_dispatcher
    assert all(
        new is not previous
        for new, previous in zip(replacement_components, old_components, strict=True)
    )
    assert old._dream._task is None
    assert old._dream._closed is True
    assert old.schedule_service._aborted is True
    assert old.schedule_service._loop_task is None
    assert not old.schedule_service._run_tasks
    assert not old.schedule_service._terminal_commit_tasks
    assert all(
        new_tool is not old_tool
        for new_tool, old_tool in zip(replacement_components[-1], old_components[-1], strict=True)
    )

    assert replacement.session.workspace_state is old.session.workspace_state
    assert replacement.session.workspace_state is host._workspace_state
    assert replacement._router._configuration is old._router._configuration is host._configuration
    assert replacement._router._provider_factory is host._provider_factory
    assert old._router._provider_factory is host._provider_factory
    assert replacement._router._clock is old._router._clock is host._retry_clock
    assert replacement.schedule_service._clock is host._schedule_scheduler_clock
    assert old.schedule_service._clock is host._schedule_scheduler_clock
    assert replacement.agent_loop._context_builder._workspace == host._workspace
    assert old.agent_loop._context_builder._workspace == host._workspace
    assert str(replacement.agent_loop._context_builder._timezone) == host._timezone_name
    assert str(old.agent_loop._context_builder._timezone) == host._timezone_name
    assert host._new_uuid is uuid4
    for generation in (old, replacement):
        assert generation.session._now is host._now
        assert generation.agent_loop._now is host._now
        assert generation.agent_loop._context_builder._clock is host._now
        assert generation._management_service._now is host._now
        assert generation._management_service._config.agent_home is host._agent_home
        status_service = generation._management_service._status_service
        assert status_service is not None
        assert status_service._monotonic is host._monotonic_now
    assert await old_bus.inbound_snapshot() == ()
    with pytest.raises(RuntimeError, match="no longer active"):
        _ = old.agent_loop.bus
    with pytest.raises(RuntimeError, match="no longer active"):
        _ = old.control.has_active_run
    old_management = await old.management_dispatcher.dispatch("/status")
    assert old_management.output is not None
    assert "route_unavailable" not in old_management.output
    assert await old_bus.inbound_snapshot() == ()
    with pytest.raises(RuntimeError, match="abandoned"):
        old.session.add_message("user", "old generation is detached")

    await host.close()


@pytest.mark.asyncio
async def test_terminal_rebinds_once_and_rebuilds_the_target_session(
    agent_home: Path,
    workspace: Path,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    old_bus = old.bus
    target = Session.create(
        old.session.workspace_state,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    target.add_message("user", "Target display")
    target.close()

    app = _RuntimeAppDriver(host)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/resume"), "enter")
        async with asyncio.timeout(2):
            while app.screen.id != "session-picker":
                await pilot.pause()
        await pilot.press("enter")
        async with asyncio.timeout(2):
            while host.generation.session_id != target.session_id:
                await pilot.pause()

        assert app._bus is host.bus
        assert app._control is host.control
        messages = app.query_one("#conversation-display").query(".user-message")
        assert messages
        assert any("Target display" in str(getattr(message, "content", "")) for message in messages)
        assert app.rebind_sessions == [target.session_id]

    assert old_bus is host.bus


@pytest.mark.asyncio
async def test_runtime_handoff_preserves_lifetime_seams_and_starts_target_after_rebind(
    agent_home: Path,
    workspace: Path,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old_control = host.control
    shared_bus = host.bus
    shared_dispatcher = host.management_dispatcher
    pending = InboundMessage(content="discard during handoff")
    stale_outbound = OutboundMessage(type="model_response", content="stale")
    await shared_bus.put_inbound(pending)
    await shared_bus.put_outbound(stale_outbound)

    target = Session.create(host.generation.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Target presentation")
    target.close()
    events: list[str] = []

    async def quiesce() -> None:
        events.append("quiesce")
        assert await shared_bus.inbound_snapshot() == (pending,)

    async def rebind(presentation: object) -> None:
        events.append("rebind")
        assert isinstance(presentation, runtime_module.RuntimeGenerationPresentation)
        assert host.bus is shared_bus
        assert host.management_dispatcher is shared_dispatcher
        assert presentation.control is host.control
        assert presentation.control is not old_control
        assert presentation.session_projection.session_id == target.session_id
        assert tuple(shared_bus._outbound) == ()

    host.bind_terminal(rebind, quiesce=quiesce)
    result = await host.management_dispatcher.resume(target.session_id)

    assert result.resumed_session_id == target.session_id
    assert events == ["quiesce", "rebind"]
    assert host.bus is shared_bus
    assert host.management_dispatcher is shared_dispatcher
    assert host.control is not old_control
    assert host.generation.agent_loop._consumer_task is not None
    assert await shared_bus.inbound_snapshot() == ()
    await host.close()


@pytest.mark.asyncio
async def test_runtime_handoff_waits_for_old_tasks_before_reset_rebind_and_target_output(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    shared_bus = host.bus
    target_session = Session.create(
        old.session.workspace_state,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    target_session.add_message("user", "Target after drain barrier")
    target_session.close()
    old_cancelled = asyncio.Event()
    release_old = asyncio.Event()
    old_finished = asyncio.Event()
    rebind_started = asyncio.Event()
    events: list[str] = []
    consumed: asyncio.Task[OutboundMessage] | None = None
    original_start = runtime_module.PreparedRuntime.start

    async def old_owned_task() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            old_cancelled.set()
            await release_old.wait()
            await shared_bus.put_outbound(
                OutboundMessage(type="model_response", content="late old output")
            )
            events.append("old_finished")
            old_finished.set()

    async def rebind(presentation: RuntimeGenerationPresentation) -> None:
        nonlocal consumed
        assert old_finished.is_set()
        assert tuple(shared_bus._outbound) == ()
        events.append("rebind")
        rebind_started.set()
        consumed = asyncio.create_task(shared_bus.get_outbound())
        assert presentation.session_projection.session_id == target_session.session_id

    async def start_with_target_output(active: runtime_module.PreparedRuntime) -> None:
        await original_start(active)
        if active is not old:
            assert rebind_started.is_set()
            events.append("target_start")
            await shared_bus.put_outbound(
                OutboundMessage(type="model_response", content="target output")
            )

    old.agent_loop._consumer_task = asyncio.create_task(old_owned_task())
    monkeypatch.setattr(runtime_module.PreparedRuntime, "start", start_with_target_output)
    host.bind_terminal(rebind)
    replacement = asyncio.create_task(
        host.management_dispatcher.resume(target_session.session_id)
    )

    await old_cancelled.wait()
    await asyncio.sleep(0)
    assert not replacement.done()
    assert not rebind_started.is_set()
    release_old.set()
    result = await replacement

    assert result.resumed_session_id == target_session.session_id
    assert consumed is not None
    assert (await consumed).content == "target output"
    assert events == ["old_finished", "rebind", "target_start"]
    assert tuple(shared_bus._outbound) == ()
    await host.close()


@pytest.mark.asyncio
async def test_target_preflight_failure_preserves_current_generation_and_bus(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    shared_bus = host.bus
    shared_dispatcher = host.management_dispatcher
    snapshots: list[tuple[InboundMessage, ...]] = []
    shared_bus.set_inbound_changed_callback(snapshots.append)
    pending = InboundMessage(content="keep current pending input")
    await shared_bus.put_inbound(pending)
    snapshots.clear()

    target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Target never committed")
    target.close()
    original_validate = runtime_module.PreparedRuntime.validate_unstarted

    def fail_target_preflight(runtime: runtime_module.PreparedRuntime) -> None:
        if runtime is not old:
            raise RuntimeError("target preflight failed")
        original_validate(runtime)

    monkeypatch.setattr(
        runtime_module.PreparedRuntime,
        "validate_unstarted",
        fail_target_preflight,
    )

    result = await shared_dispatcher.resume(target.session_id)

    assert result.output is not None
    assert result.output.startswith("persistence_error:")
    assert host.generation is old
    assert host.control is old.control
    assert host.bus is shared_bus
    assert host.management_dispatcher is shared_dispatcher
    assert shared_dispatcher._management is old._management_service
    assert await shared_bus.inbound_snapshot() == (pending,)
    assert snapshots == []
    assert shared_bus._inbound_changed_callback is not None
    await host.close()


@pytest.mark.asyncio
async def test_rebind_failure_leaves_a_quiesced_target_without_mixed_generation_state(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    shared_bus = host.bus
    shared_dispatcher = host.management_dispatcher
    pending = InboundMessage(content="old pending")
    stale = OutboundMessage(type="model_response", content="old stale")
    await shared_bus.put_inbound(pending)
    await shared_bus.put_outbound(stale)
    target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Target presentation")
    target.close()
    events: list[str] = []
    presentations: list[RuntimeGenerationPresentation] = []
    target_starts: list[runtime_module.PreparedRuntime] = []
    original_start = runtime_module.PreparedRuntime.start

    async def record_target_start(runtime: runtime_module.PreparedRuntime) -> None:
        target_starts.append(runtime)
        await original_start(runtime)

    async def quiesce() -> None:
        events.append("quiesce")

    async def fail_rebind(presentation: RuntimeGenerationPresentation) -> None:
        events.append("rebind")
        presentations.append(presentation)
        assert presentation.session_projection.session_id == target.session_id
        raise RuntimeError("presentation rebind failed")

    monkeypatch.setattr(runtime_module.PreparedRuntime, "start", record_target_start)
    host.bind_terminal(fail_rebind, quiesce=quiesce)

    with pytest.raises(RuntimeError, match="presentation rebind failed"):
        await shared_dispatcher.resume(target.session_id)

    assert events == ["quiesce", "rebind", "quiesce"]
    assert target_starts == []
    assert old._lifetime.aborted
    assert len(presentations) == 1
    assert cast(AgentLoop, presentations[0].control)._aborted
    assert host.bus is shared_bus
    assert host.management_dispatcher is shared_dispatcher
    with pytest.raises(RuntimeError, match="no active Runtime Generation"):
        _ = host.generation
    assert shared_dispatcher._management is None
    unavailable = await shared_dispatcher.resume(target.session_id)
    assert unavailable.output == "route_unavailable: Runtime Generation is unavailable."
    assert await shared_bus.inbound_snapshot() == ()
    assert tuple(shared_bus._outbound) == ()
    assert host._terminal_rebind is None
    assert host._terminal_quiesce is None
    await host.close()
    await host.close()


@pytest.mark.asyncio
async def test_target_start_failure_enters_the_same_fail_closed_state(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    shared_bus = host.bus
    shared_dispatcher = host.management_dispatcher
    target_session = Session.create(
        old.session.workspace_state,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    target_session.add_message("user", "Target start fails")
    target_session.close()
    target_control: AgentLoop | None = None
    original_start = runtime_module.PreparedRuntime.start

    async def rebind(presentation: RuntimeGenerationPresentation) -> None:
        nonlocal target_control
        target_control = cast(AgentLoop, presentation.control)

    async def fail_target_start(active: runtime_module.PreparedRuntime) -> None:
        if active is old:
            await original_start(active)
            return
        raise RuntimeError("target start failed")

    monkeypatch.setattr(runtime_module.PreparedRuntime, "start", fail_target_start)
    host.bind_terminal(rebind)

    with pytest.raises(RuntimeError, match="target start failed"):
        await shared_dispatcher.resume(target_session.session_id)

    assert target_control is not None
    assert target_control._aborted
    assert old._lifetime.aborted
    assert host.bus is shared_bus
    assert host.management_dispatcher is shared_dispatcher
    assert shared_dispatcher._management is None
    with pytest.raises(RuntimeError, match="no active Runtime Generation"):
        _ = host.generation
    await host.close()


@pytest.mark.asyncio
async def test_concurrent_old_generation_resume_requests_cannot_replace_the_new_generation(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    targets: list[Session] = []
    for content in ("First target", "Second target"):
        target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
        target.add_message("user", content)
        target.close()
        targets.append(target)

    original_listing = old._management_service.resumable_sessions
    listing_call_count = 0
    both_listings_started = asyncio.Event()
    release_listings = asyncio.Event()

    async def gated_listing() -> tuple[SessionListingEntry, ...]:
        nonlocal listing_call_count
        listing = await original_listing()
        listing_call_count += 1
        if listing_call_count == 2:
            both_listings_started.set()
        await release_listings.wait()
        return listing

    monkeypatch.setattr(old._management_service, "resumable_sessions", gated_listing)
    dispatcher = cast(ManagementCommandDispatcher, old.management_dispatcher)
    requests = tuple(
        asyncio.create_task(dispatcher.resume(target.session_id)) for target in targets
    )
    await both_listings_started.wait()
    release_listings.set()
    results = await asyncio.gather(*requests)

    successes = [result for result in results if result.resumed_session_id is not None]
    assert len(successes) == 1
    assert host.generation.session_id == successes[0].resumed_session_id
    assert any(
        result.output == "route_unavailable: Runtime Generation is no longer active."
        for result in results
    )

    await host.close()


@pytest.mark.asyncio
async def test_management_command_uses_one_generation_port_across_concurrent_rebind(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    dispatcher = host.management_dispatcher
    target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Concurrent management target")
    target.close()
    old_command_started = asyncio.Event()
    release_old_command = asyncio.Event()

    async def gated_old_memory_view() -> str:
        old_command_started.set()
        await release_old_command.wait()
        return "old generation memory"

    monkeypatch.setattr(old._management_service, "memory_view", gated_old_memory_view)
    old_command = asyncio.create_task(dispatcher.dispatch("/memory"))
    await old_command_started.wait()

    replacement = await dispatcher.resume(target.session_id)
    assert replacement.resumed_session_id == target.session_id
    assert dispatcher._management is host.generation._management_service
    release_old_command.set()
    result = await old_command

    assert result.output == "old generation memory"
    assert dispatcher._management is host.generation._management_service
    await host.close()


@pytest.mark.asyncio
async def test_close_waits_for_an_in_progress_replacement_and_closes_the_committed_target(
    agent_home: Path,
    workspace: Path,
) -> None:
    host = _host(agent_home, workspace, RuntimeProvider(()))
    old = host.generation
    target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Committed before close")
    target.close()
    rebind_started = asyncio.Event()
    release_rebind = asyncio.Event()

    async def gated_rebind(presentation: RuntimeGenerationPresentation) -> None:
        del presentation
        rebind_started.set()
        await release_rebind.wait()

    host.bind_terminal(gated_rebind)
    replacement = asyncio.create_task(host.management_dispatcher.resume(target.session_id))
    await rebind_started.wait()
    closing = asyncio.create_task(host.close())
    await asyncio.sleep(0)
    assert not closing.done()

    release_rebind.set()
    result = await replacement
    await closing

    assert result.resumed_session_id == target.session_id
    assert host.generation.session_id == target.session_id
    assert host._closed
    assert host.generation.session._closed


@pytest.mark.asyncio
async def test_active_same_session_resume_requires_confirmation_before_rebuild(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = GatedProvider()
    host = _host(agent_home, workspace, provider)
    old = host.generation
    old.session.add_message("user", "Persist the current Session")
    old.session.persist()
    pending_persist = old.session._pending_persist
    assert pending_persist is not None
    await pending_persist

    app = _RuntimeAppDriver(host)
    async with app.run_test(size=(80, 24)) as pilot:
        try:
            await pilot.press(*list("active work"), "enter")
            await asyncio.wait_for(provider.stream_started.wait(), timeout=2)
            await pilot.press(*list("/resume"), "enter")
            async with asyncio.timeout(2):
                while app.screen.id != "session-picker":
                    await pilot.pause()
            await pilot.press("enter")
            async with asyncio.timeout(2):
                while app.screen.id != "session-switch-confirmation":
                    await pilot.pause()

            assert host.generation is old
            assert app._bus is old.bus
            assert any(
                "active work" in str(getattr(message, "content", ""))
                for message in app.query(".user-message")
            )
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen.id != "session-switch-confirmation"
        finally:
            provider.release_stream.set()
            provider.release_close.set()


@pytest.mark.asyncio
async def test_active_resume_decline_keeps_the_old_generation_untouched(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = GatedProvider()
    host = _host(agent_home, workspace, provider)
    old = host.generation
    target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Decline target")
    target.close()

    app = _RuntimeAppDriver(host)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("active work"), "enter")
        await asyncio.wait_for(provider.stream_started.wait(), timeout=2)
        old_bus = app._bus
        await pilot.press(*list("/resume"), "enter")
        async with asyncio.timeout(2):
            while app.screen.id != "session-picker":
                await pilot.pause()
        await pilot.press("enter")
        async with asyncio.timeout(2):
            while app.screen.id != "session-switch-confirmation":
                await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert host.generation is old
        assert app._bus is old_bus
        assert not provider.release_stream.is_set()
        old.session.add_message("user", "Old generation is still usable")

        provider.release_stream.set()
        provider.release_close.set()


@pytest.mark.asyncio
async def test_active_resume_approval_detaches_without_waiting_for_provider_close(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = GatedProvider()
    host = _host(agent_home, workspace, provider)
    old = host.generation
    old_bus = old.bus
    target = Session.create(old.session.workspace_state, now=lambda: NOW, new_uuid=uuid4)
    target.add_message("user", "Approve target")
    target.close()

    app = _RuntimeAppDriver(host)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("active work"), "enter")
        await asyncio.wait_for(provider.stream_started.wait(), timeout=2)
        await pilot.press(*list("/resume"), "enter")
        async with asyncio.timeout(2):
            while app.screen.id != "session-picker":
                await pilot.pause()
        await pilot.press("enter")
        async with asyncio.timeout(2):
            while app.screen.id != "session-switch-confirmation":
                await pilot.pause()
        await pilot.press("right", "enter")

        async with asyncio.timeout(2):
            while host.generation is old:
                await pilot.pause()
        assert host.generation.session_id == target.session_id
        await asyncio.wait_for(provider.close_started.wait(), timeout=2)
        assert not provider.close_finished.is_set()
        assert app._bus is host.bus
        assert app._bus is old_bus is host.bus
        await old_bus.put_outbound(
            OutboundMessage(
                type="model_response",
                content="late old output",
                metadata={"_streamed": True},
            )
        )
        await pilot.pause()
        assert all(
            "late old output" not in str(getattr(message, "content", ""))
            for message in app.query(".assistant-message")
        )

        provider.release_close.set()
        await asyncio.wait_for(provider.close_finished.wait(), timeout=2)
