from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pytest

from myclaw.agent.context_budget import estimate_run_slice_tokens
from myclaw.agent.memory.conversation_compactor import (
    AgentRunContextController,
    AgentRunContextPreparation,
    AgentRunContextSnapshot,
)
from myclaw.agent.memory.manager import MemoryManager
from myclaw.agent.session.session import Session
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouteStatus
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelResponse,
    ModelUsage,
)
from tests.fixtures import ScriptedFakeProvider, ScriptedFakeRouter

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 4, 16, 0, 0, tzinfo=LOCAL_OFFSET)


def _state(workspace: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _response(content: str, *, input_tokens: int = 20, output_tokens: int = 5) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        finish_reason="stop",
    )


def _usage(input_tokens: int = 4, output_tokens: int = 2) -> dict[str, int]:
    return {
        "model_calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _add_assistant(session: Session, content: str) -> None:
    session.add_message(
        "assistant",
        content,
        tool_calls=[],
        status="completed",
        error=None,
        token_usage=_usage(),
    )


def _add_tool_run(session: Session, label: str, *, result_size: int = 240) -> None:
    session.add_message(
        "assistant",
        f"{label} tool call",
        tool_calls=[{"id": f"call-{label}", "name": "read_file", "arguments": "{}"}],
        status="completed",
        error=None,
        token_usage=_usage(),
    )
    session.add_message(
        "tool",
        f"{label} tool result " + "r" * result_size,
        tool_call_id=f"call-{label}",
        name="read_file",
        status="success",
        artifact=None,
    )


def _add_run(session: Session, label: str, *, size: int = 500) -> None:
    session.add_message("user", f"{label} user " + "u" * size)
    _add_assistant(session, f"{label} assistant " + "a" * size)


def _controller(
    workspace: Path,
    session: Session,
    provider: ScriptedFakeProvider,
) -> AgentRunContextController:
    state = session.workspace_state
    return AgentRunContextController(
        snapshot=AgentRunContextSnapshot.from_session(session),
        provider=ScriptedFakeRouter(provider),
        memory_manager=MemoryManager(state),
        now=lambda: NOW,
    )


async def _prepare_controller(
    controller: AgentRunContextController,
    *,
    current_user: str = "new user",
    context_window: int = 1_800,
    max_output: int = 200,
    compact_ratio: float = 0.5,
    tools: Sequence[dict[str, Any]] = (),
    memory_route_status: ModelRouteStatus | None = None,
    provider_id: str = "",
    model: str = "",
) -> AgentRunContextPreparation:
    return await controller.prepare_run_start(
        project_messages=_project_messages,
        current_user={"role": "user", "content": current_user},
        route_context_window=context_window,
        route_max_output=max_output,
        compact_ratio=compact_ratio,
        tools=tools,
        memory_route_status=memory_route_status,
        provider_id=provider_id,
        model=model,
    )


def _memory_status(*, context_window: int, max_output: int = 100) -> ModelRouteStatus:
    return ModelRouteStatus(
        requested_route="memory",
        selected_route="memory",
        provider_id="memory-provider",
        model="memory-model",
        context_window=context_window,
        max_output=max_output,
        used_default=False,
    )


def _project_messages(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "SYSTEM"},
        *[
            {
                "role": message["role"],
                "content": message["content"],
                **(
                    {"tool_call_id": message["tool_call_id"], "name": message["name"]}
                    if message["role"] == "tool"
                    else {}
                ),
            }
            for message in messages
        ],
    ]


@pytest.mark.asyncio
async def test_controller_stages_run_start_compaction_from_detached_snapshot(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    session.add_message("user", "Old user " + "u" * 500)
    _add_assistant(session, "Old assistant " + "a" * 500)
    session.add_message("user", "Current persisted history " + "h" * 500)
    _add_assistant(session, "Current history assistant " + "c" * 500)
    session.update_metadata(summary="Prior action")
    snapshot = AgentRunContextSnapshot.from_session(session)
    provider = ScriptedFakeProvider(
        completions=(_response("Facts"), _response("Updated action")),
    )
    manager = AgentRunContextController(
        snapshot=snapshot,
        provider=ScriptedFakeRouter(provider),
        memory_manager=MemoryManager(state),
        now=lambda: NOW,
    )

    result = await manager.prepare_run_start(
        project_messages=_project_messages,
        current_user={"role": "user", "content": "New user must stay out"},
        route_context_window=1200,
        route_max_output=100,
        compact_ratio=0.5,
        tools=(),
    )

    assert result.selected_batch
    assert all(message["content"] != "New user must stay out" for message in result.selected_batch)
    assert "New user must stay out" not in str(provider.complete_requests[0].messages)
    assert manager.pending_action_summary == "Updated action"
    assert manager.pending_last_compacted > snapshot.last_compacted
    assert manager.pending_compaction_usage == {
        "model_calls": 2,
        "input_tokens": 40,
        "output_tokens": 10,
        "total_tokens": 50,
    }
    terminal = manager.terminal_commit_values()
    assert terminal.pending_last_compacted == manager.pending_last_compacted
    assert terminal.pending_action_summary == "Updated action"
    assert terminal.usage_delta == manager.pending_compaction_usage
    terminal.usage_delta["model_calls"] = 99
    assert manager.pending_compaction_usage["model_calls"] == 2


def test_snapshot_copies_transcript_and_metadata_without_retaining_session(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "before", size=20)
    snapshot = AgentRunContextSnapshot.from_session(session)
    session.add_message("user", "after")
    session.metadata["summary"] = "changed"

    assert len(snapshot.messages) == 2
    assert "after" not in str(snapshot.messages)
    assert snapshot.metadata.get("summary") == ""


@pytest.mark.asyncio
async def test_controller_detaches_from_the_supplied_snapshot(workspace: Path) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "original", size=800)
    session.update_metadata(summary="original action")
    snapshot = AgentRunContextSnapshot.from_session(session)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = AgentRunContextController(
        snapshot=snapshot,
        provider=ScriptedFakeRouter(provider),
        memory_manager=MemoryManager(state),
        now=lambda: NOW,
    )

    snapshot.messages[0]["content"] = "mutated outside controller"
    assert isinstance(snapshot.metadata, dict)
    snapshot.metadata["summary"] = "mutated action"
    result = await _prepare_controller(
        controller,
        context_window=1_000,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    assert "original user" in str(result.selected_batch)
    assert "mutated outside controller" not in str(result.selected_batch)
    assert "original action" in str(provider.complete_requests[1].messages)
    assert "mutated action" not in str(provider.complete_requests[1].messages)


@pytest.mark.asyncio
async def test_multiple_runs_under_ten_percent_keep_latest_run(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=1_400)
    _add_run(session, "middle", size=1_400)
    _add_run(session, "latest", size=320)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    latest_tokens = _estimate_latest_run(session)
    available = latest_tokens * 10

    result = await _prepare_controller(
        controller,
        context_window=available + 100,
        max_output=100,
        memory_route_status=_memory_status(context_window=4_000),
    )

    assert result.compacted is True
    fact_payload = str(provider.complete_requests[0].messages[1]["content"])
    assert "old user" in fact_payload
    assert "middle user" in fact_payload
    assert "latest user" not in fact_payload


@pytest.mark.asyncio
async def test_multiple_runs_at_ten_percent_keep_latest_run(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=1_400)
    _add_run(session, "middle", size=1_400)
    _add_run(session, "latest", size=320)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    latest_tokens = _estimate_latest_run(session)

    await _prepare_controller(
        controller,
        context_window=latest_tokens * 10 + 100,
        max_output=100,
        memory_route_status=_memory_status(context_window=4_000),
    )

    fact_payload = str(provider.complete_requests[0].messages[1]["content"])
    assert "latest user" not in fact_payload


@pytest.mark.asyncio
async def test_multiple_runs_over_ten_percent_select_latest_run(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=1_400)
    _add_run(session, "middle", size=1_400)
    _add_run(session, "latest", size=320)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    latest_tokens = _estimate_latest_run(session)

    await _prepare_controller(
        controller,
        context_window=latest_tokens * 10 - 1 + 100,
        max_output=100,
        memory_route_status=_memory_status(context_window=4_000),
    )

    fact_payload = str(provider.complete_requests[0].messages[1]["content"])
    assert "latest user" in fact_payload
    assert "new user" not in fact_payload


@pytest.mark.asyncio
async def test_single_completed_run_selects_its_entire_cursor_suffix(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "only", size=900)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)

    result = await _prepare_controller(
        controller,
        context_window=800,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    assert [message["role"] for message in result.selected_batch] == ["user", "assistant"]
    assert "only user" in str(provider.complete_requests[0].messages[1]["content"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "cursor", "expected_roles"),
    (
        ("before-user", 0, ("user", "assistant")),
        ("after-user", 1, ("assistant",)),
        ("inside-tools", 2, ("tool",)),
        ("run-boundary", 2, ("user", "assistant")),
    ),
    ids=("before-user", "after-user", "inside-tools", "run-boundary"),
)
async def test_cursor_intersects_recovered_run_boundary_without_selecting_fragments(
    workspace: Path,
    case: str,
    cursor: int,
    expected_roles: tuple[str, ...],
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    if case == "inside-tools":
        session.add_message("user", "tool run user")
        _add_tool_run(session, "tool")
    else:
        _add_run(session, "first", size=1_000)
        _add_run(session, "second", size=10)
    session.last_compacted = cursor
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)

    result = await _prepare_controller(
        controller,
        current_user="cursor current " + "c" * 2_000,
        context_window=1_200,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    assert tuple(message["role"] for message in result.selected_batch) == expected_roles
    assert controller.pending_last_compacted == cursor + len(expected_roles)


@pytest.mark.asyncio
async def test_fact_summary_contains_complete_tool_results_and_current_user_is_excluded(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    session.add_message("user", "old user " + "u" * 800)
    _add_tool_run(session, "complete", result_size=900)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)

    await _prepare_controller(
        controller,
        current_user="must not be summarized " + "n" * 1_800,
        context_window=1_200,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    fact_payload = str(provider.complete_requests[0].messages[1]["content"])
    action_payload = str(provider.complete_requests[1].messages[1]["content"])
    assert "complete tool result" in fact_payload
    assert "must not be summarized" not in fact_payload
    assert "must not be summarized" not in action_payload


@pytest.mark.asyncio
async def test_consecutive_staging_consumes_only_new_batch_and_replaces_action_summary(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "first", size=1_000)
    _add_run(session, "second", size=1_000)
    _add_run(session, "latest", size=50)
    provider = ScriptedFakeProvider(
        completions=(
            _response("facts one"),
            _response("action one"),
            _response("facts two"),
            _response("action two"),
        )
    )
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "context_window": 1_800,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=4_000),
    }

    first = await _prepare_controller(
        controller,
        current_user="new user " + "n" * 3_000,
        **kwargs,
    )
    second = await _prepare_controller(
        controller,
        current_user="new user changed " + "n" * 3_000,
        **kwargs,
    )

    assert first.selected_batch
    assert second.selected_batch
    assert "first user" in str(provider.complete_requests[0].messages[1]["content"])
    assert "first user" not in str(provider.complete_requests[2].messages[1]["content"])
    assert "latest user" in str(provider.complete_requests[2].messages[1]["content"])
    assert "action one" in str(provider.complete_requests[3].messages[1]["content"])
    assert controller.pending_action_summary == "action two"
    assert len(provider.complete_requests) == 4


@pytest.mark.asyncio
async def test_action_none_stages_removal_of_previous_action_summary(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=800)
    session.update_metadata(summary="previous action")
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("None")))
    controller = _controller(workspace, session, provider)

    await _prepare_controller(
        controller,
        context_window=1_000,
        max_output=200,
        memory_route_status=_memory_status(context_window=4_000),
    )

    assert controller.pending_action_summary is None
    assert "previous action" in str(provider.complete_requests[1].messages[1]["content"])


@pytest.mark.asyncio
async def test_fact_model_failure_does_not_stage_cursor_action_or_summary(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=800)
    session.update_metadata(summary="previous action")
    failure = ModelCallError(ErrorInfo(code="model_failed", message="fact failed"))
    provider = ScriptedFakeProvider(completions=(failure,))
    controller = _controller(workspace, session, provider)

    with pytest.raises(ModelCallError):
        await _prepare_controller(
            controller,
            context_window=1_000,
            max_output=200,
            memory_route_status=_memory_status(context_window=4_000),
        )
    with pytest.raises(ModelCallError):
        await _prepare_controller(
            controller,
            context_window=1_000,
            max_output=200,
            memory_route_status=_memory_status(context_window=4_000),
        )

    assert controller.pending_last_compacted == 0
    assert controller.pending_action_summary == "previous action"
    assert controller.pending_compaction_usage["model_calls"] == 0
    assert not (state.memory_directory / "summary.jsonl").exists()
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_fact_persistence_failure_keeps_batch_uncommitted_but_stages_usage(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=800)
    session.update_metadata(summary="previous action")
    (state.memory_directory / "summary.jsonl").mkdir()
    provider = ScriptedFakeProvider(
        completions=(_response("facts", input_tokens=7, output_tokens=3),)
    )
    controller = _controller(workspace, session, provider)

    with pytest.raises(ModelCallError, match="could not be persisted"):
        await _prepare_controller(
            controller,
            context_window=1_000,
            max_output=200,
            memory_route_status=_memory_status(context_window=4_000),
        )

    assert controller.pending_last_compacted == 0
    assert controller.pending_action_summary == "previous action"
    assert controller.pending_compaction_usage == {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_action_failure_keeps_fact_and_earlier_staged_state_without_advancing_cursor(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=800)
    session.update_metadata(summary="previous action")
    failure = ModelCallError(ErrorInfo(code="model_failed", message="action failed"))
    provider = ScriptedFakeProvider(
        completions=(_response("facts"), failure, _response("recovered action"))
    )
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "context_window": 1_000,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=4_000),
    }

    with pytest.raises(ModelCallError):
        await _prepare_controller(controller, **kwargs)

    assert controller.pending_last_compacted == 0
    assert controller.pending_action_summary == "previous action"
    assert controller.pending_compaction_usage["model_calls"] == 1
    assert "facts" in (state.memory_directory / "summary.jsonl").read_text(encoding="utf-8")
    assert len(provider.complete_requests) == 2

    with pytest.raises(ModelCallError):
        await _prepare_controller(controller, **kwargs)
    assert len(provider.complete_requests) == 2

    recovered = await _prepare_controller(controller, current_user="changed user", **kwargs)

    summary_content = (state.memory_directory / "summary.jsonl").read_text(encoding="utf-8")
    assert recovered.compacted is True
    assert summary_content.count('"content":"facts"') == 1
    assert controller.pending_last_compacted == len(session.messages)
    assert controller.pending_action_summary == "recovered action"
    assert controller.pending_compaction_usage["model_calls"] == 2
    assert len(provider.complete_requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "error_code"),
    (("length", "model_failed"), ("cancelled", "turn_cancelled")),
)
async def test_action_finish_failure_keeps_orphan_fact_and_usage_without_advancing(
    workspace: Path,
    finish_reason: Literal["length", "cancelled"],
    error_code: str,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=800)
    session.update_metadata(summary="previous action")
    action_response = ModelResponse(
        message=AssistantModelMessage(content="incomplete action"),
        usage=ModelUsage(input_tokens=8, output_tokens=3, total_tokens=11),
        finish_reason=finish_reason,
    )
    provider = ScriptedFakeProvider(completions=(_response("facts"), action_response))
    controller = _controller(workspace, session, provider)

    with pytest.raises(ModelCallError) as raised:
        await _prepare_controller(
            controller,
            context_window=1_000,
            max_output=200,
            memory_route_status=_memory_status(context_window=4_000),
        )

    assert raised.value.error.code == error_code
    assert controller.pending_last_compacted == 0
    assert controller.pending_action_summary == "previous action"
    assert controller.pending_compaction_usage == {
        "model_calls": 2,
        "input_tokens": 28,
        "output_tokens": 8,
        "total_tokens": 36,
    }
    assert "facts" in (state.memory_directory / "summary.jsonl").read_text(encoding="utf-8")
    assert len(provider.complete_requests) == 2


@pytest.mark.asyncio
async def test_summary_hard_overflow_is_rejected_before_provider_call(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=800)
    provider = ScriptedFakeProvider()
    controller = _controller(workspace, session, provider)

    with pytest.raises(ModelCallError) as raised:
        await _prepare_controller(
            controller,
            context_window=1_000,
            max_output=200,
            memory_route_status=_memory_status(context_window=20, max_output=10),
        )

    assert raised.value.error.code == "model_context_overflow"
    assert provider.complete_requests == []
    assert controller.checked_context_revision is not None


@pytest.mark.asyncio
async def test_action_replacement_is_checked_against_the_final_hard_limit(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=800)
    oversized_action = "action " + "x" * 4_000
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response(oversized_action)))
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "context_window": 1_000,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=10_000),
    }

    with pytest.raises(ModelCallError) as raised:
        await _prepare_controller(controller, **kwargs)

    assert raised.value.error.code == "model_context_overflow"
    assert controller.pending_last_compacted == len(session.messages)
    assert controller.pending_action_summary == oversized_action
    assert controller.pending_compaction_usage["model_calls"] == 2
    assert len(provider.complete_requests) == 2

    with pytest.raises(ModelCallError):
        await _prepare_controller(controller, **kwargs)
    assert len(provider.complete_requests) == 2


@pytest.mark.asyncio
async def test_latest_usage_context_uses_only_compatible_main_agent_assistant(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    session.add_message(
        "user",
        "old user",
    )
    session.add_message(
        "assistant",
        "old answer",
        tool_calls=[],
        status="completed",
        error=None,
        token_usage=_usage(100, 10),
        context_usage={
            "requested_route": "chat",
            "selected_route": "chat",
            "provider_id": "provider",
            "model": "model",
            "context_window": 1_600,
            "max_output": 200,
            "anchor_estimated_tokens": 20,
            "estimator_version": "utf8-bytes-div4-v1",
            "run_projected_tokens": 80,
            "run_projection_source": "estimated",
        },
    )
    controller = _controller(workspace, session, ScriptedFakeProvider())

    result = await _prepare_controller(
        controller,
        context_window=1_600,
        max_output=200,
        provider_id="provider",
        model="model",
    )

    assert controller.latest_usage_context is not None
    assert controller.latest_usage_context.provider_id == "provider"
    assert result.projection_source == "reported_delta"
    assert controller.pending_compaction_usage["model_calls"] == 0


@pytest.mark.asyncio
async def test_incompatible_latest_main_usage_does_not_fall_back_to_an_older_anchor(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    for label, provider_id in (("older", "provider"), ("latest", "other-provider")):
        session.add_message("user", f"{label} user")
        session.add_message(
            "assistant",
            f"{label} answer",
            tool_calls=[],
            status="completed",
            error=None,
            token_usage=_usage(100, 10),
            context_usage={
                "requested_route": "chat",
                "selected_route": "chat",
                "provider_id": provider_id,
                "model": "model",
                "context_window": 1_600,
                "max_output": 200,
                "anchor_estimated_tokens": 20,
                "estimator_version": "utf8-bytes-div4-v1",
                "run_projected_tokens": 80,
                "run_projection_source": "estimated",
            },
        )
    controller = _controller(workspace, session, ScriptedFakeProvider())

    result = await _prepare_controller(
        controller,
        context_window=1_600,
        max_output=200,
        provider_id="provider",
        model="model",
    )

    assert controller.latest_usage_context is not None
    assert controller.latest_usage_context.provider_id == "other-provider"
    assert result.projection_source == "estimated"


@pytest.mark.asyncio
async def test_same_context_revision_is_a_noop_after_successful_staging(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "old", size=800)
    provider = ScriptedFakeProvider(completions=(_response("facts"), _response("action")))
    controller = _controller(workspace, session, provider)
    kwargs: dict[str, Any] = {
        "context_window": 1_000,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=4_000),
    }

    first = await _prepare_controller(controller, **kwargs)
    second = await _prepare_controller(controller, **kwargs)

    assert first.compacted is True
    assert second.compacted is False
    assert second.selected_batch == ()
    assert second.context_revision == first.context_revision
    assert controller.checked_context_revision == first.context_revision
    assert len(provider.complete_requests) == 2


@pytest.mark.asyncio
async def test_visible_tool_schema_change_rechecks_a_new_revision(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    _add_run(session, "first", size=800)
    _add_run(session, "second", size=800)
    _add_run(session, "latest", size=50)
    provider = ScriptedFakeProvider(
        completions=(
            _response("facts one"),
            _response("action one"),
            _response("facts two"),
            _response("action two"),
        )
    )
    controller = _controller(workspace, session, provider)
    base_kwargs: dict[str, Any] = {
        "context_window": 1_800,
        "max_output": 200,
        "memory_route_status": _memory_status(context_window=4_000),
    }
    first = await _prepare_controller(controller, **base_kwargs)
    second = await _prepare_controller(
        controller,
        **base_kwargs,
        current_user="changed user " + "c" * 1_200,
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "new_tool",
                    "description": "x" * 2_000,
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
    )

    assert first.context_revision != second.context_revision
    assert second.compacted is True
    assert len(provider.complete_requests) == 4


def _estimate_latest_run(session: Session) -> int:
    user_indices = [
        index for index, message in enumerate(session.messages) if message["role"] == "user"
    ]
    return estimate_run_slice_tokens(session.messages[user_indices[-1] :])
