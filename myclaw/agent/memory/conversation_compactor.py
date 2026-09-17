"""Synchronous Conversation Compaction selection and Conversation Summary persistence."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Protocol, cast

from myclaw.agent.context_budget import (
    CONTEXT_ESTIMATOR_VERSION,
    ContextBudget,
    ContextProjection,
    ContextUsageSnapshot,
    ProjectionSource,
    estimate_request_tokens,
    project_next_request_tokens,
)
from myclaw.agent.memory.manager import MemoryManager
from myclaw.agent.session.session import Session
from myclaw.errors import MODEL_CONTEXT_OVERFLOW_MESSAGE, TURN_CANCELLED_MESSAGE, ErrorInfo
from myclaw.logging.session import without_session_log
from myclaw.management.service import RuntimeStatusInput, estimate_input_tokens
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelAttemptGuard, ModelRouteStatus
from myclaw.provider.models import ModelMessages, ModelResponse, ModelRoute
from myclaw.templates import render_template

type CompactionProjection = Callable[
    [Sequence[dict[str, Any]]],
    list[dict[str, Any]],
]

_COMPACTION_JSON_TRANSLATION = str.maketrans({"`": r"\u0060"})

__all__ = [
    "AgentRunContextController",
    "AgentRunContextModelRouter",
    "AgentRunContextPreparation",
    "AgentRunContextSnapshot",
    "AgentRunStagedValues",
    "AgentRunTerminalCommitValues",
    "CompactionModelRouter",
    "ConversationCompactor",
]


class CompactionModelRouter(Protocol):
    """The direct Router seam used for the specialized memory model call."""

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse: ...


class AgentRunContextModelRouter(Protocol):
    """Router seam for one guarded logical Memory completion."""

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[dict[str, Any]],
        guard: ModelAttemptGuard | None = None,
    ) -> ModelResponse: ...


@dataclass(frozen=True, slots=True)
class AgentRunContextSnapshot:
    """Detached Session state captured at Agent Run start."""

    messages: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]
    last_compacted: int

    def __post_init__(self) -> None:
        if isinstance(self.messages, (str, bytes)):
            raise TypeError("Agent Run snapshot messages must be a sequence")
        if self.last_compacted < 0 or self.last_compacted > len(self.messages):
            raise ValueError("Agent Run snapshot cursor is outside its transcript")
        object.__setattr__(self, "messages", tuple(deepcopy(list(self.messages))))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @classmethod
    def from_session(cls, session: Session) -> AgentRunContextSnapshot:
        """Copy the complete raw Session transcript and staged metadata."""
        return cls(
            messages=tuple(session.messages),
            metadata=session.metadata,
            last_compacted=session.last_compacted,
        )


@dataclass(frozen=True, slots=True)
class AgentRunStagedValues:
    """Detached values that a later terminal Session commit may publish."""

    pending_last_compacted: int
    pending_action_summary: str | None
    pending_compaction_usage: Mapping[str, int]
    current_user_compacted: bool
    latest_usage_context: ContextUsageSnapshot | None
    checked_context_revision: str | None


@dataclass(frozen=True, slots=True)
class AgentRunTerminalCommitValues:
    """Detached values accepted by ``Session.commit_agent_run``."""

    pending_last_compacted: int
    pending_action_summary: str | None
    usage_delta: dict[str, int]


@dataclass(frozen=True, slots=True)
class AgentRunContextPreparation:
    """Result of checking one run-start request against the staged controller."""

    selected_batch: tuple[dict[str, Any], ...]
    projected_tokens: int
    projection_source: ProjectionSource
    context_revision: str
    compacted: bool

    @property
    def selected_messages(self) -> tuple[dict[str, Any], ...]:
        """Alias used by callers that name the selected raw batch messages."""
        return self.selected_batch


@dataclass(frozen=True, slots=True)
class _PendingFactBatch:
    batch: tuple[dict[str, Any], ...]
    cutoff: int
    selected_payload: str


class AgentRunContextController:
    """Run-local staged context state shared by Run-start and future ReAct preparation."""

    def __init__(
        self,
        *,
        snapshot: AgentRunContextSnapshot,
        provider: AgentRunContextModelRouter,
        memory_manager: MemoryManager,
        now: Callable[[], datetime],
    ) -> None:
        if not isinstance(snapshot, AgentRunContextSnapshot):
            raise TypeError("Agent Run controller requires a detached snapshot")
        if not callable(now):
            raise TypeError("Agent Run controller requires a clock")
        self._snapshot = AgentRunContextSnapshot(
            messages=snapshot.messages,
            metadata=snapshot.metadata,
            last_compacted=snapshot.last_compacted,
        )
        self._provider = provider
        self._memory_manager = memory_manager
        self._now = now
        self._pending_last_compacted = snapshot.last_compacted
        self._pending_action_summary = _normalized_staged_action_summary(
            snapshot.metadata.get("summary")
        )
        self._pending_compaction_usage = _empty_usage()
        self._current_user_compacted = False
        self._usage_history = _main_agent_usage_history(snapshot.messages)
        self._latest_usage_context = self._usage_history[0][0] if self._usage_history else None
        self._base_context_revision: str | None = None
        self._checked_context_revision: str | None = None
        self._pending_fact: _PendingFactBatch | None = None
        self._failed_context_revision: str | None = None
        self._failed_exception: Exception | None = None

    @classmethod
    def from_session(
        cls,
        session: Session,
        *,
        provider: AgentRunContextModelRouter,
        memory_manager: MemoryManager,
        now: Callable[[], datetime],
    ) -> AgentRunContextController:
        """Create a controller without retaining the writable Session object."""
        return cls(
            snapshot=AgentRunContextSnapshot.from_session(session),
            provider=provider,
            memory_manager=memory_manager,
            now=now,
        )

    @property
    def pending_last_compacted(self) -> int:
        return self._pending_last_compacted

    @property
    def pending_action_summary(self) -> str | None:
        return self._pending_action_summary

    @property
    def pending_compaction_usage(self) -> dict[str, int]:
        return dict(self._pending_compaction_usage)

    @property
    def current_user_compacted(self) -> bool:
        return self._current_user_compacted

    @property
    def latest_usage_context(self) -> ContextUsageSnapshot | None:
        return self._latest_usage_context

    @property
    def checked_context_revision(self) -> str | None:
        return self._checked_context_revision

    @property
    def base_context_revision(self) -> str | None:
        return self._base_context_revision

    def staged_values(self) -> AgentRunStagedValues:
        """Return detached staged values for a later terminal commit."""
        return AgentRunStagedValues(
            pending_last_compacted=self._pending_last_compacted,
            pending_action_summary=self._pending_action_summary,
            pending_compaction_usage=dict(self._pending_compaction_usage),
            current_user_compacted=self._current_user_compacted,
            latest_usage_context=self._latest_usage_context,
            checked_context_revision=self._checked_context_revision,
        )

    def terminal_commit_values(self) -> AgentRunTerminalCommitValues:
        """Return only values accepted by the dormant terminal Session commit."""
        return AgentRunTerminalCommitValues(
            pending_last_compacted=self._pending_last_compacted,
            pending_action_summary=self._pending_action_summary,
            usage_delta=dict(self._pending_compaction_usage),
        )

    async def prepare_run_start(
        self,
        *,
        project_messages: CompactionProjection,
        route_context_window: int,
        route_max_output: int,
        tools: Sequence[dict[str, Any]] = (),
        current_user: dict[str, Any] | None = None,
        compact_ratio: float = 0.9,
        route_status: ModelRouteStatus | None = None,
        memory_route_status: ModelRouteStatus | None = None,
        requested_route: str = "chat",
        selected_route: str | None = None,
        provider_id: str = "",
        model: str = "",
        estimator_version: str = CONTEXT_ESTIMATOR_VERSION,
    ) -> AgentRunContextPreparation:
        """Check and stage Run-start history compression without publishing Session state."""
        budget = ContextBudget(
            context_window=route_context_window,
            max_output=route_max_output,
            compact_ratio=compact_ratio,
        )
        route_values = _route_projection_values(
            route_status=route_status,
            requested_route=requested_route,
            selected_route=selected_route,
            provider_id=provider_id,
            model=model,
            context_window=route_context_window,
            max_output=route_max_output,
        )
        effective_tools = tuple(deepcopy(list(tools)))
        copied_user = None if current_user is None else deepcopy(current_user)
        projected = self._project_candidate(
            project_messages,
            current_user=copied_user,
            tools=effective_tools,
        )
        projection = self._projection_from_candidate(
            projected,
            tools=effective_tools,
            route_values=route_values,
            estimator_version=estimator_version,
        )
        compatible_context = self._compatible_usage_context(route_values, estimator_version)
        if compatible_context is not None:
            self._latest_usage_context = compatible_context
        revision = self._context_revision(
            project_messages=project_messages,
            current_user=copied_user,
            tools=effective_tools,
            projected=projected,
            route_values=route_values,
            memory_route_status=memory_route_status,
            compact_ratio=compact_ratio,
            estimator_version=estimator_version,
        )
        if self._base_context_revision is None or self._base_context_revision != revision:
            self._base_context_revision = revision
        if self._failed_context_revision == revision:
            assert self._failed_exception is not None
            raise self._failed_exception
        self._failed_context_revision = None
        self._failed_exception = None
        if self._checked_context_revision == revision:
            return AgentRunContextPreparation(
                selected_batch=(),
                projected_tokens=projection.projected_tokens,
                projection_source=projection.source,
                context_revision=revision,
                compacted=False,
            )

        protected_projection = self._projected_tokens(
            project_messages,
            raw_messages=(),
            current_user=copied_user,
            tools=effective_tools,
            route_values=route_values,
            estimator_version=estimator_version,
        )
        if protected_projection.projected_tokens >= budget.available_context:
            overflow_error = _model_context_overflow()
            self._record_failure(revision, overflow_error)
            raise overflow_error
        if self._pending_fact is None and not budget.should_compact(projection.projected_tokens):
            self._checked_context_revision = revision
            return AgentRunContextPreparation(
                selected_batch=(),
                projected_tokens=projection.projected_tokens,
                projection_source=projection.source,
                context_revision=revision,
                compacted=False,
            )

        pending_fact = self._pending_fact
        if pending_fact is None:
            batch, cutoff = self._select_run_start_batch(budget)
        else:
            batch = tuple(deepcopy(list(pending_fact.batch)))
            cutoff = pending_fact.cutoff
        if batch and pending_fact is None:
            retained_projection = self._projected_tokens(
                project_messages,
                raw_messages=self._snapshot.messages[cutoff:],
                current_user=copied_user,
                tools=effective_tools,
                route_values=route_values,
                estimator_version=estimator_version,
            )
            if retained_projection.projected_tokens >= budget.available_context:
                all_batch, all_cutoff = self._all_eligible_batch()
                if all_batch and all_cutoff != cutoff:
                    all_projection = self._projected_tokens(
                        project_messages,
                        raw_messages=self._snapshot.messages[all_cutoff:],
                        current_user=copied_user,
                        tools=effective_tools,
                        route_values=route_values,
                        estimator_version=estimator_version,
                    )
                    if all_projection.projected_tokens < budget.available_context:
                        batch, cutoff = all_batch, all_cutoff
                    else:
                        overflow_error = _model_context_overflow()
                        self._record_failure(revision, overflow_error)
                        raise overflow_error
                else:
                    overflow_error = _model_context_overflow()
                    self._record_failure(revision, overflow_error)
                    raise overflow_error
        else:
            if not batch:
                self._checked_context_revision = revision
                if projection.projected_tokens >= budget.available_context:
                    overflow_error = _model_context_overflow()
                    self._record_failure(revision, overflow_error)
                    raise overflow_error
                return AgentRunContextPreparation(
                    selected_batch=(),
                    projected_tokens=projection.projected_tokens,
                    projection_source=projection.source,
                    context_revision=revision,
                    compacted=False,
                )

        selected_payload = (
            _compaction_user_context(list(batch))
            if pending_fact is None
            else pending_fact.selected_payload
        )
        memory_context_window, memory_max_output = _memory_budget_values(
            memory_route_status=memory_route_status,
            fallback_context_window=route_context_window,
            fallback_max_output=route_max_output,
        )
        if pending_fact is None:
            fact_messages = _summary_request_messages(
                template_name="conversation-compaction-system-prompt.md",
                selected_payload=selected_payload,
            )
            if estimate_request_tokens(fact_messages) >= memory_context_window - memory_max_output:
                overflow_error = _model_context_overflow()
                self._record_failure(revision, overflow_error)
                raise overflow_error

            try:
                fact_response = await self._provider.complete(
                    "memory",
                    messages=fact_messages,
                    tools=(),
                    guard=_summary_hard_guard,
                )
            except Exception as provider_error:
                self._record_failure(revision, provider_error)
                raise
            _add_pending_usage(self._pending_compaction_usage, fact_response)
            response_error = _summary_response_error(fact_response)
            if response_error is not None:
                self._record_failure(revision, response_error)
                raise response_error
            try:
                await self._memory_manager.append_summary(
                    content=fact_response.message.content,
                    timestamp=self._persisted_now(),
                )
            except (OSError, UnicodeError, ValueError) as persistence_cause:
                persistence_error = ModelCallError(
                    ErrorInfo(
                        code="persistence_error",
                        message="Conversation Summary could not be persisted.",
                    )
                )
                self._record_failure(revision, persistence_error)
                raise persistence_error from persistence_cause
            self._pending_fact = _PendingFactBatch(
                batch=tuple(deepcopy(list(batch))),
                cutoff=cutoff,
                selected_payload=selected_payload,
            )

        action_messages = _summary_request_messages(
            template_name="conversation-summary-system-prompt.md",
            selected_payload=_action_summary_user_context(
                self._pending_action_summary,
                selected_payload,
            ),
        )
        if estimate_request_tokens(action_messages) >= memory_context_window - memory_max_output:
            overflow_error = _model_context_overflow()
            self._record_failure(revision, overflow_error)
            raise overflow_error
        try:
            action_response = await self._provider.complete(
                "memory",
                messages=action_messages,
                tools=(),
                guard=_summary_hard_guard,
            )
        except Exception as provider_error:
            self._record_failure(revision, provider_error)
            raise
        _add_pending_usage(self._pending_compaction_usage, action_response)
        response_error = _summary_response_error(action_response)
        if response_error is not None:
            self._record_failure(revision, response_error)
            raise response_error
        self._pending_action_summary = _normalize_action_summary(action_response.message.content)
        self._pending_last_compacted = cutoff
        self._pending_fact = None
        final_projected = self._project_candidate(
            project_messages,
            current_user=copied_user,
            tools=effective_tools,
        )
        final_projection = self._projection_from_candidate(
            final_projected,
            tools=effective_tools,
            route_values=route_values,
            estimator_version=estimator_version,
        )
        final_revision = self._context_revision(
            project_messages=project_messages,
            current_user=copied_user,
            tools=effective_tools,
            projected=final_projected,
            route_values=route_values,
            memory_route_status=memory_route_status,
            compact_ratio=compact_ratio,
            estimator_version=estimator_version,
        )
        self._base_context_revision = final_revision
        self._checked_context_revision = final_revision
        if final_projection.projected_tokens >= budget.available_context:
            overflow_error = _model_context_overflow()
            self._record_failure(final_revision, overflow_error)
            raise overflow_error
        return AgentRunContextPreparation(
            selected_batch=tuple(deepcopy(list(batch))),
            projected_tokens=final_projection.projected_tokens,
            projection_source=final_projection.source,
            context_revision=final_revision,
            compacted=True,
        )

    def _project_candidate(
        self,
        project_messages: CompactionProjection,
        *,
        current_user: dict[str, Any] | None,
        tools: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        raw_messages = list(self._snapshot.messages[self._pending_last_compacted :])
        if current_user is not None:
            raw_messages.append(deepcopy(current_user))
        return _insert_action_summary(
            project_messages(deepcopy(raw_messages)),
            self._pending_action_summary,
        )

    def _projected_tokens(
        self,
        project_messages: CompactionProjection,
        *,
        raw_messages: Sequence[dict[str, Any]],
        current_user: dict[str, Any] | None,
        tools: Sequence[dict[str, Any]],
        route_values: _RouteProjectionValues,
        estimator_version: str,
    ) -> ContextProjection:
        source = list(deepcopy(list(raw_messages)))
        if current_user is not None:
            source.append(deepcopy(current_user))
        projected = _insert_action_summary(project_messages(source), self._pending_action_summary)
        return self._projection_from_candidate(
            projected,
            tools=tools,
            route_values=route_values,
            estimator_version=estimator_version,
        )

    def _projection_from_candidate(
        self,
        projected: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]],
        route_values: _RouteProjectionValues,
        estimator_version: str,
    ) -> ContextProjection:
        return project_next_request_tokens(
            estimate_request_tokens(projected, tools),
            snapshot=self._compatible_usage_context(route_values, estimator_version),
            reported_usage=self._compatible_usage(route_values, estimator_version),
            requested_route=route_values.requested_route,
            selected_route=route_values.selected_route,
            provider_id=route_values.provider_id,
            model=route_values.model,
            context_window=route_values.context_window,
            max_output=route_values.max_output,
            estimator_version=estimator_version,
        )

    def _record_failure(self, revision: str, error: Exception) -> None:
        self._base_context_revision = revision
        self._checked_context_revision = revision
        self._failed_context_revision = revision
        self._failed_exception = error

    def _select_run_start_batch(
        self, budget: ContextBudget
    ) -> tuple[tuple[dict[str, Any], ...], int]:
        eligible = self._eligible_runs()
        if not eligible:
            return (), self._pending_last_compacted
        if len(eligible) == 1:
            selected = eligible
        else:
            latest_start, latest_end = eligible[-1]
            latest_suffix = self._snapshot.messages[
                max(self._pending_last_compacted, latest_start) : latest_end
            ]
            selected = (
                eligible[:-1]
                if budget.can_retain_run_slice(
                    latest_suffix,
                    percentage=10,
                )
                else eligible
            )
        return self._batch_from_runs(selected)

    def _all_eligible_batch(self) -> tuple[tuple[dict[str, Any], ...], int]:
        return self._batch_from_runs(self._eligible_runs())

    def _batch_from_runs(
        self,
        runs: Sequence[tuple[int, int]],
    ) -> tuple[tuple[dict[str, Any], ...], int]:
        if not runs:
            return (), self._pending_last_compacted
        start = max(self._pending_last_compacted, runs[0][0])
        cutoff = runs[-1][1]
        if cutoff <= start:
            return (), self._pending_last_compacted
        return tuple(deepcopy(self._snapshot.messages[start:cutoff])), cutoff

    def _eligible_runs(self) -> list[tuple[int, int]]:
        runs = _completed_run_ranges(self._snapshot.messages)
        return [
            (max(self._pending_last_compacted, start), end)
            for start, end in runs
            if end > self._pending_last_compacted
        ]

    def _compatible_usage_context(
        self,
        route_values: _RouteProjectionValues,
        estimator_version: str,
    ) -> ContextUsageSnapshot | None:
        return next(
            (
                context
                for context, _usage in self._usage_history
                if _usage_context_matches(context, route_values, estimator_version)
            ),
            None,
        )

    def _compatible_usage(
        self,
        route_values: _RouteProjectionValues,
        estimator_version: str,
    ) -> dict[str, int] | None:
        return next(
            (
                usage
                for context, usage in self._usage_history
                if _usage_context_matches(context, route_values, estimator_version)
            ),
            None,
        )

    def _context_revision(
        self,
        *,
        project_messages: CompactionProjection,
        current_user: dict[str, Any] | None,
        tools: Sequence[dict[str, Any]],
        projected: Sequence[dict[str, Any]],
        route_values: _RouteProjectionValues,
        memory_route_status: ModelRouteStatus | None,
        compact_ratio: float,
        estimator_version: str,
    ) -> str:
        value = {
            "transcript": self._snapshot.messages,
            "pending_last_compacted": self._pending_last_compacted,
            "pending_action_summary": self._pending_action_summary,
            "current_user": current_user,
            "tools": list(tools),
            "projected": list(projected),
            "route": route_values.to_dict(),
            "compact_ratio": compact_ratio,
            "memory_route": (
                None
                if memory_route_status is None
                else {
                    "requested_route": memory_route_status.requested_route,
                    "selected_route": memory_route_status.selected_route,
                    "provider_id": memory_route_status.provider_id,
                    "model": memory_route_status.model,
                    "context_window": memory_route_status.context_window,
                    "max_output": memory_route_status.max_output,
                }
            ),
            "estimator_version": estimator_version,
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)


@dataclass(frozen=True, slots=True)
class _RouteProjectionValues:
    requested_route: str
    selected_route: str
    provider_id: str
    model: str
    context_window: int
    max_output: int

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_route": self.requested_route,
            "selected_route": self.selected_route,
            "provider_id": self.provider_id,
            "model": self.model,
            "context_window": self.context_window,
            "max_output": self.max_output,
        }


def _route_projection_values(
    *,
    route_status: ModelRouteStatus | None,
    requested_route: str,
    selected_route: str | None,
    provider_id: str,
    model: str,
    context_window: int,
    max_output: int,
) -> _RouteProjectionValues:
    if route_status is not None:
        return _RouteProjectionValues(
            requested_route=route_status.requested_route,
            selected_route=route_status.selected_route,
            provider_id=route_status.provider_id,
            model=route_status.model,
            context_window=route_status.context_window,
            max_output=route_status.max_output,
        )
    return _RouteProjectionValues(
        requested_route=requested_route,
        selected_route=requested_route if selected_route is None else selected_route,
        provider_id=provider_id,
        model=model,
        context_window=context_window,
        max_output=max_output,
    )


def _memory_budget_values(
    *,
    memory_route_status: ModelRouteStatus | None,
    fallback_context_window: int,
    fallback_max_output: int,
) -> tuple[int, int]:
    if memory_route_status is not None:
        return memory_route_status.context_window, memory_route_status.max_output
    return fallback_context_window, fallback_max_output


def _completed_run_ranges(messages: Sequence[dict[str, Any]]) -> list[tuple[int, int]]:
    starts = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else len(messages))
        for index, start in enumerate(starts)
    ]


def _normalized_staged_action_summary(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Session action summary must be a string or None")
    return None if not value or value.strip() == "None" else value


def _main_agent_usage_history(
    messages: Sequence[dict[str, Any]],
) -> list[tuple[ContextUsageSnapshot, dict[str, int]]]:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        context_value = message.get("context_usage")
        usage_value = message.get("token_usage")
        if context_value is None or not _valid_main_agent_usage(usage_value):
            return []
        try:
            context = ContextUsageSnapshot.from_dict(context_value)
        except (TypeError, ValueError):
            return []
        if context.requested_route not in {"chat", "schedule"}:
            return []
        assert isinstance(usage_value, dict)
        return [(context, deepcopy(usage_value))]
    return []


def _valid_main_agent_usage(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value) != {"model_calls", "input_tokens", "output_tokens", "total_tokens"}:
        return False
    values = tuple(
        value[field] for field in ("model_calls", "input_tokens", "output_tokens", "total_tokens")
    )
    if any(type(item) is not int or item < 0 for item in values):
        return False
    model_calls = cast(int, values[0])
    input_tokens = cast(int, values[1])
    output_tokens = cast(int, values[2])
    total_tokens = cast(int, values[3])
    return (
        model_calls == 1
        and input_tokens > 0
        and output_tokens > 0
        and total_tokens == input_tokens + output_tokens
    )


def _usage_context_matches(
    context: ContextUsageSnapshot,
    route_values: _RouteProjectionValues,
    estimator_version: str,
) -> bool:
    return (
        context.requested_route == route_values.requested_route
        and context.selected_route == route_values.selected_route
        and context.provider_id == route_values.provider_id
        and context.model == route_values.model
        and context.context_window == route_values.context_window
        and context.max_output == route_values.max_output
        and context.estimator_version == estimator_version
    )


def _empty_usage() -> dict[str, int]:
    return {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _add_pending_usage(target: dict[str, int], response: ModelResponse) -> None:
    target["model_calls"] += 1
    target["input_tokens"] += response.usage.input_tokens
    target["output_tokens"] += response.usage.output_tokens
    target["total_tokens"] += response.usage.total_tokens


def _summary_response_error(response: ModelResponse) -> ModelCallError | None:
    if response.finish_reason == "stop" and not response.message.tool_calls:
        return None
    if response.finish_reason == "cancelled":
        return ModelCallError(ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE))
    return ModelCallError(
        ErrorInfo("model_failed", "Summary model response did not complete normally.")
    )


def _summary_hard_guard(
    status: ModelRouteStatus,
    messages: ModelMessages,
    tools: Sequence[dict[str, Any]],
) -> bool:
    available_context = status.context_window - status.max_output
    return estimate_request_tokens(messages, tools) < available_context


def _normalize_action_summary(content: str) -> str | None:
    normalized = content.strip()
    return None if normalized == "None" else content


def _action_summary_user_context(previous: str | None, selected_payload: str) -> str:
    if previous is None:
        return selected_payload
    return f"## Previous Action Summary\n\n{previous}\n\n{selected_payload}"


def _insert_action_summary(
    projected: Sequence[dict[str, Any]],
    action_summary: str | None,
) -> list[dict[str, Any]]:
    result = deepcopy(list(projected))
    if action_summary is None:
        return result
    index = 1 if result and result[0].get("role") == "system" else 0
    result.insert(index, {"role": "user", "content": deepcopy(action_summary)})
    return result


class ConversationCompactor:
    """Compress eligible early Session messages before an Agent Run model call."""

    def __init__(
        self,
        *,
        provider: CompactionModelRouter,
        memory_manager: MemoryManager,
        compaction_message_threshold: int,
        now: Callable[[], datetime],
    ) -> None:
        self._provider = provider
        self._memory_manager = memory_manager
        self._message_threshold = compaction_message_threshold
        self._now = now

    async def prepare(
        self,
        session: Session,
        *,
        project_messages: CompactionProjection,
        route_context_window: int,
        route_max_output: int,
        tools: Sequence[dict[str, Any]],
        current_user: dict[str, Any] | None = None,
        continuation: Sequence[dict[str, Any]] = (),
    ) -> Session:
        with without_session_log():
            return await self._prepare(
                session,
                current_user=current_user,
                continuation=continuation,
                project_messages=project_messages,
                route_context_window=route_context_window,
                route_max_output=route_max_output,
                tools=tools,
            )

    async def _prepare(
        self,
        session: Session,
        *,
        project_messages: CompactionProjection,
        route_context_window: int,
        route_max_output: int,
        tools: Sequence[dict[str, Any]],
        current_user: dict[str, Any] | None,
        continuation: Sequence[dict[str, Any]],
    ) -> Session:
        effective_tools = tuple(tools)
        short_term = _short_term_messages(
            session,
            current_user=current_user,
            continuation=continuation,
        )
        current_user_index = _last_user_message_index(short_term)
        complete_messages = project_messages(short_term)
        available_input = route_context_window - route_max_output
        fixed_request_tokens = _estimate_messages(
            complete_messages[:1],
            tools=effective_tools,
        )
        if fixed_request_tokens > available_input:
            raise _model_context_overflow()
        if current_user_index < len(short_term):
            non_compactable_messages = project_messages(short_term[current_user_index:])
            if (
                _estimate_messages(non_compactable_messages, tools=effective_tools)
                >= available_input
            ):
                raise _model_context_overflow()
        token_triggered = (
            _estimate_messages(complete_messages, tools=effective_tools) >= available_input
        )
        message_triggered = len(short_term) >= self._message_threshold
        if not token_triggered and not message_triggered:
            return session
        initial_cutoff = 0
        if token_triggered:
            initial_cutoff = _token_cutoff(
                short_term,
                current_user_index,
                available_input,
                project_messages,
                tools=effective_tools,
            )
        if message_triggered:
            initial_cutoff = max(
                initial_cutoff,
                min(self._message_threshold // 2, len(short_term) - 1),
            )
        cutoff = _aligned_cutoff(short_term, initial_cutoff)
        if cutoff == 0:
            raise _model_context_overflow()
        selected = short_term[:cutoff]
        selected_payload = _compaction_user_context(selected)
        fact_response = await self._provider.complete(
            "memory",
            messages=_summary_request_messages(
                template_name="conversation-compaction-system-prompt.md",
                selected_payload=selected_payload,
            ),
            tools=(),
        )
        session.update_metadata(usage_delta={"model_calls": 1, **fact_response.usage.to_dict()})
        try:
            new_last_compacted = session.last_compacted + cutoff
            await self._memory_manager.append_summary(
                content=fact_response.message.content,
                timestamp=self._persisted_now(),
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise ModelCallError(
                ErrorInfo(
                    code="persistence_error",
                    message="Conversation Summary could not be persisted.",
                )
            ) from error

        action_response = await self._provider.complete(
            "memory",
            messages=_summary_request_messages(
                template_name="conversation-summary-system-prompt.md",
                selected_payload=selected_payload,
            ),
            tools=(),
        )
        session.update_metadata(usage_delta={"model_calls": 1, **action_response.usage.to_dict()})
        action_summary = action_response.message.content
        if action_summary.strip() == "None":
            action_summary = ""
        session.update_metadata(summary=action_summary)
        session.last_compacted = new_last_compacted
        return session

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)


def _summary_request_messages(*, template_name: str, selected_payload: str) -> ModelMessages:
    return [
        {"role": "system", "content": render_template(template_name)},
        {"role": "user", "content": selected_payload},
    ]


def _short_term_messages(
    session: Session,
    *,
    current_user: dict[str, Any] | None = None,
    continuation: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    messages = list(session.messages[session.last_compacted :])
    if current_user is not None:
        messages.append(current_user)
    messages.extend(continuation)
    return messages


def _last_user_message_index(messages: Sequence[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return len(messages)


def _aligned_cutoff(messages: list[dict[str, Any]], initial: int) -> int:
    for index in range(initial, len(messages)):
        if messages[index].get("role") == "user":
            return index
    for index in range(initial - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return 0


def _token_cutoff(
    messages: Sequence[dict[str, Any]],
    current_user_index: int,
    input_budget: int,
    project_messages: CompactionProjection,
    tools: Sequence[dict[str, Any]],
) -> int:
    if current_user_index == len(messages):
        return len(messages)
    current_user = messages[current_user_index]
    continuation = messages[current_user_index + 1 :]
    tool_tokens = _estimate_messages((), tools=tools)
    target_bytes = max(input_budget - tool_tokens, 0) // 2 * 4
    for index in range(current_user_index):
        projected = project_messages([*messages[: index + 1], current_user, *continuation])
        selected_bytes = _projected_history_bytes(projected)
        if selected_bytes >= target_bytes:
            return index + 1
    return current_user_index


def _estimate_messages(
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[dict[str, Any]] = (),
) -> int:
    system_prompt = ""
    retained = messages
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if not isinstance(content, str):
            raise TypeError("Projected system message content must be a string")
        system_prompt = content
        retained = messages[1:]
    retained_messages = tuple(
        json.dumps(message, ensure_ascii=False, separators=(",", ":")) for message in retained
    )
    tool_definitions = tuple(
        json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools
    )
    return estimate_input_tokens(
        RuntimeStatusInput(
            system_prompt=system_prompt,
            retained_messages=retained_messages,
            tool_definitions=tool_definitions,
            runtime_context="",
        )
    )


def _model_context_overflow() -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="model_context_overflow",
            message=MODEL_CONTEXT_OVERFLOW_MESSAGE,
        )
    )


def _projected_history_bytes(messages: Sequence[dict[str, Any]]) -> int:
    current_user_index = _last_user_message_index(messages)
    history_end = len(messages) if current_user_index == len(messages) else current_user_index
    return sum(
        len(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        for message in messages[1:history_end]
    )


def _project_compaction_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message["role"]
    if role == "user":
        return {"role": "user", "content": deepcopy(message["content"])}

    if role == "assistant":
        content = message["content"]
        tool_calls = [
            {
                "id": deepcopy(tool_call["id"]),
                "name": deepcopy(tool_call["name"]),
                "arguments": deepcopy(tool_call["arguments"]),
            }
            for tool_call in message["tool_calls"]
        ]
        if message["status"] == "error" and not content and not tool_calls:
            return None
        if message["status"] == "interrupted":
            content = f"{content}\n\n[Turn interrupted by user.]"
        return {
            "role": "assistant",
            "content": deepcopy(content),
            "tool_calls": tool_calls,
        }

    return {
        "role": "tool",
        "tool_call_id": deepcopy(message["tool_call_id"]),
        "name": deepcopy(message["name"]),
        "content": deepcopy(message["content"]),
    }


def _compaction_user_context(messages: list[dict[str, Any]]) -> str:
    records = [
        projected
        for message in messages
        if (projected := _project_compaction_message(message)) is not None
    ]
    serialized = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).translate(_COMPACTION_JSON_TRANSLATION)
    return f"## Conversation Messages\n\n```json\n{serialized}\n```"
