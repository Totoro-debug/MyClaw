"""Model Route resolution and one shared Provider attempt budget."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Protocol, cast

from loguru import logger

from myclaw.config.config import ProviderConfiguration, ResolvedModelRoute, UserConfiguration
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    ModelContinuation,
    ModelMessages,
    ModelProvider,
    ModelResponse,
    ModelRoute,
    ModelStreamEvent,
)
from myclaw.tools.base import OpenAIToolSchema

_MAX_ATTEMPTS = 5
_RETRYABLE_CODES = frozenset({"provider_rate_limited", "provider_timeout", "provider_unavailable"})
_FALLBACK_CODES = frozenset({"route_unavailable", "provider_auth_error"})


class RetryClock(Protocol):
    async def sleep(self, seconds: float) -> None: ...


type ProviderImplementation = ModelProvider
type ProviderFactory = Callable[[ProviderConfiguration], ProviderImplementation]
type Jitter = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class ModelRouteStatus:
    """Secret-free snapshot of the route selected for one logical purpose."""

    requested_route: ModelRoute
    selected_route: ModelRoute
    provider_id: str
    model: str
    context_window: int
    used_default: bool


class ModelRouter:
    """Resolve a logical Model Route and coordinate Provider attempts."""

    def __init__(
        self,
        *,
        configuration: UserConfiguration,
        provider_factory: ProviderFactory,
        clock: RetryClock | None = None,
        jitter: Jitter | None = None,
    ) -> None:
        self._configuration = configuration
        self._provider_factory = provider_factory
        self._clock = clock
        self._jitter = jitter
        self._providers: dict[str, ProviderImplementation] = {}
        self._route_statuses: dict[ModelRoute, ModelRouteStatus] = {}
        self._current_call_statuses: ContextVar[dict[ModelRoute, ModelRouteStatus] | None] = (
            ContextVar("myclaw_model_router_call_statuses", default=None)
        )
        self._close_task: asyncio.Task[None] | None = None

    def route_status(self, requested_route: ModelRoute) -> ModelRouteStatus:
        """Return the current concrete route identity without provider credentials."""
        status = self._route_statuses.get(requested_route)
        if status is None:
            status = _route_status(
                requested_route,
                self._configuration.resolve_route(requested_route),
            )
            self._route_statuses[requested_route] = status
        return status

    def current_call_status(self, requested_route: ModelRoute) -> ModelRouteStatus | None:
        """Return the route selected by the current task's latest logical call."""
        statuses = self._current_call_statuses.get()
        return None if statuses is None else statuses.get(requested_route)

    def stream(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        return self._stream_direct(
            route,
            messages=messages,
            tools=tools,
            continuation=continuation,
        )

    async def _stream_direct(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        resolved = self._begin_call(route)
        if continuation is not None and continuation.provider_id != resolved.provider.provider_id:
            continuation = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            provider = self._provider(resolved.provider)
            emitted = False
            try:
                events = provider.stream(
                    messages=messages,
                    tools=tools,
                    model=resolved.route.model,
                    max_output=resolved.route.max_output,
                    temperature=resolved.route.temperature,
                    reasoning_effort=resolved.route.reasoning_effort,
                    timeout=resolved.route.timeout,
                    **(
                        {"continuation": continuation}
                        if continuation is not None
                        and continuation.provider_id == resolved.provider.provider_id
                        else {}
                    ),
                )
                async for event in events:
                    emitted = True
                    yield event
                return
            except ModelCallError as failure:
                if emitted:
                    raise
                resolved = await self._recover_attempt(resolved, failure, attempt=attempt)

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        return await self._complete_direct(
            route,
            messages=messages,
            tools=tools,
            continuation=continuation,
        )

    async def _complete_direct(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None,
    ) -> ModelResponse:
        resolved = self._begin_call(route)
        if continuation is not None and continuation.provider_id != resolved.provider.provider_id:
            continuation = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            provider = self._provider(resolved.provider)
            try:
                return await provider.complete(
                    messages=messages,
                    tools=tools,
                    model=resolved.route.model,
                    max_output=resolved.route.max_output,
                    temperature=resolved.route.temperature,
                    reasoning_effort=resolved.route.reasoning_effort,
                    timeout=resolved.route.timeout,
                    **(
                        {"continuation": continuation}
                        if continuation is not None
                        and continuation.provider_id == resolved.provider.provider_id
                        else {}
                    ),
                )
            except ModelCallError as failure:
                resolved = await self._recover_attempt(resolved, failure, attempt=attempt)

        raise AssertionError("Provider attempt budget exhausted without a terminal result")

    async def close(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_providers())
            self._close_task = task
        await asyncio.shield(task)

    async def _close_providers(self) -> None:
        providers = tuple(self._providers.values())
        self._providers.clear()
        closed: set[int] = set()
        unique: list[ProviderImplementation] = []
        for provider in providers:
            identity = id(provider)
            if identity in closed:
                continue
            closed.add(identity)
            unique.append(provider)
        results = await asyncio.gather(
            *(provider.close() for provider in unique),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Model Provider shutdown failed", failures)

    async def _recover_attempt(
        self,
        current: ResolvedModelRoute,
        failure: ModelCallError,
        *,
        attempt: int,
    ) -> ResolvedModelRoute:
        if attempt == _MAX_ATTEMPTS:
            raise failure
        code = failure.error.code
        if failure.error.retryable and code in _RETRYABLE_CODES:
            backoff = min(30.0, 0.5 * 2.0 ** (attempt - 1))
            if self._jitter is not None:
                backoff += min(backoff, max(0.0, self._jitter(backoff)))
            retry_after = float(failure.error.retry_after_seconds or 0.0)
            delay = min(60.0, max(backoff, retry_after))
            _log_retry(current, failure, attempt=attempt, delay=delay)
            if self._clock is None:
                await asyncio.sleep(delay)
            else:
                await self._clock.sleep(delay)
            return current
        allows_fallback = code in _FALLBACK_CODES or (
            code == "provider_unavailable" and not failure.error.retryable
        )
        if current.selected_route == "default" or not allows_fallback:
            raise failure
        fallback = self._configuration.resolve_route("default")
        if fallback.provider == current.provider and fallback.route == current.route:
            raise failure
        requested_route = cast(ModelRoute, current.requested_route)
        fallback = replace(
            fallback,
            requested_route=requested_route,
            used_default=True,
        )
        status = _route_status(requested_route, fallback)
        self._route_statuses[requested_route] = status
        self._remember_current_call_status(requested_route, status)
        _log_fallback(current, fallback, failure, attempt=attempt)
        return fallback

    def _provider(self, configuration: ProviderConfiguration) -> ProviderImplementation:
        if self._close_task is not None:
            raise RuntimeError("Model Router is closed")
        provider = self._providers.get(configuration.provider_id)
        if provider is None:
            provider = self._provider_factory(configuration)
            self._providers[configuration.provider_id] = provider
        return provider

    def _begin_call(self, requested_route: ModelRoute) -> ResolvedModelRoute:
        resolved = self._configuration.resolve_route(requested_route)
        status = _route_status(requested_route, resolved)
        self._route_statuses[requested_route] = status
        self._remember_current_call_status(requested_route, status)
        if resolved.used_default:
            logger.warning(
                "Default Model Route selected code=route_unavailable requested_route={} "
                "provider={} selected_route={} model={}",
                resolved.requested_route,
                resolved.provider.provider_id,
                resolved.selected_route,
                resolved.route.model,
            )
        return resolved

    def _remember_current_call_status(
        self,
        requested_route: ModelRoute,
        status: ModelRouteStatus,
    ) -> None:
        current = self._current_call_statuses.get()
        statuses = {} if current is None else dict(current)
        statuses[requested_route] = status
        self._current_call_statuses.set(statuses)


def _log_retry(
    resolved: ResolvedModelRoute,
    failure: ModelCallError,
    *,
    attempt: int,
    delay: float,
) -> None:
    logger.opt(exception=failure).warning(
        "Provider attempt failed; retrying attempt={}/{} code={} provider={} "
        "requested_route={} selected_route={} model={} planned_delay_seconds={}",
        attempt,
        _MAX_ATTEMPTS,
        failure.error.code,
        resolved.provider.provider_id,
        resolved.requested_route,
        resolved.selected_route,
        resolved.route.model,
        delay,
    )


def _log_fallback(
    failed: ResolvedModelRoute,
    fallback: ResolvedModelRoute,
    failure: ModelCallError,
    *,
    attempt: int,
) -> None:
    logger.opt(exception=failure).warning(
        "Provider attempt failed; recovering attempt={}/{} code={} provider={} "
        "requested_route={} selected_route={} model={} planned_delay_seconds=0.0",
        attempt,
        _MAX_ATTEMPTS,
        failure.error.code,
        failed.provider.provider_id,
        failed.requested_route,
        failed.selected_route,
        failed.route.model,
    )
    logger.warning(
        "Default Model Route selected code={} requested_route={} provider={} "
        "selected_route={} model={}",
        failure.error.code,
        fallback.requested_route,
        fallback.provider.provider_id,
        fallback.selected_route,
        fallback.route.model,
    )


def _route_status(
    requested_route: ModelRoute,
    resolved: ResolvedModelRoute,
) -> ModelRouteStatus:
    return ModelRouteStatus(
        requested_route=requested_route,
        selected_route=cast(ModelRoute, resolved.selected_route),
        provider_id=resolved.provider.provider_id,
        model=resolved.route.model,
        context_window=resolved.route.context_window,
        used_default=resolved.used_default,
    )
