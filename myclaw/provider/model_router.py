"""Model Route resolution and one shared Provider attempt budget."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast

from myclaw.config.config import ProviderConfiguration, ResolvedModelRoute, UserConfiguration
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import ModelRequest, ModelResponse, ModelRoute, ModelStreamEvent
from myclaw.provider.ports import ModelProvider

_MAX_ATTEMPTS = 5
_RETRYABLE_CODES = frozenset({"provider_rate_limited", "provider_timeout", "provider_unavailable"})
_FALLBACK_CODES = frozenset({"route_unavailable", "provider_auth_error"})

logger = logging.getLogger(__name__)


class RetryClock(Protocol):
    async def sleep(self, seconds: float) -> None: ...


class AsyncioRetryClock:
    """Production retry clock backed by the event loop's monotonic timer."""

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


type ProviderFactory = Callable[[ProviderConfiguration], ModelProvider]
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
        clock: RetryClock,
        jitter: Jitter | None = None,
    ) -> None:
        self._configuration = configuration
        self._provider_factory = provider_factory
        self._clock = clock
        self._jitter = jitter
        self._providers: dict[str, ModelProvider] = {}
        self._route_statuses: dict[ModelRoute, ModelRouteStatus] = {}
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

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        resolved = self._begin_call(request.route)

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            provider = self._provider(resolved.provider)
            concrete_request = _concrete_request(request, resolved)
            emitted = False
            try:
                async for event in provider.stream(concrete_request):
                    emitted = True
                    yield event
                return
            except ModelCallError as failure:
                if emitted:
                    raise
                resolved = await self._recover_attempt(resolved, failure, attempt=attempt)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        resolved = self._begin_call(request.route)

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            provider = self._provider(resolved.provider)
            concrete_request = _concrete_request(request, resolved)
            try:
                return await provider.complete(concrete_request)
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
        unique: list[ModelProvider] = []
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
        self._route_statuses[requested_route] = _route_status(requested_route, fallback)
        _log_fallback(current, fallback, failure, attempt=attempt)
        return fallback

    def _provider(self, configuration: ProviderConfiguration) -> ModelProvider:
        if self._close_task is not None:
            raise RuntimeError("Model Router is closed")
        provider = self._providers.get(configuration.provider_id)
        if provider is None:
            provider = self._provider_factory(configuration)
            self._providers[configuration.provider_id] = provider
        return provider

    def _begin_call(self, requested_route: ModelRoute) -> ResolvedModelRoute:
        resolved = self._configuration.resolve_route(requested_route)
        self._route_statuses[requested_route] = _route_status(requested_route, resolved)
        if resolved.used_default:
            logger.warning(
                "Default Model Route selected code=route_unavailable requested_route=%s "
                "provider=%s selected_route=%s model=%s",
                resolved.requested_route,
                resolved.provider.provider_id,
                resolved.selected_route,
                resolved.route.model,
            )
        return resolved


def _log_retry(
    resolved: ResolvedModelRoute,
    failure: ModelCallError,
    *,
    attempt: int,
    delay: float,
) -> None:
    safe_failure = _safe_provider_failure(failure)
    logger.warning(
        "Provider attempt failed; retrying attempt=%d/%d code=%s provider=%s "
        "requested_route=%s selected_route=%s model=%s planned_delay_seconds=%s",
        attempt,
        _MAX_ATTEMPTS,
        failure.error.code,
        resolved.provider.provider_id,
        resolved.requested_route,
        resolved.selected_route,
        resolved.route.model,
        delay,
        exc_info=(type(safe_failure), safe_failure, safe_failure.__traceback__),
    )


def _log_fallback(
    failed: ResolvedModelRoute,
    fallback: ResolvedModelRoute,
    failure: ModelCallError,
    *,
    attempt: int,
) -> None:
    safe_failure = _safe_provider_failure(failure)
    logger.warning(
        "Provider attempt failed; recovering attempt=%d/%d code=%s provider=%s "
        "requested_route=%s selected_route=%s model=%s planned_delay_seconds=0.0",
        attempt,
        _MAX_ATTEMPTS,
        failure.error.code,
        failed.provider.provider_id,
        failed.requested_route,
        failed.selected_route,
        failed.route.model,
        exc_info=(type(safe_failure), safe_failure, safe_failure.__traceback__),
    )
    logger.warning(
        "Default Model Route selected code=%s requested_route=%s provider=%s "
        "selected_route=%s model=%s",
        failure.error.code,
        fallback.requested_route,
        fallback.provider.provider_id,
        fallback.selected_route,
        fallback.route.model,
    )


def _safe_provider_failure(failure: BaseException) -> BaseException:
    if isinstance(failure, ModelCallError):
        safe_failure: BaseException = ModelCallError(
            ErrorInfo(
                code=failure.error.code,
                message="The Provider attempt failed.",
            )
        )
    else:
        try:
            safe_failure = type(failure)("Diagnostic detail redacted.")
        except Exception:
            safe_failure = RuntimeError(f"{type(failure).__name__}: Diagnostic detail redacted.")
    if failure.__cause__ is not None:
        safe_failure.__cause__ = _safe_provider_failure(failure.__cause__)
    elif failure.__context__ is not None and not failure.__suppress_context__:
        safe_failure.__context__ = _safe_provider_failure(failure.__context__)
        safe_failure.__suppress_context__ = False
    return safe_failure.with_traceback(failure.__traceback__)


def _concrete_request(request: ModelRequest, resolved: ResolvedModelRoute) -> ModelRequest:
    route = resolved.route
    return replace(
        request,
        model=route.model,
        max_output=route.max_output,
        temperature=route.temperature,
        reasoning_effort=route.reasoning_effort,
        timeout_seconds=route.timeout,
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
