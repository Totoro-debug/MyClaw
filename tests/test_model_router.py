import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.config.config import (
    MemoryConfiguration,
    ModelsConfiguration,
    ProviderConfiguration,
    RouteConfiguration,
    RuntimeConfiguration,
    UserConfiguration,
)
from myclaw.errors import (
    ErrorCode,
    ErrorInfo,
)
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouter, ModelRouteStatus
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelRoute,
    ModelUsage,
    TextDelta,
)
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript
from tests.fixtures.diagnostic_capture import capture_diagnostics

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
REQUEST_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
SESSION_ID = "20260711-153012-123000_550e8400-e29b-41d4-a716-446655440000"


async def collect(stream: AsyncIterator[object]) -> list[object]:
    return [event async for event in stream]


def configuration() -> UserConfiguration:
    provider = ProviderConfiguration(
        provider_id="default-provider",
        protocol="anthropic",
        base_url="https://default.example/v1",
        api_key="default-secret",
        models=("default-model",),
    )
    route = RouteConfiguration(
        provider_id=provider.provider_id,
        model="default-model",
        context_window=100_000,
        max_output=4096,
        temperature=0.2,
        reasoning_effort=None,
        timeout=120,
    )
    return UserConfiguration(
        runtime=RuntimeConfiguration(max_tool_result_chars=50_000),
        memory=MemoryConfiguration(
            consolidation_message_threshold=40,
            batch_size=10,
            schedule="0 * * * *",
        ),
        models=ModelsConfiguration(
            providers={provider.provider_id: provider},
            routes={"default": route},
        ),
    )


def routed_configuration() -> UserConfiguration:
    default = configuration()
    chat_provider = ProviderConfiguration(
        provider_id="chat-provider",
        protocol="openai-compatible",
        base_url="https://chat.example/v1",
        api_key="chat-secret",
        models=("chat-model",),
    )
    chat_route = RouteConfiguration(
        provider_id=chat_provider.provider_id,
        model="chat-model",
        context_window=200_000,
        max_output=8192,
        temperature=0.1,
        reasoning_effort="high",
        timeout=90,
    )
    return UserConfiguration(
        runtime=default.runtime,
        memory=default.memory,
        models=ModelsConfiguration(
            providers={**default.models.providers, chat_provider.provider_id: chat_provider},
            routes={**default.models.routes, "chat": chat_route},
        ),
    )


def memory_configuration() -> UserConfiguration:
    default = configuration()
    memory_provider = ProviderConfiguration(
        provider_id="memory-provider",
        protocol="openai-compatible",
        base_url="https://memory.example/v1",
        api_key="memory-secret",
        models=("memory-model",),
    )
    memory_route = RouteConfiguration(
        provider_id=memory_provider.provider_id,
        model="memory-model",
        context_window=80_000,
        max_output=2048,
        temperature=0,
        reasoning_effort="low",
        timeout=60,
    )
    return UserConfiguration(
        runtime=default.runtime,
        memory=default.memory,
        models=ModelsConfiguration(
            providers={**default.models.providers, memory_provider.provider_id: memory_provider},
            routes={**default.models.routes, "memory": memory_route},
        ),
    )


def request(*, route: ModelRoute = "default", stream: bool = True) -> ModelRequest:
    return ModelRequest(
        request_id=REQUEST_UUID,
        route=route,
        system_prompt="You are MyClaw.",
        messages=(),
        tools=(),
        stream=stream,
        model="caller-placeholder",
        max_output=1,
        temperature=0,
        reasoning_effort=None,
        timeout_seconds=1,
    )


def completed(content: str = "Done") -> ModelCompleted:
    return ModelCompleted(response=response(content))


def response(content: str = "Done") -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=4, output_tokens=1, total_tokens=5),
        finish_reason="stop",
    )


def retryable_timeout(*, retry_after: float | None = None) -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="provider_timeout",
            message="The provider timed out.",
            retryable=True,
            retry_after_seconds=retry_after,
        )
    )


def permanent_failure(
    code: ErrorCode = "provider_auth_error",
) -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code=code,
            message="The configured route is permanently unavailable.",
        )
    )


@pytest.mark.asyncio
async def test_router_close_settles_every_cached_provider_when_one_close_fails() -> None:
    close_order: list[str] = []

    class CloseTrackingProvider(ScriptedFakeProvider):
        def __init__(self, name: str, *, fail_close: bool = False) -> None:
            super().__init__(completions=(response(name),))
            self._name = name
            self._fail_close = fail_close

        async def close(self) -> None:
            close_order.append(self._name)
            if self._fail_close:
                raise RuntimeError(f"{self._name} close failed")

    providers = {
        "default-provider": CloseTrackingProvider("default", fail_close=True),
        "memory-provider": CloseTrackingProvider("memory"),
    }
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    await router.complete(request(route="default", stream=False))
    await router.complete(request(route="memory", stream=False))

    with pytest.raises(RuntimeError, match="default close failed"):
        await router.close()

    assert close_order == ["default", "memory"]


@pytest.mark.asyncio
async def test_router_close_settles_every_provider_when_one_close_is_cancelled() -> None:
    close_order: list[str] = []

    class CloseTrackingProvider(ScriptedFakeProvider):
        def __init__(self, name: str, *, cancel_close: bool = False) -> None:
            super().__init__(completions=(response(name),))
            self._name = name
            self._cancel_close = cancel_close

        async def close(self) -> None:
            close_order.append(self._name)
            if self._cancel_close:
                raise asyncio.CancelledError

    providers = {
        "default-provider": CloseTrackingProvider("default", cancel_close=True),
        "memory-provider": CloseTrackingProvider("memory"),
    }
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    await router.complete(request(route="default", stream=False))
    await router.complete(request(route="memory", stream=False))

    with pytest.raises(asyncio.CancelledError):
        await router.close()

    assert close_order == ["default", "memory"]


@pytest.mark.asyncio
async def test_router_starts_every_provider_close_before_waiting_for_one_to_finish() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    class FirstProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            first_started.set()
            await release_first.wait()

    class SecondProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            second_started.set()

    providers: dict[str, ScriptedFakeProvider] = {
        "default-provider": FirstProvider(completions=(response("default"),)),
        "memory-provider": SecondProvider(completions=(response("memory"),)),
    }
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
    )
    await router.complete(request(route="default", stream=False))
    await router.complete(request(route="memory", stream=False))
    closing = asyncio.create_task(router.close())
    await first_started.wait()
    try:
        await asyncio.sleep(0)
        assert second_started.is_set()
    finally:
        release_first.set()
        await closing


@pytest.mark.asyncio
async def test_concurrent_router_close_waits_for_the_same_provider_shutdown() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    close_calls = 0

    class BlockingCloseProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            started.set()
            await release.wait()

    provider = BlockingCloseProvider(completions=(response(),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
    )
    await router.complete(request(stream=False))

    first = asyncio.create_task(router.close())
    await started.wait()
    second = asyncio.create_task(router.close())
    await asyncio.sleep(0)
    try:
        assert not second.done()
    finally:
        release.set()
        await asyncio.gather(first, second)

    assert close_calls == 1


@pytest.mark.asyncio
async def test_cancelling_one_router_close_caller_does_not_cancel_provider_shutdown() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class BlockingCloseProvider(ScriptedFakeProvider):
        async def close(self) -> None:
            started.set()
            await release.wait()
            finished.set()

    provider = BlockingCloseProvider(completions=(response(),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
    )
    await router.complete(request(stream=False))

    caller = asyncio.create_task(router.close())
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert not finished.is_set()
    release.set()
    await router.close()

    assert finished.is_set()


@pytest.mark.asyncio
async def test_router_cannot_create_a_new_provider_after_close() -> None:
    provider = ScriptedFakeProvider(completions=(response(),))
    factory_calls = 0

    def provider_factory(_configuration: ProviderConfiguration) -> ScriptedFakeProvider:
        nonlocal factory_calls
        factory_calls += 1
        return provider

    router = ModelRouter(
        configuration=configuration(),
        provider_factory=provider_factory,
        clock=FakeClock(NOW),
    )
    await router.complete(request(stream=False))
    await router.close()

    with pytest.raises(RuntimeError, match="Model Router is closed"):
        await router.complete(request(stream=False))

    assert factory_calls == 1


@pytest.mark.asyncio
async def test_model_router_caps_one_logical_stream_at_five_provider_attempts() -> None:
    failure = retryable_timeout()
    provider = ScriptedFakeProvider(
        streams=[StreamScript(events=(), error=failure) for _ in range(5)]
    )
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )

    with pytest.raises(ModelCallError) as raised:
        await collect(router.stream(request()))

    assert raised.value.error is failure.error
    assert len(provider.stream_requests) == 5


@pytest.mark.asyncio
async def test_model_router_records_only_consumed_retry_attempts(
    agent_home: Path,
) -> None:
    private_failure_detail = "user-message prompt memory tool-args tool-result provider-body"
    failure = ModelCallError(
        ErrorInfo(
            code="provider_timeout",
            message=private_failure_detail,
            retryable=True,
        )
    )
    provider = ScriptedFakeProvider(
        streams=[StreamScript(events=(), error=failure) for _ in range(5)]
    )
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    capture = capture_diagnostics()

    with capture.session(SESSION_ID), pytest.raises(ModelCallError):
        await collect(router.stream(request(route="chat")))
    capture.close()

    records = [
        line for line in capture.text.splitlines() if "myclaw.provider.model_router:" in line
    ]
    assert len(records) == 4
    for attempt, delay, record in zip(
        range(1, 5),
        (0.5, 1.0, 2.0, 4.0),
        records,
        strict=True,
    ):
        assert f"attempt={attempt}/5" in record
        assert "code=provider_timeout" in record
        assert "provider=chat-provider" in record
        assert "requested_route=chat" in record
        assert "selected_route=chat" in record
        assert "model=chat-model" in record
        assert f"planned_delay_seconds={delay}" in record
    assert "attempt=5/5" not in capture.text
    content = capture.text
    event_text = capture.event_text
    assert content.count("Traceback (most recent call last)") == 4
    assert content.count(f"ModelCallError: {private_failure_detail}") == 4
    assert private_failure_detail not in event_text


@pytest.mark.asyncio
async def test_model_router_uses_injected_clock_for_exponential_backoff_and_retry_after() -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=retryable_timeout()),
            StreamScript(events=(), error=retryable_timeout(retry_after=0.25)),
            StreamScript(events=(), error=retryable_timeout(retry_after=7)),
            StreamScript(events=(), error=retryable_timeout(retry_after=100)),
            StreamScript(events=(completed(),)),
        )
    )
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=clock,
        jitter=None,
    )

    observed = await collect(router.stream(request()))

    assert observed == [completed()]
    assert clock.sleeps == [0.5, 1.0, 7, 60]


@pytest.mark.asyncio
async def test_model_router_uses_asyncio_sleep_when_clock_is_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=retryable_timeout()),
            StreamScript(events=(completed(),)),
        )
    )
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
    )

    observed = await collect(router.stream(request()))

    assert observed == [completed()]
    assert sleeps == [0.5]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "permanent_code",
    ("provider_auth_error", "route_unavailable", "provider_unavailable"),
)
async def test_model_router_falls_back_after_a_permanent_requested_route_failure(
    permanent_code: ErrorCode,
) -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=retryable_timeout()),
            StreamScript(events=(), error=permanent_failure(permanent_code)),
        )
    )
    default_provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    factory_calls: list[str] = []

    def provider_factory(provider: ProviderConfiguration) -> ScriptedFakeProvider:
        factory_calls.append(provider.provider_id)
        return providers[provider.provider_id]

    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=provider_factory,
        clock=clock,
        jitter=None,
    )

    observed = await collect(router.stream(request(route="chat")))

    assert observed == [completed()]
    assert factory_calls == ["chat-provider", "default-provider"]
    assert len(chat_provider.stream_requests) == 2
    assert len(default_provider.stream_requests) == 1
    fallback_request = default_provider.stream_requests[0]
    assert isinstance(fallback_request, ModelRequest)
    assert (
        fallback_request.route,
        fallback_request.model,
        fallback_request.max_output,
        fallback_request.temperature,
        fallback_request.reasoning_effort,
        fallback_request.timeout_seconds,
    ) == ("chat", "default-model", 4096, 0.2, None, 120)
    assert clock.sleeps == [0.5]


@pytest.mark.asyncio
async def test_model_router_records_failed_attempt_and_default_fallback_separately(
    agent_home: Path,
) -> None:
    private_provider_body = "private provider response body"
    failure = ModelCallError(ErrorInfo(code="provider_auth_error", message=private_provider_body))
    chat_provider = ScriptedFakeProvider(streams=(StreamScript(events=(), error=failure),))
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Recovered"),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )
    capture = capture_diagnostics()

    with capture.session(SESSION_ID):
        observed = await collect(router.stream(request(route="chat")))
    capture.close()

    assert observed == [completed("Recovered")]
    records = [
        line for line in capture.text.splitlines() if "myclaw.provider.model_router:" in line
    ]
    assert len(records) == 2
    assert "Provider attempt failed" in records[0]
    assert "attempt=1/5" in records[0]
    assert "code=provider_auth_error" in records[0]
    assert "provider=chat-provider" in records[0]
    assert "requested_route=chat" in records[0]
    assert "selected_route=chat" in records[0]
    assert "model=chat-model" in records[0]
    assert "planned_delay_seconds=0.0" in records[0]
    assert "Default Model Route selected" in records[1]
    assert "requested_route=chat" in records[1]
    assert "provider=default-provider" in records[1]
    assert "selected_route=default" in records[1]
    assert "model=default-model" in records[1]
    assert all(" WARNING " in record for record in records)
    content = capture.text
    event_text = capture.event_text
    assert content.count("Traceback (most recent call last)") == 1
    assert content.count(f"ModelCallError: {private_provider_body}") == 1
    assert private_provider_body not in event_text


@pytest.mark.asyncio
async def test_model_router_route_status_updates_when_dynamic_fallback_is_selected() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=permanent_failure()),)
    )
    default_provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    initial = router.route_status("chat")
    assert initial == ModelRouteStatus(
        requested_route="chat",
        selected_route="chat",
        provider_id="chat-provider",
        model="chat-model",
        context_window=200_000,
        used_default=False,
    )
    assert "chat-secret" not in repr(initial)

    await collect(router.stream(request(route="chat")))

    fallback = router.route_status("chat")
    assert fallback == ModelRouteStatus(
        requested_route="chat",
        selected_route="default",
        provider_id="default-provider",
        model="default-model",
        context_window=100_000,
        used_default=True,
    )
    assert "default-secret" not in repr(fallback)


def test_model_router_route_status_starts_from_static_default_fallback() -> None:
    def unexpected_factory(_: ProviderConfiguration) -> ScriptedFakeProvider:
        raise AssertionError("Reading route status must not construct a Provider adapter")

    router = ModelRouter(
        configuration=configuration(),
        provider_factory=unexpected_factory,
        clock=FakeClock(NOW),
        jitter=None,
    )

    assert router.route_status("chat") == ModelRouteStatus(
        requested_route="chat",
        selected_route="default",
        provider_id="default-provider",
        model="default-model",
        context_window=100_000,
        used_default=True,
    )


@pytest.mark.asyncio
async def test_model_router_records_static_default_fallback_without_provider_attempt(
    agent_home: Path,
) -> None:
    provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed("Static fallback"),)),))
    router = ModelRouter(
        configuration=configuration(),
        provider_factory=lambda _: provider,
        clock=FakeClock(NOW),
        jitter=None,
    )
    capture = capture_diagnostics()

    with capture.session(SESSION_ID):
        observed = await collect(router.stream(request(route="chat")))
    capture.close()

    assert observed == [completed("Static fallback")]
    records = [
        line for line in capture.text.splitlines() if "myclaw.provider.model_router:" in line
    ]
    assert len(records) == 1
    assert " WARNING " in records[0]
    assert "Default Model Route selected code=route_unavailable" in records[0]
    assert "requested_route=chat" in records[0]
    assert "provider=default-provider" in records[0]
    assert "selected_route=default" in records[0]
    assert "model=default-model" in records[0]
    assert "Provider attempt failed" not in records[0]


@pytest.mark.asyncio
async def test_model_router_route_status_recovers_on_the_next_logical_stream() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=permanent_failure()),
            StreamScript(events=(completed("Chat recovered."),)),
        )
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Fallback response."),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    first = await collect(router.stream(request(route="chat")))
    assert first == [completed("Fallback response.")]
    assert router.route_status("chat").selected_route == "default"

    second = await collect(router.stream(request(route="chat")))

    assert second == [completed("Chat recovered.")]
    assert len(chat_provider.stream_requests) == 2
    assert len(default_provider.stream_requests) == 1
    assert router.route_status("chat") == ModelRouteStatus(
        requested_route="chat",
        selected_route="chat",
        provider_id="chat-provider",
        model="chat-model",
        context_window=200_000,
        used_default=False,
    )


@pytest.mark.asyncio
async def test_model_router_complete_shares_one_budget_across_requested_and_default() -> None:
    memory_provider = ScriptedFakeProvider(
        completions=(
            retryable_timeout(),
            retryable_timeout(),
            permanent_failure(),
        )
    )
    expected = response("Summary complete.")
    default_provider = ScriptedFakeProvider(completions=(retryable_timeout(), expected))
    providers = {
        "memory-provider": memory_provider,
        "default-provider": default_provider,
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )

    observed = await router.complete(request(route="memory", stream=False))

    assert observed is expected
    assert len(memory_provider.complete_requests) == 3
    assert len(default_provider.complete_requests) == 2
    fallback_request = default_provider.complete_requests[0]
    assert isinstance(fallback_request, ModelRequest)
    assert (fallback_request.route, fallback_request.model, fallback_request.stream) == (
        "memory",
        "default-model",
        False,
    )
    assert clock.sleeps == [0.5, 1.0, 4.0]


@pytest.mark.asyncio
async def test_model_router_route_status_recovers_on_the_next_logical_completion() -> None:
    memory_provider = ScriptedFakeProvider(
        completions=(permanent_failure(), response("Memory recovered."))
    )
    default_provider = ScriptedFakeProvider(completions=(response("Fallback summary."),))
    providers = {
        "memory-provider": memory_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=memory_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )

    first = await router.complete(request(route="memory", stream=False))
    assert first == response("Fallback summary.")
    assert router.route_status("memory").selected_route == "default"

    second = await router.complete(request(route="memory", stream=False))

    assert second == response("Memory recovered.")
    assert len(memory_provider.complete_requests) == 2
    assert len(default_provider.complete_requests) == 1
    assert router.route_status("memory") == ModelRouteStatus(
        requested_route="memory",
        selected_route="memory",
        provider_id="memory-provider",
        model="memory-model",
        context_window=80_000,
        used_default=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    ("model_invalid_request", "model_context_overflow", "turn_cancelled", "model_failed"),
)
async def test_model_router_does_not_retry_or_fallback_terminal_model_errors(
    code: ErrorCode,
) -> None:
    failure = ModelCallError(
        ErrorInfo(
            code=code,
            message="The model call cannot continue.",
            retryable=True,
        )
    )
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(), error=failure),
            StreamScript(events=(completed("Unexpected retry"),)),
        )
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Unexpected fallback"),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )

    with pytest.raises(ModelCallError) as raised:
        await collect(router.stream(request(route="chat")))

    assert raised.value is failure
    assert len(chat_provider.stream_requests) == 1
    assert default_provider.stream_requests == []
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_model_router_never_retries_after_streaming_becomes_observable() -> None:
    partial = TextDelta(delta="Visible partial text")
    chat_provider = ScriptedFakeProvider(
        streams=(
            StreamScript(events=(partial,), error=retryable_timeout()),
            StreamScript(events=(completed("Duplicated response"),)),
        )
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Unexpected fallback"),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )
    observed: list[object] = []

    with pytest.raises(ModelCallError):
        async for event in router.stream(request(route="chat")):
            observed.append(event)

    assert observed == [partial]
    assert len(chat_provider.stream_requests) == 1
    assert default_provider.stream_requests == []
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_model_router_propagates_task_cancellation_without_another_attempt() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=asyncio.CancelledError()),)
    )
    default_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(completed("Unexpected fallback"),)),)
    )
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    clock = FakeClock(NOW)
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=clock,
        jitter=None,
    )

    with pytest.raises(asyncio.CancelledError):
        await collect(router.stream(request(route="chat")))

    assert len(chat_provider.stream_requests) == 1
    assert default_provider.stream_requests == []
    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_model_router_closes_each_provider_adapter_it_constructed() -> None:
    chat_provider = ScriptedFakeProvider(
        streams=(StreamScript(events=(), error=permanent_failure()),)
    )
    default_provider = ScriptedFakeProvider(streams=(StreamScript(events=(completed(),)),))
    providers = {
        "chat-provider": chat_provider,
        "default-provider": default_provider,
    }
    router = ModelRouter(
        configuration=routed_configuration(),
        provider_factory=lambda provider: providers[provider.provider_id],
        clock=FakeClock(NOW),
        jitter=None,
    )
    await collect(router.stream(request(route="chat")))

    await router.close()

    assert chat_provider.closed is True
    assert default_provider.closed is True
