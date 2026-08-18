"""Scripted provider boundary for deterministic offline tests."""

from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass

from myclaw.agent.prompts import session_title_prompt
from myclaw.config.config import ProviderConfiguration
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ReasoningEffort,
)
from myclaw.tools.base import OpenAIToolSchema


def unexpected_provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
    del configuration
    raise AssertionError("Provider factory was unexpectedly called")


@dataclass(frozen=True, slots=True)
class StreamScript:
    """Events yielded by one provider stream call."""

    events: tuple[ModelStreamEvent, ...]
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class ProviderCall:
    """Arguments captured from one direct keyword-only provider call."""

    messages: list[dict[str, object]]
    tools: tuple[OpenAIToolSchema, ...]
    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout: int


class ScriptedFakeProvider:
    """Replay provider behavior without loading an SDK or using the network."""

    def __init__(
        self,
        *,
        streams: Iterable[StreamScript] = (),
        completions: Iterable[ModelResponse | BaseException] = (),
    ) -> None:
        self._streams = deque(streams)
        self._completions = deque(completions)
        self.stream_requests: list[ProviderCall] = []
        self.unscripted_title_requests: list[ProviderCall] = []
        self.complete_requests: list[ProviderCall] = []
        self.closed = False

    async def stream(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[OpenAIToolSchema],
        model: str = "test-model",
        max_output: int = 1024,
        temperature: float = 0.2,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: int = 30,
    ) -> AsyncIterator[ModelStreamEvent]:
        call = _provider_call(
            messages=messages,
            tools=tools,
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        if call.messages and call.messages[0] == {
            "role": "system",
            "content": session_title_prompt(),
        }:
            self.unscripted_title_requests.append(call)
            raise ModelCallError(
                ErrorInfo(code="model_failed", message="No title response was scripted.")
            )
        self.stream_requests.append(call)
        if not self._streams:
            msg = "No scripted stream remains"
            raise AssertionError(msg)
        script = self._streams.popleft()
        for event in script.events:
            yield event
        if script.error is not None:
            raise script.error

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[OpenAIToolSchema],
        model: str = "test-model",
        max_output: int = 1024,
        temperature: float = 0.2,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: int = 30,
    ) -> ModelResponse:
        self.complete_requests.append(
            _provider_call(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
        )
        if not self._completions:
            msg = "No scripted completion remains"
            raise AssertionError(msg)
        outcome = self._completions.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


class ScriptedFakeRouter:
    """Adapt a direct provider test double to the route-only test seam."""

    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    def stream(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[OpenAIToolSchema],
    ) -> AsyncIterator[ModelStreamEvent]:
        del route
        return self._provider.stream(
            messages=messages,
            tools=tools,
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout=30,
        )

    async def complete(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[OpenAIToolSchema],
    ) -> ModelResponse:
        del route
        return await self._provider.complete(
            messages=messages,
            tools=tools,
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout=30,
        )

    async def close(self) -> None:
        await self._provider.close()


def _provider_call(
    *,
    messages: Sequence[dict[str, object]],
    tools: Sequence[OpenAIToolSchema],
    model: str,
    max_output: int,
    temperature: float,
    reasoning_effort: ReasoningEffort | None,
    timeout: int,
) -> ProviderCall:
    return ProviderCall(
        messages=deepcopy(list(messages)),
        tools=tuple(tools),
        model=model,
        max_output=max_output,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
