"""Scripted provider boundary for deterministic offline tests."""

from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

from myclaw.agent.prompts import session_title_prompt
from myclaw.config.config import ProviderConfiguration
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelRoute,
    ModelStreamEvent,
    ReasoningEffort,
    ToolModelMessage,
    UserModelMessage,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall


def unexpected_provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
    del configuration
    raise AssertionError("Provider factory was unexpectedly called")


@dataclass(frozen=True, slots=True)
class StreamScript:
    """Events yielded by one provider stream call."""

    events: tuple[ModelStreamEvent, ...]
    error: BaseException | None = None


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
        self.stream_requests: list[object] = []
        self.unscripted_title_requests: list[ModelRequest] = []
        self.complete_requests: list[object] = []
        self.closed = False

    async def stream(self, request: object) -> AsyncIterator[ModelStreamEvent]:
        if isinstance(request, ModelRequest) and request.system_prompt == session_title_prompt():
            self.unscripted_title_requests.append(request)
            raise ModelCallError(
                ErrorInfo(code="model_failed", message="No title response was scripted.")
            )
        self.stream_requests.append(request)
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
        request: object | None = None,
        *,
        messages: Sequence[dict[str, object]] | None = None,
        tools: Sequence[OpenAIToolSchema] | None = None,
        model: str | None = None,
        max_output: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        timeout: int | None = None,
    ) -> ModelResponse:
        if messages is not None:
            request = _legacy_request_from_direct(
                route="schedule" if len(tools or ()) == 10 else "memory",
                messages=messages,
                tools=() if tools is None else tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
        elif request is None:
            raise TypeError("Provider calls require a request or direct messages")
        self.complete_requests.append(request)
        if not self._completions:
            msg = "No scripted completion remains"
            raise AssertionError(msg)
        outcome = self._completions.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


def _legacy_request_from_direct(
    *,
    route: ModelRoute,
    messages: Sequence[dict[str, object]],
    tools: Sequence[OpenAIToolSchema],
    model: str | None,
    max_output: int | None,
    temperature: float | None,
    reasoning_effort: str | None,
    timeout: int | None,
) -> ModelRequest:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("direct fake Provider request must start with a system message")
    system_prompt = messages[0].get("content")
    if not isinstance(system_prompt, str):
        raise TypeError("fake Provider system content must be a string")
    projected: tuple[ModelMessage, ...] = tuple(_model_message(message) for message in messages[1:])
    return ModelRequest(
        request_id=uuid4(),
        route=route,
        system_prompt=system_prompt,
        messages=projected,
        tools=tuple(tools),
        stream=False,
        model="" if model is None else model,
        max_output=0 if max_output is None else max_output,
        temperature=0.0 if temperature is None else temperature,
        reasoning_effort=cast(ReasoningEffort | None, reasoning_effort),
        timeout_seconds=0 if timeout is None else timeout,
    )


def _model_message(message: dict[str, object]) -> ModelMessage:
    role = message.get("role")
    if role == "user":
        return UserModelMessage(content=_require_string(message, "content"))
    if role == "assistant":
        raw_tool_calls = message.get("tool_calls", [])
        if not isinstance(raw_tool_calls, list):
            raise TypeError("fake Provider assistant tool calls must be a list")
        return AssistantModelMessage(
            content=_require_string(message, "content"),
            tool_calls=tuple(
                ModelToolCall(
                    id=_require_string(tool_call, "id"),
                    name=_require_string(tool_call, "name"),
                    arguments=_require_string(tool_call, "arguments"),
                )
                for tool_call in raw_tool_calls
                if isinstance(tool_call, dict)
            ),
        )
    if role == "tool":
        return ToolModelMessage(
            tool_call_id=_require_string(message, "tool_call_id"),
            name=_require_string(message, "name"),
            content=_require_string(message, "content"),
        )
    raise ValueError("fake Provider message role is unsupported")


def _require_string(message: dict[str, object], field: str) -> str:
    value = message.get(field)
    if not isinstance(value, str):
        raise TypeError(f"fake Provider {field} must be a string")
    return value
