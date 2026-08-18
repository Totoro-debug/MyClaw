"""OpenAI-compatible Model Provider implemented through the official async SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from math import isfinite
from typing import Protocol, cast

from myclaw.config.config import ProviderConfiguration
from myclaw.errors import ErrorCode, ErrorInfo
from myclaw.provider.errors import EmptyModelResponseError, ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    FinishReason,
    ModelCompleted,
    ModelMessages,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    TextDelta,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall


class _Completions(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _Chat(Protocol):
    completions: _Completions


class _OpenAIClient(Protocol):
    chat: _Chat

    async def close(self) -> None: ...


class _CompletionMessage(Protocol):
    content: object
    tool_calls: list[object] | None


class _CompletionChoice(Protocol):
    message: _CompletionMessage
    finish_reason: object


class _FunctionCall(Protocol):
    name: object
    arguments: object


class _CompleteToolCall(Protocol):
    id: object
    function: _FunctionCall


type OpenAIClientFactory = Callable[..., object]


@dataclass(slots=True)
class _ToolCallParts:
    id: list[str] = field(default_factory=list)
    name: list[str] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)


def _official_client_factory(**kwargs: object) -> object:
    openai = import_module("openai")
    client_type = openai.AsyncOpenAI
    return client_type(**kwargs)


class OpenAICompatibleProvider:
    """Translate the provider-neutral contract to OpenAI chat completions."""

    def __init__(
        self,
        configuration: ProviderConfiguration,
        *,
        client_factory: OpenAIClientFactory | None = None,
    ) -> None:
        factory = _official_client_factory if client_factory is None else client_factory
        self._client = cast(
            _OpenAIClient,
            factory(
                api_key=configuration.api_key,
                base_url=configuration.base_url,
                max_retries=0,
            ),
        )

    def stream(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: int,
    ) -> AsyncIterator[ModelStreamEvent]:
        return self._stream_with_arguments(
            messages=messages,
            tools=tools,
            model=model,
            max_output=max_output,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )

    async def _stream_with_arguments(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            async for event in self._stream_once(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            ):
                yield event
        except ModelCallError:
            raise
        except Exception as failure:
            raise _model_call_error(failure) from failure

    async def _stream_once(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
    ) -> AsyncIterator[ModelStreamEvent]:
        result = await self._client.chat.completions.create(
            **_request_arguments(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                stream=True,
            )
        )
        chunks = cast(AsyncIterator[object], result)
        content_parts: list[str] = []
        finish_reason: FinishReason = "stop"
        input_tokens = 0
        output_tokens = 0
        tool_call_parts: dict[int, _ToolCallParts] = {}

        async for chunk in chunks:
            choices = cast(list[object], getattr(chunk, "choices", []))
            for choice in choices:
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    yield TextDelta(delta=content)
                for tool_call_delta in getattr(delta, "tool_calls", None) or ():
                    index = int(getattr(tool_call_delta, "index", 0))
                    parts = tool_call_parts.setdefault(index, _ToolCallParts())
                    _append_string(parts.id, getattr(tool_call_delta, "id", None))
                    function = getattr(tool_call_delta, "function", None)
                    _append_string(parts.name, getattr(function, "name", None))
                    _append_string(parts.arguments, getattr(function, "arguments", None))
                observed_finish = getattr(choice, "finish_reason", None)
                if observed_finish is not None:
                    finish_reason = _finish_reason(str(observed_finish))

            usage = getattr(chunk, "usage", None)
            if usage is not None:
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        content = "".join(content_parts)
        tool_calls = tuple(
            _model_tool_call(tool_call_parts[index]) for index in sorted(tool_call_parts)
        )

        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(
                    content=content,
                    tool_calls=tool_calls,
                ),
                usage=ModelUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
                finish_reason=finish_reason,
            )
        )

    async def complete(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: int,
    ) -> ModelResponse:
        try:
            return await self._complete_once(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
        except ModelCallError:
            raise
        except Exception as failure:
            raise _model_call_error(failure) from failure

    async def _complete_once(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
    ) -> ModelResponse:
        result = await self._client.chat.completions.create(
            **_request_arguments(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                stream=False,
            )
        )
        choices = cast(list[object], getattr(result, "choices", []))
        if not choices:
            raise _empty_response_error()
        choice = cast(_CompletionChoice, choices[0])
        message = choice.message
        content = message.content
        usage = getattr(result, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return ModelResponse(
            message=AssistantModelMessage(
                content=content if isinstance(content, str) else "",
                tool_calls=tuple(
                    _complete_tool_call(tool_call) for tool_call in (message.tool_calls or ())
                ),
            ),
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            finish_reason=_finish_reason(str(choice.finish_reason or "stop")),
        )

    async def close(self) -> None:
        await self._client.close()


def _request_arguments(
    *,
    messages: ModelMessages,
    tools: Sequence[OpenAIToolSchema],
    model: str,
    max_output: int,
    temperature: float,
    reasoning_effort: ReasoningEffort | None,
    timeout: int,
    stream: bool,
) -> dict[str, object]:
    arguments: dict[str, object] = {
        "max_tokens": max_output,
        "messages": [_openai_message(message) for message in messages],
        "model": model,
        "stream": stream,
        "temperature": temperature,
        "timeout": timeout,
        "tools": list(tools),
    }
    if reasoning_effort is not None:
        arguments["reasoning_effort"] = reasoning_effort
    if stream:
        arguments["stream_options"] = {"include_usage": True}
    return arguments


def _append_string(parts: list[str], value: object) -> None:
    if isinstance(value, str) and value:
        parts.append(value)


def _model_tool_call(parts: _ToolCallParts) -> ModelToolCall:
    return ModelToolCall(
        id="".join(parts.id),
        name="".join(parts.name),
        arguments="".join(parts.arguments),
    )


def _complete_tool_call(tool_call: object) -> ModelToolCall:
    complete_tool_call = cast(_CompleteToolCall, tool_call)
    function = complete_tool_call.function
    return ModelToolCall(
        id=str(complete_tool_call.id),
        name=str(function.name),
        arguments=str(function.arguments or ""),
    )


def _openai_message(message: Mapping[str, object]) -> dict[str, object]:
    role = message.get("role")
    content = message.get("content", "")
    if role == "assistant":
        result: dict[str, object] = {
            "role": "assistant",
            "content": content,
        }
        tool_calls = message.get("tool_calls")
        if tool_calls:
            result["tool_calls"] = [
                _openai_tool_call(tool_call) for tool_call in _tool_call_sequence(tool_calls)
            ]
        return result
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.get("tool_call_id", ""),
            "content": content,
        }
    if role in {"system", "user"}:
        return {"role": role, "content": content}
    raise TypeError("Model message role is unsupported")


def _tool_call_sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("assistant tool_calls must be a sequence")
    return value


def _openai_tool_call(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("assistant tool call must be a mapping")
    function = value.get("function")
    function_mapping = function if isinstance(function, Mapping) else value
    name = function_mapping.get("name")
    arguments = function_mapping.get("arguments", "")
    if not isinstance(name, str) or not isinstance(arguments, str):
        raise TypeError("assistant tool call name and arguments must be strings")
    return {
        "id": value.get("id", ""),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _finish_reason(value: str) -> FinishReason:
    if value in {"tool_calls", "function_call"}:
        return "tool_calls"
    if value == "length":
        return "length"
    return "stop"


_CONTEXT_ERROR_CODES = frozenset(
    {
        "context_length_exceeded",
        "context_window_exceeded",
        "max_tokens_exceeded",
    }
)
_ROUTE_ERROR_CODES = frozenset(
    {
        "deployment_not_found",
        "invalid_model",
        "model_not_found",
        "not_found",
        "unsupported",
        "unsupported_model",
    }
)


def _model_call_error(failure: Exception) -> ModelCallError:
    if isinstance(failure, EmptyModelResponseError):
        return _empty_response_error()
    status = _status_code(failure)
    provider_code = _provider_error_code(failure)
    class_name = type(failure).__name__.lower()

    if status == 413 or provider_code in _CONTEXT_ERROR_CODES:
        return _failure(
            "model_context_overflow",
            "OpenAI-compatible request exceeds the model context window.",
        )
    if status in {401, 403} or class_name in {
        "authenticationerror",
        "permissiondeniederror",
    }:
        return _failure(
            "provider_auth_error",
            "OpenAI-compatible provider authentication failed.",
        )
    if status == 429 or class_name == "ratelimiterror":
        return _failure(
            "provider_rate_limited",
            "OpenAI-compatible provider rate limited the request.",
            retryable=True,
            retry_after_seconds=_retry_after_seconds(failure),
        )
    if status == 408 or isinstance(failure, TimeoutError) or "timeout" in class_name:
        return _failure(
            "provider_timeout",
            "OpenAI-compatible provider request timed out.",
            retryable=True,
            retry_after_seconds=_retry_after_seconds(failure),
        )
    if status in {404, 501} or provider_code in _ROUTE_ERROR_CODES or class_name == "notfounderror":
        return _failure(
            "route_unavailable",
            "OpenAI-compatible model or capability is unavailable.",
        )
    if isinstance(failure, (ConnectionError, OSError)) or class_name == "apiconnectionerror":
        return _failure(
            "provider_unavailable",
            "OpenAI-compatible provider is temporarily unavailable.",
            retryable=True,
            retry_after_seconds=_retry_after_seconds(failure),
        )
    if status is not None and status >= 500:
        return _failure(
            "provider_unavailable",
            "OpenAI-compatible provider is temporarily unavailable.",
            retryable=True,
            retry_after_seconds=_retry_after_seconds(failure),
        )
    if status is not None and 400 <= status < 500:
        return _failure(
            "model_invalid_request",
            "OpenAI-compatible provider rejected the request.",
        )
    return _failure("model_failed", "OpenAI-compatible provider call failed.")


def _empty_response_error() -> ModelCallError:
    return _failure(
        "model_failed",
        "OpenAI-compatible provider returned an empty response. "
        "Check its API base URL and model configuration.",
    )


def _failure(
    code: ErrorCode,
    message: str,
    *,
    retryable: bool = False,
    retry_after_seconds: float | None = None,
) -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code=code,
            message=message,
            retryable=retryable,
            retry_after_seconds=retry_after_seconds,
        )
    )


def _status_code(failure: Exception) -> int | None:
    value = getattr(failure, "status_code", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _provider_error_code(failure: Exception) -> str | None:
    direct = getattr(failure, "code", None)
    if isinstance(direct, str):
        return direct.lower()
    body = getattr(failure, "body", None)
    if not isinstance(body, Mapping):
        return None
    error = body.get("error", body)
    if not isinstance(error, Mapping):
        return None
    value = error.get("code")
    return value.lower() if isinstance(value, str) else None


def _retry_after_seconds(failure: Exception) -> float | None:
    response = getattr(failure, "response", None)
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    raw_value = next(
        (value for key, value in headers.items() if str(key).lower() == "retry-after"),
        None,
    )
    try:
        seconds = float(str(raw_value))
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 and isfinite(seconds) else None
