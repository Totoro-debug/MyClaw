"""Anthropic SDK adapter for the provider-neutral model contract."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol, cast

from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
)

from myclaw.config.config import ProviderConfiguration
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    FinishReason,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
    ToolModelMessage,
    UserModelMessage,
)
from myclaw.tools.models import ModelToolCall
from myclaw.utils.json_types import JsonObject, JsonValue


class AnthropicMessages(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class AnthropicClient(Protocol):
    @property
    def messages(self) -> AnthropicMessages: ...

    async def close(self) -> None: ...


class AnthropicClientFactory(Protocol):
    def __call__(
        self,
        *,
        api_key: str,
        base_url: str,
        max_retries: int,
    ) -> AnthropicClient: ...


def _official_client_factory(
    *,
    api_key: str,
    base_url: str,
    max_retries: int,
) -> AnthropicClient:
    return cast(
        AnthropicClient,
        AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=max_retries),
    )


class AnthropicProvider:
    """Translate Anthropic SDK calls at the public ModelProvider boundary."""

    def __init__(
        self,
        configuration: ProviderConfiguration,
        *,
        client_factory: AnthropicClientFactory = _official_client_factory,
    ) -> None:
        self._client = client_factory(
            api_key=configuration.api_key,
            base_url=configuration.base_url,
            max_retries=0,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        try:
            async for event in self._stream_once(request):
                yield event
        except APIError as error:
            raise _normalized_error(error) from error
        except (TypeError, ValueError) as error:
            raise _model_failed_error() from error

    async def _stream_once(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        sdk_stream = await self._client.messages.create(**_request_arguments(request, stream=True))
        if not hasattr(sdk_stream, "__aiter__"):
            msg = "Anthropic streaming response is not asynchronously iterable"
            raise TypeError(msg)

        text_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None
        tool_uses: dict[int, _StreamingToolUse] = {}
        async for event in cast(AsyncIterator[object], sdk_stream):
            event_type = _string_field(event, "type")
            if event_type == "message_start":
                usage = _field(_field(event, "message"), "usage")
                input_tokens = _integer_field(usage, "input_tokens")
            elif event_type == "content_block_delta":
                delta = _field(event, "delta")
                delta_type = _string_field(delta, "type")
                if delta_type == "text_delta":
                    text = _string_field(delta, "text")
                    if text:
                        text_parts.append(text)
                        yield TextDelta(delta=text)
                elif delta_type == "input_json_delta":
                    index = _integer_field(event, "index")
                    partial_json = _string_field(delta, "partial_json")
                    if index in tool_uses and partial_json:
                        tool_uses[index].json_parts.append(partial_json)
            elif event_type == "content_block_start":
                content_block = _field(event, "content_block")
                if _string_field(content_block, "type") == "tool_use":
                    index = _integer_field(event, "index")
                    tool_uses[index] = _StreamingToolUse(
                        id=_string_field(content_block, "id") or "",
                        name=_string_field(content_block, "name") or "",
                        initial_input=_field(content_block, "input"),
                    )
            elif event_type == "message_delta":
                stop_reason = _string_field(_field(event, "delta"), "stop_reason")
                output_tokens = _integer_field(_field(event, "usage"), "output_tokens")

        response = ModelResponse(
            message=AssistantModelMessage(
                content="".join(text_parts),
                tool_calls=tuple(
                    tool_uses[index].to_model_tool_call() for index in sorted(tool_uses)
                ),
            ),
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            finish_reason=_finish_reason(stop_reason),
        )
        yield ModelCompleted(response=response)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            message = await self._client.messages.create(
                **_request_arguments(request, stream=False)
            )
            return _response_from_message(message)
        except APIError as error:
            raise _normalized_error(error) from error
        except (TypeError, ValueError) as error:
            raise _model_failed_error() from error

    async def close(self) -> None:
        await self._client.close()


def _request_arguments(request: ModelRequest, *, stream: bool) -> dict[str, object]:
    messages = [_message_argument(message) for message in request.messages]
    tools: list[dict[str, object]] = []
    for tool in request.tools:
        function = tool["function"]
        tools.append(
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            }
        )
    return {
        "max_tokens": request.max_output,
        "messages": messages,
        "model": request.model,
        "stream": stream,
        "system": request.system_prompt,
        "temperature": request.temperature,
        "timeout": request.timeout_seconds,
        "tools": tools,
    }


def _message_argument(
    message: UserModelMessage | AssistantModelMessage | ToolModelMessage,
) -> dict[str, object]:
    if isinstance(message, UserModelMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantModelMessage):
        content: list[dict[str, object]] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        content.extend(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.name,
                "input": _argument_object(tool_call.arguments),
            }
            for tool_call in message.tool_calls
        )
        return {"role": "assistant", "content": content}
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
        ],
    }


def _response_from_message(message: object) -> ModelResponse:
    text_parts: list[str] = []
    tool_calls: list[ModelToolCall] = []
    content = _field(message, "content")
    if isinstance(content, (list, tuple)):
        for block in content:
            block_type = _string_field(block, "type")
            if block_type == "text":
                text = _string_field(block, "text")
                if text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ModelToolCall(
                        id=_string_field(block, "id") or "",
                        name=_string_field(block, "name") or "",
                        arguments=_json_text(_field(block, "input")),
                    )
                )
    usage = _field(message, "usage")
    input_tokens = _integer_field(usage, "input_tokens")
    output_tokens = _integer_field(usage, "output_tokens")
    return ModelResponse(
        message=AssistantModelMessage(content="".join(text_parts), tool_calls=tuple(tool_calls)),
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        finish_reason=_finish_reason(_string_field(message, "stop_reason")),
    )


@dataclass(slots=True)
class _StreamingToolUse:
    id: str
    name: str
    initial_input: object
    json_parts: list[str] = field(default_factory=list)

    def to_model_tool_call(self) -> ModelToolCall:
        if self.json_parts:
            arguments = "".join(self.json_parts)
            _argument_object(arguments)
        else:
            arguments = _json_text(self.initial_input)
        return ModelToolCall(id=self.id, name=self.name, arguments=arguments)


def _argument_object(arguments: str | JsonObject) -> JsonObject:
    if isinstance(arguments, dict):
        # Temporary support for pre-migration test fixtures; removed by #48.
        return arguments
    try:
        value = cast(object, json.loads(arguments))
    except json.JSONDecodeError as error:
        raise ValueError("Anthropic tool arguments must be a JSON object") from error
    return _json_object(value)


def _json_text(value: object) -> str:
    return json.dumps(
        _json_object(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _json_object(value: object) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        msg = "Anthropic tool arguments must be a JSON object"
        raise ValueError(msg)
    return {cast(str, key): _json_value(item) for key, item in value.items()}


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {cast(str, key): _json_value(item) for key, item in value.items()}
    msg = "Anthropic tool arguments contain a non-JSON value"
    raise ValueError(msg)


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _string_field(value: object, name: str) -> str | None:
    field = _field(value, name)
    return field if isinstance(field, str) else None


def _integer_field(value: object, name: str) -> int:
    field = _field(value, name)
    return field if isinstance(field, int) and not isinstance(field, bool) else 0


def _finish_reason(stop_reason: str | None) -> FinishReason:
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "tool_use":
        return "tool_calls"
    return "stop"


def _normalized_error(
    error: APIError,
) -> ModelCallError:
    if isinstance(error, APITimeoutError):
        return ModelCallError(
            ErrorInfo(
                code="provider_timeout",
                message="Anthropic request timed out.",
                retryable=True,
            )
        )
    if isinstance(error, APIConnectionError):
        return ModelCallError(
            ErrorInfo(
                code="provider_unavailable",
                message="Anthropic is temporarily unavailable.",
                retryable=True,
            )
        )
    if not isinstance(error, APIStatusError):
        return _model_failed_error()

    status = error.status_code
    if status in {401, 403}:
        return ModelCallError(
            ErrorInfo(
                code="provider_auth_error",
                message="Anthropic authentication failed.",
            )
        )
    if status == 408:
        return ModelCallError(
            ErrorInfo(
                code="provider_timeout",
                message="Anthropic request timed out.",
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error),
            )
        )
    if status == 429:
        return ModelCallError(
            ErrorInfo(
                code="provider_rate_limited",
                message="Anthropic rate limit was reached.",
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error),
            )
        )
    if status >= 500:
        return ModelCallError(
            ErrorInfo(
                code="provider_unavailable",
                message="Anthropic is temporarily unavailable.",
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error),
            )
        )
    provider_message = _provider_error_message(error).lower()
    if status == 404 or _describes_unsupported_capability(provider_message):
        return ModelCallError(
            ErrorInfo(
                code="route_unavailable",
                message="Anthropic model or capability is unavailable.",
            )
        )
    if status == 413 or _describes_context_overflow(provider_message):
        return ModelCallError(
            ErrorInfo(
                code="model_context_overflow",
                message="Anthropic model context was exceeded.",
            )
        )
    if status in {400, 422}:
        return ModelCallError(
            ErrorInfo(
                code="model_invalid_request",
                message="Anthropic rejected the model request.",
            )
        )
    return _model_failed_error()


def _model_failed_error() -> ModelCallError:
    return ModelCallError(ErrorInfo(code="model_failed", message="Anthropic model call failed."))


def _retry_after_seconds(error: APIStatusError) -> float | None:
    raw_value = error.response.headers.get("retry-after")
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    return value if value >= 0 and isfinite(value) else None


def _provider_error_message(error: APIStatusError) -> str:
    body_error = _field(error.body, "error")
    return _string_field(body_error, "message") or ""


def _describes_unsupported_capability(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "does not support",
            "not supported",
            "unsupported",
        )
    )


def _describes_context_overflow(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "prompt is too long",
            "too many tokens",
            "context window",
            "maximum context",
            "maximum allowed number of tokens",
        )
    )
