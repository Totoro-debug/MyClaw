"""Provider-neutral model request and response values."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from inspect import signature
from typing import Any, ClassVar, Literal, Protocol, cast
from uuid import UUID, uuid4

from myclaw.provider.errors import EmptyModelResponseError
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall
from myclaw.utils.validation import require_nonnegative_int, require_uuid4

type ModelRoute = Literal["default", "chat", "memory", "schedule"]
type ReasoningEffort = Literal["low", "medium", "high"]
type FinishReason = Literal["stop", "tool_calls", "length", "cancelled"]
type ModelMessageDictionary = dict[str, Any]
type ModelMessages = Sequence[ModelMessageDictionary]


@dataclass(frozen=True, slots=True)
class _ProviderCallArguments:
    messages: ModelMessages
    tools: Sequence[OpenAIToolSchema]
    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout: int
    from_legacy_request: bool


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Actual token usage reported by one model call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        require_nonnegative_int(self.input_tokens, field="input_tokens")
        require_nonnegative_int(self.output_tokens, field="output_tokens")
        require_nonnegative_int(self.total_tokens, field="total_tokens")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            msg = "total_tokens must equal input_tokens + output_tokens"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class UserModelMessage:
    """User input sent through a Model Provider."""

    role: ClassVar[Literal["user"]] = "user"
    content: str

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class AssistantModelMessage:
    """Aggregated assistant content and provider tool calls."""

    role: ClassVar[Literal["assistant"]] = "assistant"
    content: str
    tool_calls: tuple[ModelToolCall, ...] = ()

    def __post_init__(self) -> None:
        tool_call_ids = [tool_call.id for tool_call in self.tool_calls]
        if len(set(tool_call_ids)) != len(tool_call_ids):
            raise ValueError("assistant tool call IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
        }


@dataclass(frozen=True, slots=True)
class ToolModelMessage:
    """A normalized Tool result returned to a model."""

    role: ClassVar[Literal["tool"]] = "tool"
    tool_call_id: str
    name: str
    content: str

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
        }


type ModelMessage = UserModelMessage | AssistantModelMessage | ToolModelMessage


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """A fully resolved request passed to one Model Provider."""

    request_id: UUID
    route: ModelRoute
    system_prompt: str
    messages: tuple[ModelMessage, ...]
    tools: tuple[OpenAIToolSchema, ...]
    stream: bool
    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int

    def __post_init__(self) -> None:
        require_uuid4(self.request_id, field="request_id")
        if self.route == "chat" and not self.stream:
            msg = "chat requests must stream"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": str(self.request_id),
            "route": self.route,
            "system_prompt": self.system_prompt,
            "messages": [message.to_dict() for message in self.messages],
            "tools": list(self.tools),
            "stream": self.stream,
            "model": self.model,
            "max_output": self.max_output,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "timeout_seconds": self.timeout_seconds,
        }


def accepts_direct_provider_call(method: object) -> bool:
    """Detect whether a Provider method exposes the direct message seam."""
    if not callable(method):
        return True
    try:
        method_signature = signature(method)
    except (TypeError, ValueError):
        return True
    return "messages" in method_signature.parameters


def legacy_request_from_direct(
    *,
    route: ModelRoute,
    messages: ModelMessages,
    tools: Sequence[OpenAIToolSchema],
    model: str,
    max_output: int,
    temperature: float,
    reasoning_effort: ReasoningEffort | None,
    timeout: int,
    stream: bool,
) -> ModelRequest:
    """Adapt direct dictionaries for Providers that remain on the old seam."""
    if not messages or messages[0].get("role") != "system":
        raise TypeError("direct model messages must start with a system message")
    system_prompt = messages[0].get("content")
    if not isinstance(system_prompt, str):
        raise TypeError("direct system message content must be a string")
    return ModelRequest(
        request_id=uuid4(),
        route=route,
        system_prompt=system_prompt,
        messages=tuple(_legacy_model_message(message) for message in messages[1:]),
        tools=tuple(tools),
        stream=stream,
        model=model,
        max_output=max_output,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout,
    )


def _legacy_model_message(message: ModelMessageDictionary) -> ModelMessage:
    role = message.get("role")
    if role == "user":
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("direct user message content must be a string")
        return UserModelMessage(content=content)
    if role == "assistant":
        content = message.get("content")
        raw_tool_calls = message.get("tool_calls", [])
        if not isinstance(content, str) or not isinstance(raw_tool_calls, list):
            raise TypeError("direct assistant message is malformed")
        return AssistantModelMessage(
            content=content,
            tool_calls=tuple(
                ModelToolCall(
                    id=tool_call["id"],
                    name=tool_call["name"],
                    arguments=tool_call["arguments"],
                )
                for tool_call in raw_tool_calls
                if isinstance(tool_call, dict)
            ),
        )
    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        name = message.get("name")
        content = message.get("content")
        if not all(isinstance(value, str) for value in (tool_call_id, name, content)):
            raise TypeError("direct Tool message is malformed")
        return ToolModelMessage(
            tool_call_id=cast(str, tool_call_id),
            name=cast(str, name),
            content=cast(str, content),
        )
    raise TypeError("direct model message role is unsupported")


def resolve_provider_call_arguments(
    request: ModelRequest | None,
    *,
    messages: ModelMessages | None,
    tools: Sequence[OpenAIToolSchema] | None,
    model: str | None,
    max_output: int | None,
    temperature: float | None,
    reasoning_effort: ReasoningEffort | None,
    timeout: int | None,
) -> _ProviderCallArguments:
    """Normalize the temporary request-object and direct Provider call seams."""
    if request is not None:
        if any(
            value is not None
            for value in (
                messages,
                tools,
                model,
                max_output,
                temperature,
                reasoning_effort,
                timeout,
            )
        ):
            raise TypeError("legacy ModelRequest and direct Provider arguments cannot be mixed")
        return _ProviderCallArguments(
            messages=[
                {"role": "system", "content": request.system_prompt},
                *(message.to_dict() for message in request.messages),
            ],
            tools=request.tools,
            model=request.model,
            max_output=request.max_output,
            temperature=request.temperature,
            reasoning_effort=request.reasoning_effort,
            timeout=request.timeout_seconds,
            from_legacy_request=True,
        )
    if messages is None:
        raise TypeError("direct Provider calls require messages")
    if tools is None:
        raise TypeError("direct Provider calls require tools")
    if model is None:
        raise TypeError("direct Provider calls require model")
    if max_output is None:
        raise TypeError("direct Provider calls require max_output")
    if temperature is None:
        raise TypeError("direct Provider calls require temperature")
    if timeout is None:
        raise TypeError("direct Provider calls require timeout")
    return _ProviderCallArguments(
        messages=messages,
        tools=tools,
        model=model,
        max_output=max_output,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        from_legacy_request=False,
    )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One complete provider-neutral model response."""

    message: AssistantModelMessage
    usage: ModelUsage
    finish_reason: FinishReason

    def __post_init__(self) -> None:
        if not self.message.content.strip() and not self.message.tool_calls:
            raise EmptyModelResponseError("model response requires content or tool calls")

    def to_dict(self) -> dict[str, object]:
        return {
            "message": self.message.to_dict(),
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True, slots=True)
class TextDelta:
    """One ordered, non-provider-specific streaming text chunk."""

    type: ClassVar[Literal["text_delta"]] = "text_delta"
    delta: str

    def __post_init__(self) -> None:
        if not self.delta:
            msg = "delta must not be empty"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "delta": self.delta}


@dataclass(frozen=True, slots=True)
class ModelCompleted:
    """The sole successful terminal event in a provider stream."""

    type: ClassVar[Literal["completed"]] = "completed"
    response: ModelResponse

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "response": self.response.to_dict()}


type ModelStreamEvent = TextDelta | ModelCompleted


class ModelProvider(Protocol):
    """Legacy request-object Model Provider boundary during the migration stack."""

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    async def close(self) -> None: ...


class DirectModelProvider(Protocol):
    """Temporary direct-call seam for provider-neutral message dictionaries."""

    def stream(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(
        self,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: ReasoningEffort | None,
        timeout: int,
    ) -> ModelResponse: ...

    async def close(self) -> None: ...
