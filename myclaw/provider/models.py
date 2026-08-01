"""Provider-neutral model request and response values."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, runtime_checkable
from uuid import UUID

from myclaw.tools.models import ModelToolCall
from myclaw.tools.schema import OpenAIToolSchema
from myclaw.utils.validation import require_nonnegative_int, require_uuid4

type ModelRoute = Literal["default", "chat", "memory", "cron"]
type ReasoningEffort = Literal["low", "medium", "high"]
type FinishReason = Literal["stop", "tool_calls", "length", "cancelled"]


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


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One complete provider-neutral model response."""

    message: AssistantModelMessage
    usage: ModelUsage
    finish_reason: FinishReason

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


@runtime_checkable
class ModelProvider(Protocol):
    """Execute provider-neutral model requests."""

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    async def close(self) -> None: ...
