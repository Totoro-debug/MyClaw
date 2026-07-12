"""Scripted provider boundary for deterministic offline tests."""

from collections import deque
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

from myclaw.contracts import (
    ErrorInfo,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
)
from myclaw.prompts import session_title_prompt


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

    async def complete(self, request: object) -> ModelResponse:
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
