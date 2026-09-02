"""Test-only adapters for Task Framing requests."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from typing import Any

from myclaw.agent.runner import AgentRunnerRoute, AgentRunnerRouter
from myclaw.provider.models import (
    ModelContinuation,
    ModelMessages,
    ModelResponse,
    ModelRoute,
    ModelStreamEvent,
)
from myclaw.tools.base import OpenAIToolSchema


class TaskFramingRouterAdapter:
    """Script Task Framing completions while delegating every other Router call."""

    def __init__(
        self,
        delegate: AgentRunnerRouter,
        outcomes: Sequence[ModelResponse | BaseException] = (),
    ) -> None:
        self._delegate = delegate
        self._outcomes = deque(outcomes)
        self.framing_requests: list[ModelMessages] = []

    def stream(
        self,
        route: AgentRunnerRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        return self._delegate.stream(
            route,
            messages=messages,
            tools=tools,
            continuation=continuation,
        )

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        if _is_task_framing_request(route, messages=messages, tools=tools):
            self.framing_requests.append(deepcopy(messages))
            return await self._complete_task_framing()
        return await self._delegate.complete(
            route,  # type: ignore[arg-type]
            messages=messages,
            tools=tools,
            continuation=continuation,
        )

    async def _complete_task_framing(self) -> ModelResponse:
        if not self._outcomes:
            raise RuntimeError("Task Framing response is not scripted for this test")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class BlockingTaskFramingRouterAdapter(TaskFramingRouterAdapter):
    """Block a Task Framing completion until the caller cancels it."""

    def __init__(self, delegate: AgentRunnerRouter) -> None:
        super().__init__(delegate)
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def _complete_task_framing(self) -> ModelResponse:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


def _is_task_framing_request(
    route: ModelRoute,
    *,
    messages: ModelMessages,
    tools: Sequence[OpenAIToolSchema],
) -> bool:
    return (
        route == "chat"
        and not tools
        and len(messages) == 1
        and messages[0].get("role") == "system"
    )
