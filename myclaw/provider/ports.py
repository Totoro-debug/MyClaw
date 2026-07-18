"""Model Provider boundary."""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from myclaw.provider.models import ModelRequest, ModelResponse, ModelStreamEvent


@runtime_checkable
class ModelProvider(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    async def close(self) -> None: ...
