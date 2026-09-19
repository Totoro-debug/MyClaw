"""Test-only request preparers for the Session-independent Agent Runner."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from myclaw.provider.models import ModelContinuation, ModelResponse


class DetachedRequestPreparer:
    """Return a detached request without adding a context projection."""

    def __init__(self, initial_messages: Sequence[dict[str, Any]] = ()) -> None:
        self._initial_messages = deepcopy(list(initial_messages))

    async def prepare(
        self,
        *,
        increment: Sequence[dict[str, Any]],
        latest_cycle_start: int | None,
        tools: Sequence[dict[str, Any]],
        continuation: ModelContinuation | None,
        continuation_revision: int,
    ) -> list[dict[str, Any]]:
        del latest_cycle_start, tools, continuation, continuation_revision
        return deepcopy([*self._initial_messages, *increment])

    def observe_request_projection(
        self,
        _messages: Sequence[dict[str, Any]],
        *,
        micro_compression_enabled: bool,
    ) -> None:
        del micro_compression_enabled

    def record_response(
        self,
        *,
        request_messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        response: ModelResponse,
        increment: Sequence[dict[str, Any]],
    ) -> dict[str, object] | None:
        del request_messages, tools, response, increment
        return None
