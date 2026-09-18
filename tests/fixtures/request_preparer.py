"""Test-only request preparers for the Session-independent Agent Runner."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from myclaw.provider.models import ModelResponse


class DetachedRequestPreparer:
    """Return a detached request without adding a context projection."""

    @property
    def recounts_retained_tool_calls(self) -> bool:
        return False

    async def prepare(
        self,
        candidate: Sequence[dict[str, Any]],
        *,
        increment: Sequence[dict[str, Any]],
        latest_cycle_start: int | None,
        tools: Sequence[dict[str, Any]],
        continuation_revision: int,
    ) -> list[dict[str, Any]]:
        del increment, latest_cycle_start, tools, continuation_revision
        return deepcopy(list(candidate))

    def observe_request_projection(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        micro_compression_enabled: bool,
    ) -> None:
        del messages, micro_compression_enabled

    def record_response(
        self,
        *,
        request_messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        response: ModelResponse,
        increment: Sequence[dict[str, Any]],
    ) -> None:
        del request_messages, tools, response, increment
