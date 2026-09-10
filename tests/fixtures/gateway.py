"""Small test adapters for exercising one concrete Tool through the common pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from myclaw.agent.tools.base import BaseTool
from myclaw.agent.tools.tool_gateway import (
    ConfirmationRequester,
    ModelToolCall,
    ToolGateway,
    ToolResult,
)


class SingleToolGateway(ToolGateway):
    """Bind a test Tool to the production preparation and result pipeline."""

    def __init__(
        self,
        tools: Iterable[BaseTool],
        *,
        confirmation: ConfirmationRequester | None = None,
    ) -> None:
        self._gateway = ToolGateway._for_memory(tuple(tools))
        self._confirmation = confirmation

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._gateway.schemas

    async def call(
        self,
        tool_call: ModelToolCall,
        *,
        confirmation: ConfirmationRequester | None = None,
    ) -> ToolResult:
        requester = self._confirmation if confirmation is None else confirmation
        return await self._gateway.call(tool_call, confirmation=requester)
