"""Tool execution boundary."""

from typing import Protocol, runtime_checkable

from myclaw.tools.models import ToolDefinition, ToolExecutionContext
from myclaw.utils.json_types import JsonObject


@runtime_checkable
class Tool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str: ...
