"""The fixed Core Tool Catalog and its only invocation boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, cast
from uuid import UUID, uuid4

from loguru import logger

from myclaw.agent.workspace import Workspace
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.tools.base import (
    ArtifactReference,
    BaseTool,
    OpenAIToolSchema,
    PreparedToolCall,
    ToolError,
)
from myclaw.tools.core.edit_file import EditFileTool
from myclaw.tools.core.exec import ExecTool
from myclaw.tools.core.glob import GlobTool
from myclaw.tools.core.grep import GrepTool
from myclaw.tools.core.list_dir import ListDirTool
from myclaw.tools.core.read_file import ReadFileTool
from myclaw.tools.core.schedule import ScheduleTool
from myclaw.tools.core.web_fetch import WebFetchTool
from myclaw.tools.core.web_search import WebSearchTool
from myclaw.tools.core.write_file import WriteFileTool
from myclaw.utils.json_types import JsonObject, JsonValue
from myclaw.utils.validation import require_uuid4

type ConfirmationDecision = Literal["approved", "declined"]
type ConfirmationOutcome = ConfirmationDecision | None
type ToolResultStatus = Literal["success", "error", "refused"]
type ConfirmationRequester = Callable[["ConfirmationRequest"], Awaitable[ConfirmationDecision]]


@dataclass(frozen=True, slots=True, init=False)
class ConfirmationRequest:
    """One immutable confirmation request bound to one normalized Tool call."""

    confirmation_id: UUID
    tool_call_id: str
    tool_name: str
    reason: str
    summary: str
    _details: JsonObject = field(repr=False)
    warnings: tuple[str, ...] = ()

    def __init__(
        self,
        confirmation_id: UUID,
        tool_call_id: str,
        tool_name: str,
        summary: str,
        details: JsonObject,
        warnings: tuple[str, ...] = (),
        *,
        reason: str = "",
    ) -> None:
        require_uuid4(confirmation_id, field="confirmation_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise TypeError("confirmation tool_call_id must be a non-empty string")
        if not isinstance(tool_name, str) or not tool_name:
            raise TypeError("confirmation tool_name must be a non-empty string")
        if not isinstance(summary, str) or not summary or len(summary) > 240:
            raise ValueError("confirmation summary must contain 1 through 240 characters")
        if not isinstance(reason, str):
            raise TypeError("confirmation reason must be a string")
        if not isinstance(details, dict):
            raise TypeError("confirmation details must be a JSON object")
        if not isinstance(warnings, (tuple, list)) or any(
            not isinstance(item, str) for item in warnings
        ):
            raise TypeError("confirmation warnings must be a sequence of strings")
        object.__setattr__(self, "confirmation_id", confirmation_id)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "_details", deepcopy(details))
        object.__setattr__(self, "warnings", tuple(warnings))

    @property
    def details(self) -> JsonObject:
        """Return a detached view of the normalized operation details."""
        return deepcopy(self._details)

    def to_dict(self) -> dict[str, object]:
        return {
            "confirmation_id": str(self.confirmation_id),
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "reason": self.reason,
            "summary": self.summary,
            "details": deepcopy(self._details),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ToolConfirmationMetadata:
    """The request and decision carried by a Tool Result."""

    request: ConfirmationRequest
    decision: ConfirmationOutcome

    def to_dict(self) -> dict[str, object]:
        return {"request": self.request.to_dict(), "decision": self.decision}


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """A provider Tool call preserving its raw JSON argument text."""

    id: str
    name: str
    arguments: str

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The normalized result returned by the Tool Gateway."""

    tool_call_id: str
    name: str
    status: ToolResultStatus
    content: str
    artifact: ArtifactReference | None = None
    confirmation: ToolConfirmationMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("Tool result name must be a non-empty string")
        if self.status not in {"success", "error", "refused"}:
            raise ValueError("Tool result status is invalid")
        if not isinstance(self.content, str):
            raise TypeError("Tool result content must be a string")
        if self.artifact is not None and not isinstance(self.artifact, ArtifactReference):
            raise TypeError("Tool result artifact must be an ArtifactReference")
        if self.status != "success" and self.artifact is not None:
            raise ValueError("only successful Tool results may contain an artifact")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "status": self.status,
            "content": self.content,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
        }
        if self.confirmation is not None:
            result["confirmation"] = self.confirmation.to_dict()
        return result


class ConfirmationChannel:
    """In-memory interactive channel for one live Agent Run."""

    def __init__(self) -> None:
        self._requests: asyncio.Queue[ConfirmationRequest | None] = asyncio.Queue()
        self._pending: dict[UUID, asyncio.Future[ConfirmationDecision]] = {}
        self._consumed: set[UUID] = set()
        self._closed = False

    async def __call__(self, request: ConfirmationRequest) -> ConfirmationDecision:
        if self._closed:
            raise RuntimeError("Confirmation channel is closed")
        if request.confirmation_id in self._pending or request.confirmation_id in self._consumed:
            raise ValueError("Confirmation request is already pending or consumed")

        future: asyncio.Future[ConfirmationDecision] = asyncio.get_running_loop().create_future()
        self._pending[request.confirmation_id] = future
        await self._requests.put(request)
        try:
            return await future
        finally:
            self._pending.pop(request.confirmation_id, None)
            self._consumed.add(request.confirmation_id)

    async def next_request(self) -> ConfirmationRequest:
        """Return the next live request for the interactive host."""
        while True:
            if self._closed:
                raise RuntimeError("Confirmation channel is closed")
            request = await self._requests.get()
            if request is None:
                self._requests.put_nowait(None)
                raise RuntimeError("Confirmation channel is closed")
            future = self._pending.get(request.confirmation_id)
            if future is not None and not future.done():
                return request

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        if decision not in {"approved", "declined"}:
            raise ValueError("confirmation decision must be approved or declined")
        future = self._pending.get(confirmation_id)
        if future is None or future.done():
            raise ValueError("Confirmation response is late or unknown")
        future.set_result(decision)

    def close(self) -> None:
        """Invalidate pending requests and wake request observers."""
        if self._closed:
            return
        self._closed = True
        for future in self._pending.values():
            future.cancel()
        self._requests.put_nowait(None)


class ToolGateway:
    """Create and invoke the fixed ten-Tool Catalog."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        schedule_store: WorkspaceScheduleStore,
        scheduled_agent: bool = False,
    ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("Tool Gateway requires a Workspace")
        if not isinstance(schedule_store, WorkspaceScheduleStore):
            raise TypeError("Tool Gateway requires a WorkspaceScheduleStore")
        if not isinstance(scheduled_agent, bool):
            raise TypeError("scheduled_agent must be a boolean")

        tools: tuple[BaseTool, ...] = (
            ReadFileTool(workspace=workspace),
            WriteFileTool(workspace=workspace),
            EditFileTool(workspace=workspace),
            ListDirTool(workspace=workspace),
            GlobTool(workspace=workspace),
            GrepTool(workspace=workspace),
            ExecTool(workspace=workspace),
            WebSearchTool(),
            WebFetchTool(),
            ScheduleTool(store=schedule_store, scheduled_agent=scheduled_agent),
        )
        self._tools = {tool.name: tool for tool in tools}
        self._schemas = [tool.to_schema() for tool in tools]
        self._failure_observer: Callable[[Exception], None] | None = None

    @classmethod
    def _for_memory(
        cls,
        tools: tuple[BaseTool, ...],
        *,
        on_failure: Callable[[Exception], None] | None = None,
    ) -> ToolGateway:
        """Build the isolated Long-term Memory catalog without widening the public API."""
        if not tools or len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Memory Tool names must be unique and non-empty")
        gateway = object.__new__(cls)
        catalog: tuple[BaseTool, ...] = tools
        gateway._tools = {tool.name: tool for tool in catalog}
        gateway._schemas = [tool.to_schema() for tool in catalog]
        gateway._failure_observer = on_failure
        return gateway

    @property
    def schemas(self) -> list[OpenAIToolSchema]:
        """Return a detached JSON list in fixed Catalog order."""
        return deepcopy(self._schemas)

    async def call(
        self,
        tool_call: ModelToolCall,
        *,
        confirmation: ConfirmationRequester | None = None,
    ) -> ToolResult:
        """Parse, prepare, confirm when needed, execute, and normalize one call."""
        raw_arguments = tool_call.arguments
        if not isinstance(raw_arguments, str):
            return _result(tool_call, "error", "Tool arguments could not be parsed.")
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return _result(tool_call, "error", "Tool arguments could not be parsed.")
        if not isinstance(parsed, dict):
            return _result(tool_call, "error", "Tool arguments could not be parsed.")

        tool = self._tools.get(tool_call.name)
        if tool is None:
            return _result(tool_call, "error", "The requested tool is not available.")

        try:
            preparation = await tool.prepare(cast(JsonObject, parsed))
            if not isinstance(preparation, PreparedToolCall):
                raise TypeError("Tool preparation returned an invalid value")
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool.name))

        normalized = preparation.arguments
        try:
            refusal = self._refusal_reason(tool, normalized)
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool.name))
        if refusal is not None:
            return _result(tool_call, "refused", refusal)

        if preparation.safety_reason is None:
            return await self._execute(tool_call, tool, normalized, confirmation=None)

        try:
            confirmation_details = cast(JsonObject, _project_confirmation_details(normalized))
            if tool.name == "exec":
                for name in ("command", "cwd", "timeout"):
                    confirmation_details[name] = deepcopy(normalized[name])
            request = ConfirmationRequest(
                confirmation_id=uuid4(),
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                reason=preparation.safety_reason,
                summary=f"Confirm {tool.name}"[:240],
                details=confirmation_details,
            )
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            return _result(tool_call, "error", error.message)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(tool_call, "error", _generic_tool_failure(tool.name))

        if confirmation is None:
            return _refused_confirmation_result(
                tool_call,
                "Tool confirmation is unavailable.",
                request=request,
                decision=None,
            )

        try:
            decision = await confirmation(request)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _refused_confirmation_result(
                tool_call,
                "Tool confirmation was expired or invalid.",
                request=request,
                decision=None,
            )
        if decision not in {"approved", "declined"}:
            return _refused_confirmation_result(
                tool_call,
                "Tool confirmation was expired or invalid.",
                request=request,
                decision=None,
            )
        metadata = ToolConfirmationMetadata(request=request, decision=decision)
        if decision == "declined":
            return _result(
                tool_call,
                "refused",
                "Tool confirmation was declined.",
                confirmation=metadata,
            )
        return await self._execute(tool_call, tool, normalized, confirmation=metadata)

    @staticmethod
    def _refusal_reason(tool: BaseTool, normalized: JsonObject) -> str | None:
        refusal = getattr(tool, "refusal_reason", None)
        if refusal is None:
            return None
        reason = cast(Callable[..., object], refusal)(**deepcopy(normalized))
        if reason is not None and not isinstance(reason, str):
            raise TypeError("Tool refusal checks must return a string reason or None")
        return reason

    async def _execute(
        self,
        tool_call: ModelToolCall,
        tool: BaseTool,
        normalized: JsonObject,
        *,
        confirmation: ToolConfirmationMetadata | None,
    ) -> ToolResult:
        try:
            content = await tool.execute(**_declared_arguments(tool, normalized))
            if not isinstance(content, str):
                raise TypeError("Tool execution must return a string")
        except asyncio.CancelledError:
            raise
        except ToolError as error:
            if self._failure_observer is not None:
                self._failure_observer(error)
            return _result(tool_call, "error", error.message, confirmation=confirmation)
        except Exception as error:
            self._record_unexpected_failure(tool, error)
            return _result(
                tool_call,
                "error",
                _generic_tool_failure(tool.name),
                confirmation=confirmation,
            )
        return _result(tool_call, "success", content, confirmation=confirmation)

    def _record_unexpected_failure(self, tool: BaseTool, error: Exception) -> None:
        if self._failure_observer is None:
            logger.opt(exception=error).error(
                "Tool execution failed name={} type={}",
                tool.name,
                type(error).__name__,
            )
            return
        self._failure_observer(error)


def _declared_arguments(tool: BaseTool, arguments: JsonObject) -> JsonObject:
    return {
        name: deepcopy(arguments[name]) for name in tool.parameters.properties if name in arguments
    }


def _project_confirmation_details(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        if len(value) <= 256:
            return value
        return {"value": value[:256], "original_length": len(value)}
    if isinstance(value, list):
        return [_project_confirmation_details(item) for item in value]
    if isinstance(value, dict):
        return {name: _project_confirmation_details(item) for name, item in value.items()}
    return deepcopy(value)


def _generic_tool_failure(tool_name: str) -> str:
    return f"{tool_name} could not complete the request."


def _refused_confirmation_result(
    tool_call: ModelToolCall,
    content: str,
    *,
    request: ConfirmationRequest,
    decision: ConfirmationOutcome,
) -> ToolResult:
    return _result(
        tool_call,
        "refused",
        content,
        confirmation=ToolConfirmationMetadata(request=request, decision=decision),
    )


def _result(
    tool_call: ModelToolCall,
    status: ToolResultStatus,
    content: str,
    confirmation: ToolConfirmationMetadata | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status=status,
        content=content,
        artifact=None,
        confirmation=confirmation,
    )


__all__ = [
    "ConfirmationChannel",
    "ConfirmationDecision",
    "ConfirmationRequest",
    "ConfirmationRequester",
    "ModelToolCall",
    "ToolConfirmationMetadata",
    "ToolGateway",
    "ToolResult",
    "ToolResultStatus",
]
