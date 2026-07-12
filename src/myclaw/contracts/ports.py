"""Minimal dependency-inversion Protocols for runtime boundaries."""

from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

from myclaw.contracts.events import AgentEvent
from myclaw.contracts.json_types import JsonObject
from myclaw.contracts.management import (
    ConfigView,
    MemoryTaskResult,
    ResumeResult,
    RuntimeStatus,
)
from myclaw.contracts.memory import SummaryEntry
from myclaw.contracts.models import ModelRequest, ModelResponse, ModelStreamEvent
from myclaw.contracts.sessions import (
    ConversationSession,
    MetadataUpdate,
    SessionMessage,
    SessionSummary,
)
from myclaw.contracts.tools import ToolDefinition, ToolExecutionContext


@runtime_checkable
class ConversationPort(Protocol):
    def submit(self, text: str) -> AsyncIterator[AgentEvent]: ...

    async def resolve_permission(self, request_id: UUID, approved: bool) -> None: ...

    async def cancel_active_turn(self) -> None: ...


@runtime_checkable
class ManagementPort(Protocol):
    async def config_view(self) -> ConfigView: ...

    async def status(self) -> RuntimeStatus: ...

    async def resumable_sessions(self) -> tuple[SessionSummary, ...]: ...

    async def resume(self, session_id: str) -> ResumeResult: ...

    async def memory_view(self) -> str: ...

    async def dream(self) -> MemoryTaskResult: ...


@runtime_checkable
class SessionStore(Protocol):
    async def append_message(self, session_id: str, message: SessionMessage) -> None: ...

    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None: ...

    async def load(self, session_id: str) -> ConversationSession: ...

    async def list_for_workspace(self, workspace: Path) -> tuple[SessionSummary, ...]: ...


@runtime_checkable
class SummaryStore(Protocol):
    async def append(self, content: str, timestamp: datetime) -> SummaryEntry: ...

    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]: ...


@runtime_checkable
class MemoryStore(Protocol):
    async def read_long_term(self) -> str: ...

    async def replace_long_term(self, content: str) -> None: ...

    async def read_summary_cursor(self) -> int: ...

    async def write_summary_cursor(self, index: int) -> None: ...


@runtime_checkable
class ModelProvider(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    async def close(self) -> None: ...


@runtime_checkable
class Tool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str: ...
