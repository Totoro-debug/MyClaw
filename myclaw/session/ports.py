"""Conversation Session persistence boundary."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from myclaw.session.records import (
    ConversationSession,
    MetadataUpdate,
    SessionMessage,
    SessionSummary,
)


@runtime_checkable
class SessionStore(Protocol):
    async def append_message(self, session_id: str, message: SessionMessage) -> None: ...

    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None: ...

    async def load(self, session_id: str) -> ConversationSession: ...

    async def list_for_workspace(self, workspace: Path) -> tuple[SessionSummary, ...]: ...
