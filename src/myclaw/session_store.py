"""Workspace-scoped JSONL Conversation Session persistence."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from myclaw.agent_home import AgentHome
from myclaw.atomic_files import atomic_replace_bytes, atomic_replace_text
from myclaw.contracts import (
    AssistantSessionMessage,
    ConversationSession,
    CumulativeUsage,
    MetadataUpdate,
    ModelUsage,
    SessionMessage,
    SessionMetadata,
    SessionSummary,
    UserSessionMessage,
    make_session_id,
)
from myclaw.contracts.common import require_session_id
from myclaw.workspace import Workspace


class JsonlSessionStore:
    """Persist complete Session records beneath one Workspace directory."""

    def __init__(
        self,
        *,
        agent_home: AgentHome,
        workspace: Workspace,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        self.agent_home = agent_home
        self.workspace = workspace
        self._now = now
        self._new_uuid = new_uuid
        self._prepared: dict[str, SessionMetadata] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def directory(self) -> Path:
        return self.agent_home.path / "sessions" / self.workspace.slug

    def path_for(self, session_id: str) -> Path:
        require_session_id(session_id)
        return self.directory / f"{session_id}.jsonl"

    def prepare(self) -> SessionMetadata:
        """Prepare a new Session identity without creating its file or directory."""
        created_at = self._now()
        session_id = make_session_id(created_at, self._new_uuid())
        persisted_created_at = created_at.replace(microsecond=created_at.microsecond // 1000 * 1000)
        metadata = SessionMetadata(
            id=session_id,
            title="Untitled session",
            created_at=persisted_created_at,
            updated_at=persisted_created_at,
            consolidation_cursor=0,
            cumulative_usage=CumulativeUsage(
                model_calls=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
        )
        self._prepared[session_id] = metadata
        return metadata

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        path = self.path_for(session_id)
        io_path = _io_path(path)
        async with self._lock_for(session_id):
            if not io_path.exists():
                metadata = self._prepared.get(session_id)
                if metadata is None:
                    msg = "Session must be prepared before its first message is appended"
                    raise ValueError(msg)
                io_path.parent.mkdir(parents=True, exist_ok=True)
                updated = _metadata_after_message(metadata, message)
                atomic_replace_text(io_path, updated.to_json_line() + message.to_json_line())
                self._prepared[session_id] = updated
                return
            records = _read_complete_records(io_path)
            metadata = _parse_metadata(records[0])
            if metadata.id != session_id:
                msg = "Session metadata ID does not match its file name"
                raise ValueError(msg)
            updated = _metadata_after_message(metadata, message)
            _append_complete_line(io_path, message.to_json_line())
            content = io_path.read_bytes()
            first_line_end = content.index(b"\n") + 1
            atomic_replace_bytes(
                io_path,
                updated.to_json_line().encode("utf-8") + content[first_line_end:],
            )

    async def load(self, session_id: str) -> ConversationSession:
        path = self.path_for(session_id)
        async with self._lock_for(session_id):
            records = _read_complete_records(_io_path(path))
        metadata = _parse_metadata(records[0])
        if metadata.id != session_id:
            msg = "Session metadata ID does not match its file name"
            raise ValueError(msg)
        messages = tuple(_parse_message(record) for record in records[1:])
        return ConversationSession(metadata=metadata, messages=messages)

    async def current_session(self, session_id: str) -> ConversationSession:
        """Return persisted state or an unmaterialized prepared Session snapshot."""
        path = _io_path(self.path_for(session_id))
        async with self._lock_for(session_id):
            if not path.exists():
                metadata = self._prepared.get(session_id)
                if metadata is None:
                    msg = "Session has not been prepared"
                    raise ValueError(msg)
                return ConversationSession(metadata=metadata, messages=())
            records = _read_complete_records(path)
        metadata = _parse_metadata(records[0])
        if metadata.id != session_id:
            msg = "Session metadata ID does not match its file name"
            raise ValueError(msg)
        messages = tuple(_parse_message(record) for record in records[1:])
        return ConversationSession(metadata=metadata, messages=messages)

    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None:
        path = _io_path(self.path_for(session_id))
        async with self._lock_for(session_id):
            content = path.read_bytes()
            records = _read_complete_records(path)
            metadata = _parse_metadata(records[0])
            if metadata.id != session_id:
                msg = "Session metadata ID does not match its file name"
                raise ValueError(msg)
            updated = replace(
                metadata,
                title=metadata.title if update.title is None else update.title,
                updated_at=(
                    metadata.updated_at if update.updated_at is None else update.updated_at
                ),
                consolidation_cursor=(
                    metadata.consolidation_cursor
                    if update.consolidation_cursor is None
                    else update.consolidation_cursor
                ),
                cumulative_usage=(
                    metadata.cumulative_usage
                    if update.cumulative_usage is None
                    else update.cumulative_usage
                ),
            )
            messages = tuple(_parse_message(record) for record in records[1:])
            ConversationSession(metadata=updated, messages=messages)
            first_line_end = content.index(b"\n") + 1
            atomic_replace_bytes(
                path,
                updated.to_json_line().encode("utf-8") + content[first_line_end:],
            )
            self._prepared[session_id] = updated

    async def list_for_workspace(self, workspace: Path) -> tuple[SessionSummary, ...]:
        raise NotImplementedError

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())


def _metadata_after_message(metadata: SessionMetadata, message: SessionMessage) -> SessionMetadata:
    usage = metadata.cumulative_usage
    if isinstance(message, AssistantSessionMessage):
        usage = CumulativeUsage(
            model_calls=usage.model_calls + 1,
            input_tokens=usage.input_tokens + message.usage.input_tokens,
            output_tokens=usage.output_tokens + message.usage.output_tokens,
            total_tokens=usage.total_tokens + message.usage.total_tokens,
        )
    return replace(metadata, updated_at=message.created_at, cumulative_usage=usage)


def _append_complete_line(path: Path, line: str) -> None:
    content = line.encode("utf-8")
    with path.open("ab") as stream:
        written = stream.write(content)
        if written != len(content):
            raise OSError("session append did not write the complete JSONL record")
        stream.flush()
        os.fsync(stream.fileno())


def _io_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    native = str(path.absolute())
    if native.startswith("\\\\"):
        unc_native = native.lstrip("\\")
        return Path(f"\\\\?\\UNC\\{unc_native}")
    return Path(f"\\\\?\\{native}")


def _read_complete_records(path: Path) -> tuple[dict[str, object], ...]:
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        msg = "Session must contain complete JSONL records"
        raise ValueError(msg)
    records: list[dict[str, object]] = []
    for line in lines:
        decoded: object = json.loads(line.decode("utf-8"))
        if not isinstance(decoded, dict):
            msg = "Session records must be JSON objects"
            raise ValueError(msg)
        records.append(cast(dict[str, object], decoded))
    return tuple(records)


def _parse_metadata(record: dict[str, object]) -> SessionMetadata:
    usage = _object(record, "cumulative_usage")
    return SessionMetadata(
        id=_string(record, "id"),
        title=_string(record, "title"),
        created_at=_datetime(record, "created_at"),
        updated_at=_datetime(record, "updated_at"),
        consolidation_cursor=_integer(record, "consolidation_cursor"),
        cumulative_usage=CumulativeUsage(
            model_calls=_integer(usage, "model_calls"),
            input_tokens=_integer(usage, "input_tokens"),
            output_tokens=_integer(usage, "output_tokens"),
            total_tokens=_integer(usage, "total_tokens"),
        ),
    )


def _parse_message(record: dict[str, object]) -> SessionMessage:
    role = record.get("role")
    if role == "user":
        return UserSessionMessage(
            id=_string(record, "id"),
            created_at=_datetime(record, "created_at"),
            content=_string(record, "content"),
        )
    if role == "assistant":
        if record.get("status") != "completed" or record.get("error") is not None:
            msg = "Only completed assistant records are supported"
            raise ValueError(msg)
        if record.get("tool_calls") != []:
            msg = "Assistant tool calls are not supported yet"
            raise ValueError(msg)
        usage = _object(record, "usage")
        return AssistantSessionMessage(
            id=_string(record, "id"),
            created_at=_datetime(record, "created_at"),
            content=_string(record, "content"),
            tool_calls=(),
            status="completed",
            error=None,
            usage=ModelUsage(
                input_tokens=_integer(usage, "input_tokens"),
                output_tokens=_integer(usage, "output_tokens"),
                total_tokens=_integer(usage, "total_tokens"),
            ),
        )
    msg = f"Unsupported Session message role: {role!r}"
    raise ValueError(msg)


def _object(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if not isinstance(value, dict):
        msg = f"Session field '{field}' must be an object"
        raise ValueError(msg)
    return cast(dict[str, object], value)


def _string(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        msg = f"Session field '{field}' must be a string"
        raise ValueError(msg)
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Session field '{field}' must be an integer"
        raise ValueError(msg)
    return value


def _datetime(record: dict[str, object], field: str) -> datetime:
    return datetime.fromisoformat(_string(record, field))
