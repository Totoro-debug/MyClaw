"""Workspace-scoped JSONL Conversation Session persistence."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast, runtime_checkable
from uuid import UUID

from loguru import logger

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import STABLE_ERROR_CODES, ErrorCode
from myclaw.provider.models import ModelUsage
from myclaw.session.identifiers import make_session_id, require_session_id
from myclaw.session.records import (
    AssistantMessageStatus,
    AssistantSessionMessage,
    ConversationSession,
    CumulativeUsage,
    MetadataUpdate,
    SessionError,
    SessionMessage,
    SessionMetadata,
    SessionSummary,
    ToolSessionMessage,
    UserSessionMessage,
)
from myclaw.tools.artifacts import ArtifactReference
from myclaw.tools.models import ModelToolCall, ToolResultStatus
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

type AtomicReplaceBytes = Callable[[Path, bytes], None]


@runtime_checkable
class SessionStore(Protocol):
    """Persist and query Conversation Sessions."""

    async def append_message(self, session_id: str, message: SessionMessage) -> None: ...

    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None: ...

    async def load(self, session_id: str) -> ConversationSession: ...


_TOOL_MESSAGE_FIELDS = frozenset(
    {
        "record_type",
        "id",
        "created_at",
        "role",
        "tool_call_id",
        "name",
        "content",
        "status",
        "artifact",
    }
)


@dataclass(frozen=True, slots=True)
class SessionListingReport:
    sessions: tuple[SessionSummary, ...]
    skipped_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.skipped_count, bool)
            or not isinstance(self.skipped_count, int)
            or self.skipped_count < 0
        ):
            raise ValueError("skipped_count must be a nonnegative integer")


class JsonlSessionStore:
    """Persist complete Session records beneath one Workspace directory."""

    def __init__(
        self,
        *,
        workspace_state: WorkspaceState,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        replace_bytes: AtomicReplaceBytes = HOST_FILESYSTEM.atomic_replace_bytes,
    ) -> None:
        self.workspace_state = workspace_state
        self.workspace = workspace_state.workspace
        self._now = now
        self._new_uuid = new_uuid
        self._replace_bytes = replace_bytes
        self._prepared: dict[str, SessionMetadata] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def directory(self) -> Path:
        return self.workspace_state.sessions_directory

    def path_for(self, session_id: str) -> Path:
        require_session_id(session_id)
        return self.directory / f"{session_id}.jsonl"

    def prepare(self) -> SessionMetadata:
        """Prepare a new Session identity without creating its file or directory."""
        created_at = self._now()
        session_id = make_session_id(created_at, self._new_uuid())
        return self.prepare_with_id(
            session_id=session_id,
            title="Untitled session",
            created_at=created_at,
        )

    def prepare_with_id(
        self,
        *,
        session_id: str,
        title: str,
        created_at: datetime,
    ) -> SessionMetadata:
        """Prepare a caller-owned Session identity without materializing its file."""
        require_session_id(session_id)
        persisted_created_at = created_at.replace(microsecond=created_at.microsecond // 1000 * 1000)
        metadata = SessionMetadata(
            id=session_id,
            title=title,
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
        return self._prepared.setdefault(session_id, metadata)

    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        path = self.path_for(session_id)
        io_path = HOST_FILESYSTEM.path_for_io(path)
        async with self._lock_for(session_id):
            if not io_path.exists():
                metadata = self._prepared.get(session_id)
                if metadata is None:
                    msg = "Session must be prepared before its first message is appended"
                    raise ValueError(msg)
                io_path.parent.mkdir(parents=True, exist_ok=True)
                updated = _metadata_after_message(metadata, message)
                HOST_FILESYSTEM.atomic_replace_text(
                    io_path, updated.to_json_line() + message.to_json_line()
                )
                self._prepared[session_id] = updated
                return
            records, complete_content = _read_recoverable_records(io_path)
            metadata = _parse_metadata(records[0])
            if metadata.id != session_id:
                msg = "Session metadata ID does not match its file name"
                raise ValueError(msg)
            updated = _metadata_after_message(metadata, message)
            if len(complete_content) != io_path.stat().st_size:
                first_line_end = complete_content.index(b"\n") + 1
                self._replace_bytes(
                    io_path,
                    updated.to_json_line().encode("utf-8")
                    + complete_content[first_line_end:]
                    + message.to_json_line().encode("utf-8"),
                )
                return
            _append_complete_line(io_path, message.to_json_line())
            content = io_path.read_bytes()
            first_line_end = content.index(b"\n") + 1
            self._replace_bytes(
                io_path,
                updated.to_json_line().encode("utf-8") + content[first_line_end:],
            )

    async def load(self, session_id: str) -> ConversationSession:
        path = self.path_for(session_id)
        async with self._lock_for(session_id):
            records, _ = _read_recoverable_records(HOST_FILESYSTEM.path_for_io(path))
        metadata = _parse_metadata(records[0])
        if metadata.id != session_id:
            msg = "Session metadata ID does not match its file name"
            raise ValueError(msg)
        messages = tuple(_parse_message(record) for record in records[1:])
        return ConversationSession(metadata=metadata, messages=messages)

    async def current_session(self, session_id: str) -> ConversationSession:
        """Return persisted state or an unmaterialized prepared Session snapshot."""
        path = HOST_FILESYSTEM.path_for_io(self.path_for(session_id))
        async with self._lock_for(session_id):
            if not path.exists():
                metadata = self._prepared.get(session_id)
                if metadata is None:
                    msg = "Session has not been prepared"
                    raise ValueError(msg)
                return ConversationSession(metadata=metadata, messages=())
            records, _ = _read_recoverable_records(path)
        metadata = _parse_metadata(records[0])
        if metadata.id != session_id:
            msg = "Session metadata ID does not match its file name"
            raise ValueError(msg)
        messages = tuple(_parse_message(record) for record in records[1:])
        return ConversationSession(metadata=metadata, messages=messages)

    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None:
        path = HOST_FILESYSTEM.path_for_io(self.path_for(session_id))
        async with self._lock_for(session_id):
            records, complete_content = _read_recoverable_records(path)
            metadata = _parse_metadata(records[0])
            if metadata.id != session_id:
                msg = "Session metadata ID does not match its file name"
                raise ValueError(msg)
            cumulative_usage = metadata.cumulative_usage
            if update.cumulative_usage is not None:
                cumulative_usage = update.cumulative_usage
            elif update.usage_delta is not None:
                cumulative_usage = CumulativeUsage(
                    model_calls=cumulative_usage.model_calls + 1,
                    input_tokens=(cumulative_usage.input_tokens + update.usage_delta.input_tokens),
                    output_tokens=(
                        cumulative_usage.output_tokens + update.usage_delta.output_tokens
                    ),
                    total_tokens=(cumulative_usage.total_tokens + update.usage_delta.total_tokens),
                )
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
                cumulative_usage=cumulative_usage,
            )
            messages = tuple(_parse_message(record) for record in records[1:])
            ConversationSession(metadata=updated, messages=messages)
            first_line_end = complete_content.index(b"\n") + 1
            self._replace_bytes(
                path,
                updated.to_json_line().encode("utf-8") + complete_content[first_line_end:],
            )
            self._prepared[session_id] = updated

    async def recover_consolidation_cursor(
        self,
        session_id: str,
        *,
        old_cursor: int,
        new_cursor: int,
    ) -> None:
        """Idempotently advance a journaled Session cursor in this Workspace."""
        require_session_id(session_id)
        path = HOST_FILESYSTEM.path_for_io(self.path_for(session_id))
        async with self._lock_for(session_id):
            records, complete_content = _read_recoverable_records(path)
            metadata = _parse_metadata(records[0])
            if metadata.id != session_id:
                raise ValueError("Session metadata ID does not match its file name")
            if metadata.consolidation_cursor == new_cursor:
                return
            if metadata.consolidation_cursor != old_cursor:
                raise ValueError("session consolidation cursor conflicts with journal")
            updated = replace(metadata, consolidation_cursor=new_cursor)
            messages = tuple(_parse_message(record) for record in records[1:])
            ConversationSession(metadata=updated, messages=messages)
            first_line_end = complete_content.index(b"\n") + 1
            self._replace_bytes(
                path,
                updated.to_json_line().encode("utf-8") + complete_content[first_line_end:],
            )
            self._prepared[session_id] = updated

    async def scan_for_workspace(self, workspace: Path) -> SessionListingReport:
        if Workspace.from_path(workspace) != self.workspace or not self.directory.exists():
            return SessionListingReport(sessions=(), skipped_count=0)
        summaries: list[SessionSummary] = []
        skipped_count = 0
        for path in self.directory.glob("*.jsonl"):
            try:
                session = await self.load(path.stem)
            except (OSError, UnicodeError, ValueError) as error:
                logger.opt(exception=error).warning(
                    "Skipped corrupt or unreadable Conversation Session entry path={} type={}",
                    path,
                    type(error).__name__,
                )
                skipped_count += 1
                continue
            summaries.append(
                SessionSummary(
                    id=session.metadata.id,
                    title=session.metadata.title,
                    created_at=session.metadata.created_at,
                    updated_at=session.metadata.updated_at,
                    message_count=len(session.messages),
                )
            )
        return SessionListingReport(
            sessions=tuple(
                sorted(
                    summaries,
                    key=lambda summary: (
                        summary.updated_at,
                        summary.created_at,
                        summary.id,
                    ),
                    reverse=True,
                )
            ),
            skipped_count=skipped_count,
        )

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


def _read_recoverable_records(
    path: Path,
) -> tuple[tuple[dict[str, object], ...], bytes]:
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    if not lines:
        msg = "Session must contain complete JSONL records"
        raise ValueError(msg)
    if not lines[-1].endswith(b"\n"):
        incomplete_tail = lines.pop()
        try:
            json.loads(incomplete_tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        else:
            msg = "Session final JSONL record must end with a newline"
            raise ValueError(msg)
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
    return tuple(records), b"".join(lines)


def _parse_metadata(record: dict[str, object]) -> SessionMetadata:
    schema_version = record.get("schema_version")
    if (
        record.get("record_type") != "metadata"
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        msg = "Session metadata record type or schema version is not supported"
        raise ValueError(msg)
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
    if record.get("record_type") != "message":
        msg = "Session message record type is not supported"
        raise ValueError(msg)
    role = record.get("role")
    if role == "user":
        return UserSessionMessage(
            id=_string(record, "id"),
            created_at=_datetime(record, "created_at"),
            content=_string(record, "content"),
        )
    if role == "assistant":
        status_value = record.get("status")
        if status_value not in {"completed", "interrupted", "error"}:
            msg = "Assistant status is not supported"
            raise ValueError(msg)
        status = cast(AssistantMessageStatus, status_value)
        usage = _object(record, "usage")
        return AssistantSessionMessage(
            id=_string(record, "id"),
            created_at=_datetime(record, "created_at"),
            content=_string(record, "content"),
            tool_calls=_model_tool_calls(record),
            status=status,
            error=_session_error(record),
            usage=ModelUsage(
                input_tokens=_integer(usage, "input_tokens"),
                output_tokens=_integer(usage, "output_tokens"),
                total_tokens=_integer(usage, "total_tokens"),
            ),
        )
    if role == "tool":
        if set(record) != _TOOL_MESSAGE_FIELDS:
            msg = "Tool Session message fields do not match the current schema"
            raise ValueError(msg)
        status_value = record.get("status")
        if status_value not in {"success", "error", "refused"}:
            msg = "Tool status is not supported"
            raise ValueError(msg)
        artifact_record = record.get("artifact")
        artifact = None
        if artifact_record is not None:
            if not isinstance(artifact_record, dict):
                msg = "Session field 'artifact' must be an object or null"
                raise ValueError(msg)
            artifact_object = cast(dict[str, object], artifact_record)
            artifact = ArtifactReference(
                path=_string(artifact_object, "path"),
                total_chars=_integer(artifact_object, "total_chars"),
                preview_chars=_integer(artifact_object, "preview_chars"),
            )
        return ToolSessionMessage(
            id=_string(record, "id"),
            created_at=_datetime(record, "created_at"),
            tool_call_id=_string(record, "tool_call_id"),
            name=_string(record, "name"),
            content=_string(record, "content"),
            status=cast(ToolResultStatus, status_value),
            artifact=artifact,
        )
    msg = f"Unsupported Session message role: {role!r}"
    raise ValueError(msg)


def _model_tool_calls(record: dict[str, object]) -> tuple[ModelToolCall, ...]:
    value = record.get("tool_calls")
    if not isinstance(value, list):
        msg = "Session field 'tool_calls' must be an array"
        raise ValueError(msg)
    calls: list[ModelToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            msg = "Session tool calls must be objects"
            raise ValueError(msg)
        tool_call = cast(dict[str, object], item)
        calls.append(
            ModelToolCall(
                id=_string(tool_call, "id"),
                name=_string(tool_call, "name"),
                arguments=_string(tool_call, "arguments"),
            )
        )
    return tuple(calls)


def _session_error(record: dict[str, object]) -> SessionError | None:
    error_record = record.get("error")
    if error_record is None:
        return None
    if not isinstance(error_record, dict):
        msg = "Session field 'error' must be an object or null"
        raise ValueError(msg)
    error_object = cast(dict[str, object], error_record)
    code_value = _string(error_object, "code")
    if code_value not in STABLE_ERROR_CODES:
        msg = "Session error code must be stable"
        raise ValueError(msg)
    return SessionError(
        code=cast(ErrorCode, code_value),
        message=_string(error_object, "message"),
    )


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
