"""Stateful Conversation Session public interface."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, cast
from uuid import UUID, uuid4

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.tools.base import ArtifactReference
from myclaw.utils.async_tasks import await_task_preserving_cancellation
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.time import format_rfc3339_milliseconds, local_now
from myclaw.utils.validation import (
    require_aware_datetime,
    require_nonnegative_int,
    require_uuid4,
    require_uuid4_string,
    token_usage_validation_issue,
)

__all__ = ["Session", "SessionStoragePartition"]


class SessionStoragePartition(StrEnum):
    """The Workspace-owned storage partition used by one Conversation Session."""

    FOREGROUND = "foreground"
    SCHEDULE = "schedule"


_HEADER_FIELDS = frozenset(
    {"session_id", "created_at", "updated_at", "last_consolidated", "metadata"}
)
_TOKEN_USAGE_PATCH_KEYS = frozenset({"token_usage", "token_usage_delta", "usage_delta"})
_SESSION_ID_PATTERN = re.compile(
    r"(?P<timestamp>\d{8}-\d{6}-\d{6})_"
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_SCHEDULE_SESSION_ID_PATTERN = re.compile(
    r"schedule_(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})"
)
_TITLE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
    ("\u300c", "\u300d"),
    ("\u300e", "\u300f"),
    ("\u00ab", "\u00bb"),
)


class Session:
    """Own the in-memory state and identity of one Conversation Session."""

    _workspace_state: WorkspaceState
    _session_id: str
    _storage_partition: SessionStoragePartition
    _created_at: datetime
    _updated_at: datetime
    _now: Callable[[], datetime] | None
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    last_consolidated: int
    _pending_persist: asyncio.Task[None] | None
    _persist_tasks: set[asyncio.Task[None]]
    _closed: bool
    _abandoned: bool

    def __init__(self) -> None:
        raise TypeError("Use Session.create() or Session.load()")

    @classmethod
    def _from_state(
        cls,
        *,
        workspace_state: WorkspaceState,
        session_id: str,
        created_at: datetime,
        updated_at: datetime,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
        last_consolidated: int,
        partition: SessionStoragePartition | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> Self:
        resolved_partition = _resolve_partition(session_id, partition)
        require_aware_datetime(created_at, field="created_at")
        require_aware_datetime(updated_at, field="updated_at")
        require_nonnegative_int(last_consolidated, field="last_consolidated")
        session = object.__new__(cls)
        session._workspace_state = workspace_state
        session._session_id = session_id
        session._storage_partition = resolved_partition
        session._created_at = created_at
        session._updated_at = updated_at
        session._now = now
        session.messages = messages
        session.metadata = metadata
        session.last_consolidated = last_consolidated
        session._pending_persist = None
        session._persist_tasks = set()
        session._closed = False
        session._abandoned = False
        return session

    @classmethod
    def create(
        cls,
        workspace_state: WorkspaceState,
        *,
        now: Callable[[], datetime] | None = None,
        new_uuid: Callable[[], UUID] | None = None,
        partition: SessionStoragePartition = SessionStoragePartition.FOREGROUND,
        job_id: UUID | str | None = None,
    ) -> Self:
        """Create a memory-only Session in the requested storage partition."""
        created_at = _clock_now(now)
        resolved_partition = _coerce_partition(partition)
        if resolved_partition is SessionStoragePartition.SCHEDULE:
            if job_id is None:
                raise ValueError("Schedule Session requires a canonical UUID4 job_id")
            session_id = cls.schedule_session_id(job_id)
        else:
            if job_id is not None:
                raise ValueError("job_id is only valid for Schedule Sessions")
            allocate_uuid = uuid4 if new_uuid is None else new_uuid
            session_id = _make_id(created_at, allocate_uuid())
        return cls._from_state(
            workspace_state=workspace_state,
            session_id=session_id,
            created_at=created_at,
            updated_at=created_at,
            messages=[],
            metadata=_initial_metadata(),
            last_consolidated=0,
            partition=resolved_partition,
            now=now,
        )

    @classmethod
    def create_schedule(
        cls,
        workspace_state: WorkspaceState,
        job_id: UUID | str,
        *,
        now: Callable[[], datetime] | None = None,
        title: str = "Untitled session",
    ) -> Self:
        """Create a memory-only Schedule Session derived from one Job UUID4."""
        session = cls.create(
            workspace_state,
            now=now,
            partition=SessionStoragePartition.SCHEDULE,
            job_id=job_id,
        )
        session.update_metadata(title=title)
        return session

    @classmethod
    def load(
        cls,
        workspace_state: WorkspaceState,
        session_id: str,
        *,
        partition: SessionStoragePartition | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> Self:
        """Load one current-format Session synchronously from Workspace State."""
        resolved_partition = _resolve_partition(session_id, partition)
        sessions_directory = _existing_sessions_directory(workspace_state, resolved_partition)
        if sessions_directory is None:
            raise FileNotFoundError(_storage_directory(workspace_state, resolved_partition))
        path = sessions_directory / f"{session_id}.jsonl"
        owned_path = HOST_FILESYSTEM.require_owned_regular_file(
            path,
            within=sessions_directory,
        )
        records = _read_jsonl_records(owned_path)
        if not records:
            raise ValueError("Session must contain a header record")
        try:
            header = records[0]
            loaded_id, created_at, updated_at, last_consolidated, metadata = _parse_header(header)
            if loaded_id != session_id:
                raise ValueError("Session metadata ID does not match its file name")
            messages = [_parse_message(record) for record in records[1:]]
        except TypeError as error:
            raise ValueError("Session JSONL contains malformed persisted data") from error
        return cls._from_state(
            workspace_state=workspace_state,
            session_id=loaded_id,
            created_at=created_at,
            updated_at=updated_at,
            messages=messages,
            metadata=metadata,
            last_consolidated=last_consolidated,
            partition=resolved_partition,
            now=now,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def workspace_state(self) -> WorkspaceState:
        return self._workspace_state

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    def add_message(self, role: str, content: str, **fields: Any) -> None:
        """Append one validated JSON-native user, assistant, or tool message."""
        self._ensure_not_abandoned()
        if role not in {"user", "assistant", "tool"}:
            raise ValueError("role must be user, assistant, or tool")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        reserved = {"role", "content", "timestamp"}
        if reserved.intersection(fields):
            raise ValueError("role, content, and timestamp are reserved message fields")
        if "id" in fields:
            raise ValueError("unsupported Session message identifiers")
        copied_fields = _copy_json_object(fields, field="message")
        message: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": format_rfc3339_milliseconds(self._clock_now()),
            **copied_fields,
        }
        _validate_message(message)
        updated_usage = self._usage_after_assistant(message)
        self.messages.append(message)
        if updated_usage is not None:
            self.metadata["token_usage"] = updated_usage

    def append_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        metadata_updates: dict[str, Any] | None = None,
        metadata_removals: tuple[str, ...] = (),
        usage_delta: dict[str, int] | None = None,
    ) -> None:
        """Atomically append a validated Agent Run increment."""
        self._ensure_not_abandoned()
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")

        copied_updates = _copy_metadata_updates(metadata_updates)
        removals = _validate_metadata_removals(metadata_removals)
        conflict = set(copied_updates).intersection(removals)
        if conflict:
            raise ValueError("metadata updates and removals cannot target the same key")
        required_removals = {"title", "token_usage"}.intersection(removals)
        if required_removals:
            raise ValueError("required Session metadata cannot be removed")
        usage_patch = _TOKEN_USAGE_PATCH_KEYS.intersection(copied_updates)
        if usage_patch:
            raise ValueError("token usage must be supplied through usage_delta")

        copied_usage_delta: dict[str, Any] | None = None
        if usage_delta is not None:
            if not isinstance(usage_delta, dict):
                raise TypeError("usage_delta must be a dictionary")
            copied_usage_delta = _copy_json_object(usage_delta, field="usage_delta")
            _validate_token_usage(copied_usage_delta, field="usage_delta")

        candidate_metadata = copy.deepcopy(self.metadata)
        _validate_metadata(candidate_metadata)
        candidate_metadata.update(copied_updates)
        for key in removals:
            candidate_metadata.pop(key, None)
        _validate_metadata(candidate_metadata)

        updated_usage = copy.deepcopy(candidate_metadata["token_usage"])
        if copied_usage_delta is not None:
            updated_usage = _accumulate_token_usage(updated_usage, copied_usage_delta)

        prepared: list[dict[str, Any]] = []
        for index, record in enumerate(messages):
            if not isinstance(record, dict):
                raise TypeError(f"messages[{index}] must be a dictionary")
            copied = _copy_json_object(record, field="message")
            if "timestamp" in copied:
                raise ValueError("timestamp is reserved for Session message timestamps")
            copied["timestamp"] = format_rfc3339_milliseconds(self._clock_now())
            try:
                _validate_message(copied)
            except KeyError as error:
                raise ValueError(f"Session message is missing {error.args[0]}") from error
            prepared.append(copied)

            if copied["role"] == "assistant":
                updated_usage = _accumulate_token_usage(
                    updated_usage,
                    copied["token_usage"],
                )

        candidate_metadata["token_usage"] = updated_usage
        _validate_metadata(candidate_metadata)
        metadata_changed = candidate_metadata != self.metadata

        self.messages.extend(prepared)
        if metadata_changed:
            self.metadata.clear()
            self.metadata.update(candidate_metadata)

    def update_metadata(self, metadata: dict[str, Any] | None = None, **updates: Any) -> None:
        """Apply a copied shallow metadata patch and accumulate token usage deltas."""
        self._ensure_not_abandoned()
        patch: dict[str, Any] = {}
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("metadata patch must be a dictionary")
            patch.update(metadata)
        patch.update(updates)
        copied_patch = _copy_json_object(patch, field="metadata")
        _normalize_blackboard_metadata(copied_patch, invalid_is_absent=False)

        token_delta = copied_patch.pop("token_usage_delta", None)
        if token_delta is None and "usage_delta" in copied_patch:
            token_delta = copied_patch.pop("usage_delta")
        if "token_usage" in copied_patch:
            if token_delta is not None:
                raise ValueError("token usage delta was provided more than once")
            token_delta = copied_patch.pop("token_usage")
        if token_delta is not None:
            _validate_token_usage(token_delta, field="token_usage_delta")

        if "title" in copied_patch:
            title = copied_patch["title"]
            if not isinstance(title, str):
                raise TypeError("title must be a string")
            copied_patch["title"] = _normalize_title(title)

        updated_usage = self._usage_after_delta(token_delta)
        self.metadata.update(copied_patch)
        if updated_usage is not None:
            self.metadata["token_usage"] = updated_usage

    def persist(self) -> None:
        """Schedule a silent, ordered write of the current complete Session snapshot."""
        if self._closed:
            return
        self._updated_at = self._clock_now()
        if not self.messages:
            return
        try:
            content = self._serialized_state()
            loop = asyncio.get_running_loop()
            previous = self._pending_persist
            pending = loop.create_task(self._persist_after(previous, content))
            self._pending_persist = pending
            self._persist_tasks.add(pending)
            pending.add_done_callback(self._persist_task_finished)
        except Exception:
            return

    async def wait_for_pending_persist(self) -> None:
        """Wait for every already-scheduled ordered snapshot without starting a new save."""
        drain = asyncio.create_task(self._drain_pending_persist())
        await await_task_preserving_cancellation(drain)

    async def _drain_pending_persist(self) -> None:
        while True:
            done_tasks = tuple(task for task in self._persist_tasks if task.done())
            for task in done_tasks:
                self._consume_persist_task(task)
            pending_tasks = tuple(self._persist_tasks)
            if not pending_tasks:
                return
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    def _persist_task_finished(self, task: asyncio.Task[None]) -> None:
        self._consume_persist_task(task)

    def _consume_persist_task(self, task: asyncio.Task[None]) -> None:
        self._persist_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except BaseException:
            return

    def close(self) -> None:
        """Synchronously make a bounded best-effort final save and close the Session."""
        if self._closed:
            return
        self._closed = True
        if not self.messages:
            return

        for attempt in range(3):
            try:
                self._updated_at = self._clock_now()
                self._write_content(self._serialized_state())
                return
            except Exception:
                if attempt < 2:
                    time.sleep((0.1, 0.2)[attempt])

    def abandon(self) -> None:
        """Synchronously abandon the Session without a final persistence attempt."""
        if self._abandoned:
            return
        self._abandoned = True
        self._closed = True
        self._pending_persist = None
        pending_tasks = tuple(self._persist_tasks)
        for pending in pending_tasks:
            if not pending.done():
                pending.cancel()

    def _serialized_state(self) -> bytes:
        header = {
            "session_id": self._session_id,
            "created_at": format_rfc3339_milliseconds(self._created_at),
            "updated_at": format_rfc3339_milliseconds(self._updated_at),
            "last_consolidated": self.last_consolidated,
            "metadata": copy.deepcopy(self.metadata),
        }
        records = (header, *copy.deepcopy(self.messages))
        return "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            for record in records
        ).encode("utf-8")

    async def _persist_after(
        self,
        previous: asyncio.Task[None] | None,
        content: bytes,
    ) -> None:
        if previous is not None and not previous.done():
            try:
                await previous
            except Exception:
                pass
        for attempt in range(3):
            if self._closed:
                return
            try:
                self._write_content(content)
                return
            except Exception:
                if attempt == 2 or self._closed:
                    return
                await asyncio.sleep((0.1, 0.2)[attempt])

    def _ensure_not_abandoned(self) -> None:
        if self._abandoned:
            raise RuntimeError("Session has been abandoned")

    def _write_content(self, content: bytes) -> None:
        if self._storage_partition is SessionStoragePartition.FOREGROUND:
            sessions_directory = self._workspace_state.prepare_sessions_directory()
        else:
            sessions_directory = self._workspace_state.prepare_schedule_sessions_directory()
        path = sessions_directory / f"{self._session_id}.jsonl"
        io_path = HOST_FILESYSTEM.path_for_io(path)
        try:
            io_path.lstat()
        except FileNotFoundError:
            pass
        else:
            HOST_FILESYSTEM.require_owned_regular_file(
                io_path,
                within=sessions_directory,
            )
        HOST_FILESYSTEM.atomic_replace_bytes(path, content)
        HOST_FILESYSTEM.require_owned_regular_file(path, within=sessions_directory)

    def _usage_after_assistant(self, message: dict[str, Any]) -> dict[str, int] | None:
        if message["role"] != "assistant" or "token_usage" not in message:
            return None
        return self._usage_after_delta(message["token_usage"])

    def _usage_after_delta(self, delta: Any) -> dict[str, int] | None:
        if delta is None:
            return None
        return _accumulate_token_usage(self.metadata.get("token_usage"), delta)

    def _clock_now(self) -> datetime:
        return _clock_now(self._now)

    @classmethod
    def schedule_session_id(cls, job_id: UUID | str) -> str:
        """Return the canonical Schedule Session ID for one Job UUID4."""
        if isinstance(job_id, UUID):
            require_uuid4(job_id, field="job_id")
            canonical = str(job_id)
        elif isinstance(job_id, str):
            require_uuid4_string(job_id, field="job_id")
            canonical = job_id
        else:
            raise ValueError("job_id must be a canonical UUID4")
        return f"schedule_{canonical}"

    @classmethod
    def _require_id(
        cls,
        value: str,
        *,
        field: str = "session_id",
        partition: SessionStoragePartition | None = None,
    ) -> None:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a valid Session ID")
        resolved_partition = _coerce_partition(partition) if partition is not None else None
        if resolved_partition is SessionStoragePartition.FOREGROUND or (
            resolved_partition is None and _SCHEDULE_SESSION_ID_PATTERN.fullmatch(value) is None
        ):
            match = _SESSION_ID_PATTERN.fullmatch(value)
            if match is None:
                raise ValueError(f"{field} must be a valid Session ID")
            try:
                datetime.strptime(match.group("timestamp"), "%Y%m%d-%H%M%S-%f")
                require_uuid4_string(match.group("uuid"), field=field)
            except ValueError as error:
                raise ValueError(f"{field} must be a valid Session ID") from error
            return
        match = _SCHEDULE_SESSION_ID_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(f"{field} must be a valid Schedule Session ID")
        try:
            require_uuid4_string(match.group("uuid"), field=field)
        except ValueError as error:
            raise ValueError(f"{field} must be a valid Schedule Session ID") from error

    @staticmethod
    def _normalize_title(value: str) -> str:
        return _normalize_title(value)

    @staticmethod
    def _normalize_title_candidate(value: str) -> str:
        return _normalize_title(value, fallback="")


def _coerce_partition(value: SessionStoragePartition | str) -> SessionStoragePartition:
    if isinstance(value, SessionStoragePartition):
        return value
    try:
        return SessionStoragePartition(value)
    except (TypeError, ValueError) as error:
        raise ValueError("unknown Session storage partition") from error


def _resolve_partition(
    session_id: str,
    partition: SessionStoragePartition | str | None,
) -> SessionStoragePartition:
    if not isinstance(session_id, str):
        raise ValueError("session_id must be a valid Session ID")
    if partition is None:
        resolved = (
            SessionStoragePartition.SCHEDULE
            if _SCHEDULE_SESSION_ID_PATTERN.fullmatch(session_id) is not None
            else SessionStoragePartition.FOREGROUND
        )
    else:
        resolved = _coerce_partition(partition)
    Session._require_id(session_id, partition=resolved)
    return resolved


def _storage_directory(
    workspace_state: WorkspaceState,
    partition: SessionStoragePartition,
) -> Path:
    if partition is SessionStoragePartition.FOREGROUND:
        return workspace_state.sessions_directory
    return workspace_state.schedule_sessions_directory


def _existing_sessions_directory(
    workspace_state: WorkspaceState,
    partition: SessionStoragePartition,
) -> Path | None:
    if partition is SessionStoragePartition.FOREGROUND:
        return workspace_state.existing_sessions_directory()
    return workspace_state.existing_schedule_sessions_directory()


def _clock_now(now: Callable[[], datetime] | None) -> datetime:
    return local_now() if now is None else now()


def _make_id(created_at: datetime, session_uuid: UUID) -> str:
    require_aware_datetime(created_at, field="created_at")
    require_uuid4(session_uuid, field="session_uuid")
    return f"{created_at:%Y%m%d-%H%M%S-%f}_{session_uuid}"


def _initial_metadata() -> dict[str, Any]:
    return {
        "title": "Untitled session",
        "token_usage": {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    try:
        content = HOST_FILESYSTEM.path_for_io(path).read_bytes()
    except OSError:
        raise
    if not content.endswith(b"\n"):
        raise ValueError("Session JSONL must end with a newline")
    lines = content.splitlines(keepends=False)
    if not lines or any(not line for line in lines):
        raise ValueError("Session JSONL must contain complete records")
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            decoded: Any = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Session JSONL contains invalid JSON") from error
        if not isinstance(decoded, dict):
            raise ValueError("Session JSONL records must be objects")
        records.append(cast(dict[str, Any], decoded))
    return records


def _parse_header(
    record: dict[str, Any],
) -> tuple[str, datetime, datetime, int, dict[str, Any]]:
    if set(record) != _HEADER_FIELDS:
        raise ValueError("Session header fields do not match the current format")
    session_id = record["session_id"]
    if not isinstance(session_id, str):
        raise ValueError("Session header session_id must be a string")
    Session._require_id(session_id)
    created_at = _parse_datetime(record["created_at"], field="created_at")
    updated_at = _parse_datetime(record["updated_at"], field="updated_at")
    last_consolidated = record["last_consolidated"]
    require_nonnegative_int(last_consolidated, field="last_consolidated")
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Session metadata must be an object")
    metadata_copy = _copy_loaded_metadata(cast(dict[str, Any], metadata))
    _validate_metadata(metadata_copy)
    return session_id, created_at, updated_at, last_consolidated, metadata_copy


def _parse_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Session field '{field}' must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Session field '{field}' must be ISO 8601") from error
    require_aware_datetime(parsed, field=field)
    return parsed


def _parse_message(record: dict[str, Any]) -> dict[str, Any]:
    message = _copy_json_object(record, field="message")
    if any(key in message for key in ("record_" + "type", "schema_" + "version")):
        raise ValueError("legacy Session message fields are unsupported")
    try:
        _validate_message(message)
    except KeyError as error:
        raise ValueError(f"Session message is missing {error.args[0]}") from error
    return message


def _validate_metadata(metadata: dict[str, Any]) -> None:
    _validate_json_value(metadata, field="metadata")
    title = metadata.get("title")
    if not isinstance(title, str):
        raise ValueError("metadata.title must be a string")
    if not title or " ".join(title.split()) != title or len(title) > 60:
        raise ValueError("metadata.title is not normalized")
    _validate_token_usage(metadata.get("token_usage"), field="metadata.token_usage")
    _normalize_blackboard_metadata(metadata, invalid_is_absent=False)


def _copy_loaded_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(metadata)
    _normalize_blackboard_metadata(copied, invalid_is_absent=True)
    return _copy_json_object(copied, field="metadata")


def _normalize_blackboard_metadata(
    metadata: dict[str, Any],
    *,
    invalid_is_absent: bool,
) -> None:
    if "blackboard" not in metadata:
        return
    from myclaw.agent.blackboard import Blackboard

    blackboard = Blackboard.from_dict(metadata["blackboard"])
    if blackboard is None:
        if invalid_is_absent:
            del metadata["blackboard"]
            return
        raise ValueError("metadata.blackboard must be a valid Blackboard")
    metadata["blackboard"] = blackboard.to_dict()


def _copy_metadata_updates(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata_updates must be a dictionary")
    return _copy_json_object(value, field="metadata_updates")


def _validate_metadata_removals(value: tuple[str, ...]) -> frozenset[str]:
    if not isinstance(value, tuple):
        raise TypeError("metadata_removals must be a tuple")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("metadata_removals must contain only strings")
    return frozenset(value)


def _copy_json_object(value: dict[str, Any], *, field: str) -> dict[str, Any]:
    copied = copy.deepcopy(value)
    _validate_json_value(copied, field=field)
    return copied


def _validate_json_value(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"{field} must contain only JSON-compatible values")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, field=f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} must contain only JSON-compatible values")
            _validate_json_value(item, field=f"{field}.{key}")
        return
    raise TypeError(f"{field} must contain only JSON-compatible values")


def _validate_message(message: dict[str, Any]) -> None:
    _validate_json_value(message, field="message")
    if "id" in message:
        raise ValueError("unsupported Session message identifiers")
    role = message["role"]
    if not isinstance(role, str):
        raise TypeError("message role must be a string")
    if not isinstance(message["content"], str):
        raise TypeError("message content must be a string")
    if not isinstance(message["timestamp"], str):
        raise TypeError("message timestamp must be a string")
    try:
        timestamp = datetime.fromisoformat(message["timestamp"])
    except ValueError as error:
        raise ValueError("message timestamp must be ISO 8601") from error
    require_aware_datetime(timestamp, field="message timestamp")
    if role == "user":
        if not message["content"].strip():
            raise ValueError("user message content must not be blank")
        return
    if role == "assistant":
        _validate_assistant_message(message)
        return
    if role == "tool":
        _validate_tool_message(message)
        return
    raise ValueError("role must be user, assistant, or tool")


def _validate_assistant_message(message: dict[str, Any]) -> None:
    required = {"tool_calls", "status", "error", "token_usage"}
    missing = required.difference(message)
    if missing:
        raise ValueError(f"assistant message is missing {', '.join(sorted(missing))}")
    status = message["status"]
    if status not in {"completed", "interrupted", "error"}:
        raise ValueError("assistant status is not supported")
    error = message["error"]
    if status == "completed" and error is not None:
        raise ValueError("completed assistant must not have an error")
    if status != "completed" and error is None:
        raise ValueError("non-completed assistant requires an error")
    tool_calls = message["tool_calls"]
    if not isinstance(tool_calls, list):
        raise TypeError("assistant tool_calls must be a list")
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise TypeError("assistant tool_calls must contain dictionaries")
        for field in ("id", "name", "arguments"):
            if not isinstance(tool_call.get(field), str):
                raise ValueError(f"assistant tool_calls require {field}")
    if error is not None:
        if not isinstance(error, dict) or not isinstance(error.get("code"), str):
            raise ValueError("assistant error must contain a code")
        if not isinstance(error.get("message"), str):
            raise ValueError("assistant error must contain a message")
    token_usage = message["token_usage"]
    _validate_token_usage(token_usage, field="assistant.token_usage")
    if token_usage["model_calls"] != 1 and not (
        status == "error"
        and token_usage["model_calls"] == 0
        and error is not None
        and error["code"] == "agent_iteration_limit"
    ):
        raise ValueError("assistant.token_usage.model_calls must equal 1")
    if status != "error":
        if not message["content"] and not tool_calls:
            raise ValueError("assistant requires content or tool_calls unless status is error")


def _validate_tool_message(message: dict[str, Any]) -> None:
    for field in ("tool_call_id", "name", "status"):
        if not isinstance(message.get(field), str):
            raise ValueError(f"tool message requires {field}")
    if message["status"] not in {"success", "error", "refused"}:
        raise ValueError("tool status is not supported")
    artifact = message.get("artifact")
    if artifact is not None:
        if message["status"] != "success":
            raise ValueError("only successful tool messages may contain an artifact")
        if not isinstance(artifact, dict):
            raise ValueError("tool artifact must be an object or null")
        if set(artifact) != {"path", "total_chars", "preview_chars"}:
            raise ValueError("tool artifact has an invalid shape")
        try:
            ArtifactReference(
                path=artifact["path"],
                total_chars=artifact["total_chars"],
                preview_chars=artifact["preview_chars"],
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("tool artifact is malformed") from error


def _validate_token_usage(value: Any, *, field: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a dictionary")
    issue = token_usage_validation_issue(value)
    if issue == "fields":
        raise ValueError(f"{field} must contain exactly model_calls and token counters")
    if issue == "values":
        for key, member in value.items():
            require_nonnegative_int(member, field=f"{field}.{key}")
        raise AssertionError("token usage value issue did not identify an invalid field")
    if issue == "total":
        raise ValueError(f"{field}.total_tokens must equal input_tokens + output_tokens")


def _accumulate_token_usage(
    current: Any,
    delta: Any,
) -> dict[str, int]:
    _validate_token_usage(current, field="metadata.token_usage")
    _validate_token_usage(delta, field="token_usage_delta")
    assert isinstance(current, dict)
    assert isinstance(delta, dict)
    return {
        key: current[key] + delta[key]
        for key in ("model_calls", "input_tokens", "output_tokens", "total_tokens")
    }


def _normalize_title(value: str, *, fallback: str = "Untitled session") -> str:
    for line in value.splitlines():
        title = " ".join(line.split())
        if not title:
            continue
        for opening, closing in _TITLE_PAIRS:
            if len(title) >= 2 and title.startswith(opening) and title.endswith(closing):
                title = " ".join(title[1:-1].split())
                break
        return title[:60] or fallback
    return fallback
