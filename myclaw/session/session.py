"""Stateful Conversation Session public interface."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self, cast
from uuid import UUID, uuid4

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.session.identifiers import make_session_id as _make_session_id
from myclaw.session.identifiers import require_session_id as _require_session_id
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.utils.validation import require_aware_datetime, require_nonnegative_int

__all__ = ["Session"]

_HEADER_FIELDS = frozenset(
    {"session_id", "created_at", "updated_at", "last_consolidated", "metadata"}
)
_UNSUPPORTED_MESSAGE_FIELDS = frozenset({"id", "record_type", "schema_version"})


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """A complete immutable-in-practice copy of one persistence attempt."""

    session_id: str
    created_at: datetime
    updated_at: datetime
    last_consolidated: int
    metadata: dict[str, Any]
    messages: list[dict[str, Any]]


class Session:
    """Own the in-memory state and identity of one Conversation Session."""

    _workspace_state: WorkspaceState
    _session_id: str
    _created_at: datetime
    _updated_at: datetime
    _now: Callable[[], datetime] | None
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    last_consolidated: int
    _pending_persist: asyncio.Task[None] | None
    _closed: bool

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
        now: Callable[[], datetime] | None = None,
    ) -> Self:
        _require_session_id(session_id)
        require_aware_datetime(created_at, field="created_at")
        require_aware_datetime(updated_at, field="updated_at")
        require_nonnegative_int(last_consolidated, field="last_consolidated")
        session = object.__new__(cls)
        session._workspace_state = workspace_state
        session._session_id = session_id
        session._created_at = created_at
        session._updated_at = updated_at
        session._now = now
        session.messages = messages
        session.metadata = metadata
        session.last_consolidated = last_consolidated
        session._pending_persist = None
        session._closed = False
        return session

    @classmethod
    def create(
        cls,
        workspace_state: WorkspaceState,
        *,
        now: Callable[[], datetime] | None = None,
        new_uuid: Callable[[], UUID] | None = None,
    ) -> Self:
        """Create a memory-only Session with a new local timestamp-plus-UUID4 ID."""
        created_at = _clock_now(now)
        allocate_uuid = uuid4 if new_uuid is None else new_uuid
        return cls._from_state(
            workspace_state=workspace_state,
            session_id=_make_session_id(created_at, allocate_uuid()),
            created_at=created_at,
            updated_at=created_at,
            messages=[],
            metadata=_initial_metadata(),
            last_consolidated=0,
            now=now,
        )

    @classmethod
    def _create_with_id(
        cls,
        workspace_state: WorkspaceState,
        session_id: str,
        created_at: datetime,
        *,
        title: str,
        now: Callable[[], datetime] | None = None,
    ) -> Self:
        """Create a memory-only Session for an owner with an existing identity."""
        _require_session_id(session_id)
        require_aware_datetime(created_at, field="created_at")
        metadata = _initial_metadata()
        metadata["title"] = title
        _validate_metadata(metadata)
        return cls._from_state(
            workspace_state=workspace_state,
            session_id=session_id,
            created_at=created_at,
            updated_at=created_at,
            messages=[],
            metadata=metadata,
            last_consolidated=0,
            now=now,
        )

    @classmethod
    def load(
        cls,
        workspace_state: WorkspaceState,
        session_id: str,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> Self:
        """Load one current-format Session synchronously from Workspace State."""
        _require_session_id(session_id)
        path = workspace_state.sessions_directory / f"{session_id}.jsonl"
        records = _read_jsonl_records(path)
        if not records:
            raise ValueError("Session must contain a header record")
        header = records[0]
        loaded_id, created_at, updated_at, last_consolidated, metadata = _parse_header(header)
        if loaded_id != session_id:
            raise ValueError("Session metadata ID does not match its file name")
        messages = [_parse_message(record) for record in records[1:]]
        return cls._from_state(
            workspace_state=workspace_state,
            session_id=loaded_id,
            created_at=created_at,
            updated_at=updated_at,
            messages=messages,
            metadata=metadata,
            last_consolidated=last_consolidated,
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
        if role not in {"user", "assistant", "tool"}:
            raise ValueError("role must be user, assistant, or tool")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        reserved = {"role", "content", "timestamp"}
        if reserved.intersection(fields):
            raise ValueError("role, content, and timestamp are reserved message fields")
        if _UNSUPPORTED_MESSAGE_FIELDS.intersection(fields):
            raise ValueError("message id, record_type, and schema_version are unsupported")
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

    def update_metadata(self, metadata: dict[str, Any] | None = None, **updates: Any) -> None:
        """Apply a copied shallow metadata patch and accumulate token usage deltas."""
        patch: dict[str, Any] = {}
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("metadata patch must be a dictionary")
            patch.update(metadata)
        patch.update(updates)
        copied_patch = _copy_json_object(patch, field="metadata")

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
            snapshot = self._snapshot()
            loop = asyncio.get_running_loop()
            previous = self._pending_persist
            self._pending_persist = loop.create_task(self._persist_after(previous, snapshot))
        except Exception:
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
                self._write_snapshot(self._snapshot())
                return
            except Exception:
                if attempt < 2:
                    time.sleep((0.1, 0.2)[attempt])

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(
            session_id=self._session_id,
            created_at=self._created_at,
            updated_at=self._updated_at,
            last_consolidated=self.last_consolidated,
            metadata=copy.deepcopy(self.metadata),
            messages=copy.deepcopy(self.messages),
        )

    async def _persist_after(
        self,
        previous: asyncio.Task[None] | None,
        snapshot: _Snapshot,
    ) -> None:
        if previous is not None and not previous.done():
            try:
                await previous
            except Exception:
                pass
        if self._closed:
            return
        try:
            self._write_snapshot(snapshot)
        except Exception:
            pass

    def _write_snapshot(self, snapshot: _Snapshot) -> None:
        path = self._workspace_state.sessions_directory / f"{snapshot.session_id}.jsonl"
        content = _serialize_snapshot(snapshot)
        HOST_FILESYSTEM.path_for_io(path.parent).mkdir(parents=True, exist_ok=True)
        HOST_FILESYSTEM.atomic_replace_bytes(path, content)

    def _usage_after_assistant(self, message: dict[str, Any]) -> dict[str, int] | None:
        if message["role"] != "assistant" or "token_usage" not in message:
            return None
        return self._usage_after_delta(message["token_usage"])

    def _usage_after_delta(self, delta: Any) -> dict[str, int] | None:
        if delta is None:
            return None
        current = self.metadata.get("token_usage")
        _validate_token_usage(current, field="metadata.token_usage")
        _validate_token_usage(delta, field="token_usage_delta")
        assert isinstance(current, dict)
        assert isinstance(delta, dict)
        return {
            key: current[key] + delta[key]
            for key in ("model_calls", "input_tokens", "output_tokens", "total_tokens")
        }

    def _clock_now(self) -> datetime:
        return _clock_now(self._now)


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _clock_now(now: Callable[[], datetime] | None) -> datetime:
    return _local_now() if now is None else now()


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


def _serialize_snapshot(snapshot: _Snapshot) -> bytes:
    header = {
        "session_id": snapshot.session_id,
        "created_at": format_rfc3339_milliseconds(snapshot.created_at),
        "updated_at": format_rfc3339_milliseconds(snapshot.updated_at),
        "last_consolidated": snapshot.last_consolidated,
        "metadata": snapshot.metadata,
    }
    records = (header, *snapshot.messages)
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")


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
    _require_session_id(session_id)
    created_at = _parse_datetime(record["created_at"], field="created_at")
    updated_at = _parse_datetime(record["updated_at"], field="updated_at")
    last_consolidated = record["last_consolidated"]
    require_nonnegative_int(last_consolidated, field="last_consolidated")
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Session metadata must be an object")
    metadata_copy = _copy_json_object(cast(dict[str, Any], metadata), field="metadata")
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
    if "record_type" in record or "schema_version" in record:
        raise ValueError("Typed or versioned Session messages are unsupported")
    message = _copy_json_object(record, field="message")
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
    if _UNSUPPORTED_MESSAGE_FIELDS.intersection(message):
        raise ValueError("message id, record_type, and schema_version are unsupported")
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
    if token_usage["model_calls"] != 1:
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
    if "artifact" in message and message["artifact"] is not None:
        if not isinstance(message["artifact"], dict):
            raise ValueError("tool artifact must be an object or null")


def _validate_token_usage(value: Any, *, field: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a dictionary")
    expected = {"model_calls", "input_tokens", "output_tokens", "total_tokens"}
    if set(value) != expected:
        raise ValueError(f"{field} must contain exactly model_calls and token counters")
    for key in expected:
        require_nonnegative_int(value[key], field=f"{field}.{key}")
    if value["total_tokens"] != value["input_tokens"] + value["output_tokens"]:
        raise ValueError(f"{field}.total_tokens must equal input_tokens + output_tokens")


def _normalize_title(value: str) -> str:
    pairs = (
        ('"', '"'),
        ("'", "'"),
        ("\u201c", "\u201d"),
        ("\u2018", "\u2019"),
        ("\u300c", "\u300d"),
        ("\u300e", "\u300f"),
        ("\u00ab", "\u00bb"),
    )
    for line in value.splitlines():
        title = " ".join(line.split())
        if not title:
            continue
        for opening, closing in pairs:
            if len(title) >= 2 and title.startswith(opening) and title.endswith(closing):
                title = " ".join(title[1:-1].split())
                break
        return title[:60] or "Untitled session"
    return "Untitled session"
