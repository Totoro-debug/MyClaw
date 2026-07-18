"""Creation and persistence of user-approved Scheduled Work."""

import asyncio
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from myclaw.config.agent_home import AgentHome
from myclaw.contracts import JsonObject, ScheduledWork, ToolDefinition, ToolExecutionContext
from myclaw.contracts.common import make_session_id
from myclaw.contracts.scheduling import serialize_scheduled_work
from myclaw.utils.atomic_files import atomic_replace_text

_RECORD_FIELDS = frozenset({"id", "title", "cron", "prompt", "created_at", "enabled", "session_id"})


class ScheduledWorkInvalidError(ValueError):
    """Raised when create_scheduled_work receives invalid task fields."""


class ScheduledWorkPersistenceError(RuntimeError):
    """Raised when Scheduled Work persisted state cannot be read or replaced."""


class JsonScheduledWorkStore:
    """Atomically persist the complete Scheduled Work JSON array."""

    def __init__(
        self,
        agent_home: AgentHome,
        *,
        replace_text: Callable[[Path, str], None] = atomic_replace_text,
    ) -> None:
        self._agent_home = agent_home
        self._replace_text = replace_text
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._agent_home.path / "scheduled-work.json"

    async def load(self) -> tuple[ScheduledWork, ...]:
        async with self._lock:
            try:
                return self._load_unlocked()
            except Exception as error:
                raise ScheduledWorkPersistenceError(
                    "Scheduled Work state could not be read"
                ) from error

    async def append(self, record: ScheduledWork) -> None:
        async with self._lock:
            try:
                records = (*self._load_unlocked(), record)
                self._agent_home.initialize()
                self._replace_text(self.path, serialize_scheduled_work(records))
            except Exception as error:
                raise ScheduledWorkPersistenceError(
                    "Scheduled Work state could not be replaced"
                ) from error

    def _load_unlocked(self) -> tuple[ScheduledWork, ...]:
        if not self.path.exists():
            return ()
        document: object = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(document, list):
            raise ValueError("Scheduled Work file must contain a JSON array")
        return tuple(_parse_record(value) for value in document)


class CreateScheduledWorkTool:
    """Create one enabled Scheduled Work record after Gateway approval."""

    _definition = ToolDefinition(
        name="create_scheduled_work",
        description="Create recurring work with a five-field cron schedule.",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 120},
                "cron": {"type": "string", "minLength": 1},
                "prompt": {"type": "string", "minLength": 1, "maxLength": 20000},
            },
            "required": ["title", "cron", "prompt"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        *,
        store: JsonScheduledWorkStore,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        self._store = store
        self._now = now
        self._new_uuid = new_uuid

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str:
        created_at = self._now()
        try:
            record = ScheduledWork(
                id=str(self._new_uuid()),
                title=_required_string(arguments, "title"),
                cron=_required_string(arguments, "cron"),
                prompt=_required_string(arguments, "prompt"),
                created_at=created_at,
                enabled=True,
                session_id=make_session_id(created_at, self._new_uuid()),
            )
        except ValueError as error:
            raise ScheduledWorkInvalidError("invalid Scheduled Work fields") from error
        await self._store.append(record)
        return f"Created Scheduled Work {record.id}."


def _parse_record(value: object) -> ScheduledWork:
    if not isinstance(value, dict):
        raise ValueError("Scheduled Work records must be JSON objects")
    record = cast(dict[str, object], value)
    if set(record) != _RECORD_FIELDS:
        raise ValueError("Scheduled Work record fields do not match the persisted schema")
    created_at_value = _required_string(record, "created_at")
    enabled = record["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    parsed = ScheduledWork(
        id=_required_string(record, "id"),
        title=_required_string(record, "title"),
        cron=_required_string(record, "cron"),
        prompt=_required_string(record, "prompt"),
        created_at=datetime.fromisoformat(created_at_value),
        enabled=enabled,
        session_id=_required_string(record, "session_id"),
    )
    if parsed.to_dict() != record:
        raise ValueError("Scheduled Work record must use canonical persisted values")
    return parsed


def _required_string(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ScheduledWorkInvalidError(f"{name} must be a non-empty string")
    return value
