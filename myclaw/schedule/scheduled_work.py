"""Creation and persistence of user-approved Scheduled Work."""

import asyncio
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, cast
from uuid import UUID

from myclaw.config.agent_home import AgentHome
from myclaw.schedule.records import ScheduledWork, serialize_scheduled_work
from myclaw.session.identifiers import make_session_id
from myclaw.tools.base import BaseTool
from myclaw.tools.schema import ToolParam
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


class CreateScheduledWorkTool(BaseTool):
    """Declare and directly persist one enabled Scheduled Work record."""

    name = "create_scheduled_work"
    description = "Create recurring work with a five-field cron schedule."
    required = ("title", "cron", "prompt")

    title: Annotated[
        str,
        ToolParam(description="Short task title.", min_length=1, max_length=120),
    ]
    cron: Annotated[
        str,
        ToolParam(description="Five-field cron schedule.", min_length=1),
    ]
    prompt: Annotated[
        str,
        ToolParam(description="Task prompt.", min_length=1, max_length=20000),
    ]

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

    def refusal_reason(self, *, title: str, cron: str, prompt: str) -> str:
        del title, cron, prompt
        return "Scheduled Work creation is unavailable because confirmation is not implemented."

    async def execute(self, *, title: str, cron: str, prompt: str) -> str:
        created_at = self._now()
        try:
            record = ScheduledWork(
                id=str(self._new_uuid()),
                title=title,
                cron=cron,
                prompt=prompt,
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
