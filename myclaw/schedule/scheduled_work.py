"""Creation and persistence primitives for Scheduled Work."""

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, cast

from myclaw.config.agent_home import AgentHome
from myclaw.schedule.records import ScheduledWork
from myclaw.tools.base import BaseTool
from myclaw.tools.schema import ToolParam

_RECORD_FIELDS = frozenset({"id", "title", "cron", "prompt", "created_at", "enabled", "session_id"})


class ScheduledWorkPersistenceError(RuntimeError):
    """Raised when Scheduled Work persisted state cannot be read."""


class JsonScheduledWorkStore:
    """Load the complete Scheduled Work JSON array."""

    def __init__(self, agent_home: AgentHome) -> None:
        self._agent_home = agent_home

    @property
    def path(self) -> Path:
        return self._agent_home.path / "scheduled-work.json"

    def load(self) -> tuple[ScheduledWork, ...]:
        try:
            if not self.path.exists():
                return ()
            document: object = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(document, list):
                raise ValueError("Scheduled Work file must contain a JSON array")
            return tuple(_parse_record(value) for value in document)
        except Exception as error:
            raise ScheduledWorkPersistenceError(
                "Scheduled Work state could not be read"
            ) from error


class CreateScheduledWorkTool(BaseTool):
    """Declare unavailable Scheduled Work creation."""

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

    def refusal_reason(self, *, title: str, cron: str, prompt: str) -> str:
        del title, cron, prompt
        return "Scheduled Work creation is unavailable because confirmation is not implemented."

    async def execute(self, *, title: str, cron: str, prompt: str) -> str:
        raise AssertionError("Refusal-only Tool reached execution")


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
        raise ValueError(f"{name} must be a non-empty string")
    return value
