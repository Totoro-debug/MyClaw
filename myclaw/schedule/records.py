"""Scheduled Work persisted records."""

from dataclasses import dataclass
from datetime import datetime

from croniter import croniter  # type: ignore[import-untyped]

from myclaw.session.session import Session
from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.utils.validation import require_aware_datetime, require_uuid4_string


@dataclass(frozen=True, slots=True)
class ScheduledWork:
    """One exact element of a Workspace Scheduled Work JSON array."""

    id: str
    title: str
    cron: str
    prompt: str
    created_at: datetime
    enabled: bool
    session_id: str

    def __post_init__(self) -> None:
        require_uuid4_string(self.id, field="id")
        if not self.title:
            msg = "title must not be empty"
            raise ValueError(msg)
        if len(self.title) > 120:
            msg = "title must not exceed 120 characters"
            raise ValueError(msg)
        if not self.prompt:
            msg = "prompt must not be empty"
            raise ValueError(msg)
        if len(self.prompt) > 20000:
            msg = "prompt must not exceed 20000 characters"
            raise ValueError(msg)
        if len(self.cron.split()) != 5 or not croniter.is_valid(self.cron):
            msg = "cron must be a valid 5-field expression"
            raise ValueError(msg)
        require_aware_datetime(self.created_at, field="created_at")
        if not isinstance(self.enabled, bool):
            msg = "enabled must be a boolean"
            raise ValueError(msg)
        Session._require_id(self.session_id)
        if self.session_id.startswith("schedule_"):
            expected_session_id = Session.schedule_session_id(self.id)
            if self.session_id != expected_session_id:
                msg = "Schedule Session ID must be derived from the Job UUID4"
                raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "cron": self.cron,
            "prompt": self.prompt,
            "created_at": format_rfc3339_milliseconds(self.created_at),
            "enabled": self.enabled,
            "session_id": self.session_id,
        }
