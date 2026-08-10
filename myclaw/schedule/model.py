"""Strict immutable Schedule Job persistence models."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter  # type: ignore[import-untyped]

from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.utils.validation import require_nonnegative_int, require_uuid4_string

ScheduleKind = Literal["at", "every", "cron"]
JobSource = Literal["user", "system"]
JobStatus = Literal["ok", "error"]

_JOB_FIELDS = frozenset(
    {
        "job_id",
        "source",
        "message",
        "schedule",
        "state",
        "created_at_ms",
        "updated_at_ms",
    }
)
_SCHEDULE_FIELDS = frozenset({"kind", "at_time", "every_seconds", "cron_expr", "timezone"})
_STATE_FIELDS = frozenset({"last_finished_at_ms", "last_status", "last_error"})
_RFC3339_MILLISECONDS = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"\.[0-9]{3}(?:[+-][0-9]{2}:[0-9]{2})$"
)


@dataclass(frozen=True, slots=True)
class JobSchedule:
    """One immutable, fully expanded Schedule definition."""

    kind: ScheduleKind
    at_time: str | None = None
    every_seconds: int | None = None
    cron_expr: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in {"at", "every", "cron"}:
            raise ValueError("schedule.kind must be at, every, or cron")
        if self.kind == "at":
            if self.at_time is None or self.every_seconds is not None:
                raise ValueError("at Schedule must select only at_time")
            if self.cron_expr is not None or self.timezone is not None:
                raise ValueError("at Schedule must select only at_time")
            _require_canonical_rfc3339_milliseconds(self.at_time, field="schedule.at_time")
            return
        if self.at_time is not None:
            raise ValueError(f"{self.kind} Schedule must not select at_time")
        if self.kind == "every":
            if (
                self.every_seconds is None
                or self.cron_expr is not None
                or self.timezone is not None
            ):
                raise ValueError("every Schedule must select only every_seconds")
            require_nonnegative_int(self.every_seconds, field="schedule.every_seconds")
            if self.every_seconds < 1:
                raise ValueError("schedule.every_seconds must be positive")
            return
        if self.every_seconds is not None:
            raise ValueError("cron Schedule must not select every_seconds")
        if self.cron_expr is None or self.timezone is None:
            raise ValueError("cron Schedule must select cron_expr and timezone")
        _require_canonical_cron(self.cron_expr)
        _require_iana_timezone(self.timezone)

    @classmethod
    def at(cls, at_time: str) -> JobSchedule:
        return cls(kind="at", at_time=at_time)

    @classmethod
    def every(cls, every_seconds: int) -> JobSchedule:
        return cls(kind="every", every_seconds=every_seconds)

    @classmethod
    def cron(cls, cron_expr: str, timezone: str) -> JobSchedule:
        return cls(kind="cron", cron_expr=cron_expr, timezone=timezone)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "at_time": self.at_time,
            "every_seconds": self.every_seconds,
            "cron_expr": self.cron_expr,
            "timezone": self.timezone,
        }

    @property
    def at_datetime(self) -> datetime | None:
        if self.at_time is None:
            return None
        return _parse_canonical_rfc3339_milliseconds(self.at_time, field="schedule.at_time")


@dataclass(frozen=True, slots=True)
class ScheduleJobState:
    """The latest terminal result persisted for one Schedule Job."""

    last_finished_at_ms: int | None = None
    last_status: JobStatus | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        if self.last_finished_at_ms is not None:
            require_nonnegative_int(self.last_finished_at_ms, field="state.last_finished_at_ms")
        if self.last_status is not None and (
            not isinstance(self.last_status, str) or self.last_status not in {"ok", "error"}
        ):
            raise ValueError("state.last_status must be null, ok, or error")
        if self.last_status is None:
            if self.last_finished_at_ms is not None or self.last_error is not None:
                raise ValueError("never-run state must contain only null values")
            return
        if self.last_finished_at_ms is None:
            raise ValueError("terminal state must include last_finished_at_ms")
        if self.last_status == "ok":
            if self.last_error is not None:
                raise ValueError("successful state must not contain last_error")
            return
        if not isinstance(self.last_error, str) or not self.last_error:
            raise ValueError("error state must contain a non-empty last_error")

    def to_dict(self) -> dict[str, object]:
        return {
            "last_finished_at_ms": self.last_finished_at_ms,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }


@dataclass(frozen=True, slots=True)
class ScheduleJob:
    """One immutable Workspace-owned Schedule Job."""

    job_id: str
    message: str
    schedule: JobSchedule
    created_at_ms: int
    updated_at_ms: int
    source: JobSource = "user"
    state: ScheduleJobState = field(default_factory=ScheduleJobState)

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str):
            raise ValueError("job_id must be a string")
        require_uuid4_string(self.job_id, field="job_id")
        if not isinstance(self.source, str) or self.source not in {"user", "system"}:
            raise ValueError("source must be user or system")
        if not isinstance(self.message, str):
            raise ValueError("message must be a string")
        if not self.message or self.message != self.message.strip():
            raise ValueError("message must be non-empty and trimmed")
        if len(self.message) > 20_000:
            raise ValueError("message must not exceed 20000 characters")
        if not isinstance(self.schedule, JobSchedule):
            raise ValueError("schedule must be a JobSchedule")
        if not isinstance(self.state, ScheduleJobState):
            raise ValueError("state must be a ScheduleJobState")
        require_nonnegative_int(self.created_at_ms, field="created_at_ms")
        require_nonnegative_int(self.updated_at_ms, field="updated_at_ms")
        if self.updated_at_ms < self.created_at_ms:
            raise ValueError("updated_at_ms must not precede created_at_ms")
        if self.state.last_status is None and self.updated_at_ms != self.created_at_ms:
            raise ValueError("never-run Schedule Job timestamps must be equal")

    @property
    def session_id(self) -> str:
        return f"schedule_{self.job_id}"

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "source": self.source,
            "message": self.message,
            "schedule": self.schedule.to_dict(),
            "state": self.state.to_dict(),
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ScheduleJob:
        if set(value) != _JOB_FIELDS:
            raise ValueError("Schedule Job fields do not match the persisted schema")
        job_id = value["job_id"]
        source = value["source"]
        message = value["message"]
        schedule_value = value["schedule"]
        state_value = value["state"]
        created_at_ms = value["created_at_ms"]
        updated_at_ms = value["updated_at_ms"]
        if not isinstance(job_id, str):
            raise ValueError("job_id must be a string")
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        if not isinstance(schedule_value, dict):
            raise ValueError("schedule must be an object")
        if not isinstance(state_value, dict):
            raise ValueError("state must be an object")
        schedule = _schedule_from_dict(schedule_value)
        state = _state_from_dict(state_value)
        if isinstance(created_at_ms, bool) or not isinstance(created_at_ms, int):
            raise ValueError("created_at_ms must be a nonnegative integer")
        if isinstance(updated_at_ms, bool) or not isinstance(updated_at_ms, int):
            raise ValueError("updated_at_ms must be a nonnegative integer")
        return cls(
            job_id=job_id,
            message=message,
            schedule=schedule,
            created_at_ms=created_at_ms,
            updated_at_ms=updated_at_ms,
            source=cast(JobSource, source),
            state=state,
        )


def _schedule_from_dict(value: dict[str, object]) -> JobSchedule:
    if set(value) != _SCHEDULE_FIELDS:
        raise ValueError("Schedule fields do not match the persisted schema")
    kind = value["kind"]
    at_time = value["at_time"]
    every_seconds = value["every_seconds"]
    cron_expr = value["cron_expr"]
    timezone = value["timezone"]
    if not isinstance(kind, str):
        raise ValueError("schedule.kind must be a string")
    if at_time is not None and not isinstance(at_time, str):
        raise ValueError("schedule.at_time must be a string or null")
    if every_seconds is not None and (
        isinstance(every_seconds, bool) or not isinstance(every_seconds, int)
    ):
        raise ValueError("schedule.every_seconds must be an integer or null")
    if cron_expr is not None and not isinstance(cron_expr, str):
        raise ValueError("schedule.cron_expr must be a string or null")
    if timezone is not None and not isinstance(timezone, str):
        raise ValueError("schedule.timezone must be a string or null")
    return JobSchedule(
        kind=cast(ScheduleKind, kind),
        at_time=at_time,
        every_seconds=every_seconds,
        cron_expr=cron_expr,
        timezone=timezone,
    )


def _state_from_dict(value: dict[str, object]) -> ScheduleJobState:
    if set(value) != _STATE_FIELDS:
        raise ValueError("Schedule Job state fields do not match the persisted schema")
    finished = value["last_finished_at_ms"]
    status = value["last_status"]
    error = value["last_error"]
    if finished is not None and (isinstance(finished, bool) or not isinstance(finished, int)):
        raise ValueError("state.last_finished_at_ms must be an integer or null")
    if status is not None and not isinstance(status, str):
        raise ValueError("state.last_status must be a string or null")
    if error is not None and not isinstance(error, str):
        raise ValueError("state.last_error must be a string or null")
    return ScheduleJobState(
        last_finished_at_ms=finished,
        last_status=cast(JobStatus | None, status),
        last_error=error,
    )


def _require_canonical_cron(value: str) -> None:
    if not isinstance(value, str):
        raise ValueError("schedule.cron_expr must be a canonical five-field Cron")
    if value != " ".join(value.split()) or len(value.split()) != 5:
        raise ValueError("schedule.cron_expr must be a canonical five-field Cron")
    if not croniter.is_valid(value):
        raise ValueError("schedule.cron_expr must be a valid five-field Cron")


def _require_iana_timezone(value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("schedule.timezone must be a valid IANA timezone")
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError("schedule.timezone must be a valid IANA timezone") from error


def _require_canonical_rfc3339_milliseconds(value: str, *, field: str) -> None:
    _parse_canonical_rfc3339_milliseconds(value, field=field)


def _parse_canonical_rfc3339_milliseconds(value: str, *, field: str) -> datetime:
    if not isinstance(value, str) or not _RFC3339_MILLISECONDS.fullmatch(value):
        raise ValueError(f"{field} must be canonical RFC 3339 milliseconds")
    if value.endswith("-00:00"):
        raise ValueError(
            f"{field} must be canonical RFC 3339 milliseconds with a known numeric UTC offset"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be canonical RFC 3339 milliseconds") from error
    if format_rfc3339_milliseconds(parsed) != value:
        raise ValueError(f"{field} must be canonical RFC 3339 milliseconds")
    return parsed


__all__ = [
    "JobSchedule",
    "JobSource",
    "JobStatus",
    "ScheduleJob",
    "ScheduleJobState",
    "ScheduleKind",
]
