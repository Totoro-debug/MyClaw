"""Model-visible Schedule Job management Tool."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, cast
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter  # type: ignore[import-untyped]
from tzlocal import get_localzone_name

from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.store import (
    ScheduleStaleRemovalError,
    ScheduleStore,
)
from myclaw.tools.base import BaseTool
from myclaw.tools.confirmation import ConfirmationPrompt
from myclaw.tools.errors import ToolError
from myclaw.tools.schema import ToolParam
from myclaw.utils.json_types import JsonObject
from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.utils.validation import require_uuid4_string

_INVALID_ARGUMENTS = "Invalid arguments for schedule."
_NOT_FOUND = "Schedule Job was not found."
_STALE_REMOVAL = "Schedule Job changed before removal. Request removal again."
_STATE_READ_FAILED = "Schedule state could not be read."
_STATE_UPDATE_FAILED = "Schedule state could not be updated."
_TIMEZONE_FAILED = "Schedule timezone could not be resolved."


@dataclass(frozen=True, slots=True)
class _ConfirmedAdd:
    message: str
    schedule: JobSchedule


class ScheduleTool(BaseTool):
    """Add, list, and remove user-owned Schedule Jobs."""

    name = "schedule"
    description = "Manage one-time and recurring Schedule Jobs."
    required = ("action",)

    action: Annotated[
        str,
        ToolParam(description="The exact action: add, list, or remove.", min_length=1),
    ]
    message: Annotated[
        str | None,
        ToolParam(description="The message to run for an added Schedule Job."),
    ] = None
    every_seconds: Annotated[
        int | None,
        ToolParam(description="Run again this many seconds after completion.", minimum=1),
    ] = None
    cron_expr: Annotated[
        str | None,
        ToolParam(description="A canonical five-field Cron expression."),
    ] = None
    timezone: Annotated[
        str | None,
        ToolParam(description="The IANA timezone for a Cron expression."),
    ] = None
    at_time: Annotated[
        str | None,
        ToolParam(description="An absolute ISO time with a UTC offset."),
    ] = None
    job_id: Annotated[
        str | None,
        ToolParam(description="The canonical UUID4 of a Job to remove."),
    ] = None

    def __init__(
        self,
        *,
        store: ScheduleStore,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID] = uuid4,
    ) -> None:
        self._store = store
        self._now = now
        self._new_uuid = new_uuid
        self._confirmed_add: ContextVar[_ConfirmedAdd | None] = ContextVar(
            "schedule_confirmed_add",
            default=None,
        )
        self._confirmed_remove: ContextVar[ScheduleJob | None] = ContextVar(
            "schedule_confirmed_remove",
            default=None,
        )

    def prepare(self, arguments: JsonObject) -> JsonObject:
        """Keep only fields used by the exact action and schedule priority."""
        action = arguments.get("action")
        if not isinstance(action, str):
            return {"action": action}
        if action == "list":
            return {"action": action}
        if action == "remove":
            effective: JsonObject = {"action": action}
            job_id = arguments.get("job_id")
            if job_id is not None:
                effective["job_id"] = job_id
            return effective
        if action != "add":
            return {"action": action}

        effective = {"action": action}
        message = arguments.get("message")
        if message is not None:
            effective["message"] = message
        for schedule_name in ("every_seconds", "cron_expr", "at_time"):
            value = arguments.get(schedule_name)
            if value is not None:
                effective[schedule_name] = value
                if schedule_name == "cron_expr":
                    timezone = arguments.get("timezone")
                    if timezone is not None:
                        effective["timezone"] = timezone
                break
        return effective

    def confirmation_finished(self) -> None:
        """Release the frozen mutation after this confirmation path settles."""
        self._confirmed_add.set(None)
        self._confirmed_remove.set(None)

    async def confirmation_request(
        self,
        *,
        action: str,
        message: str | None = None,
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        timezone: str | None = None,
        at_time: str | None = None,
        job_id: str | None = None,
    ) -> ConfirmationPrompt | None:
        if action == "add":
            normalized_message, schedule = self._normalize_add(
                message=message,
                every_seconds=every_seconds,
                cron_expr=cron_expr,
                timezone=timezone,
                at_time=at_time,
            )
            self._confirmed_add.set(_ConfirmedAdd(normalized_message, schedule))
            warnings: tuple[str, ...] = ()
            if schedule.kind == "at" and schedule.at_datetime is not None:
                if schedule.at_datetime <= self._aware_now():
                    warnings = (
                        "This at Schedule Job is already due and will run as soon as possible.",
                    )
            return ConfirmationPrompt(
                summary="Add Schedule Job",
                details={
                    "action": "add",
                    "message": normalized_message,
                    "schedule": _public_schedule(schedule),
                },
                warnings=warnings,
            )
        if action == "remove":
            if job_id is None:
                raise ToolError(_INVALID_ARGUMENTS)
            try:
                require_uuid4_string(job_id, field="job_id")
            except ValueError as error:
                raise ToolError(_INVALID_ARGUMENTS) from error
            job = await self._current_public_job(job_id)
            if job is None:
                raise ToolError(_NOT_FOUND)
            self._confirmed_remove.set(job)
            return ConfirmationPrompt(
                summary="Remove Schedule Job",
                details={
                    "action": "remove",
                    "job_id": job.job_id,
                    "message": job.message,
                    "schedule": _public_schedule(job.schedule),
                },
                warnings=(
                    "Only the Schedule Job definition is deleted; its Conversation Session is retained.",
                ),
            )
        if action == "list":
            return None
        raise ToolError(_INVALID_ARGUMENTS)

    async def execute(
        self,
        *,
        action: str,
        message: str | None = None,
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        timezone: str | None = None,
        at_time: str | None = None,
        job_id: str | None = None,
    ) -> str:
        if action == "add":
            confirmed = self._confirmed_add.get()
            if confirmed is None:
                raise ToolError(_INVALID_ARGUMENTS)
            normalized_message, schedule = confirmed.message, confirmed.schedule
            timestamp = self._epoch_milliseconds(self._aware_now())
            job = ScheduleJob(
                job_id=str(self._new_uuid()),
                message=normalized_message,
                schedule=schedule,
                created_at_ms=timestamp,
                updated_at_ms=timestamp,
            )
            try:
                await self._store.add_user_job(job)
            except Exception as error:
                raise ToolError(_STATE_UPDATE_FAILED) from error
            return _json_content({"action": "add", "job": _public_job(job)})

        if action == "list":
            try:
                jobs = sorted(
                    await self._store.public_snapshot(),
                    key=lambda job: job.job_id,
                )
            except Exception as error:
                raise ToolError(_STATE_READ_FAILED) from error
            return _json_content({"jobs": [_public_job(job) for job in jobs]})

        if action == "remove":
            if job_id is None:
                raise ToolError(_INVALID_ARGUMENTS)
            expected = self._confirmed_remove.get()
            if expected is None or expected.job_id != job_id:
                raise ToolError(_INVALID_ARGUMENTS)
            try:
                removed = await self._store.remove_user_job(job_id, expected=expected)
            except ScheduleStaleRemovalError as error:
                raise ToolError(_STALE_REMOVAL) from error
            except Exception as error:
                raise ToolError(_STATE_UPDATE_FAILED) from error
            if not removed:
                raise ToolError(_STALE_REMOVAL)
            return _json_content({"action": "remove", "job": _public_job(expected)})

        raise ToolError(_INVALID_ARGUMENTS)

    def _normalize_add(
        self,
        *,
        message: str | None,
        every_seconds: int | None,
        cron_expr: str | None,
        timezone: str | None,
        at_time: str | None,
    ) -> tuple[str, JobSchedule]:
        if not isinstance(message, str):
            raise ToolError(_INVALID_ARGUMENTS)
        normalized_message = message.strip()
        if not normalized_message or len(normalized_message) > 20_000:
            raise ToolError(_INVALID_ARGUMENTS)

        if every_seconds is not None:
            if isinstance(every_seconds, bool) or not isinstance(every_seconds, int):
                raise ToolError(_INVALID_ARGUMENTS)
            try:
                schedule = JobSchedule.every(every_seconds)
            except ValueError as error:
                raise ToolError(_INVALID_ARGUMENTS) from error
            return normalized_message, schedule

        if cron_expr is not None:
            if not isinstance(cron_expr, str):
                raise ToolError(_INVALID_ARGUMENTS)
            normalized_cron = " ".join(cron_expr.split())
            if len(normalized_cron.split()) != 5 or not croniter.is_valid(normalized_cron):
                raise ToolError(_INVALID_ARGUMENTS)
            resolved_timezone = self._resolve_timezone(timezone)
            try:
                schedule = JobSchedule.cron(normalized_cron, resolved_timezone)
            except ValueError as error:
                raise ToolError(_INVALID_ARGUMENTS) from error
            return normalized_message, schedule

        if at_time is not None:
            try:
                normalized_at = _normalize_at_time(at_time)
                schedule = JobSchedule.at(normalized_at)
            except (TypeError, ValueError) as error:
                raise ToolError(_INVALID_ARGUMENTS) from error
            return normalized_message, schedule

        raise ToolError(_INVALID_ARGUMENTS)

    def _resolve_timezone(self, value: str | None) -> str:
        if value is None:
            try:
                value = get_localzone_name()
            except Exception as error:
                raise ToolError(_TIMEZONE_FAILED) from error
        if not isinstance(value, str):
            raise ToolError(_INVALID_ARGUMENTS)
        try:
            zone = ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ToolError(_TIMEZONE_FAILED) from error
        if zone.key != value:
            raise ToolError(_TIMEZONE_FAILED)
        return value

    def _aware_now(self) -> datetime:
        current = self._now()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ToolError(_INVALID_ARGUMENTS)
        return current

    @staticmethod
    def _epoch_milliseconds(value: datetime) -> int:
        milliseconds = int(value.timestamp() * 1000)
        if milliseconds < 0:
            raise ToolError(_INVALID_ARGUMENTS)
        return milliseconds

    async def _current_public_job(self, job_id: str) -> ScheduleJob | None:
        try:
            jobs = await self._store.public_snapshot()
        except Exception as error:
            raise ToolError(_STATE_READ_FAILED) from error
        return next((job for job in jobs if job.job_id == job_id), None)


def _normalize_at_time(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("at_time must be a string")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("at_time must include an offset")
    return format_rfc3339_milliseconds(parsed)


def _public_schedule(schedule: JobSchedule) -> JsonObject:
    if schedule.kind == "at":
        return {"type": "at", "at_time": cast(str, schedule.at_time)}
    if schedule.kind == "every":
        return {"type": "every", "every_seconds": cast(int, schedule.every_seconds)}
    return {
        "type": "cron",
        "cron_expr": cast(str, schedule.cron_expr),
        "timezone": cast(str, schedule.timezone),
    }


def _public_job(job: ScheduleJob) -> JsonObject:
    return {
        "job_id": job.job_id,
        "message": job.message,
        "schedule": _public_schedule(job.schedule),
    }


def _json_content(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = ["ScheduleTool"]
