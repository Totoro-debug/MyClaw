"""Reusable boundary fakes and pytest fixtures."""

import json
from collections.abc import Iterable
from pathlib import Path

from myclaw.schedule.records import ScheduledWork
from tests.fixtures.clock import FakeClock
from tests.fixtures.events import validate_agent_event_sequence
from tests.fixtures.provider import (
    ScriptedFakeProvider,
    StreamScript,
    unexpected_provider_factory,
)
from tests.fixtures.tool import FakeTool, FakeToolCall


def persist_scheduled_work(state_root: Path, records: Iterable[ScheduledWork]) -> None:
    """Write canonical Scheduled Work test state without a production mutation seam."""
    (state_root / "scheduled-work.json").write_text(
        json.dumps(
            [record.to_dict() for record in records],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


__all__ = [
    "FakeClock",
    "FakeTool",
    "FakeToolCall",
    "ScriptedFakeProvider",
    "StreamScript",
    "persist_scheduled_work",
    "unexpected_provider_factory",
    "validate_agent_event_sequence",
]
