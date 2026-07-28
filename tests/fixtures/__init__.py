"""Reusable boundary fakes and pytest fixtures."""

import json
from collections.abc import Iterable
from pathlib import Path

from myclaw.schedule.records import ScheduledWork
from tests.fixtures.clock import FakeClock
from tests.fixtures.provider import ScriptedFakeProvider, StreamScript
from tests.fixtures.tool import FakeTool, FakeToolCall


def persist_scheduled_work(agent_home: Path, records: Iterable[ScheduledWork]) -> None:
    """Write canonical Scheduled Work test state without a production mutation seam."""
    (agent_home / "scheduled-work.json").write_text(
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
]
