"""Reusable boundary fakes and pytest fixtures."""

from tests.fixtures.clock import FakeClock
from tests.fixtures.events import validate_agent_event_sequence
from tests.fixtures.provider import (
    ScriptedFakeProvider,
    StreamScript,
    unexpected_provider_factory,
)
from tests.fixtures.tool import FakeTool, FakeToolCall

__all__ = [
    "FakeClock",
    "FakeTool",
    "FakeToolCall",
    "ScriptedFakeProvider",
    "StreamScript",
    "unexpected_provider_factory",
    "validate_agent_event_sequence",
]
