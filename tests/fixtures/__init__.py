"""Reusable boundary fakes and pytest fixtures."""

from tests.fixtures.clock import FakeClock
from tests.fixtures.events import validate_agent_event_sequence
from tests.fixtures.gateway import SingleToolGateway
from tests.fixtures.provider import (
    ScriptedFakeProvider,
    StreamScript,
    unexpected_provider_factory,
)
from tests.fixtures.schedule import write_schedule_state
from tests.fixtures.tool import FakeTool, FakeToolCall

__all__ = [
    "FakeClock",
    "FakeTool",
    "FakeToolCall",
    "ScriptedFakeProvider",
    "SingleToolGateway",
    "StreamScript",
    "unexpected_provider_factory",
    "validate_agent_event_sequence",
    "write_schedule_state",
]
