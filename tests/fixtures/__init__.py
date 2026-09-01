"""Reusable boundary fakes and pytest fixtures."""

from tests.fixtures.agent_loop import collect_foreground_outbound
from tests.fixtures.clock import FakeClock
from tests.fixtures.framing import (
    BlockingBlackboardGenerator,
    DeterministicBlackboardGenerator,
)
from tests.fixtures.gateway import SingleToolGateway
from tests.fixtures.provider import (
    ProviderCall,
    ScriptedFakeProvider,
    ScriptedFakeRouter,
    StreamScript,
    unexpected_provider_factory,
)
from tests.fixtures.schedule import write_schedule_state
from tests.fixtures.tool import FakeTool, FakeToolCall

__all__ = [
    "BlockingBlackboardGenerator",
    "DeterministicBlackboardGenerator",
    "FakeClock",
    "FakeTool",
    "FakeToolCall",
    "ProviderCall",
    "ScriptedFakeProvider",
    "ScriptedFakeRouter",
    "SingleToolGateway",
    "StreamScript",
    "collect_foreground_outbound",
    "unexpected_provider_factory",
    "write_schedule_state",
]
