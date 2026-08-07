"""Assertions shared by Agent Event behavior tests."""

from collections.abc import Iterable
from itertools import pairwise

from myclaw.agent.events import AgentEvent


def validate_agent_event_sequence(events: Iterable[AgentEvent]) -> None:
    observed = tuple(events)
    assert all(current.event_id > previous.event_id for previous, current in pairwise(observed))
    if not observed:
        return
    terminal_types = {"turn_completed", "turn_failed", "turn_cancelled"}
    assert all(event.turn_id == observed[0].turn_id for event in observed[1:])
    assert observed[0].type == "turn_started"
    terminal_indexes = [
        index for index, event in enumerate(observed) if event.type in terminal_types
    ]
    assert terminal_indexes == [len(observed) - 1]
