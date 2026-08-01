"""Assertions shared by Agent Event behavior tests."""

from collections.abc import Iterable
from itertools import pairwise

from myclaw.agent.events import AgentEvent


def validate_agent_event_sequence(events: Iterable[AgentEvent]) -> None:
    observed = tuple(events)
    assert all(current.event_id > previous.event_id for previous, current in pairwise(observed))

    terminal_types = {"turn_completed", "turn_failed", "turn_cancelled"}
    foreground_active = False
    for event in observed:
        if event.type == "background_completed":
            assert not foreground_active
        elif event.type == "turn_started":
            foreground_active = True
        elif event.type in terminal_types:
            foreground_active = False

    foreground = tuple(event for event in observed if event.type != "background_completed")
    if not foreground:
        return
    assert all(event.turn_id == foreground[0].turn_id for event in foreground[1:])
    assert foreground[0].type == "turn_started"
    terminal_indexes = [
        index for index, event in enumerate(foreground) if event.type in terminal_types
    ]
    assert terminal_indexes == [len(foreground) - 1]
