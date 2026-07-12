"""Scripted Tool boundary for deterministic offline tests."""

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FakeToolCall:
    """One call observed by a FakeTool."""

    arguments: Mapping[str, object]
    context: object


class FakeTool:
    """Return scripted normalized text while recording Tool calls."""

    def __init__(
        self,
        *,
        definition: object,
        outcomes: Iterable[str | BaseException],
    ) -> None:
        self._definition = definition
        self._outcomes = deque(outcomes)
        self.calls: list[FakeToolCall] = []

    @property
    def definition(self) -> object:
        return self._definition

    async def execute(self, arguments: Mapping[str, object], context: object) -> str:
        self.calls.append(FakeToolCall(arguments=dict(arguments), context=context))
        if not self._outcomes:
            msg = "No scripted Tool outcome remains"
            raise AssertionError(msg)
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
