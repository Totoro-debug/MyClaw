"""Conversation Summary and Long-term Memory persistence boundaries."""

from datetime import datetime
from typing import Protocol, runtime_checkable

from myclaw.memory.records import SummaryEntry


@runtime_checkable
class SummaryStore(Protocol):
    async def append(self, content: str, timestamp: datetime) -> SummaryEntry: ...

    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]: ...


@runtime_checkable
class MemoryStore(Protocol):
    async def read_long_term(self) -> str: ...

    async def replace_long_term(self, content: str) -> None: ...

    async def read_summary_cursor(self) -> int: ...

    async def write_summary_cursor(self, index: int) -> None: ...
