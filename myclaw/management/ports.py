"""Management boundary."""

from typing import Protocol, runtime_checkable

from myclaw.config.models import ConfigView
from myclaw.management.models import RuntimeStatus
from myclaw.memory.models import MemoryTaskResult
from myclaw.session.models import ResumeResult
from myclaw.session.records import SessionSummary


@runtime_checkable
class ManagementPort(Protocol):
    async def config_view(self) -> ConfigView: ...

    async def status(self) -> RuntimeStatus: ...

    async def resumable_sessions(self) -> tuple[SessionSummary, ...]: ...

    async def resume(self, session_id: str) -> ResumeResult: ...

    async def memory_view(self) -> str: ...

    async def dream(self) -> MemoryTaskResult: ...
