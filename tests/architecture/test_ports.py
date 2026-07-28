from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from myclaw.agent.events import AgentEvent
from myclaw.agent.ports import ConversationPort
from myclaw.config.config import ConfigView
from myclaw.management.commands import ManagementPort
from myclaw.management.service import ResumeResult, RuntimeStatus
from myclaw.memory.memory_task import MemoryTaskResult
from myclaw.memory.ports import (
    MemoryStore,
    SummaryStore,
)
from myclaw.memory.records import SummaryEntry
from myclaw.provider.ports import ModelProvider
from myclaw.session.ports import SessionStore
from myclaw.session.records import (
    ConversationSession,
    MetadataUpdate,
    SessionMessage,
    SessionSummary,
)
from myclaw.session.session_store import SessionListingReport
from myclaw.tools.base import BaseTool
from tests.fixtures.provider import ScriptedFakeProvider
from tests.fixtures.tool import FakeTool


class _ConversationPortFake:
    def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        raise NotImplementedError

    async def cancel_active_turn(self) -> None:
        raise NotImplementedError


class _ManagementPortFake:
    async def config_view(self) -> ConfigView:
        raise NotImplementedError

    async def status(self) -> RuntimeStatus:
        raise NotImplementedError

    async def resumable_listing(self) -> SessionListingReport:
        raise NotImplementedError

    async def resume(self, session_id: str) -> ResumeResult:
        raise NotImplementedError

    async def memory_view(self) -> str:
        raise NotImplementedError

    async def dream(self) -> MemoryTaskResult:
        raise NotImplementedError


class _SessionStoreFake:
    async def append_message(self, session_id: str, message: SessionMessage) -> None:
        raise NotImplementedError

    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None:
        raise NotImplementedError

    async def load(self, session_id: str) -> ConversationSession:
        raise NotImplementedError

    async def list_for_workspace(self, workspace: Path) -> tuple[SessionSummary, ...]:
        raise NotImplementedError


class _SummaryStoreFake:
    async def append(self, content: str, timestamp: datetime) -> SummaryEntry:
        raise NotImplementedError

    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]:
        raise NotImplementedError


class _MemoryStoreFake:
    async def read_long_term(self) -> str:
        raise NotImplementedError

    async def replace_long_term(self, content: str) -> None:
        raise NotImplementedError

    async def read_summary_cursor(self) -> int:
        raise NotImplementedError

    async def write_summary_cursor(self, index: int) -> None:
        raise NotImplementedError


def test_runtime_boundaries_are_structurally_substitutable_protocols() -> None:
    conversation: ConversationPort = _ConversationPortFake()
    management: ManagementPort = _ManagementPortFake()
    sessions: SessionStore = _SessionStoreFake()
    summaries: SummaryStore = _SummaryStoreFake()
    memory: MemoryStore = _MemoryStoreFake()
    provider: ModelProvider = ScriptedFakeProvider()
    tool = FakeTool(
        name="read_file",
        description="Read a UTF-8 file.",
        required=("path",),
        outcomes=(),
    )

    assert isinstance(conversation, ConversationPort)
    assert isinstance(management, ManagementPort)
    assert isinstance(sessions, SessionStore)
    assert isinstance(summaries, SummaryStore)
    assert isinstance(memory, MemoryStore)
    assert isinstance(provider, ModelProvider)
    assert isinstance(tool, BaseTool)
