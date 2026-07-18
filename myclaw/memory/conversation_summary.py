"""Synchronous Conversation Summary selection and persistence."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from myclaw.agent.prompts import current_user_input
from myclaw.agent.turn import model_message_from_session
from myclaw.config.agent_home import AgentHome
from myclaw.errors import ErrorInfo
from myclaw.management.service import RuntimeStatusInput, estimate_input_tokens
from myclaw.memory.ports import SummaryStore
from myclaw.memory.records import SummaryEntry
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import ModelRequest, ReasoningEffort, UserModelMessage
from myclaw.provider.ports import ModelProvider
from myclaw.session.identifiers import require_session_id
from myclaw.session.ports import SessionStore
from myclaw.session.records import (
    ConversationSession,
    MetadataUpdate,
    SessionMessage,
    UserSessionMessage,
)
from myclaw.tools.models import ToolDefinition
from myclaw.utils.atomic_files import atomic_replace_bytes
from myclaw.utils.time import format_rfc3339_milliseconds

_SUMMARY_SYSTEM_PROMPT = """Summarize the provided earlier conversation messages.
Preserve decisions, user intent, important facts, and unresolved work concisely."""

type AtomicReplaceBytes = Callable[[Path, bytes], None]
type UnlinkFile = Callable[[Path], None]


def _unlink_file(path: Path) -> None:
    path.unlink()


@dataclass(frozen=True, slots=True)
class _PendingConsolidation:
    session_id: str
    old_cursor: int
    new_cursor: int
    summary: SummaryEntry

    def __post_init__(self) -> None:
        require_session_id(self.session_id)
        for field, value in (
            ("old_cursor", self.old_cursor),
            ("new_cursor", self.new_cursor),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.new_cursor <= self.old_cursor:
            raise ValueError("new_cursor must advance beyond old_cursor")

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "old_cursor": self.old_cursor,
            "new_cursor": self.new_cursor,
            "summary_index": self.summary.index,
            "timestamp": format_rfc3339_milliseconds(self.summary.timestamp),
            "content": self.summary.content,
        }

    @classmethod
    def from_bytes(cls, content: bytes) -> _PendingConsolidation:
        loaded: object = json.loads(content.decode("utf-8"))
        if not isinstance(loaded, dict) or set(loaded) != {
            "session_id",
            "old_cursor",
            "new_cursor",
            "summary_index",
            "timestamp",
            "content",
        }:
            raise ValueError("consolidation journal has an invalid schema")
        session_id = loaded["session_id"]
        old_cursor = loaded["old_cursor"]
        new_cursor = loaded["new_cursor"]
        summary_index = loaded["summary_index"]
        timestamp = loaded["timestamp"]
        summary_content = loaded["content"]
        if not isinstance(session_id, str):
            raise ValueError("journal session_id must be a string")
        if not isinstance(timestamp, str):
            raise ValueError("journal timestamp must be a string")
        if not isinstance(summary_content, str):
            raise ValueError("journal content must be a string")
        if isinstance(summary_index, bool) or not isinstance(summary_index, int):
            raise ValueError("journal summary_index must be an integer")
        parsed_timestamp = datetime.fromisoformat(timestamp)
        if format_rfc3339_milliseconds(parsed_timestamp) != timestamp:
            raise ValueError("journal timestamp must use canonical RFC 3339 milliseconds")
        if isinstance(old_cursor, bool) or not isinstance(old_cursor, int):
            raise ValueError("journal old_cursor must be an integer")
        if isinstance(new_cursor, bool) or not isinstance(new_cursor, int):
            raise ValueError("journal new_cursor must be an integer")
        return cls(
            session_id=session_id,
            old_cursor=old_cursor,
            new_cursor=new_cursor,
            summary=SummaryEntry(
                index=summary_index,
                timestamp=parsed_timestamp,
                content=summary_content,
            ),
        )


class ConsolidationSessionStore(SessionStore, Protocol):
    async def recover_consolidation_cursor(
        self,
        session_id: str,
        *,
        old_cursor: int,
        new_cursor: int,
    ) -> None: ...


class ConsolidatingSummaryStore(SummaryStore, Protocol):
    async def commit_consolidation(
        self,
        *,
        sessions: ConsolidationSessionStore,
        session_id: str,
        old_cursor: int,
        new_cursor: int,
        content: str,
        timestamp: datetime,
    ) -> SummaryEntry: ...

    async def recover_pending(self, sessions: ConsolidationSessionStore) -> None: ...


@dataclass(frozen=True, slots=True)
class SummaryModelSettings:
    """Resolved provider-neutral fields for Conversation Summary generation."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int


class JsonlSummaryStore:
    """Append the global ordered Conversation Summary stream."""

    def __init__(
        self,
        agent_home: AgentHome,
        *,
        replace_bytes: AtomicReplaceBytes = atomic_replace_bytes,
        unlink_file: UnlinkFile = _unlink_file,
    ) -> None:
        self.path = agent_home.path / "memory" / "summary.jsonl"
        self.pending_directory = agent_home.path / "memory" / "pending-consolidations"
        self._replace_bytes = replace_bytes
        self._unlink_file = unlink_file
        self._lock = asyncio.Lock()

    async def append(self, content: str, timestamp: datetime) -> SummaryEntry:
        async with self._lock:
            next_index = len(self._entries()) + 1
            entry = SummaryEntry(index=next_index, timestamp=timestamp, content=content)
            self._append_exact(entry)
            return entry

    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]:
        async with self._lock:
            return self._entries()[cursor : cursor + limit]

    async def commit_consolidation(
        self,
        *,
        sessions: ConsolidationSessionStore,
        session_id: str,
        old_cursor: int,
        new_cursor: int,
        content: str,
        timestamp: datetime,
    ) -> SummaryEntry:
        async with self._lock:
            next_index = len(self._entries()) + 1
            entry = SummaryEntry(index=next_index, timestamp=timestamp, content=content)
            journal = _PendingConsolidation(
                session_id=session_id,
                old_cursor=old_cursor,
                new_cursor=new_cursor,
                summary=entry,
            )
            self.pending_directory.mkdir(parents=True, exist_ok=True)
            journal_path = self.pending_directory / f"{session_id}.json"
            self._replace_bytes(
                journal_path,
                json.dumps(journal.to_dict(), ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                ),
            )
            self._append_exact(entry)
            await sessions.update_metadata(
                session_id,
                MetadataUpdate(consolidation_cursor=new_cursor),
            )
            self._unlink_file(journal_path)
            return entry

    async def recover_pending(self, sessions: ConsolidationSessionStore) -> None:
        async with self._lock:
            if not self.pending_directory.exists():
                return
            for journal_path in sorted(self.pending_directory.glob("*.json")):
                journal = _PendingConsolidation.from_bytes(journal_path.read_bytes())
                if journal_path.stem != journal.session_id:
                    raise ValueError("journal file name must match session_id")
                self._append_exact(journal.summary)
                await sessions.recover_consolidation_cursor(
                    journal.session_id,
                    old_cursor=journal.old_cursor,
                    new_cursor=journal.new_cursor,
                )
                self._unlink_file(journal_path)

    def _append_exact(self, entry: SummaryEntry) -> None:
        entries = self._entries()
        if entry.index <= len(entries):
            if entries[entry.index - 1].to_dict() != entry.to_dict():
                raise ValueError("summary index already contains a different record")
            return
        if entry.index != len(entries) + 1:
            raise ValueError("summary index must be contiguous")
        existing = self.path.read_bytes() if self.path.exists() else b""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._replace_bytes(self.path, existing + entry.to_json_line().encode("utf-8"))

    def _entries(self) -> tuple[SummaryEntry, ...]:
        if not self.path.exists():
            return ()
        content = self.path.read_bytes()
        if content and not content.endswith(b"\n"):
            raise ValueError("summary stream must contain complete JSONL records")
        entries: list[SummaryEntry] = []
        for expected_index, line in enumerate(content.decode("utf-8").splitlines(), start=1):
            loaded: object = json.loads(line)
            if not isinstance(loaded, dict) or set(loaded) != {
                "index",
                "timestamp",
                "content",
            }:
                raise ValueError("summary record has an invalid schema")
            index = loaded["index"]
            timestamp = loaded["timestamp"]
            summary_content = loaded["content"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("summary index must be an integer")
            if index != expected_index:
                raise ValueError("summary indexes must be contiguous from 1")
            if not isinstance(timestamp, str):
                raise ValueError("summary timestamp must be a string")
            if not isinstance(summary_content, str):
                raise ValueError("summary content must be a string")
            parsed_timestamp = datetime.fromisoformat(timestamp)
            if format_rfc3339_milliseconds(parsed_timestamp) != timestamp:
                raise ValueError("summary timestamp must use canonical RFC 3339 milliseconds")
            entries.append(
                SummaryEntry(
                    index=index,
                    timestamp=parsed_timestamp,
                    content=summary_content,
                )
            )
        return tuple(entries)


class ConversationSummaryManager:
    """Compress eligible early Session messages before a chat model call."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        sessions: ConsolidationSessionStore,
        summaries: ConsolidatingSummaryStore,
        settings: SummaryModelSettings,
        chat_context_window: int,
        chat_max_output: int,
        consolidation_message_threshold: int,
        chat_system_prompt: str,
        tools: tuple[ToolDefinition, ...],
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        self._provider = provider
        self._sessions = sessions
        self._summaries = summaries
        self._settings = settings
        self._chat_context_window = chat_context_window
        self._chat_max_output = chat_max_output
        self._message_threshold = consolidation_message_threshold
        self._chat_system_prompt = chat_system_prompt
        self._tools = tools
        self._now = now
        self._new_uuid = new_uuid

    async def recover_pending(self) -> None:
        try:
            await self._summaries.recover_pending(self._sessions)
        except (OSError, UnicodeError, ValueError) as error:
            raise ModelCallError(
                ErrorInfo(
                    code="persistence_error",
                    message="Pending Conversation Summary recovery could not complete.",
                )
            ) from error

    async def prepare(self, session: ConversationSession) -> ConversationSession:
        short_term = session.short_term_messages
        available_input = self._chat_context_window - self._chat_max_output
        system_tokens = estimate_input_tokens(
            RuntimeStatusInput(
                system_prompt=self._chat_system_prompt,
                retained_messages=(),
                tool_definitions=(),
                runtime_context="",
            )
        )
        if system_tokens > available_input:
            raise ModelCallError(
                ErrorInfo(
                    code="memory_context_too_large",
                    message="Long-term Memory and system context exceed the chat input budget.",
                )
            )
        token_triggered = estimate_input_tokens(self._chat_input(session)) >= available_input
        message_triggered = len(short_term) >= self._message_threshold
        if not token_triggered and not message_triggered:
            return session
        initial_cutoff = 0
        if token_triggered:
            initial_cutoff = _token_cutoff(short_term, available_input)
        if message_triggered:
            initial_cutoff = max(
                initial_cutoff,
                min(self._message_threshold // 2, len(short_term) - 1),
            )
        cutoff = _aligned_cutoff(short_term, initial_cutoff)
        if cutoff == 0:
            raise ModelCallError(
                ErrorInfo(
                    code="model_context_overflow",
                    message="No complete earlier conversation turn can be summarized safely.",
                )
            )
        selected = short_term[:cutoff]
        request = ModelRequest(
            request_id=self._new_uuid(),
            route="memory",
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            messages=(UserModelMessage(content=_summary_input(selected)),),
            tools=(),
            stream=False,
            model=self._settings.model,
            max_output=self._settings.max_output,
            temperature=self._settings.temperature,
            reasoning_effort=self._settings.reasoning_effort,
            timeout_seconds=self._settings.timeout_seconds,
        )
        response = await self._provider.complete(request)
        try:
            await self._sessions.update_metadata(
                session.metadata.id,
                MetadataUpdate(usage_delta=response.usage),
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise ModelCallError(
                ErrorInfo(
                    code="persistence_error",
                    message="Conversation Summary usage could not be persisted.",
                )
            ) from error
        if not response.message.content:
            raise ModelCallError(
                ErrorInfo(
                    code="model_failed",
                    message="The memory model returned an empty Conversation Summary.",
                )
            )
        try:
            new_cursor = session.metadata.consolidation_cursor + cutoff
            await self._summaries.commit_consolidation(
                sessions=self._sessions,
                session_id=session.metadata.id,
                old_cursor=session.metadata.consolidation_cursor,
                new_cursor=new_cursor,
                content=response.message.content,
                timestamp=self._persisted_now(),
            )
            return await self._sessions.load(session.metadata.id)
        except (OSError, UnicodeError, ValueError) as error:
            raise ModelCallError(
                ErrorInfo(
                    code="persistence_error",
                    message="Conversation Summary could not be persisted.",
                )
            ) from error

    def _chat_input(self, session: ConversationSession) -> RuntimeStatusInput:
        short_term = session.short_term_messages
        current_user_index = _last_user_index(short_term)
        retained: list[str] = []
        for index, message in enumerate(short_term):
            model_message = model_message_from_session(message)
            if model_message is None:
                continue
            if index == current_user_index and isinstance(message, UserSessionMessage):
                model_message = UserModelMessage(
                    content=current_user_input(
                        content=message.content,
                        current_time=message.created_at,
                        session_id=session.metadata.id,
                    )
                )
            retained.append(
                json.dumps(
                    model_message.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        tool_definitions = tuple(
            json.dumps(tool.to_dict(), ensure_ascii=False, separators=(",", ":"))
            for tool in self._tools
        )
        return RuntimeStatusInput(
            system_prompt=self._chat_system_prompt,
            retained_messages=tuple(retained),
            tool_definitions=tool_definitions,
            runtime_context="",
        )

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)


def _aligned_cutoff(messages: tuple[SessionMessage, ...], initial: int) -> int:
    for index in range(initial, len(messages)):
        if isinstance(messages[index], UserSessionMessage):
            return index
    for index in range(initial - 1, -1, -1):
        if isinstance(messages[index], UserSessionMessage):
            return index
    return 0


def _token_cutoff(messages: tuple[SessionMessage, ...], input_budget: int) -> int:
    current_turn_start = _last_user_index(messages)
    target_bytes = input_budget // 2 * 4
    selected_bytes = 0
    for index, message in enumerate(messages[:current_turn_start]):
        model_message = model_message_from_session(message)
        if model_message is not None:
            selected_bytes += len(
                json.dumps(
                    model_message.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        if selected_bytes >= target_bytes:
            return index + 1
    return current_turn_start


def _last_user_index(messages: tuple[SessionMessage, ...]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], UserSessionMessage):
            return index
    return len(messages)


def _summary_input(messages: tuple[SessionMessage, ...]) -> str:
    records = [
        model_message.to_dict()
        for message in messages
        if (model_message := model_message_from_session(message)) is not None
    ]
    return (
        "<conversation_messages>\n"
        + json.dumps(
            records,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n</conversation_messages>"
    )
