"""Synchronous Conversation Summary selection and persistence."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from myclaw.agent.prompts import (
    conversation_summary_input,
    conversation_summary_prompt,
    current_user_input,
)
from myclaw.agent.run import model_message_from_session
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.logging.session import without_session_log
from myclaw.management.service import RuntimeStatusInput, estimate_input_tokens
from myclaw.memory.records import SummaryEntry
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import ModelProvider, ModelRequest, ReasoningEffort, UserModelMessage
from myclaw.session.session import Session
from myclaw.tools.base import OpenAIToolSchema
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.time import format_rfc3339_milliseconds

type AtomicReplaceBytes = Callable[[Path, bytes], None]


class SummaryStore(Protocol):
    """Append and read the ordered Conversation Summary stream."""

    async def append(self, content: str, timestamp: datetime) -> SummaryEntry: ...

    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]: ...


@dataclass(frozen=True, slots=True)
class SummaryModelSettings:
    """Resolved provider-neutral fields for Conversation Summary generation."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int


class WorkspaceJsonlSummaryStore:
    """Persist Conversation Summary entries in one Workspace State."""

    def __init__(
        self,
        workspace_state: WorkspaceState,
        *,
        replace_bytes: AtomicReplaceBytes = HOST_FILESYSTEM.atomic_replace_bytes,
    ) -> None:
        self.workspace_state = workspace_state
        self._state_root = workspace_state.path.resolve(strict=True)
        self._memory_directory = workspace_state.memory_directory
        self.path = workspace_state.memory_directory / "summary.jsonl"
        self._replace_bytes = replace_bytes
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

    def _append_exact(self, entry: SummaryEntry) -> None:
        entries = self._entries()
        if entry.index <= len(entries):
            if entries[entry.index - 1].to_dict() != entry.to_dict():
                raise ValueError("summary index already contains a different record")
            return
        if entry.index != len(entries) + 1:
            raise ValueError("summary index must be contiguous")
        existing = self.path.read_bytes() if self.path.exists() else b""
        self._replace_bytes(self.path, existing + entry.to_json_line().encode("utf-8"))
        HOST_FILESYSTEM.require_owned_regular_file(self.path, within=self._state_root)

    def _entries(self) -> tuple[SummaryEntry, ...]:
        self._require_memory_directory()
        if not self.path.exists() and not self.path.is_symlink():
            return ()
        HOST_FILESYSTEM.require_owned_regular_file(self.path, within=self._state_root)
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

    def _require_memory_directory(self) -> Path:
        return HOST_FILESYSTEM.require_owned_directory(
            self._memory_directory, within=self._state_root
        )


class ConversationSummaryManager:
    """Compress eligible early Session messages before a chat model call."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        summaries: SummaryStore,
        settings: SummaryModelSettings,
        chat_context_window: int,
        chat_max_output: int,
        consolidation_message_threshold: int,
        chat_system_prompt: str,
        tools: tuple[OpenAIToolSchema, ...],
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        self._provider = provider
        self._summaries = summaries
        self._settings = settings
        self._chat_context_window = chat_context_window
        self._chat_max_output = chat_max_output
        self._message_threshold = consolidation_message_threshold
        self._chat_system_prompt = chat_system_prompt
        self._tools = tools
        self._now = now
        self._new_uuid = new_uuid

    async def prepare(
        self,
        session: Session,
        *,
        system_prompt: str | None = None,
    ) -> Session:
        with without_session_log():
            return await self._prepare(session, system_prompt=system_prompt)

    async def _prepare(self, session: Session, *, system_prompt: str | None) -> Session:
        short_term = _short_term_messages(session)
        available_input = self._chat_context_window - self._chat_max_output
        effective_system_prompt = (
            self._chat_system_prompt if system_prompt is None else system_prompt
        )
        system_tokens = estimate_input_tokens(
            RuntimeStatusInput(
                system_prompt=effective_system_prompt,
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
        token_triggered = (
            estimate_input_tokens(
                self._chat_input(session, system_prompt=effective_system_prompt)
            )
            >= available_input
        )
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
            system_prompt=conversation_summary_prompt(),
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
        session.update_metadata(usage_delta={"model_calls": 1, **response.usage.to_dict()})
        if not response.message.content:
            raise ModelCallError(
                ErrorInfo(
                    code="model_failed",
                    message="The memory model returned an empty Conversation Summary.",
                )
            )
        try:
            new_last_consolidated = session.last_consolidated + cutoff
            await self._summaries.append(
                content=response.message.content,
                timestamp=self._persisted_now(),
            )
            session.last_consolidated = new_last_consolidated
            return session
        except (OSError, UnicodeError, ValueError) as error:
            raise ModelCallError(
                ErrorInfo(
                    code="persistence_error",
                    message="Conversation Summary could not be persisted.",
                )
            ) from error

    def _chat_input(
        self,
        session: Session,
        *,
        system_prompt: str,
    ) -> RuntimeStatusInput:
        short_term = _short_term_messages(session)
        current_user_index = _last_user_index(short_term)
        retained: list[str] = []
        for index, message in enumerate(short_term):
            model_message = model_message_from_session(message)
            if model_message is None:
                continue
            if index == current_user_index and message.get("role") == "user":
                content = message.get("content")
                timestamp = message.get("timestamp")
                if not isinstance(content, str) or not isinstance(timestamp, str):
                    raise TypeError("Session user message is malformed")
                model_message = UserModelMessage(
                    content=current_user_input(
                        content=content,
                        current_time=datetime.fromisoformat(timestamp),
                        session_id=session.session_id,
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
            json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in self._tools
        )
        return RuntimeStatusInput(
            system_prompt=system_prompt,
            retained_messages=tuple(retained),
            tool_definitions=tool_definitions,
            runtime_context="",
        )

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)


def _short_term_messages(session: Session) -> list[dict[str, Any]]:
    return session.messages[session.last_consolidated :]


def _aligned_cutoff(messages: list[dict[str, Any]], initial: int) -> int:
    for index in range(initial, len(messages)):
        if messages[index].get("role") == "user":
            return index
    for index in range(initial - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return 0


def _token_cutoff(messages: list[dict[str, Any]], input_budget: int) -> int:
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


def _last_user_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return len(messages)


def _summary_input(messages: list[dict[str, Any]]) -> str:
    records = [
        model_message.to_dict()
        for message in messages
        if (model_message := model_message_from_session(message)) is not None
    ]
    return conversation_summary_input(
        messages=json.dumps(
            records,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
