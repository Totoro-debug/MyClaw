"""Synchronous Conversation Summary selection and persistence."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from myclaw.agent.prompts import conversation_summary_input, conversation_summary_prompt
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.logging.session import without_session_log
from myclaw.management.service import RuntimeStatusInput, estimate_input_tokens
from myclaw.memory.records import SummaryEntry
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import ModelMessages, ModelResponse, ModelRoute
from myclaw.session.projection import project_session_message
from myclaw.session.session import Session
from myclaw.tools.base import OpenAIToolSchema
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.time import format_rfc3339_milliseconds

type AtomicReplaceBytes = Callable[[Path, bytes], None]
type SummaryProjection = Callable[
    [Sequence[dict[str, Any]]],
    list[dict[str, Any]],
]


class SummaryModelRouter(Protocol):
    """The direct Router seam used for the specialized memory model call."""

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
    ) -> ModelResponse: ...


class SummaryStore(Protocol):
    """Append and read the ordered Conversation Summary stream."""

    async def append(self, content: str, timestamp: datetime) -> SummaryEntry: ...

    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]: ...


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
    """Compress eligible early Session messages before an Agent Run model call."""

    def __init__(
        self,
        *,
        provider: SummaryModelRouter,
        summaries: SummaryStore,
        route_context_window: int,
        route_max_output: int,
        consolidation_message_threshold: int,
        tools: tuple[OpenAIToolSchema, ...],
        now: Callable[[], datetime],
        project_messages: SummaryProjection,
    ) -> None:
        self._provider = provider
        self._summaries = summaries
        self._route_context_window = route_context_window
        self._route_max_output = route_max_output
        self._message_threshold = consolidation_message_threshold
        self._tools = tools
        self._now = now
        self._project_messages = project_messages

    async def prepare(
        self,
        session: Session,
        *,
        current_user: dict[str, Any] | None = None,
        continuation: Sequence[dict[str, Any]] = (),
    ) -> Session:
        with without_session_log():
            return await self._prepare(
                session,
                current_user=current_user,
                continuation=continuation,
            )

    async def _prepare(
        self,
        session: Session,
        *,
        current_user: dict[str, Any] | None,
        continuation: Sequence[dict[str, Any]],
    ) -> Session:
        short_term = _short_term_messages(
            session,
            current_user=current_user,
            continuation=continuation,
        )
        current_user_index = _last_user_index(short_term)
        complete_messages = self._project_messages(short_term)
        available_input = self._route_context_window - self._route_max_output
        system_tokens = _estimate_messages(complete_messages[:1])
        if system_tokens > available_input:
            raise ModelCallError(
                ErrorInfo(
                    code="memory_context_too_large",
                    message="Long-term Memory and system context exceed the chat input budget.",
                )
            )
        if current_user_index < len(short_term):
            non_summarizable_messages = self._project_messages(
                short_term[current_user_index:]
            )
            if _estimate_messages(non_summarizable_messages) >= available_input:
                raise ModelCallError(
                    ErrorInfo(
                        code="model_context_overflow",
                        message="System context and current input exceed the model input budget.",
                    )
                )
        token_triggered = (
            _estimate_messages(complete_messages, tools=self._tools) >= available_input
        )
        message_triggered = len(short_term) >= self._message_threshold
        if not token_triggered and not message_triggered:
            return session
        initial_cutoff = 0
        if token_triggered:
            initial_cutoff = _token_cutoff(
                short_term,
                current_user_index,
                available_input,
                self._project_messages,
            )
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
        response = await self._provider.complete(
            "memory",
            messages=[
                {"role": "system", "content": conversation_summary_prompt()},
                {"role": "user", "content": _summary_input(selected)},
            ],
            tools=(),
        )
        session.update_metadata(usage_delta={"model_calls": 1, **response.usage.to_dict()})
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

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)


def _short_term_messages(
    session: Session,
    *,
    current_user: dict[str, Any] | None = None,
    continuation: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    messages = list(session.messages[session.last_consolidated :])
    if current_user is not None:
        messages.append(current_user)
    messages.extend(continuation)
    return messages


def _aligned_cutoff(messages: list[dict[str, Any]], initial: int) -> int:
    for index in range(initial, len(messages)):
        if messages[index].get("role") == "user":
            return index
    for index in range(initial - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return 0


def _token_cutoff(
    messages: Sequence[dict[str, Any]],
    current_user_index: int,
    input_budget: int,
    project_messages: SummaryProjection,
) -> int:
    target_bytes = input_budget // 2 * 4
    if current_user_index == len(messages):
        return len(messages)
    current_user = messages[current_user_index]
    continuation = messages[current_user_index + 1 :]
    for index in range(current_user_index):
        projected = project_messages([*messages[: index + 1], current_user, *continuation])
        selected_bytes = _projected_history_bytes(projected)
        if selected_bytes >= target_bytes:
            return index + 1
    return current_user_index


def _estimate_messages(
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[OpenAIToolSchema] = (),
) -> int:
    system_prompt = ""
    retained = messages
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if not isinstance(content, str):
            raise TypeError("Projected system message content must be a string")
        system_prompt = content
        retained = messages[1:]
    retained_messages = tuple(
        json.dumps(message, ensure_ascii=False, separators=(",", ":")) for message in retained
    )
    tool_definitions = tuple(
        json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools
    )
    return estimate_input_tokens(
        RuntimeStatusInput(
            system_prompt=system_prompt,
            retained_messages=retained_messages,
            tool_definitions=tool_definitions,
            runtime_context="",
        )
    )


def _projected_history_bytes(messages: Sequence[dict[str, Any]]) -> int:
    if len(messages) <= 2:
        return 0
    return sum(
        len(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        for message in messages[1:-1]
    )


def _last_user_index(messages: Sequence[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return len(messages)


def _summary_input(messages: list[dict[str, Any]]) -> str:
    records = [
        projected
        for message in messages
        if (projected := project_session_message(message)) is not None
    ]
    return conversation_summary_input(
        messages=json.dumps(
            records,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
