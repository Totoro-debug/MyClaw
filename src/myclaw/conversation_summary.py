"""Synchronous Conversation Summary selection and persistence."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from myclaw.agent_home import AgentHome
from myclaw.contracts import (
    ConversationSession,
    ErrorInfo,
    MetadataUpdate,
    ModelCallError,
    ModelProvider,
    ModelRequest,
    ReasoningEffort,
    SessionMessage,
    SessionStore,
    SummaryEntry,
    SummaryStore,
    ToolDefinition,
    UserModelMessage,
    UserSessionMessage,
)
from myclaw.conversation import model_message_from_session
from myclaw.management import RuntimeStatusInput, estimate_input_tokens
from myclaw.prompts import current_user_input

_SUMMARY_SYSTEM_PROMPT = """Summarize the provided earlier conversation messages.
Preserve decisions, user intent, important facts, and unresolved work concisely."""


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

    def __init__(self, agent_home: AgentHome) -> None:
        self.path = agent_home.path / "memory" / "summary.jsonl"
        self._lock = asyncio.Lock()

    async def append(self, content: str, timestamp: datetime) -> SummaryEntry:
        async with self._lock:
            next_index = 1
            if self.path.exists():
                lines = self.path.read_text(encoding="utf-8").splitlines()
                if lines:
                    record = json.loads(lines[-1])
                    if not isinstance(record, dict):
                        raise ValueError("summary record must be an object")
                    previous_index = record.get("index")
                    if isinstance(previous_index, bool) or not isinstance(previous_index, int):
                        raise ValueError("summary index must be an integer")
                    next_index = previous_index + 1
            entry = SummaryEntry(index=next_index, timestamp=timestamp, content=content)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = entry.to_json_line().encode("utf-8")
            with self.path.open("ab") as stream:
                written = stream.write(encoded)
                if written != len(encoded):
                    raise OSError("summary append did not write the complete JSONL record")
                stream.flush()
                os.fsync(stream.fileno())
            return entry

    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]:
        raise NotImplementedError


class ConversationSummaryManager:
    """Compress eligible early Session messages before a chat model call."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        sessions: SessionStore,
        summaries: SummaryStore,
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
            await self._summaries.append(response.message.content, self._persisted_now())
            new_cursor = session.metadata.consolidation_cursor + cutoff
            await self._sessions.update_metadata(
                session.metadata.id,
                MetadataUpdate(
                    consolidation_cursor=new_cursor,
                ),
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
