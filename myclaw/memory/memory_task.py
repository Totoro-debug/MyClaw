"""Manual Memory Task orchestration and Agent Home persistence."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import Annotated, Protocol, runtime_checkable
from uuid import uuid4

from myclaw.agent.prompts import memory_task_input, memory_task_prompt
from myclaw.config.agent_home import AgentHome
from myclaw.errors import ErrorInfo
from myclaw.memory.records import SummaryEntry
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ReasoningEffort,
    ToolModelMessage,
    UserModelMessage,
)
from myclaw.runtime_log import log_sanitized_exception
from myclaw.tools.base import BaseTool
from myclaw.tools.errors import ToolError
from myclaw.tools.schema import ToolParam
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.utils.atomic_files import atomic_replace_text
from myclaw.utils.validation import require_nonnegative_int

logger = logging.getLogger(__name__)


@runtime_checkable
class MemoryStore(Protocol):
    """Persist Long-term Memory and its Summary Cursor."""

    async def read_long_term(self) -> str: ...

    async def replace_long_term(self, content: str) -> None: ...

    async def read_summary_cursor(self) -> int: ...

    async def write_summary_cursor(self, index: int) -> None: ...


class _SummaryReader(Protocol):
    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]: ...


@dataclass(frozen=True, slots=True)
class MemoryTaskResult:
    """Observable summary returned by a manual Memory Task run."""

    status: str
    processed_count: int
    memory_updated: bool
    cursor: int
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        require_nonnegative_int(self.processed_count, field="processed_count")
        require_nonnegative_int(self.cursor, field="cursor")
        if not isinstance(self.memory_updated, bool):
            msg = "memory_updated must be a boolean"
            raise ValueError(msg)


class MemoryPathDeniedError(PermissionError):
    """Raised when Long-term Memory aliases or identifies another file kind."""


class MemoryReadFileTool(BaseTool):
    """Read the injected Long-term Memory through its dedicated store."""

    name = "read_file"
    description = "Read the current Long-term Memory UTF-8 text."
    required = ("path",)

    path: Annotated[str, ToolParam(description="Exact Long-term Memory file path.")]
    offset: Annotated[int, ToolParam(description="Zero-based first line.", minimum=0)] = 0
    limit: Annotated[
        int,
        ToolParam(description="Maximum lines to return.", minimum=1, maximum=10000),
    ] = 2000

    def __init__(self, *, memory: MemoryStore, long_term_path: Path) -> None:
        self._memory = memory
        self._long_term_path = long_term_path

    async def execute(self, *, path: str, offset: int, limit: int) -> str:
        _require_long_term_path(path, expected=self._long_term_path)
        try:
            content = await self._memory.read_long_term()
        except MemoryPathDeniedError as error:
            raise ToolError("Long-term Memory must be a regular Agent Home file.") from error
        except (OSError, UnicodeError, ValueError) as error:
            raise ToolError("Long-term Memory could not be read.") from error
        return "\n".join(content.splitlines()[offset : offset + limit])


class MemoryEditFileTool(BaseTool):
    """Edit only the injected Long-term Memory through its dedicated store."""

    name = "edit_file"
    description = "Replace exact text only in the current Long-term Memory file."
    required = ("path", "old_text", "new_text")

    path: Annotated[str, ToolParam(description="Exact Long-term Memory file path.")]
    old_text: Annotated[str, ToolParam(description="Exact text to replace.", min_length=1)]
    new_text: Annotated[str, ToolParam(description="Replacement text.")]
    replace_all: Annotated[bool, ToolParam(description="Replace every exact match.")] = False

    def __init__(self, *, memory: MemoryStore, long_term_path: Path) -> None:
        self._memory = memory
        self._long_term_path = long_term_path

    async def execute(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str:
        _require_long_term_path(path, expected=self._long_term_path)
        try:
            content = await self._memory.read_long_term()
        except MemoryPathDeniedError as error:
            raise ToolError("Long-term Memory must be a regular Agent Home file.") from error
        except (OSError, UnicodeError, ValueError) as error:
            raise ToolError("Long-term Memory could not be read.") from error
        match_count = content.count(old_text)
        if match_count == 0 or (not replace_all and match_count != 1):
            raise ToolError("The requested Long-term Memory text did not match precisely.")
        replacement = (
            content.replace(old_text, new_text)
            if replace_all
            else content.replace(old_text, new_text, 1)
        )
        try:
            await self._memory.replace_long_term(replacement)
        except MemoryPathDeniedError as error:
            raise ToolError("Long-term Memory must be a regular Agent Home file.") from error
        except (OSError, UnicodeError, ValueError) as error:
            raise ToolError("Long-term Memory could not be updated.") from error
        return "Long-term Memory updated."


def _require_long_term_path(path: str, *, expected: Path) -> None:
    if Path(path) != expected:
        raise ToolError("Memory Tasks may access only Long-term Memory.")


class FileMemoryStore:
    """Persist Long-term Memory and its Summary Cursor under Agent Home."""

    def __init__(
        self,
        agent_home: AgentHome,
        *,
        replace_text: Callable[[Path, str], None] = atomic_replace_text,
    ) -> None:
        self._agent_home_root = agent_home.path.resolve(strict=True)
        self._long_term_path = agent_home.path / "memory" / "memory.md"
        self._cursor_path = agent_home.path / "memory" / ".cursor"
        self._replace_text = replace_text
        self._lock = asyncio.Lock()

    async def read_long_term(self) -> str:
        async with self._lock:
            self._require_private_long_term_file()
            return self._long_term_path.read_text(encoding="utf-8")

    async def replace_long_term(self, content: str) -> None:
        async with self._lock:
            self._require_private_long_term_file()
            self._replace_text(self._long_term_path, content)

    def _require_private_long_term_file(self) -> None:
        self._require_private_regular_file(self._long_term_path)

    def _require_private_regular_file(self, path: Path) -> None:
        resolved = path.resolve(strict=True)
        file_status = path.lstat()
        if (
            not resolved.is_relative_to(self._agent_home_root)
            or not S_ISREG(file_status.st_mode)
            or file_status.st_nlink != 1
        ):
            raise MemoryPathDeniedError("Agent Home state must be an unaliased regular file")

    def _require_private_cursor_location(self) -> None:
        resolved_parent = self._cursor_path.parent.resolve(strict=True)
        if not resolved_parent.is_relative_to(self._agent_home_root):
            raise MemoryPathDeniedError("Summary Cursor must remain inside Agent Home")
        if self._cursor_path.exists() or self._cursor_path.is_symlink():
            self._require_private_regular_file(self._cursor_path)

    async def read_summary_cursor(self) -> int:
        async with self._lock:
            self._require_private_cursor_location()
            if not self._cursor_path.exists():
                return 0
            content = self._cursor_path.read_bytes()
            digits = content[:-1]
            if not content.endswith(b"\n") or not digits.isdigit():
                raise ValueError("Summary Cursor must use canonical ASCII decimal text")
            index = int(digits)
            if digits != str(index).encode("ascii"):
                raise ValueError("Summary Cursor must use canonical ASCII decimal text")
            return index

    async def write_summary_cursor(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Summary Cursor must be a nonnegative integer")
        async with self._lock:
            self._require_private_cursor_location()
            self._replace_text(self._cursor_path, f"{index}\n")


@dataclass(frozen=True, slots=True)
class MemoryTaskModelSettings:
    """Resolved provider-neutral settings for a Memory Task model call."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int


class MemoryManager:
    """Process pending Conversation Summaries into Long-term Memory."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        summaries: _SummaryReader,
        memory: MemoryStore,
        long_term_path: Path,
        settings: MemoryTaskModelSettings,
        batch_size: int,
    ) -> None:
        self._provider = provider
        self._summaries = summaries
        self._memory = memory
        self._long_term_path = long_term_path
        self._settings = settings
        self._batch_size = batch_size
        self._failure_diagnostic: Exception | None = None
        self._tools = ToolGateway(
            owns_terminal_failures=False,
            on_terminal_failure=self._capture_terminal_failure,
        )
        self._tools.register_tools(
            (
                MemoryReadFileTool(memory=memory, long_term_path=long_term_path),
                MemoryEditFileTool(memory=memory, long_term_path=long_term_path),
            )
        )
        self._running = False
        self._running_cursor = 0

    async def run_manual(self) -> MemoryTaskResult:
        if self._running:
            return MemoryTaskResult(
                status="Memory Task is already running.",
                processed_count=0,
                memory_updated=False,
                cursor=self._running_cursor,
                error=ErrorInfo(
                    code="memory_task_running",
                    message="A Memory Task is already running.",
                ),
            )
        self._running = True
        self._running_cursor = 0
        self._failure_diagnostic = None
        try:
            result = await self._run_once()
            self._log_failure(result)
            return result
        finally:
            self._running = False

    async def run_periodic(self) -> MemoryTaskResult | None:
        if self._running:
            return None
        self._running = True
        self._running_cursor = 0
        self._failure_diagnostic = None
        try:
            result = await self._run_once()
            self._log_failure(result)
            return result
        finally:
            self._running = False

    async def _run_once(self) -> MemoryTaskResult:
        try:
            cursor = await self._memory.read_summary_cursor()
        except (OSError, UnicodeError, ValueError) as error:
            self._capture_terminal_failure(error)
            return _state_read_failure(cursor=0)
        self._running_cursor = cursor
        try:
            pending = await self._summaries.after(cursor, self._batch_size)
        except (OSError, UnicodeError, ValueError) as error:
            self._capture_terminal_failure(error)
            return _state_read_failure(cursor=cursor)
        if not pending:
            return MemoryTaskResult(
                status="No pending summaries",
                processed_count=0,
                memory_updated=False,
                cursor=cursor,
            )
        messages: list[ModelMessage] = [
            UserModelMessage(
                content=memory_task_input(cursor=cursor, summaries=pending),
            )
        ]
        memory_updated = False
        while True:
            request = ModelRequest(
                request_id=uuid4(),
                route="memory",
                system_prompt=memory_task_prompt(long_term_path=self._long_term_path),
                messages=tuple(messages),
                tools=self._tools.schemas,
                stream=False,
                model=self._settings.model,
                max_output=self._settings.max_output,
                temperature=self._settings.temperature,
                reasoning_effort=self._settings.reasoning_effort,
                timeout_seconds=self._settings.timeout_seconds,
            )
            try:
                response = await self._provider.complete(request)
            except ModelCallError as failure:
                self._capture_terminal_failure(failure)
                return MemoryTaskResult(
                    status="Memory Task failed.",
                    processed_count=0,
                    memory_updated=memory_updated,
                    cursor=cursor,
                    error=failure.error,
                )
            messages.append(response.message)
            if not response.message.tool_calls:
                break
            for tool_call in response.message.tool_calls:
                tool_result = await self._tools.call(tool_call)
                messages.append(
                    ToolModelMessage(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=tool_result.content,
                    )
                )
                if tool_result.status != "success":
                    return MemoryTaskResult(
                        status="Memory Task failed.",
                        processed_count=0,
                        memory_updated=memory_updated,
                        cursor=cursor,
                        error=ErrorInfo(code="tool_failed", message=tool_result.content),
                    )
                if tool_call.name == "edit_file":
                    memory_updated = True
        new_cursor = pending[-1].index
        try:
            await self._memory.write_summary_cursor(new_cursor)
        except (OSError, UnicodeError, ValueError) as error:
            self._capture_terminal_failure(error)
            return MemoryTaskResult(
                status="Memory Task failed.",
                processed_count=0,
                memory_updated=memory_updated,
                cursor=cursor,
                error=ErrorInfo(
                    code="persistence_error",
                    message="Summary Cursor could not be updated.",
                ),
            )
        count = len(pending)
        noun = "summary" if count == 1 else "summaries"
        outcome = "updated" if memory_updated else "unchanged"
        return MemoryTaskResult(
            status=f"Processed {count} {noun}; Long-term Memory {outcome}.",
            processed_count=count,
            memory_updated=memory_updated,
            cursor=new_cursor,
        )

    def _capture_terminal_failure(self, error: Exception) -> None:
        self._failure_diagnostic = error

    def _log_failure(self, result: MemoryTaskResult) -> None:
        if result.error is None:
            return
        message = f"Memory Task failed code={result.error.code}"
        if self._failure_diagnostic is None:
            logger.error(message)
            return
        log_sanitized_exception(
            logger,
            logging.ERROR,
            message,
            self._failure_diagnostic,
        )


def _state_read_failure(*, cursor: int) -> MemoryTaskResult:
    return MemoryTaskResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=cursor,
        error=ErrorInfo(
            code="persistence_error",
            message="Memory Task state could not be read.",
        ),
    )
