"""Manual Memory Task orchestration and persistent-state adapters."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol

from loguru import logger

from myclaw.agent.prompts import memory_task_input, memory_task_prompt
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.logging.session import without_session_log
from myclaw.memory.records import SummaryEntry
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    ModelMessages,
    ModelResponse,
    ModelRoute,
)
from myclaw.tools.base import BaseTool, OpenAIToolSchema, ToolError, ToolParam
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.validation import require_nonnegative_int


class RuntimeMemory:
    """Hold the Runtime-local Long-term Memory snapshot."""

    def __init__(self, content: str) -> None:
        self._content = content

    def snapshot(self) -> str:
        """Return the immutable text captured by the caller."""
        return self._content

    def replace(self, content: str) -> None:
        """Publish a later snapshot without a persistence failure path."""
        self._content = content


class MemoryStore(Protocol):
    """Persist Long-term Memory and its Summary Cursor."""

    async def read_long_term(self) -> str: ...

    async def replace_long_term(self, content: str) -> None: ...

    async def read_summary_cursor(self) -> int: ...

    async def write_summary_cursor(self, index: int) -> None: ...


class _SummaryReader(Protocol):
    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]: ...


class MemoryTaskModelRouter(Protocol):
    """The direct Router seam used for the specialized Memory Task call."""

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
    ) -> ModelResponse: ...


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
            raise ToolError("Long-term Memory must be a regular Workspace State file.") from error
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

    def __init__(
        self,
        *,
        memory: MemoryStore,
        long_term_path: Path,
        runtime_memory: RuntimeMemory | None = None,
    ) -> None:
        self._memory = memory
        self._long_term_path = long_term_path
        self._runtime_memory = runtime_memory

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
            raise ToolError("Long-term Memory must be a regular Workspace State file.") from error
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
            raise ToolError("Long-term Memory must be a regular Workspace State file.") from error
        except (OSError, UnicodeError, ValueError) as error:
            raise ToolError("Long-term Memory could not be updated.") from error
        if self._runtime_memory is not None:
            self._runtime_memory.replace(replacement)
        return "Long-term Memory updated."


def _require_long_term_path(path: str, *, expected: Path) -> None:
    if Path(path) != expected:
        raise ToolError("Memory Tasks may access only Long-term Memory.")


class WorkspaceFileMemoryStore:
    """Persist Long-term Memory and its Summary Cursor in one Workspace State."""

    def __init__(
        self,
        workspace_state: WorkspaceState,
        *,
        replace_text: Callable[[Path, str], None] = HOST_FILESYSTEM.atomic_replace_text,
    ) -> None:
        self.workspace_state = workspace_state
        self._state_root = workspace_state.path.resolve(strict=True)
        self._memory_directory = workspace_state.memory_directory
        self._long_term_path = workspace_state.long_term_memory_path
        self._cursor_path = workspace_state.memory_directory / ".cursor"
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
        self._require_private_memory_directory()
        try:
            HOST_FILESYSTEM.require_owned_regular_file(path, within=self._state_root)
        except PermissionError as error:
            raise MemoryPathDeniedError(
                "Workspace State must be an unaliased regular file"
            ) from error

    def _require_private_memory_directory(self) -> None:
        try:
            HOST_FILESYSTEM.require_owned_directory(self._memory_directory, within=self._state_root)
        except PermissionError as error:
            raise MemoryPathDeniedError(
                "Workspace State Memory directory must remain unaliased"
            ) from error

    def _require_private_cursor_location(self) -> None:
        self._require_private_memory_directory()
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


class MemoryManager:
    """Process pending Conversation Summaries into Long-term Memory."""

    def __init__(
        self,
        *,
        router: MemoryTaskModelRouter,
        summaries: _SummaryReader,
        memory: MemoryStore,
        long_term_path: Path,
        batch_size: int,
        runtime_memory: RuntimeMemory | None = None,
    ) -> None:
        self._router = router
        self._summaries = summaries
        self._memory = memory
        self._long_term_path = long_term_path
        self._batch_size = batch_size
        self._failure_diagnostic: Exception | None = None
        self._tools = ToolGateway._for_memory(
            (
                MemoryReadFileTool(memory=memory, long_term_path=long_term_path),
                MemoryEditFileTool(
                    memory=memory,
                    long_term_path=long_term_path,
                    runtime_memory=runtime_memory,
                ),
            ),
            on_failure=self._capture_terminal_failure,
        )
        self._running = False
        self._running_cursor = 0
        self._aborted = False
        self._task: asyncio.Task[MemoryTaskResult | None] | None = None

    def abort(self) -> None:
        """Synchronously cancel an in-flight Memory Task for an abandoned generation."""
        if self._aborted:
            return
        self._aborted = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()

    async def run_manual(self) -> MemoryTaskResult:
        with without_session_log():
            if self._aborted:
                raise RuntimeError("Memory Task is no longer active")
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
            task = asyncio.current_task()
            self._task = task
            try:
                result = await self._run_once()
                self._log_failure(result)
                return result
            finally:
                self._running = False
                if self._task is task:
                    self._task = None

    async def run_periodic(self) -> MemoryTaskResult | None:
        with without_session_log():
            if self._aborted:
                raise RuntimeError("Memory Task is no longer active")
            if self._running:
                return None
            self._running = True
            self._running_cursor = 0
            self._failure_diagnostic = None
            task = asyncio.current_task()
            self._task = task
            try:
                result = await self._run_once()
                self._log_failure(result)
                return result
            finally:
                self._running = False
                if self._task is task:
                    self._task = None

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
        new_cursor = pending[-1].index
        try:
            await self._memory.write_summary_cursor(new_cursor)
        except (OSError, UnicodeError, ValueError) as error:
            self._capture_terminal_failure(error)
            return MemoryTaskResult(
                status="Memory Task failed.",
                processed_count=0,
                memory_updated=False,
                cursor=cursor,
                error=ErrorInfo(
                    code="persistence_error",
                    message="Summary Cursor could not be updated.",
                ),
            )
        self._running_cursor = new_cursor
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": memory_task_prompt(long_term_path=self._long_term_path),
            },
            {
                "role": "user",
                "content": memory_task_input(cursor=cursor, summaries=pending),
            },
        ]
        memory_updated = False
        while True:
            try:
                response = await self._router.complete(
                    "memory",
                    messages=messages,
                    tools=self._tools.schemas,
                )
            except ModelCallError as failure:
                self._capture_terminal_failure(failure)
                return MemoryTaskResult(
                    status="Memory Task failed.",
                    processed_count=0,
                    memory_updated=memory_updated,
                    cursor=new_cursor,
                    error=failure.error,
                )
            messages.append(response.message.to_dict())
            if not response.message.tool_calls:
                break
            for tool_call in response.message.tool_calls:
                tool_result = await self._tools.call(tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": tool_result.content,
                    }
                )
                if tool_result.status != "success":
                    return MemoryTaskResult(
                        status="Memory Task failed.",
                        processed_count=0,
                        memory_updated=memory_updated,
                        cursor=new_cursor,
                        error=ErrorInfo(code="tool_failed", message=tool_result.content),
                    )
                if tool_call.name == "edit_file":
                    memory_updated = True
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
        if self._failure_diagnostic is None:
            logger.error("Memory Task failed code={}", result.error.code)
            return
        diagnostic = self._failure_diagnostic
        logger.opt(
            exception=(type(diagnostic), diagnostic, diagnostic.__traceback__),
        ).error("Memory Task failed code={}", result.error.code)


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
