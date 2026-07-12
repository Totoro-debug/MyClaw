"""Manual Memory Task orchestration and Agent Home persistence."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker

from myclaw.agent_home import AgentHome
from myclaw.atomic_files import atomic_replace_text
from myclaw.contracts import (
    ErrorCode,
    ErrorInfo,
    MemoryStore,
    MemoryTaskResult,
    ModelCallError,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelToolCall,
    ReasoningEffort,
    SummaryStore,
    ToolDefinition,
    ToolModelMessage,
    ToolResult,
    UserModelMessage,
)
from myclaw.prompts import memory_task_input, memory_task_prompt

_MEMORY_READ_DEFINITION = ToolDefinition(
    name="read_file",
    description="Read the current Long-term Memory UTF-8 text.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "default": 2000,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)

_MEMORY_EDIT_DEFINITION = ToolDefinition(
    name="edit_file",
    description="Replace exact text only in the current Long-term Memory file.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "old_text": {"type": "string", "minLength": 1},
            "new_text": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
)


class MemoryPathDeniedError(PermissionError):
    """Raised when Long-term Memory aliases or identifies another file kind."""


class RestrictedMemoryToolGateway:
    """Expose only exact Long-term Memory reads and edits to a Memory Task."""

    def __init__(self, *, memory: MemoryStore, long_term_path: Path) -> None:
        self._memory = memory
        self._long_term_path = long_term_path
        self._definitions = (_MEMORY_READ_DEFINITION, _MEMORY_EDIT_DEFINITION)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._definitions

    async def execute(self, tool_call: ModelToolCall) -> ToolResult:
        definition = next(
            (item for item in self._definitions if item.name == tool_call.name),
            None,
        )
        if definition is None:
            return _tool_error(
                tool_call,
                code="tool_not_found",
                message="The requested tool is not available to Memory Tasks.",
            )
        if not Draft202012Validator(
            definition.input_schema,
            format_checker=FormatChecker(),
        ).is_valid(tool_call.arguments):
            return _tool_error(
                tool_call,
                code="tool_invalid_arguments",
                message=f"Invalid arguments for {tool_call.name}.",
            )
        requested = tool_call.arguments.get("path")
        if not isinstance(requested, str) or Path(requested) != self._long_term_path:
            return _tool_error(
                tool_call,
                code="tool_denied",
                message="Memory Tasks may access only Long-term Memory.",
            )
        if tool_call.name == "read_file":
            return await self._read(tool_call)
        return await self._edit(tool_call)

    async def _read(self, tool_call: ModelToolCall) -> ToolResult:
        offset = tool_call.arguments.get("offset", 0)
        limit = tool_call.arguments.get("limit", 2000)
        if not isinstance(offset, int) or not isinstance(limit, int):
            raise AssertionError("validated read_file integer arguments changed type")
        try:
            content = await self._memory.read_long_term()
        except MemoryPathDeniedError:
            return _tool_error(
                tool_call,
                code="tool_denied",
                message="Long-term Memory must be a regular Agent Home file.",
            )
        except (OSError, UnicodeError, ValueError):
            return _tool_error(
                tool_call,
                code="persistence_error",
                message="Long-term Memory could not be read.",
            )
        return _tool_success(tool_call, "\n".join(content.splitlines()[offset : offset + limit]))

    async def _edit(self, tool_call: ModelToolCall) -> ToolResult:
        old_text = tool_call.arguments.get("old_text")
        new_text = tool_call.arguments.get("new_text")
        replace_all = tool_call.arguments.get("replace_all", False)
        if (
            not isinstance(old_text, str)
            or not isinstance(new_text, str)
            or not isinstance(replace_all, bool)
        ):
            raise AssertionError("validated edit_file arguments changed type")
        try:
            content = await self._memory.read_long_term()
        except MemoryPathDeniedError:
            return _tool_error(
                tool_call,
                code="tool_denied",
                message="Long-term Memory must be a regular Agent Home file.",
            )
        except (OSError, UnicodeError, ValueError):
            return _tool_error(
                tool_call,
                code="persistence_error",
                message="Long-term Memory could not be read.",
            )
        match_count = content.count(old_text)
        if match_count == 0 or (not replace_all and match_count != 1):
            return _tool_error(
                tool_call,
                code="tool_invalid_arguments",
                message="The requested Long-term Memory text did not match precisely.",
            )
        replacement = (
            content.replace(old_text, new_text)
            if replace_all
            else content.replace(old_text, new_text, 1)
        )
        try:
            await self._memory.replace_long_term(replacement)
        except MemoryPathDeniedError:
            return _tool_error(
                tool_call,
                code="tool_denied",
                message="Long-term Memory must be a regular Agent Home file.",
            )
        except (OSError, UnicodeError, ValueError):
            return _tool_error(
                tool_call,
                code="persistence_error",
                message="Long-term Memory could not be updated.",
            )
        return _tool_success(tool_call, "Long-term Memory updated.")


def _tool_success(tool_call: ModelToolCall, content: str) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="success",
        content=content,
        error=None,
        artifact=None,
    )


def _tool_error(
    tool_call: ModelToolCall,
    *,
    code: ErrorCode,
    message: str,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="error",
        content=message,
        error=ErrorInfo(code=code, message=message),
        artifact=None,
    )


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
        summaries: SummaryStore,
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
        self._tools = RestrictedMemoryToolGateway(
            memory=memory,
            long_term_path=long_term_path,
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
        try:
            return await self._run_once()
        finally:
            self._running = False

    async def run_periodic(self) -> MemoryTaskResult | None:
        if self._running:
            return None
        self._running = True
        self._running_cursor = 0
        try:
            return await self._run_once()
        finally:
            self._running = False

    async def _run_once(self) -> MemoryTaskResult:
        try:
            cursor = await self._memory.read_summary_cursor()
        except (OSError, UnicodeError, ValueError):
            return _state_read_failure(cursor=0)
        self._running_cursor = cursor
        try:
            pending = await self._summaries.after(cursor, self._batch_size)
        except (OSError, UnicodeError, ValueError):
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
                tools=self._tools.definitions,
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
                tool_result = await self._tools.execute(tool_call)
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
                        error=tool_result.error,
                    )
                if tool_call.name == "edit_file":
                    memory_updated = True
        new_cursor = pending[-1].index
        try:
            await self._memory.write_summary_cursor(new_cursor)
        except (OSError, UnicodeError, ValueError):
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
