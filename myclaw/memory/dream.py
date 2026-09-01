"""Dedicated Dream execution for semantic Long-term Memory updates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from loguru import logger

from myclaw.agent.runner import (
    AgentRunner,
    AgentRunnerMemoryRouter,
)
from myclaw.errors import ErrorInfo
from myclaw.logging.session import without_session_log
from myclaw.memory.manager import (
    MemoryEditMismatchError,
    MemoryEditReadError,
    MemoryEditWriteError,
    MemoryManager,
    MemoryPathDeniedError,
    SummaryClaimError,
)
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import ModelContinuation, ModelMessages, ModelResponse
from myclaw.templates import render_template
from myclaw.tools.base import BaseTool, OpenAIToolSchema, ToolError, ToolParam
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.utils.validation import require_nonnegative_int

_MEMORY_JSON_TRANSLATION = str.maketrans({"`": r"\u0060"})


@dataclass(frozen=True, slots=True)
class DreamResult:
    """Observable summary returned by one Dream execution."""

    status: str
    processed_count: int
    memory_updated: bool
    cursor: int
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        require_nonnegative_int(self.processed_count, field="processed_count")
        require_nonnegative_int(self.cursor, field="cursor")
        if not isinstance(self.memory_updated, bool):
            raise ValueError("memory_updated must be a boolean")


class DreamReadFileTool(BaseTool):
    """Read only the Long-term Memory target owned by MemoryManager."""

    name = "read_file"
    description = "Read the current Long-term Memory UTF-8 text."
    required = ("path",)

    path: Annotated[str, ToolParam(description="Exact Long-term Memory file path.")]
    offset: Annotated[int, ToolParam(description="Zero-based first line.", minimum=0)] = 0
    limit: Annotated[
        int,
        ToolParam(description="Maximum lines to return.", minimum=1, maximum=10000),
    ] = 2000

    def __init__(self, *, memory_manager: MemoryManager) -> None:
        self._memory_manager = memory_manager

    async def execute(self, *, path: str, offset: int, limit: int) -> str:
        _require_long_term_path(path, expected=self._memory_manager.long_term_path)
        try:
            content = await self._memory_manager.read_long_term()
        except MemoryPathDeniedError as error:
            raise ToolError("Long-term Memory must be a regular Workspace State file.") from error
        except (OSError, UnicodeError, ValueError) as error:
            raise ToolError("Long-term Memory could not be read.") from error
        return "\n".join(content.splitlines()[offset : offset + limit])


class DreamEditFileTool(BaseTool):
    """Edit only the Long-term Memory target owned by MemoryManager."""

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
        memory_manager: MemoryManager,
        on_edit: Callable[[], None] | None = None,
    ) -> None:
        self._memory_manager = memory_manager
        self._on_edit = on_edit

    async def execute(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str:
        _require_long_term_path(path, expected=self._memory_manager.long_term_path)
        try:
            await self._memory_manager.edit_long_term(
                old=old_text,
                new=new_text,
                replace_all=replace_all,
            )
        except MemoryPathDeniedError as error:
            raise ToolError("Long-term Memory must be a regular Workspace State file.") from error
        except MemoryEditMismatchError as error:
            raise ToolError(str(error)) from error
        except MemoryEditReadError as error:
            raise ToolError("Long-term Memory could not be read.") from error
        except MemoryEditWriteError as error:
            raise ToolError("Long-term Memory could not be updated.") from error
        if self._on_edit is not None:
            self._on_edit()
        return "Long-term Memory updated."


class DreamModelRouter(Protocol):
    """The memory-route seam consumed only by Dream."""

    async def complete(
        self,
        route: Literal["memory"],
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
    ) -> ModelResponse: ...


class _DreamModelFailure(Exception):
    def __init__(self, failure: ModelCallError) -> None:
        self.error = failure.error
        super().__init__(failure.error.message)


class _DreamRouter(AgentRunnerMemoryRouter):
    """Adapt the Runner's non-streaming lane to the dedicated memory route."""

    def __init__(
        self,
        router: DreamModelRouter,
        capture_failure: Callable[[Exception], None],
    ) -> None:
        self._router = router
        self._capture_failure = capture_failure

    async def complete(
        self,
        route: Literal["memory"],
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse:
        if route != "memory":
            raise ValueError("Dream Runner must use the memory route")
        del continuation
        try:
            return await self._router.complete(
                route,
                messages=_provider_memory_messages(messages),
                tools=tools,
            )
        except asyncio.CancelledError:
            raise
        except ModelCallError as failure:
            self._capture_failure(failure)
            raise _DreamModelFailure(failure) from failure
        except Exception as error:
            self._capture_failure(error)
            raise


async def _ignore_output(event: object) -> None:
    del event


class Dream:
    """Own one dedicated memory Agent Runner and restricted Tool Gateway."""

    def __init__(
        self,
        *,
        memory_manager: MemoryManager,
        model_router: DreamModelRouter,
        batch_size: int,
        max_iterations: int,
    ) -> None:
        require_nonnegative_int(batch_size, field="batch_size")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._memory_manager = memory_manager
        self._batch_size = batch_size
        self._max_iterations = max_iterations
        self._failure_diagnostic: Exception | None = None
        self._memory_updated = False
        self._tool_gateway = ToolGateway._for_memory(
            (
                DreamReadFileTool(memory_manager=memory_manager),
                DreamEditFileTool(
                    memory_manager=memory_manager,
                    on_edit=self._record_memory_update,
                ),
            ),
            on_failure=self._capture_terminal_failure,
        )
        self._runner = AgentRunner(
            _DreamRouter(model_router, self._capture_terminal_failure)
        )
        self._running = False
        self._running_cursor = 0
        self._task: asyncio.Task[DreamResult | None] | None = None
        self._closed = False
        self._aborted = False

    async def run(self) -> DreamResult:
        with without_session_log():
            if self._closed or self._aborted:
                raise RuntimeError("Dream is no longer active")
            if self._running:
                return DreamResult(
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

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.wait_until_idle()

    async def wait_until_idle(self) -> None:
        """Wait for an active Dream run without cancelling its work."""
        task = self._task
        if task is None or task is asyncio.current_task() or task.done():
            return
        await asyncio.shield(task)

    def abort(self) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._closed = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()

    async def abort_and_wait(self) -> None:
        """Cancel an active Dream and wait until its run releases ownership."""
        self.abort()
        task = self._task
        if task is None or task is asyncio.current_task() or task.done():
            return
        await asyncio.gather(task, return_exceptions=True)

    async def _run_once(self) -> DreamResult:
        self._memory_updated = False
        try:
            claim = await self._memory_manager.claim_summaries(self._batch_size)
        except SummaryClaimError as failure:
            self._capture_terminal_failure(failure.cause)
            if failure.phase == "write_cursor":
                return _cursor_write_failure(cursor=failure.cursor)
            return _state_read_failure(cursor=failure.cursor)
        except (OSError, UnicodeError, ValueError) as error:
            self._capture_terminal_failure(error)
            return _state_read_failure(cursor=self._running_cursor)

        self._running_cursor = claim.cursor
        if not claim.entries:
            return DreamResult(
                status="No pending summaries",
                processed_count=0,
                memory_updated=False,
                cursor=claim.cursor,
            )

        records = "\n".join(
            json.dumps(
                entry.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).translate(_MEMORY_JSON_TRANSLATION)
            for entry in claim.entries
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": render_template(
                    "memory-task-prompt.md",
                    long_term_path=self._memory_manager.long_term_path
                ),
            },
            {
                "role": "user",
                "content": (
                    "## Summary Cursor\n\n"
                    f"{claim.previous_cursor}\n\n"
                    "## Conversation Summaries\n\n"
                    "```jsonl\n"
                    f"{records}\n"
                    "```"
                ),
            },
        ]
        try:
            result = await self._runner.run(
                messages,
                model="memory",
                tool_gateway=self._tool_gateway,
                on_output=_ignore_output,
                confirmation=None,
                externalize_result=None,
                cancel_requested=None,
                max_iterations=self._max_iterations,
                stop_on_tool_error=True,
                propagate_unexpected_errors=True,
                tool_calls_as_tasks=False,
            )
        except _DreamModelFailure as failure:
            return DreamResult(
                status="Memory Task failed.",
                processed_count=0,
                memory_updated=self._memory_updated,
                cursor=claim.cursor,
                error=failure.error,
            )
        memory_updated = self._memory_updated
        if result.finish_reason == "completed":
            count = len(claim.entries)
            noun = "summary" if count == 1 else "summaries"
            outcome = "updated" if memory_updated else "unchanged"
            return DreamResult(
                status=f"Processed {count} {noun}; Long-term Memory {outcome}.",
                processed_count=count,
                memory_updated=memory_updated,
                cursor=claim.cursor,
            )
        return DreamResult(
            status="Memory Task failed.",
            processed_count=0,
            memory_updated=memory_updated,
            cursor=claim.cursor,
            error=result.error
            or ErrorInfo("model_failed", "The model request failed."),
        )

    def _capture_terminal_failure(self, error: Exception) -> None:
        self._failure_diagnostic = error

    def _record_memory_update(self) -> None:
        self._memory_updated = True

    def _log_failure(self, result: DreamResult) -> None:
        if result.error is None:
            return
        if self._failure_diagnostic is None:
            logger.error("Memory Task failed code={}", result.error.code)
            return
        diagnostic = self._failure_diagnostic
        logger.opt(
            exception=(type(diagnostic), diagnostic, diagnostic.__traceback__),
        ).error("Memory Task failed code={}", result.error.code)


def _require_long_term_path(path: str, *, expected: Path) -> None:
    if Path(path) != expected:
        raise ToolError("Memory Tasks may access only Long-term Memory.")


def _provider_memory_messages(messages: ModelMessages) -> ModelMessages:
    """Preserve the pre-193 provider transcript while Runner keeps its own records."""
    projected: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            projected.append(
                {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": deepcopy(message.get("tool_calls", [])),
                }
            )
            continue
        if role == "tool":
            projected.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id"),
                    "name": message.get("name"),
                    "content": message.get("content", ""),
                }
            )
            continue
        projected.append(deepcopy(message))
    return projected


def _state_read_failure(*, cursor: int) -> DreamResult:
    return DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=cursor,
        error=ErrorInfo(
            code="persistence_error",
            message="Memory Task state could not be read.",
        ),
    )


def _cursor_write_failure(*, cursor: int) -> DreamResult:
    return DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=cursor,
        error=ErrorInfo(
            code="persistence_error",
            message="Summary Cursor could not be updated.",
        ),
    )


__all__ = [
    "Dream",
    "DreamEditFileTool",
    "DreamModelRouter",
    "DreamReadFileTool",
    "DreamResult",
]
