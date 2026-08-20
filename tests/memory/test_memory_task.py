import asyncio
import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import ManagementViewService
from myclaw.memory.conversation_summary import WorkspaceJsonlSummaryStore
from myclaw.memory.memory_task import (
    MemoryEditFileTool,
    MemoryManager,
    MemoryReadFileTool,
    MemoryTaskResult,
    RuntimeMemory,
    WorkspaceFileMemoryStore,
)
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouter
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelContinuation,
    ModelResponse,
    ModelRoute,
    ModelUsage,
    ReasoningEffort,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ModelToolCall
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock, ProviderCall, ScriptedFakeProvider, ScriptedFakeRouter
from tests.fixtures.diagnostic_capture import capture_diagnostics, configured_process_logging

NOW = datetime(2026, 7, 11, 16, 0, 0, tzinfo=timezone(timedelta(hours=8)))
SESSION_ID = "20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000"


def _state(home: AgentHome) -> WorkspaceState:
    workspace = home.path.parent / "workspace-state"
    workspace.mkdir(exist_ok=True)
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _response(
    content: str,
    *,
    tool_calls: tuple[ModelToolCall, ...] = (),
) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content, tool_calls=tool_calls),
        usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
        finish_reason="tool_calls" if tool_calls else "stop",
    )


def _provider_call(
    *,
    messages: Sequence[dict[str, object]],
    tools: Sequence[OpenAIToolSchema],
    model: str,
    max_output: int,
    temperature: float,
    reasoning_effort: ReasoningEffort | None,
    timeout: int,
    continuation: ModelContinuation | None = None,
) -> ProviderCall:
    return ProviderCall(
        messages=list(messages),
        tools=tuple(tools),
        model=model,
        max_output=max_output,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        continuation=continuation,
    )


def _router(provider: ScriptedFakeProvider) -> ScriptedFakeRouter:
    return ScriptedFakeRouter(provider)


class _RecordingMemoryTaskRouter:
    def __init__(self, *responses: ModelResponse) -> None:
        self._responses = list(responses)
        self.calls: list[
            tuple[ModelRoute, Sequence[dict[str, Any]], Sequence[OpenAIToolSchema]]
        ] = []

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
    ) -> ModelResponse:
        self.calls.append((route, list(messages), tuple(tools)))
        if not self._responses:
            raise AssertionError("No scripted Memory Task response remains")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_summary_store_returns_the_limited_batch_after_the_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    for content in ("Summary one.", "Summary two.", "Summary three."):
        await summaries.append(content, NOW)

    pending = await summaries.after(cursor=1, limit=1)

    assert [(entry.index, entry.content) for entry in pending] == [
        (2, "Summary two."),
    ]


@pytest.mark.asyncio
async def test_memory_store_treats_a_missing_summary_cursor_as_zero(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()

    cursor = await WorkspaceFileMemoryStore(_state(home)).read_summary_cursor()

    assert cursor == 0


@pytest.mark.asyncio
async def test_memory_store_atomically_persists_the_canonical_summary_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()

    await WorkspaceFileMemoryStore(_state(home)).write_summary_cursor(12)

    assert (_state(home).memory_directory / ".cursor").read_bytes() == b"12\n"
    assert await WorkspaceFileMemoryStore(_state(home)).read_summary_cursor() == 12


@pytest.mark.asyncio
async def test_memory_store_atomically_replaces_exact_long_term_memory(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    replacement = "# Long-term Memory\n\n## User Info\n\nUses UTF-8: \u4f60\u597d\n"

    await WorkspaceFileMemoryStore(_state(home)).replace_long_term(replacement)

    assert await WorkspaceFileMemoryStore(_state(home)).read_long_term() == replacement
    assert (_state(home).long_term_memory_path).read_bytes() == replacement.encode("utf-8")


def test_runtime_memory_keeps_snapshots_immutable_and_replaces_without_failure() -> None:
    memory = RuntimeMemory("before")

    first_snapshot = memory.snapshot()
    memory.replace("after")

    assert first_snapshot == "before"
    assert memory.snapshot() == "after"


def test_memory_tools_export_common_schemas_with_zero_retries(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    memory = WorkspaceFileMemoryStore(_state(home))
    path = _state(home).long_term_memory_path
    read_tool = MemoryReadFileTool(memory=memory, long_term_path=path)
    edit_tool = MemoryEditFileTool(memory=memory, long_term_path=path)

    assert read_tool.to_schema() == {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the current Long-term Memory UTF-8 text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Exact Long-term Memory file path.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Zero-based first line.",
                        "minimum": 0,
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum lines to return.",
                        "minimum": 1,
                        "maximum": 10000,
                        "default": 2000,
                    },
                },
                "required": ["path"],
            },
        },
    }
    assert edit_tool.to_schema() == {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text only in the current Long-term Memory file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Exact Long-term Memory file path.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace.",
                        "minLength": 1,
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every exact match.",
                        "default": False,
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    }
    assert not hasattr(read_tool, "max_retries")
    assert not hasattr(edit_tool, "max_retries")


@pytest.mark.asyncio
async def test_manual_memory_task_returns_exact_zero_work_result_without_a_model_call(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    provider = ScriptedFakeProvider()
    manager = MemoryManager(
        router=_router(provider),
        summaries=WorkspaceJsonlSummaryStore(_state(home)),
        memory=WorkspaceFileMemoryStore(_state(home)),
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )
    capture = capture_diagnostics()

    result = await manager.run_manual()
    capture.close()

    assert result == MemoryTaskResult(
        status="No pending summaries",
        processed_count=0,
        memory_updated=False,
        cursor=0,
    )
    assert provider.complete_requests == []
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_memory_task_uses_the_direct_memory_route_and_dictionary_messages(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("The user prefers concise status reports.", NOW)
    router = _RecordingMemoryTaskRouter(_response("No stable update is needed."))
    manager = MemoryManager(
        router=router,
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(state),
        long_term_path=state.long_term_memory_path,
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result.status == "Processed 1 summary; Long-term Memory unchanged."
    assert len(router.calls) == 1
    route, messages, tools = router.calls[0]
    assert route == "memory"
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "User Info" in messages[0]["content"]
    assert "The user prefers concise status reports." in messages[1]["content"]
    assert [schema["function"]["name"] for schema in tools] == [
        "read_file",
        "edit_file",
    ]
    assert all("id" not in message for message in messages)


@pytest.mark.asyncio
async def test_memory_task_direct_router_receives_tool_results_as_follow_up_dictionaries(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("Inspect Long-term Memory before deciding.", NOW)
    first_response = _response(
        "",
        tool_calls=(
            ModelToolCall(
                id="read-memory",
                name="read_file",
                arguments=json.dumps({"path": str(state.long_term_memory_path)}),
            ),
        ),
    )
    router = _RecordingMemoryTaskRouter(
        first_response,
        _response("No stable update is needed."),
    )
    manager = MemoryManager(
        router=router,
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(state),
        long_term_path=state.long_term_memory_path,
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result.status == "Processed 1 summary; Long-term Memory unchanged."
    assert len(router.calls) == 2
    assert [route for route, _, _ in router.calls] == ["memory", "memory"]
    _, follow_up_messages, follow_up_tools = router.calls[1]
    assert [message["role"] for message in follow_up_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert follow_up_messages[2] == first_response.message.to_dict()
    assert follow_up_messages[3] == {
        "role": "tool",
        "tool_call_id": "read-memory",
        "name": "read_file",
        "content": state.long_term_memory_path.read_text(encoding="utf-8").rstrip("\n"),
    }
    assert [schema["function"]["name"] for schema in follow_up_tools] == [
        "read_file",
        "edit_file",
    ]


@pytest.mark.asyncio
async def test_manual_memory_task_does_not_borrow_a_foreground_session_log(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("PRIVATE SUMMARY CONTENT", NOW)
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo(code="model_failed", message="PRIVATE PROVIDER OUTPUT")),
        )
    )
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    router = ModelRouter(
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
    )
    manager = MemoryManager(
        router=router,
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(state),
        long_term_path=state.long_term_memory_path,
        batch_size=10,
    )
    with configured_process_logging(), session_log(state, SESSION_ID):
        result = await manager.run_manual()

    assert result.error == ErrorInfo(
        code="model_failed",
        message="PRIVATE PROVIDER OUTPUT",
    )
    assert result.cursor == 1
    assert capsys.readouterr().err == "Memory Task failed code=model_failed\n"
    assert not (state.logs_directory / f"{SESSION_ID}.log").exists()
    await router.close()


@pytest.mark.asyncio
async def test_memory_task_without_an_edit_advances_the_summary_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("The user prefers concise status reports.", NOW)
    memory = WorkspaceFileMemoryStore(_state(home))
    original_memory = await memory.read_long_term()
    provider = ScriptedFakeProvider(
        completions=(_response("No stable Long-term Memory update is needed."),)
    )
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=memory,
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )
    assert await memory.read_summary_cursor() == 1
    assert await memory.read_long_term() == original_memory
    request = provider.complete_requests[0]
    assert [schema["function"]["name"] for schema in request.tools] == [
        "read_file",
        "edit_file",
    ]
    assert "The user prefers concise status reports." in cast(str, request.messages[1]["content"])
    for section in ("User Info", "User Preference", "Project Fact", "Lesson"):
        assert section in cast(str, request.messages[0]["content"])


@pytest.mark.asyncio
async def test_memory_task_advances_the_summary_cursor_before_model_work(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("A pending summary.", NOW)
    memory = WorkspaceFileMemoryStore(_state(home))

    class CursorObservingProvider(ScriptedFakeProvider):
        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str = "test-model",
            max_output: int = 1024,
            temperature: float = 0.2,
            reasoning_effort: ReasoningEffort | None = None,
            timeout: int = 30,
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            assert await memory.read_summary_cursor() == 1
            return await super().complete(
                messages=messages,
                tools=tools,
                model=model,
                max_output=max_output,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                continuation=continuation,
            )

    provider = CursorObservingProvider(completions=(_response("No update needed."),))
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=memory,
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result.cursor == 1
    assert await memory.read_summary_cursor() == 1


@pytest.mark.asyncio
async def test_memory_task_preadvances_summary_cursor_before_exact_edit(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("The user prefers concise status reports.", NOW)
    memory_path = state.long_term_memory_path
    old_text = "## User Preference\n"
    new_text = "## User Preference\n\nPrefers concise status reports.\n"
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="read-memory",
                        name="read_file",
                        arguments=json.dumps({"path": str(memory_path)}),
                    ),
                ),
            ),
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(memory_path),
                                "old_text": old_text,
                                "new_text": new_text,
                                "replace_all": "false",
                                "ignored": "projected away",
                            }
                        ),
                    ),
                ),
            ),
            _response("Long-term Memory updated."),
        )
    )
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(_state(home)),
        long_term_path=memory_path,
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="Processed 1 summary; Long-term Memory updated.",
        processed_count=1,
        memory_updated=True,
        cursor=1,
    )
    assert new_text in await WorkspaceFileMemoryStore(_state(home)).read_long_term()
    assert await WorkspaceFileMemoryStore(_state(home)).read_summary_cursor() == 1
    second_request = provider.complete_requests[1]
    read_result = second_request.messages[-1]
    assert read_result["role"] == "tool"
    assert read_result["name"] == "read_file"
    assert "# Long-term Memory" in cast(str, read_result["content"])
    assert list((agent_home / "sessions").rglob("*.jsonl")) == []
    assert list((agent_home / "sessions").rglob("artifacts")) == []


@pytest.mark.asyncio
async def test_memory_task_catalog_denies_every_non_long_term_memory_path(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("A pending summary.", NOW)
    outside = agent_home.parent / "outside-memory.md"
    secret = "OUTSIDE MEMORY MUST NOT REACH THE MODEL"
    outside.write_text(secret, encoding="utf-8")
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="read-outside",
                        name="read_file",
                        arguments=json.dumps({"path": str(outside)}),
                    ),
                ),
            ),
        )
    )
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(state),
        long_term_path=state.long_term_memory_path,
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="tool_failed",
            message="Memory Tasks may access only Long-term Memory.",
        ),
    )
    assert len(provider.complete_requests) == 1
    request = provider.complete_requests[0]
    assert secret not in json.dumps(request.messages)
    assert await WorkspaceFileMemoryStore(state).read_summary_cursor() == 1


@pytest.mark.asyncio
async def test_required_memory_edit_failure_keeps_the_advanced_summary_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("The user prefers concise status reports.", NOW)
    memory_path = state.long_term_memory_path
    original_memory = memory_path.read_bytes()

    def fail_replace(_target: Path, _content: str) -> None:
        if _target == memory_path:
            raise OSError("injected atomic replacement failure")
        HOST_FILESYSTEM.atomic_replace_text(_target, _content)

    memory = WorkspaceFileMemoryStore(state, replace_text=fail_replace)
    runtime_memory = RuntimeMemory("cached memory")
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(memory_path),
                                "old_text": "## User Preference\n",
                                "new_text": (
                                    "## User Preference\n\nPrefers concise status reports.\n"
                                ),
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=memory,
        long_term_path=memory_path,
        batch_size=10,
        runtime_memory=runtime_memory,
    )
    capture = capture_diagnostics()

    result = await manager.run_manual()
    capture.close()

    assert result == MemoryTaskResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="tool_failed",
            message="Long-term Memory could not be updated.",
        ),
    )
    assert memory_path.read_bytes() == original_memory
    assert runtime_memory.snapshot() == "cached memory"
    assert await WorkspaceFileMemoryStore(state).read_summary_cursor() == 1
    content = capture.text
    event_text = capture.event_text
    assert content.count(" ERROR ") == 1
    assert "Memory Task failed code=tool_failed" in content
    assert "Traceback (most recent call last):" in content
    assert "ToolError: Long-term Memory could not be updated." in content
    assert "OSError: injected atomic replacement failure" in content
    assert "The above exception was the direct cause" in content
    assert "The user prefers concise status reports." not in event_text
    assert "Prefers concise status reports." not in event_text


@pytest.mark.asyncio
async def test_unexpected_memory_tool_failure_is_logged_once_at_the_task_boundary(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("A pending summary.", NOW)
    memory_path = state.long_term_memory_path

    class FailingMemoryStore:
        def __init__(self) -> None:
            self._cursor = 0

        async def read_long_term(self) -> str:
            return "## User Info\n"

        async def replace_long_term(self, content: str) -> None:
            del content
            raise RuntimeError("PRIVATE unexpected memory failure")

        async def read_summary_cursor(self) -> int:
            return self._cursor

        async def write_summary_cursor(self, index: int) -> None:
            self._cursor = index

    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="unexpected-edit-memory",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": str(memory_path),
                                "old_text": "## User Info\n",
                                "new_text": "## User Info\n\nUpdated.\n",
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=FailingMemoryStore(),
        long_term_path=memory_path,
        batch_size=10,
    )
    capture = capture_diagnostics()

    result = await manager.run_manual()
    capture.close()

    assert result.error == ErrorInfo(
        code="tool_failed",
        message="edit_file could not complete the request.",
    )
    assert capture.text.count(" ERROR ") == 1
    assert "Memory Task failed code=tool_failed" in capture.text
    assert "Tool execution failed name=edit_file" not in capture.text
    assert "RuntimeError: PRIVATE unexpected memory failure" in capture.text


@pytest.mark.asyncio
async def test_conversation_summary_read_failure_is_logged_only_at_memory_task_boundary(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    summaries.path.write_text("PRIVATE INVALID SUMMARY STREAM", encoding="utf-8")
    manager = MemoryManager(
        router=_router(ScriptedFakeProvider()),
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(_state(home)),
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )
    capture = capture_diagnostics()

    result = await manager.run_manual()
    capture.close()

    assert result.error is not None
    assert result.error.code == "persistence_error"
    content = capture.text
    assert content.count(" ERROR ") == 1
    assert "Memory Task failed code=persistence_error" in content
    assert "Traceback (most recent call last):" in content
    assert "ValueError: summary stream must contain complete JSONL records" in content
    assert "PRIVATE INVALID SUMMARY STREAM" not in content


@pytest.mark.asyncio
async def test_restricted_memory_catalog_never_reads_through_an_external_hard_link(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(home)
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("A pending summary.", NOW)
    memory_path = state.long_term_memory_path
    outside = agent_home.parent / "outside-memory.md"
    secret = "OUTSIDE SECRET MUST NOT REACH THE MODEL"
    outside.write_text(secret, encoding="utf-8")
    memory_path.unlink()
    memory_path.hardlink_to(outside)
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="read-memory",
                        name="read_file",
                        arguments=json.dumps({"path": str(memory_path)}),
                    ),
                ),
            ),
            _response("No update needed."),
        )
    )
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(state),
        long_term_path=memory_path,
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="tool_failed",
            message="Long-term Memory must be a regular Workspace State file.",
        ),
    )
    assert provider.complete_requests
    model_payload = json.dumps(
        [request.messages for request in provider.complete_requests], ensure_ascii=False
    )

    assert secret not in model_payload
    assert outside.read_text(encoding="utf-8") == secret
    assert await WorkspaceFileMemoryStore(state).read_summary_cursor() == 1


@pytest.mark.asyncio
async def test_overlapping_manual_memory_task_is_rejected_without_a_second_model_call(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("One pending summary.", NOW)
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingFirstProvider(ScriptedFakeProvider):
        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str = "test-model",
            max_output: int = 1024,
            temperature: float = 0.2,
            reasoning_effort: ReasoningEffort | None = None,
            timeout: int = 30,
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            self.complete_requests.append(
                _provider_call(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_output=max_output,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                    continuation=continuation,
                )
            )
            if len(self.complete_requests) == 1:
                first_started.set()
                await release_first.wait()
            return _response("No update needed.")

    provider = BlockingFirstProvider()
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(_state(home)),
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )
    capture = capture_diagnostics()

    first_task = asyncio.create_task(manager.run_manual())
    await first_started.wait()
    overlapping = await manager.run_manual()
    release_first.set()
    first = await first_task
    capture.close()

    assert overlapping == MemoryTaskResult(
        status="Memory Task is already running.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="memory_task_running",
            message="A Memory Task is already running.",
        ),
    )
    assert len(provider.complete_requests) == 1
    assert first.cursor == 1
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_overlapping_manual_memory_task_ignores_a_corrupt_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("One pending summary.", NOW)
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingProvider(ScriptedFakeProvider):
        async def complete(
            self,
            *,
            messages: Sequence[dict[str, object]],
            tools: Sequence[OpenAIToolSchema],
            model: str = "test-model",
            max_output: int = 1024,
            temperature: float = 0.2,
            reasoning_effort: ReasoningEffort | None = None,
            timeout: int = 30,
            continuation: ModelContinuation | None = None,
        ) -> ModelResponse:
            self.complete_requests.append(
                _provider_call(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_output=max_output,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                    continuation=continuation,
                )
            )
            first_started.set()
            await release_first.wait()
            return _response("No update needed.")

    provider = BlockingProvider()
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(_state(home)),
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )
    first_task = asyncio.create_task(manager.run_manual())
    await first_started.wait()
    (_state(home).memory_directory / ".cursor").write_bytes(b"corrupt\n")

    try:
        overlapping = await manager.run_manual()
    finally:
        release_first.set()
        first = await first_task

    assert overlapping.error == ErrorInfo(
        code="memory_task_running",
        message="A Memory Task is already running.",
    )
    assert overlapping.cursor == 1
    assert len(provider.complete_requests) == 1
    assert first.cursor == 1


@pytest.mark.asyncio
async def test_dream_command_returns_exact_no_pending_output_without_a_model_call(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    provider = ScriptedFakeProvider()
    memory_manager = MemoryManager(
        router=_router(provider),
        summaries=WorkspaceJsonlSummaryStore(_state(home)),
        memory=WorkspaceFileMemoryStore(_state(home)),
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_manager=memory_manager)
    )

    result = await dispatcher.dispatch("/dream")

    assert result.handled is True
    assert result.output == "No pending summaries"
    assert provider.complete_requests == []


@pytest.mark.asyncio
async def test_runtime_dream_uses_memory_route_with_static_default_fallback(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    summaries = WorkspaceJsonlSummaryStore(state)
    await summaries.append("The user prefers concise status reports.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No durable update is needed."),))
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    result = await runtime.management_dispatcher.dispatch("/dream")
    await runtime.close()

    assert result.output == (
        "Processed 1 summary; Long-term Memory unchanged.\n"
        "processed_count: 1\n"
        "memory_updated: false\n"
        "cursor: 1"
    )
    assert await WorkspaceFileMemoryStore(state).read_summary_cursor() == 1
    request = provider.complete_requests[0]
    assert request.model == "claude-model"


@pytest.mark.asyncio
async def test_dream_command_renders_model_failure_after_advancing_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("One pending summary.", NOW)
    memory = WorkspaceFileMemoryStore(_state(home))
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo(code="model_failed", message="Memory model failed.")),
        )
    )
    memory_manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=memory,
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_manager=memory_manager)
    )

    result = await dispatcher.dispatch("/dream")

    assert result.handled is True
    assert result.output == (
        "model_failed: Memory model failed.\nprocessed_count: 0\nmemory_updated: false\ncursor: 1"
    )
    assert await memory.read_summary_cursor() == 1

    retry = await memory_manager.run_manual()

    assert retry.status == "No pending summaries"
    assert retry.cursor == 1
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_dream_reports_cursor_publication_failure_as_unprocessed(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("One pending summary.", NOW)

    def fail_cursor(target: Path, content: str) -> None:
        if target.name == ".cursor":
            raise OSError("injected cursor publication failure")
        HOST_FILESYSTEM.atomic_replace_text(target, content)

    memory = WorkspaceFileMemoryStore(_state(home), replace_text=fail_cursor)
    provider = ScriptedFakeProvider(completions=(_response("No durable update is needed."),))
    memory_manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=memory,
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_manager=memory_manager)
    )
    capture = capture_diagnostics()

    result = await dispatcher.dispatch("/dream")
    capture.close()

    assert result.output == (
        "persistence_error: Summary Cursor could not be updated.\n"
        "processed_count: 0\n"
        "memory_updated: false\n"
        "cursor: 0"
    )
    assert await WorkspaceFileMemoryStore(_state(home)).read_summary_cursor() == 0
    assert provider.complete_requests == []
    content = capture.text
    assert content.count(" ERROR ") == 1
    assert "Memory Task failed code=persistence_error" in content


@pytest.mark.parametrize(
    "cursor_bytes",
    (b"not-a-cursor\n", b"-1\n", b"1", b"1 \n", b"1\n2\n"),
)
@pytest.mark.asyncio
async def test_dream_reports_corrupt_cursor_without_calling_the_model(
    agent_home: Path,
    cursor_bytes: bytes,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (_state(home).memory_directory / ".cursor").write_bytes(cursor_bytes)
    provider = ScriptedFakeProvider()
    memory_manager = MemoryManager(
        router=_router(provider),
        summaries=WorkspaceJsonlSummaryStore(_state(home)),
        memory=WorkspaceFileMemoryStore(_state(home)),
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_manager=memory_manager)
    )

    result = await dispatcher.dispatch("/dream")

    assert result.output == (
        "persistence_error: Memory Task state could not be read.\n"
        "processed_count: 0\n"
        "memory_updated: false\n"
        "cursor: 0"
    )
    assert provider.complete_requests == []
    assert (_state(home).memory_directory / ".cursor").read_bytes() == cursor_bytes


@pytest.mark.asyncio
async def test_memory_task_rejects_an_external_hard_linked_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = WorkspaceJsonlSummaryStore(_state(home))
    await summaries.append("One pending summary.", NOW)
    outside_cursor = agent_home.parent / "outside-cursor"
    outside_cursor.write_bytes(b"999\n")
    cursor_path = _state(home).memory_directory / ".cursor"
    cursor_path.hardlink_to(outside_cursor)
    provider = ScriptedFakeProvider()
    manager = MemoryManager(
        router=_router(provider),
        summaries=summaries,
        memory=WorkspaceFileMemoryStore(_state(home)),
        long_term_path=_state(home).long_term_memory_path,
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=0,
        error=ErrorInfo(
            code="persistence_error",
            message="Memory Task state could not be read.",
        ),
    )
    assert provider.complete_requests == []
    assert outside_cursor.read_bytes() == b"999\n"
