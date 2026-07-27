import asyncio
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import ManagementViewService
from myclaw.memory.conversation_summary import JsonlSummaryStore
from myclaw.memory.memory_task import (
    FileMemoryStore,
    MemoryEditFileTool,
    MemoryManager,
    MemoryReadFileTool,
    MemoryTaskModelSettings,
)
from myclaw.memory.models import MemoryTaskResult
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolModelMessage,
)
from myclaw.tools.models import ModelToolCall
from myclaw.utils.atomic_files import atomic_replace_text
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider

NOW = datetime(2026, 7, 11, 16, 0, 0, tzinfo=timezone(timedelta(hours=8)))


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


@pytest.mark.asyncio
async def test_summary_store_returns_the_limited_batch_after_the_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
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

    cursor = await FileMemoryStore(home).read_summary_cursor()

    assert cursor == 0


@pytest.mark.asyncio
async def test_memory_store_atomically_persists_the_canonical_summary_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()

    await FileMemoryStore(home).write_summary_cursor(12)

    assert (agent_home / "memory" / ".cursor").read_bytes() == b"12\n"
    assert await FileMemoryStore(home).read_summary_cursor() == 12


@pytest.mark.asyncio
async def test_memory_store_atomically_replaces_exact_long_term_memory(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    replacement = "# Long-term Memory\n\n## User Info\n\nUses UTF-8: \u4f60\u597d\n"

    await FileMemoryStore(home).replace_long_term(replacement)

    assert await FileMemoryStore(home).read_long_term() == replacement
    assert (agent_home / "memory" / "memory.md").read_bytes() == replacement.encode("utf-8")


def test_memory_tools_export_common_schemas_with_zero_retries(agent_home: Path) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    memory = FileMemoryStore(home)
    path = agent_home / "memory" / "memory.md"
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
    assert read_tool.max_retries == edit_tool.max_retries == 0


@pytest.mark.asyncio
async def test_manual_memory_task_returns_exact_zero_work_result_without_a_model_call(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    provider = ScriptedFakeProvider()
    manager = MemoryManager(
        provider=provider,
        summaries=JsonlSummaryStore(home),
        memory=FileMemoryStore(home),
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="No pending summaries",
        processed_count=0,
        memory_updated=False,
        cursor=0,
    )
    assert provider.complete_requests == []


@pytest.mark.asyncio
async def test_memory_task_without_an_edit_advances_the_summary_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("The user prefers concise status reports.", NOW)
    memory = FileMemoryStore(home)
    original_memory = await memory.read_long_term()
    provider = ScriptedFakeProvider(
        completions=(_response("No stable Long-term Memory update is needed."),)
    )
    manager = MemoryManager(
        provider=provider,
        summaries=summaries,
        memory=memory,
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
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
    assert isinstance(request, ModelRequest)
    assert request.route == "memory"
    assert request.stream is False
    assert [schema["function"]["name"] for schema in request.tools] == [
        "read_file",
        "edit_file",
    ]
    assert "The user prefers concise status reports." in request.messages[0].content
    for section in ("User Info", "User Preference", "Project Fact", "Lesson"):
        assert section in request.system_prompt


@pytest.mark.asyncio
async def test_memory_task_exact_edit_updates_long_term_memory_then_advances_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("The user prefers concise status reports.", NOW)
    memory_path = agent_home / "memory" / "memory.md"
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
        provider=provider,
        summaries=summaries,
        memory=FileMemoryStore(home),
        long_term_path=memory_path,
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="Processed 1 summary; Long-term Memory updated.",
        processed_count=1,
        memory_updated=True,
        cursor=1,
    )
    assert new_text in await FileMemoryStore(home).read_long_term()
    assert await FileMemoryStore(home).read_summary_cursor() == 1
    second_request = provider.complete_requests[1]
    assert isinstance(second_request, ModelRequest)
    read_result = second_request.messages[-1]
    assert isinstance(read_result, ToolModelMessage)
    assert read_result.name == "read_file"
    assert "# Long-term Memory" in read_result.content
    assert list((agent_home / "sessions").rglob("*.jsonl")) == []
    assert list((agent_home / "sessions").rglob("artifacts")) == []


@pytest.mark.asyncio
async def test_memory_task_catalog_denies_every_non_long_term_memory_path(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
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
        provider=provider,
        summaries=summaries,
        memory=FileMemoryStore(home),
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=0,
        error=ErrorInfo(
            code="tool_failed",
            message="Memory Tasks may access only Long-term Memory.",
        ),
    )
    assert len(provider.complete_requests) == 1
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert secret not in json.dumps(request.to_dict())
    assert await FileMemoryStore(home).read_summary_cursor() == 0


@pytest.mark.asyncio
async def test_required_memory_edit_failure_does_not_advance_summary_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("The user prefers concise status reports.", NOW)
    memory_path = agent_home / "memory" / "memory.md"
    original_memory = memory_path.read_bytes()

    def fail_replace(_target: Path, _content: str) -> None:
        raise OSError("injected atomic replacement failure")

    memory = FileMemoryStore(home, replace_text=fail_replace)
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
                                    "## User Preference\n\n"
                                    "Prefers concise status reports.\n"
                                ),
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    manager = MemoryManager(
        provider=provider,
        summaries=summaries,
        memory=memory,
        long_term_path=memory_path,
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=0,
        error=ErrorInfo(
            code="tool_failed",
            message="Long-term Memory could not be updated.",
        ),
    )
    assert memory_path.read_bytes() == original_memory
    assert await FileMemoryStore(home).read_summary_cursor() == 0


@pytest.mark.asyncio
async def test_restricted_memory_catalog_never_reads_through_an_external_hard_link(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("A pending summary.", NOW)
    memory_path = agent_home / "memory" / "memory.md"
    outside = agent_home.parent / "outside-memory.md"
    secret = "OUTSIDE SECRET MUST NOT REACH THE MODEL"
    outside.write_text(secret, encoding="utf-8")
    memory_path.unlink()
    try:
        memory_path.hardlink_to(outside)
    except OSError as error:
        pytest.skip(f"file hard links are unavailable on this host: {error}")
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
        provider=provider,
        summaries=summaries,
        memory=FileMemoryStore(home),
        long_term_path=memory_path,
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )

    result = await manager.run_manual()

    assert result == MemoryTaskResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=0,
        error=ErrorInfo(
            code="tool_failed",
            message="Long-term Memory must be a regular Agent Home file.",
        ),
    )
    model_requests = [
        request for request in provider.complete_requests if isinstance(request, ModelRequest)
    ]
    assert len(model_requests) == len(provider.complete_requests)
    model_payload = json.dumps(
        [request.to_dict() for request in model_requests], ensure_ascii=False
    )
    assert secret not in model_payload
    assert outside.read_text(encoding="utf-8") == secret
    assert await FileMemoryStore(home).read_summary_cursor() == 0


@pytest.mark.asyncio
async def test_memory_task_denies_an_external_memory_directory_alias(
    agent_home: Path,
) -> None:
    agent_home.mkdir(parents=True, exist_ok=True)
    outside = agent_home.parent / "outside-memory-directory"
    outside.mkdir()
    secret = "JUNCTION SECRET MUST NOT REACH THE MODEL"
    (outside / "memory.md").write_text(secret, encoding="utf-8")
    try:
        subprocess.run(
            ("cmd", "/c", "mklink", "/J", str(agent_home / "memory"), str(outside)),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"directory junctions are unavailable on this host: {error}")
    home = AgentHome(agent_home)

    with pytest.raises(
        PermissionError,
        match="memory directory must remain inside Agent Home",
    ):
        home.initialize()

    assert (outside / "memory.md").read_text(encoding="utf-8") == secret
    assert not (outside / ".cursor").exists()


@pytest.mark.asyncio
async def test_overlapping_manual_memory_task_is_rejected_without_a_second_model_call(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("One pending summary.", NOW)
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingFirstProvider(ScriptedFakeProvider):
        async def complete(self, request: object) -> ModelResponse:
            self.complete_requests.append(request)
            if len(self.complete_requests) == 1:
                first_started.set()
                await release_first.wait()
            return _response("No update needed.")

    provider = BlockingFirstProvider()
    manager = MemoryManager(
        provider=provider,
        summaries=summaries,
        memory=FileMemoryStore(home),
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )

    first_task = asyncio.create_task(manager.run_manual())
    await first_started.wait()
    overlapping = await manager.run_manual()
    release_first.set()
    first = await first_task

    assert overlapping == MemoryTaskResult(
        status="Memory Task is already running.",
        processed_count=0,
        memory_updated=False,
        cursor=0,
        error=ErrorInfo(
            code="memory_task_running",
            message="A Memory Task is already running.",
        ),
    )
    assert len(provider.complete_requests) == 1
    assert first.cursor == 1


@pytest.mark.asyncio
async def test_overlapping_manual_memory_task_ignores_a_corrupt_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("One pending summary.", NOW)
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class BlockingProvider(ScriptedFakeProvider):
        async def complete(self, request: object) -> ModelResponse:
            self.complete_requests.append(request)
            first_started.set()
            await release_first.wait()
            return _response("No update needed.")

    provider = BlockingProvider()
    manager = MemoryManager(
        provider=provider,
        summaries=summaries,
        memory=FileMemoryStore(home),
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )
    first_task = asyncio.create_task(manager.run_manual())
    await first_started.wait()
    (agent_home / "memory" / ".cursor").write_bytes(b"corrupt\n")

    try:
        overlapping = await manager.run_manual()
    finally:
        release_first.set()
        first = await first_task

    assert overlapping.error == ErrorInfo(
        code="memory_task_running",
        message="A Memory Task is already running.",
    )
    assert overlapping.cursor == 0
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
        provider=provider,
        summaries=JsonlSummaryStore(home),
        memory=FileMemoryStore(home),
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
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
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "[tools.shell]\nenabled = true",
            "[tools.shell]\nenabled = false",
        ),
        encoding="utf-8",
    )
    configuration = ConfigLoader(home).load()
    summaries = JsonlSummaryStore(home)
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
    assert await FileMemoryStore(home).read_summary_cursor() == 1
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert request.route == "memory"
    assert request.model == "claude-model"


@pytest.mark.asyncio
async def test_dream_command_renders_model_failure_without_advancing_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("One pending summary.", NOW)
    memory = FileMemoryStore(home)
    provider = ScriptedFakeProvider(
        completions=(
            ModelCallError(ErrorInfo(code="model_failed", message="Memory model failed.")),
        )
    )
    memory_manager = MemoryManager(
        provider=provider,
        summaries=summaries,
        memory=memory,
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_manager=memory_manager)
    )

    result = await dispatcher.dispatch("/dream")

    assert result.handled is True
    assert result.output == (
        "model_failed: Memory model failed.\nprocessed_count: 0\nmemory_updated: false\ncursor: 0"
    )
    assert await memory.read_summary_cursor() == 0


@pytest.mark.asyncio
async def test_dream_reports_cursor_publication_failure_as_unprocessed(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("One pending summary.", NOW)

    def fail_cursor(target: Path, content: str) -> None:
        if target.name == ".cursor":
            raise OSError("injected cursor publication failure")
        atomic_replace_text(target, content)

    memory = FileMemoryStore(home, replace_text=fail_cursor)
    provider = ScriptedFakeProvider(completions=(_response("No durable update is needed."),))
    memory_manager = MemoryManager(
        provider=provider,
        summaries=summaries,
        memory=memory,
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        batch_size=10,
    )
    dispatcher = ManagementCommandDispatcher(
        ManagementViewService(home, memory_manager=memory_manager)
    )

    result = await dispatcher.dispatch("/dream")

    assert result.output == (
        "persistence_error: Summary Cursor could not be updated.\n"
        "processed_count: 0\n"
        "memory_updated: false\n"
        "cursor: 0"
    )
    assert await FileMemoryStore(home).read_summary_cursor() == 0


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
    (agent_home / "memory" / ".cursor").write_bytes(cursor_bytes)
    provider = ScriptedFakeProvider()
    memory_manager = MemoryManager(
        provider=provider,
        summaries=JsonlSummaryStore(home),
        memory=FileMemoryStore(home),
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
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
    assert (agent_home / "memory" / ".cursor").read_bytes() == cursor_bytes


@pytest.mark.asyncio
async def test_memory_task_rejects_an_external_hard_linked_cursor(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    summaries = JsonlSummaryStore(home)
    await summaries.append("One pending summary.", NOW)
    outside_cursor = agent_home.parent / "outside-cursor"
    outside_cursor.write_bytes(b"999\n")
    cursor_path = agent_home / "memory" / ".cursor"
    try:
        cursor_path.hardlink_to(outside_cursor)
    except OSError as error:
        pytest.skip(f"file hard links are unavailable on this host: {error}")
    provider = ScriptedFakeProvider()
    manager = MemoryManager(
        provider=provider,
        summaries=summaries,
        memory=FileMemoryStore(home),
        long_term_path=agent_home / "memory" / "memory.md",
        settings=MemoryTaskModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
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
