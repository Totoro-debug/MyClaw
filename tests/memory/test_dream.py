import asyncio
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.memory.dream import Dream, DreamResult
from myclaw.memory.manager import MemoryManager
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouter
from myclaw.provider.models import AssistantModelMessage, ModelResponse, ModelUsage
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, ScriptedFakeRouter
from tests.fixtures.diagnostic_capture import capture_diagnostics

NOW = datetime(2026, 8, 27, 10, 0, tzinfo=timezone(timedelta(hours=8)))


def _manager(agent_home: Path) -> MemoryManager:
    home = AgentHome(agent_home)
    home.initialize()
    workspace = agent_home.parent / "dream-workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return MemoryManager(state)


async def _cursor(manager: MemoryManager) -> int:
    return await manager._cursor_store.read()


def _response(
    content: str,
    *,
    tool_calls: tuple[ModelToolCall, ...] = (),
) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content, tool_calls=tool_calls),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="tool_calls" if tool_calls else "stop",
    )


class _BlockingRouter:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.complete_calls = 0

    def stream(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Dream must use the non-streaming memory route")

    async def complete(self, *args: object, **kwargs: object) -> ModelResponse:
        del args, kwargs
        self.complete_calls += 1
        self.started.set()
        await self.release.wait()
        return self.response


@pytest.mark.asyncio
async def test_dream_returns_without_a_provider_call_when_no_summary_is_pending(
    agent_home: Path,
) -> None:
    provider = ScriptedFakeProvider()
    dream = Dream(
        memory_manager=_manager(agent_home),
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="No pending summaries",
        processed_count=0,
        memory_updated=False,
        cursor=0,
    )
    assert provider.complete_requests == []


def test_dream_derives_the_long_term_path_from_the_manager() -> None:
    assert "long_term_path" not in inspect.signature(Dream).parameters


@pytest.mark.asyncio
async def test_dream_uses_the_memory_route_with_static_default_fallback(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    configuration = ConfigLoader(home).load()
    manager = MemoryManager(state)
    await manager.append_summary("The user prefers concise status reports.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No durable update is needed."),))
    router = ModelRouter(
        configuration=configuration,
        provider_factory=lambda _configuration: provider,
    )
    dream = Dream(
        memory_manager=manager,
        model_router=router,
        batch_size=configuration.memory.batch_size,
        max_iterations=configuration.runtime.max_iterations,
    )

    try:
        result = await dream.run()
    finally:
        await dream.close()
        await router.close()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )
    assert await _cursor(manager) == 1
    assert provider.complete_requests[0].model == "claude-model"


@pytest.mark.asyncio
async def test_dream_processes_claimed_summaries_through_restricted_memory_route(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("The user prefers concise reports.", NOW)
    provider = ScriptedFakeProvider(completions=(_response("No durable update."),))
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )
    assert len(provider.complete_requests) == 1
    request = provider.complete_requests[0]
    assert [schema["function"]["name"] for schema in request.tools] == [
        "read_file",
        "edit_file",
    ]
    assert request.messages[0]["role"] == "system"
    assert "The user prefers concise reports." in str(request.messages[1]["content"])
    assert provider.stream_requests == []
    assert not manager.workspace_state.logs_directory.exists()


@pytest.mark.asyncio
async def test_dream_edit_refreshes_the_manager_snapshot_after_a_successful_edit(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("The user prefers concise reports.", NOW)
    original = manager.memory_snapshot()
    old_text = "## User Preference\n"
    new_text = "## User Preference\n\nPrefers concise reports.\n"
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
                                "path": str(manager.long_term_path),
                                "old_text": old_text,
                                "new_text": new_text,
                            }
                        ),
                    ),
                ),
            ),
            _response("Memory updated."),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory updated.",
        processed_count=1,
        memory_updated=True,
        cursor=1,
    )
    expected = original.replace(old_text, new_text, 1)
    assert manager.memory_snapshot() == expected
    assert await manager.read_long_term() == expected
    assert len(provider.complete_requests) == 2
    follow_up = provider.complete_requests[1]
    assert follow_up.messages[2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "edit-memory",
                "name": "edit_file",
                "arguments": json.dumps(
                    {
                        "path": str(manager.long_term_path),
                        "old_text": old_text,
                        "new_text": new_text,
                    }
                ),
            }
        ],
    }
    assert follow_up.messages[3] == {
        "role": "tool",
        "tool_call_id": "edit-memory",
        "name": "edit_file",
        "content": "Long-term Memory updated.",
    }
    assert follow_up.continuation is None


@pytest.mark.asyncio
async def test_dream_edit_can_replace_all_exact_matches(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager._long_term_store.replace("repeat repeat")
    await manager.append_summary("A pending summary.", NOW)
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
                                "path": str(manager.long_term_path),
                                "old_text": "repeat",
                                "new_text": "updated",
                                "replace_all": True,
                            }
                        ),
                    ),
                ),
            ),
            _response("Memory updated."),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory updated.",
        processed_count=1,
        memory_updated=True,
        cursor=1,
    )
    assert manager.memory_snapshot() == "updated updated"


@pytest.mark.asyncio
async def test_dream_model_failure_after_edit_reports_the_persisted_update(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    original = manager.memory_snapshot()
    old_text = "## User Preference\n"
    new_text = "## User Preference\n\nPersisted before provider failure.\n"
    failure = ModelCallError(ErrorInfo(code="model_failed", message="provider failed"))
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
                                "path": str(manager.long_term_path),
                                "old_text": old_text,
                                "new_text": new_text,
                            }
                        ),
                    ),
                ),
            ),
            failure,
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=True,
        cursor=1,
        error=ErrorInfo(code="model_failed", message="provider failed"),
    )
    expected = original.replace(old_text, new_text, 1)
    assert manager.memory_snapshot() == expected
    assert await manager.read_long_term() == expected
    assert len(provider.complete_requests) == 2


@pytest.mark.asyncio
async def test_dream_model_failure_keeps_the_accepted_cursor(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    failure = ModelCallError(ErrorInfo(code="model_failed", message="provider failed"))
    provider = ScriptedFakeProvider(completions=(failure,))
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(code="model_failed", message="provider failed"),
    )
    assert await _cursor(manager) == 1
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_dream_cursor_publication_failure_is_unprocessed_and_logged_once(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("One pending summary.", NOW)

    def fail_cursor(_path: Path, _content: str) -> None:
        raise OSError("injected cursor publication failure")

    monkeypatch.setattr(manager._cursor_store, "_replace_text", fail_cursor)
    provider = ScriptedFakeProvider(completions=(_response("No update."),))
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )
    capture = capture_diagnostics()

    try:
        result = await dream.run()
    finally:
        capture.close()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=0,
        error=ErrorInfo(
            code="persistence_error",
            message="Summary Cursor could not be updated.",
        ),
    )
    assert await _cursor(manager) == 0
    assert provider.complete_requests == []
    assert capture.text.count(" ERROR ") == 1
    assert "Memory Task failed code=persistence_error" in capture.text


@pytest.mark.asyncio
async def test_dream_tool_failure_keeps_the_accepted_cursor_without_retry(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="outside-memory",
                        name="read_file",
                        arguments=json.dumps(
                            {
                                "path": str(manager.long_term_path.parent / "other.txt"),
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="tool_failed",
            message="Memory Tasks may access only Long-term Memory.",
        ),
    )
    assert await _cursor(manager) == 1
    assert len(provider.complete_requests) == 1


@pytest.mark.asyncio
async def test_dream_logs_an_unexpected_tool_failure_once_at_its_boundary(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)

    async def fail_edit(
        self: MemoryManager,
        *,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> str:
        del self, old, new, replace_all
        raise RuntimeError("PRIVATE unexpected memory failure")

    monkeypatch.setattr(MemoryManager, "edit_long_term", fail_edit)
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
                                "path": str(manager.long_term_path),
                                "old_text": "## User Info\n",
                                "new_text": "## User Info\n\nUpdated.\n",
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )
    capture = capture_diagnostics()

    try:
        result = await dream.run()
    finally:
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
async def test_dream_logs_a_corrupt_summary_failure_once_without_leaking_content(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    manager._summary_store.path.write_text(
        "PRIVATE INVALID SUMMARY STREAM",
        encoding="utf-8",
    )
    provider = ScriptedFakeProvider()
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )
    capture = capture_diagnostics()

    try:
        result = await dream.run()
    finally:
        capture.close()

    assert result.error == ErrorInfo(
        code="persistence_error",
        message="Memory Task state could not be read.",
    )
    assert provider.complete_requests == []
    assert capture.text.count(" ERROR ") == 1
    assert "Memory Task failed code=persistence_error" in capture.text
    assert "ValueError: summary stream must contain complete JSONL records" in capture.text
    assert "PRIVATE INVALID SUMMARY STREAM" not in capture.text


@pytest.mark.asyncio
async def test_dream_never_reads_through_an_external_long_term_memory_hard_link(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    outside = agent_home.parent / "outside-memory.md"
    secret = "OUTSIDE SECRET MUST NOT REACH THE MODEL"
    outside.write_text(secret, encoding="utf-8")
    provider = ScriptedFakeProvider(
        completions=(
            _response(
                "",
                tool_calls=(
                    ModelToolCall(
                        id="read-memory",
                        name="read_file",
                        arguments=json.dumps({"path": str(manager.long_term_path)}),
                    ),
                ),
            ),
            _response("No update needed."),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )
    manager.long_term_path.unlink()
    manager.long_term_path.hardlink_to(outside)

    result = await dream.run()

    assert result == DreamResult(
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
        [request.messages for request in provider.complete_requests],
        ensure_ascii=False,
    )
    assert secret not in model_payload
    assert outside.read_text(encoding="utf-8") == secret
    assert await _cursor(manager) == 1


@pytest.mark.asyncio
async def test_dream_edit_failure_keeps_the_accepted_cursor(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    store = manager._long_term_store

    def fail_replace(path: Path, content: str) -> None:
        del path, content
        raise OSError("disk full")

    original_replace = store._replace_text

    def replace(path: Path, content: str) -> None:
        if path == manager.long_term_path:
            fail_replace(path, content)
        original_replace(path, content)

    monkeypatch.setattr(store, "_replace_text", replace)
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
                                "path": str(manager.long_term_path),
                                "old_text": "## User Preference\n",
                                "new_text": "## User Preference\n\nUpdated.\n",
                            }
                        ),
                    ),
                ),
            ),
        )
    )
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )

    result = await dream.run()

    assert result == DreamResult(
        status="Memory Task failed.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="tool_failed",
            message="Long-term Memory could not be updated.",
        ),
    )
    assert await _cursor(manager) == 1


@pytest.mark.asyncio
async def test_dream_cancellation_keeps_the_accepted_cursor_and_releases_run(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    router = _BlockingRouter(_response("No update."))
    dream = Dream(
        memory_manager=manager,
        model_router=router,
        batch_size=10,
        max_iterations=50,
    )
    task = asyncio.create_task(dream.run())

    await router.started.wait()
    assert await _cursor(manager) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await _cursor(manager) == 1
    result = await dream.run()
    assert result.status == "No pending summaries"
    assert router.complete_calls == 1


@pytest.mark.asyncio
async def test_dream_concurrent_runs_claim_once_and_do_not_reenter(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    router = _BlockingRouter(_response("No update."))
    dream = Dream(
        memory_manager=manager,
        model_router=router,
        batch_size=10,
        max_iterations=50,
    )
    first = asyncio.create_task(dream.run())

    await router.started.wait()
    overlapping = await dream.run()
    assert overlapping == DreamResult(
        status="Memory Task is already running.",
        processed_count=0,
        memory_updated=False,
        cursor=1,
        error=ErrorInfo(
            code="memory_task_running",
            message="A Memory Task is already running.",
        ),
    )
    assert await _cursor(manager) == 1

    router.release.set()
    result = await first

    assert result == DreamResult(
        status="Processed 1 summary; Long-term Memory unchanged.",
        processed_count=1,
        memory_updated=False,
        cursor=1,
    )
    assert router.complete_calls == 1


@pytest.mark.asyncio
async def test_dream_close_waits_for_active_work_and_releases_the_task(
    agent_home: Path,
) -> None:
    manager = _manager(agent_home)
    await manager.append_summary("A pending summary.", NOW)
    router = _BlockingRouter(_response("No update."))
    dream = Dream(
        memory_manager=manager,
        model_router=router,
        batch_size=10,
        max_iterations=50,
    )
    running = asyncio.create_task(dream.run())

    await router.started.wait()
    closing = asyncio.create_task(dream.close())
    await asyncio.sleep(0)
    assert not closing.done()

    router.release.set()
    result = await running
    await closing

    assert result.status == "Processed 1 summary; Long-term Memory unchanged."
    assert dream._task is None
    with pytest.raises(RuntimeError, match="Dream is no longer active"):
        await dream.run()
