import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import (
    MemoryConfiguration,
    ModelsConfiguration,
    ProviderConfiguration,
    RouteConfiguration,
    RuntimeConfiguration,
    ToolConfiguration,
    ToolsConfiguration,
    UserConfiguration,
)
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolModelMessage,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session import Session
from myclaw.tools.models import (
    ModelToolCall,
    ToolResult,
)
from myclaw.tools.tool_artifacts import ArtifactWriteError, externalize_tool_result
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures import FakeClock, FakeTool, ScriptedFakeProvider, StreamScript
from tests.fixtures.diagnostic_capture import capture_diagnostics

SESSION_ID = "20260711-153012-123000_550e8400-e29b-41d4-a716-446655440000"
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=timezone(timedelta(hours=8)))


def _runtime_configuration(*, max_tool_result_chars: int) -> UserConfiguration:
    provider = ProviderConfiguration(
        provider_id="test-provider",
        protocol="openai-compatible",
        base_url="https://provider.example/v1",
        api_key="test-secret",
        models=("test-model",),
    )
    route = RouteConfiguration(
        provider_id=provider.provider_id,
        model="test-model",
        context_window=100_000,
        max_output=4096,
        temperature=0.2,
        reasoning_effort=None,
        timeout=30,
    )
    return UserConfiguration(
        runtime=RuntimeConfiguration(max_tool_result_chars=max_tool_result_chars),
        memory=MemoryConfiguration(
            consolidation_message_threshold=40,
            batch_size=10,
            schedule="0 * * * *",
        ),
        tools=ToolsConfiguration(
            web=ToolConfiguration(enabled=True),
            shell=ToolConfiguration(enabled=True),
        ),
        models=ModelsConfiguration(
            providers={provider.provider_id: provider},
            routes={"default": route},
        ),
    )


def _long_path(path: Path) -> Path:
    return HOST_FILESYSTEM.path_for_io(path)


def _workspace_state(workspace: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _artifact_directory(*, workspace: Path, session_id: str) -> Path:
    state = WorkspaceState(Workspace.from_path(workspace))
    return _long_path(state.sessions_directory / "artifacts" / session_id)


def _session(state: WorkspaceState) -> Session:
    return Session.create(
        state,
        now=lambda: NOW,
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )


def test_runtime_externalizes_only_success_results_over_the_configured_threshold(
    agent_home: Path,
    workspace: Path,
) -> None:
    AgentHome(agent_home).initialize()
    workspace_state = _workspace_state(workspace)
    session = _session(workspace_state)
    exact_result = "a" * 2000
    expected_preview = "b" * 2000
    oversized_result = expected_preview + "!"
    exact = ToolResult(
        tool_call_id="call_exact",
        name="inspect",
        status="success",
        content=exact_result,
        artifact=None,
    )
    oversized = ToolResult(
        tool_call_id="call_over",
        name="inspect",
        status="success",
        content=oversized_result,
        artifact=None,
    )

    exact_projected = externalize_tool_result(
        exact,
        session=session,
        max_tool_result_chars=2000,
    )
    assert not (workspace_state.sessions_directory / "artifacts").exists()
    oversized_projected = externalize_tool_result(
        oversized,
        session=session,
        max_tool_result_chars=2000,
    )

    relative_path = f"artifacts/{SESSION_ID}/call_over.txt"
    assert exact_projected is exact
    assert oversized_projected is not oversized
    assert oversized_projected.content == (
        f"{expected_preview}\n\n...[truncated; full result stored at {relative_path}]"
    )
    assert oversized_projected.artifact is not None
    assert oversized_projected.artifact.to_dict() == {
        "path": relative_path,
        "total_chars": 2001,
        "preview_chars": 2000,
    }
    artifact_directory = _artifact_directory(workspace=workspace, session_id=SESSION_ID)
    assert sorted(path.name for path in artifact_directory.iterdir()) == ["call_over.txt"]
    assert (artifact_directory / "call_over.txt").read_text(encoding="utf-8") == oversized_result


def test_active_session_is_the_only_tool_artifact_workspace_authority(
    workspace: Path,
) -> None:
    workspace_state = _workspace_state(workspace)
    session = Session.create(
        workspace_state,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    result = ToolResult(
        tool_call_id="call_active_authority",
        name="inspect",
        status="success",
        content="oversized",
        artifact=None,
    )

    projected = externalize_tool_result(
        result,
        session=session,
        max_tool_result_chars=1,
    )

    assert projected.artifact is not None
    own_artifact = workspace_state.sessions_directory / projected.artifact.path
    assert own_artifact.read_text(encoding="utf-8") == "oversized"


@pytest.mark.asyncio
async def test_runtime_persists_unicode_preview_and_safely_encoded_artifact_reference(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    expected_preview = "界" * 1999 + "🔥"
    raw_result = expected_preview + "尾"
    (workspace / "unicode.txt").write_text(raw_result, encoding="utf-8")
    unsafe_tool_call_id = "../调用\\结果:🔥"
    encoded_tool_call_id = "..%2F%E8%B0%83%E7%94%A8%5C%E7%BB%93%E6%9E%9C%3A%F0%9F%94%A5"
    tool_call = ModelToolCall(
        id=unsafe_tool_call_id,
        name="read_file",
        arguments='{"path":"unicode.txt"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(tool_calls=(tool_call,), content=""),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Inspection complete."),
                            usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=_runtime_configuration(max_tool_result_chars=2000),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    events = [event async for event in runtime.conversation.submit("Inspect unicode.txt")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    relative_path = f"artifacts/{runtime.session_id}/{encoded_tool_call_id}.txt"
    expected_content = (
        f"{expected_preview}\n\n...[truncated; full result stored at {relative_path}]"
    )
    reloaded = runtime.session
    tool_message = reloaded.messages[2]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == expected_content
    assert tool_message["artifact"] == {
        "path": relative_path,
        "total_chars": 2001,
        "preview_chars": 2000,
    }
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    model_tool_message = second_request.messages[-1]
    assert isinstance(model_tool_message, ToolModelMessage)
    assert model_tool_message.content == expected_content
    artifact_directory = _artifact_directory(
        workspace=workspace,
        session_id=runtime.session_id,
    )
    assert [path.name for path in artifact_directory.iterdir()] == [f"{encoded_tool_call_id}.txt"]
    assert (artifact_directory / f"{encoded_tool_call_id}.txt").read_text(
        encoding="utf-8"
    ) == raw_result


@pytest.mark.asyncio
async def test_artifact_boundary_failure_becomes_safe_tool_error_without_raw_fallback(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    state = _workspace_state(workspace)
    session = Session.create(state, now=clock.now, new_uuid=uuid4)
    raw_result = "private-oversized-result-" * 100
    private_write_detail = "private-artifact-adapter-detail"

    def unavailable_artifact_boundary(_path: Path, _content: str) -> None:
        raise RuntimeError(private_write_detail)

    tool = FakeTool(
        name="inspect",
        description="Return inspection output.",
        outcomes=(raw_result,),
    )
    tool_call = ModelToolCall(id="call_failed_artifact", name="inspect", arguments="{}")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(tool_calls=(tool_call,), content=""),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Artifact storage failed."),
                            usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = ToolGateway()
    gateway.register_tools((tool,))
    conversation = StreamingConversationPort(
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=4096,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=uuid4,
        tool_gateway=gateway,
        externalize_result=lambda result: externalize_tool_result(
            result,
            session=session,
            max_tool_result_chars=2000,
            write_text=unavailable_artifact_boundary,
        ),
    )
    capture = capture_diagnostics()

    with capture.session(session.session_id):
        events = [event async for event in conversation.submit("Inspect the result")]
    capture.close()

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    tool_message = session.messages[2]
    assert tool_message["content"] == "inspect result could not be stored."
    assert tool_message["status"] == "error"
    assert tool_message["artifact"] is None
    await asyncio.sleep(0)
    persisted_content = _long_path(
        state.sessions_directory / f"{session.session_id}.jsonl"
    ).read_text(encoding="utf-8")
    assert "private-oversized-result" not in persisted_content
    assert private_write_detail not in persisted_content
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    model_tool_message = second_request.messages[-1]
    assert isinstance(model_tool_message, ToolModelMessage)
    assert model_tool_message.content == "inspect result could not be stored."
    artifact_directory = _artifact_directory(
        workspace=workspace,
        session_id=session.session_id,
    )
    assert list(artifact_directory.iterdir()) == []
    content = capture.text
    event_text = capture.event_text
    records = [line for line in content.splitlines() if "myclaw.agent.turn:" in line]
    assert len(records) == 1
    assert " ERROR " in records[0]
    assert (
        "Tool Artifact persistence failed code=persistence_error tool=inspect "
        "type=ArtifactWriteError" in records[0]
    )
    assert "Traceback (most recent call last)" in content
    assert "ArtifactWriteError" in content
    assert "RuntimeError" in content
    assert raw_result not in event_text
    assert private_write_detail not in event_text
    assert private_write_detail in content


def test_invalid_empty_tool_call_id_fails_before_writing_an_artifact(
    agent_home: Path,
    workspace: Path,
) -> None:
    writes: list[tuple[Path, str]] = []
    raw_result = "private oversized result"
    result = ToolResult(
        tool_call_id="",
        name="inspect",
        status="success",
        content=raw_result,
        artifact=None,
    )

    with pytest.raises(ArtifactWriteError):
        externalize_tool_result(
            result,
            session=_session(_workspace_state(workspace)),
            max_tool_result_chars=1,
            write_text=lambda path, content: writes.append((path, content)),
        )

    assert writes == []


def test_windows_reserved_tool_call_id_uses_canonical_safe_filename(
    agent_home: Path,
    workspace: Path,
) -> None:
    writes: list[tuple[Path, str]] = []
    raw_result = "oversized result"
    original = ToolResult(
        tool_call_id="CON",
        name="inspect",
        status="success",
        content=raw_result,
        artifact=None,
    )
    result = externalize_tool_result(
        original,
        session=_session(_workspace_state(workspace)),
        max_tool_result_chars=1,
        write_text=lambda path, content: writes.append((path, content)),
    )

    expected_path = f"artifacts/{SESSION_ID}/%43%4F%4E.txt"
    assert result.status == "success"
    assert result.tool_call_id == "CON"
    assert result.artifact is not None
    assert result.artifact.path == expected_path
    assert [path.name for path, _ in writes] == ["%43%4F%4E.txt"]
    assert writes[0][1] == raw_result


def test_same_session_artifacts_are_workspace_isolated_and_legacy_bytes_are_untouched(
    agent_home: Path,
    workspace: Path,
) -> None:
    other_workspace = workspace.parent / "other-workspace"
    other_workspace.mkdir()
    first_state = _workspace_state(workspace)
    second_state = _workspace_state(other_workspace)
    legacy = agent_home / "sessions" / "legacy-workspace" / "artifacts" / SESSION_ID / "same.txt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy artifact bytes")

    for state, content in (
        (first_state, "first workspace artifact"),
        (second_state, "second workspace artifact"),
    ):
        externalize_tool_result(
            ToolResult(
                tool_call_id="same",
                name="inspect",
                status="success",
                content=content,
                artifact=None,
            ),
            session=_session(state),
            max_tool_result_chars=1,
        )

    relative = Path("artifacts") / SESSION_ID / "same.txt"
    assert (first_state.sessions_directory / relative).read_text(encoding="utf-8") == (
        "first workspace artifact"
    )
    assert (second_state.sessions_directory / relative).read_text(encoding="utf-8") == (
        "second workspace artifact"
    )
    assert legacy.read_bytes() == b"legacy artifact bytes"
