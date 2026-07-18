import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
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
from myclaw.contracts import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    ToolDefinition,
    ToolExecutionContext,
    ToolModelMessage,
    ToolSessionMessage,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session_store import JsonlSessionStore
from myclaw.tools.tool_gateway import ToolGateway
from tests.fixtures import FakeClock, FakeTool, ScriptedFakeProvider, StreamScript

SESSION_ID = "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000"
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
    if os.name != "nt":
        return path
    return Path(f"\\\\?\\{path.absolute()}")


@pytest.mark.asyncio
async def test_tool_gateway_externalizes_only_results_over_the_configured_threshold(
    agent_home: Path,
    workspace: Path,
) -> None:
    AgentHome(agent_home).initialize()
    artifact_workspace = Path(workspace.anchor) / "artifact-workspace"
    exact_result = "a" * 2000
    expected_preview = "b" * 2000
    oversized_result = expected_preview + "!"
    tool = FakeTool(
        definition=ToolDefinition(
            name="inspect",
            description="Return inspection output.",
            input_schema={"type": "object", "additionalProperties": False},
        ),
        outcomes=(exact_result, oversized_result),
    )
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=artifact_workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        tools=(tool,),
        max_tool_result_chars=2000,
    )

    exact = await gateway.execute(ModelToolCall(id="call_exact", name="inspect", arguments={}))
    oversized = await gateway.execute(ModelToolCall(id="call_over", name="inspect", arguments={}))

    relative_path = f"artifacts/{SESSION_ID}/call_over.txt"
    assert exact.content == exact_result
    assert exact.artifact is None
    assert oversized.content == (
        f"{expected_preview}\n\n...[truncated; full result stored at {relative_path}]"
    )
    assert oversized.artifact is not None
    assert oversized.artifact.to_dict() == {
        "path": relative_path,
        "total_chars": 2001,
        "preview_chars": 2000,
    }
    artifact_directory = (
        agent_home
        / "sessions"
        / Workspace.from_path(artifact_workspace).slug
        / "artifacts"
        / SESSION_ID
    )
    assert sorted(path.name for path in artifact_directory.iterdir()) == ["call_over.txt"]
    assert (artifact_directory / "call_over.txt").read_text(encoding="utf-8") == oversized_result


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
        arguments={"path": "unicode.txt"},
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
    reloaded = await runtime.sessions.load(runtime.session_id)
    tool_message = reloaded.messages[2]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.content == expected_content
    assert tool_message.artifact is not None
    assert tool_message.artifact.to_dict() == {
        "path": relative_path,
        "total_chars": 2001,
        "preview_chars": 2000,
    }
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    model_tool_message = second_request.messages[-1]
    assert isinstance(model_tool_message, ToolModelMessage)
    assert model_tool_message.content == expected_content
    artifact_directory = _long_path(runtime.sessions.directory / "artifacts" / runtime.session_id)
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
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=uuid4,
    )
    session = store.prepare()
    raw_result = "private-oversized-result-" * 100
    private_write_detail = "private-artifact-adapter-detail"

    def unavailable_artifact_boundary(_path: Path, _content: str) -> None:
        raise RuntimeError(private_write_detail)

    tool = FakeTool(
        definition=ToolDefinition(
            name="inspect",
            description="Return inspection output.",
            input_schema={"type": "object", "additionalProperties": False},
        ),
        outcomes=(raw_result,),
    )
    tool_call = ModelToolCall(id="call_failed_artifact", name="inspect", arguments={})
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
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=4096,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=uuid4,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session.id,
            ),
            tools=(tool,),
            max_tool_result_chars=2000,
            artifact_writer=unavailable_artifact_boundary,
        ),
    )

    events = [event async for event in conversation.submit("Inspect the result")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    reloaded = await store.load(session.id)
    tool_message = reloaded.messages[2]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.content == "inspect result could not be stored."
    assert tool_message.status == "error"
    assert tool_message.error is not None
    assert tool_message.error.code == "tool_failed"
    assert tool_message.artifact is None
    persisted_content = _long_path(store.path_for(session.id)).read_text(encoding="utf-8")
    assert "private-oversized-result" not in persisted_content
    assert private_write_detail not in persisted_content
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    model_tool_message = second_request.messages[-1]
    assert isinstance(model_tool_message, ToolModelMessage)
    assert model_tool_message.content == "inspect result could not be stored."
    artifact_directory = _long_path(store.directory / "artifacts" / session.id)
    assert list(artifact_directory.iterdir()) == []


@pytest.mark.asyncio
async def test_invalid_empty_tool_call_id_fails_before_writing_an_artifact(
    agent_home: Path,
    workspace: Path,
) -> None:
    writes: list[tuple[Path, str]] = []
    raw_result = "private oversized result"
    tool = FakeTool(
        definition=ToolDefinition(
            name="inspect",
            description="Return inspection output.",
            input_schema={"type": "object", "additionalProperties": False},
        ),
        outcomes=(raw_result,),
    )
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        tools=(tool,),
        max_tool_result_chars=1,
        artifact_writer=lambda path, content: writes.append((path, content)),
    )

    result = await gateway.execute(ModelToolCall(id="", name="inspect", arguments={}))

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_failed"
    assert result.content == "inspect result could not be stored."
    assert result.artifact is None
    assert writes == []
    assert raw_result not in result.content


@pytest.mark.asyncio
async def test_windows_reserved_tool_call_id_uses_canonical_safe_filename(
    agent_home: Path,
    workspace: Path,
) -> None:
    writes: list[tuple[Path, str]] = []
    raw_result = "oversized result"
    tool = FakeTool(
        definition=ToolDefinition(
            name="inspect",
            description="Return inspection output.",
            input_schema={"type": "object", "additionalProperties": False},
        ),
        outcomes=(raw_result,),
    )
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        tools=(tool,),
        max_tool_result_chars=1,
        artifact_writer=lambda path, content: writes.append((path, content)),
    )

    result = await gateway.execute(ModelToolCall(id="CON", name="inspect", arguments={}))

    expected_path = f"artifacts/{SESSION_ID}/%43%4F%4E.txt"
    assert result.status == "success"
    assert result.tool_call_id == "CON"
    assert result.artifact is not None
    assert result.artifact.path == expected_path
    assert [path.name for path, _ in writes] == ["%43%4F%4E.txt"]
    assert writes[0][1] == raw_result
