import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent_home import AgentHome
from myclaw.config import ConfigLoader
from myclaw.contracts import (
    AssistantModelMessage,
    AssistantSessionMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    ToolExecutionContext,
    ToolModelMessage,
    ToolSessionMessage,
    validate_agent_event_sequence,
)
from myclaw.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.runtime import prepare_repl_runtime
from myclaw.session_store import JsonlSessionStore
from myclaw.tool_gateway import ToolGateway
from myclaw.workspace import Workspace
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript
from tests.test_config import VALID_CONFIG

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)

SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
USER_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
REQUEST_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")
ASSISTANT_TOOL_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
TOOL_RESULT_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
FINAL_REQUEST_UUID = UUID("a8098c1a-f86e-4f33-8a28-25f602f8e603")
FINAL_ASSISTANT_UUID = UUID("67e55044-10b1-426f-9247-bb680e5fe0c8")
EXTRA_ONE_UUID = UUID("11111111-1111-4111-8111-111111111111")
EXTRA_TWO_UUID = UUID("22222222-2222-4222-8222-222222222222")
EXTRA_THREE_UUID = UUID("33333333-3333-4333-8333-333333333333")
EXTRA_FOUR_UUID = UUID("44444444-4444-4444-8444-444444444444")


def _prepared_tool_conversation(
    *,
    home: AgentHome,
    workspace: Path,
    provider: ScriptedFakeProvider,
    clock: FakeClock,
    conversation_uuids: tuple[UUID, ...],
) -> tuple[StreamingConversationPort, JsonlSessionStore, str]:
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(conversation_uuids).__next__,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=home.path,
                session_id=session.id,
            )
        ),
    )
    return conversation, store, session.id


@pytest.mark.asyncio
async def test_list_files_tool_result_continues_the_model_loop_with_linked_history(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (workspace / "docs").mkdir()
    (workspace / "docs" / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (workspace / "zeta.txt").write_text("zeta\n", encoding="utf-8")
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    tool_call = ModelToolCall(
        id="call_list",
        name="list_files",
        arguments={"path": ".", "recursive": True},
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="I will inspect the workspace.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=10, output_tokens=4, total_tokens=14),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="The workspace has two files."),
                            usage=ModelUsage(input_tokens=20, output_tokens=5, total_tokens=25),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=session.id,
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_TOOL_UUID,
                TOOL_RESULT_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=gateway,
    )

    events = [event async for event in conversation.submit("What files are here?")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    validate_agent_event_sequence(events)
    assert len(provider.stream_requests) == 2
    first_request = provider.stream_requests[0]
    second_request = provider.stream_requests[1]
    assert isinstance(first_request, ModelRequest)
    assert isinstance(second_request, ModelRequest)
    assert [definition.name for definition in first_request.tools] == [
        "read_file",
        "list_files",
        "search_files",
    ]
    assert [message.to_dict() for message in second_request.messages[-2:]] == [
        {
            "role": "assistant",
            "content": "I will inspect the workspace.",
            "tool_calls": [tool_call.to_dict()],
        },
        {
            "role": "tool",
            "tool_call_id": "call_list",
            "name": "list_files",
            "content": "docs/\ndocs/alpha.txt\nzeta.txt",
        },
    ]

    reloaded = await store.load(session.id)
    assert [message.role for message in reloaded.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assistant_call = reloaded.messages[1]
    assert isinstance(assistant_call, AssistantSessionMessage)
    assert assistant_call.tool_calls == (tool_call,)
    tool_result = reloaded.messages[2]
    assert isinstance(tool_result, ToolSessionMessage)
    assert (tool_result.tool_call_id, tool_result.name, tool_result.content) == (
        "call_list",
        "list_files",
        "docs/\ndocs/alpha.txt\nzeta.txt",
    )
    assert reloaded.metadata.cumulative_usage.to_dict() == {
        "model_calls": 2,
        "input_tokens": 30,
        "output_tokens": 9,
        "total_tokens": 39,
    }


@pytest.mark.asyncio
async def test_read_file_returns_the_requested_utf8_line_window_to_the_model(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (workspace / "notes.txt").write_bytes("zero\r\none\r\n\u4e8c\r\nthree\r\n".encode("utf-8"))
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    tool_call = ModelToolCall(
        id="call_read",
        name="read_file",
        arguments={"path": "notes.txt", "offset": 1, "limit": 2},
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
                            message=AssistantModelMessage(content="I read the requested lines."),
                            usage=ModelUsage(input_tokens=12, output_tokens=4, total_tokens=16),
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
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_TOOL_UUID,
                TOOL_RESULT_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session.id,
            )
        ),
    )

    events = [event async for event in conversation.submit("Read two lines from my notes.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    assert second_request.messages[-1].to_dict() == {
        "role": "tool",
        "tool_call_id": "call_read",
        "name": "read_file",
        "content": "one\n\u4e8c",
    }
    reloaded = await store.load(session.id)
    persisted = reloaded.messages[2]
    assert isinstance(persisted, ToolSessionMessage)
    assert (persisted.tool_call_id, persisted.status, persisted.content) == (
        "call_read",
        "success",
        "one\n\u4e8c",
    )


@pytest.mark.asyncio
async def test_search_files_returns_stably_ordered_path_line_previews(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (workspace / "docs").mkdir()
    (workspace / "zeta.txt").write_text("needle second\n", encoding="utf-8")
    (workspace / "docs" / "alpha.txt").write_text(
        "not here\nneedle first\nneedle third\n",
        encoding="utf-8",
    )
    (workspace / "docs" / "ignored.md").write_text("needle hidden\n", encoding="utf-8")
    (workspace / "binary.txt").write_bytes(b"needle\x00private")
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    tool_call = ModelToolCall(
        id="call_search",
        name="search_files",
        arguments={
            "query": "needle",
            "path": ".",
            "glob": "*.txt",
            "max_results": 2,
        },
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
                            message=AssistantModelMessage(content="I found both matches."),
                            usage=ModelUsage(input_tokens=15, output_tokens=4, total_tokens=19),
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
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_TOOL_UUID,
                TOOL_RESULT_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session.id,
            )
        ),
    )

    events = [event async for event in conversation.submit("Find needle in text files.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    assert second_request.messages[-1].to_dict() == {
        "role": "tool",
        "tool_call_id": "call_search",
        "name": "search_files",
        "content": "docs/alpha.txt:2:needle first\ndocs/alpha.txt:3:needle third",
    }


@pytest.mark.asyncio
async def test_production_conversation_denies_parent_escape_without_reading_the_file(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    outside = workspace.parent / "outside.txt"
    outside.write_text("do-not-leak", encoding="utf-8")
    tool_call = ModelToolCall(
        id="call_escape",
        name="read_file",
        arguments={"path": "../outside.txt"},
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(tool_calls=(tool_call,), content=""),
                            usage=ModelUsage(input_tokens=7, output_tokens=2, total_tokens=9),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="I cannot read outside the workspace."
                            ),
                            usage=ModelUsage(input_tokens=11, output_tokens=5, total_tokens=16),
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
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=FakeClock(NOW).now,
        new_uuid=iter(
            (
                SESSION_UUID,
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_TOOL_UUID,
                TOOL_RESULT_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
    )

    events = [
        event
        async for event in runtime.conversation.submit("Read the file outside this workspace.")
    ]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    denied_message = second_request.messages[-1]
    assert isinstance(denied_message, ToolModelMessage)
    assert denied_message.to_dict() == {
        "role": "tool",
        "tool_call_id": "call_escape",
        "name": "read_file",
        "content": "The requested path is outside the allowed Workspace.",
    }
    assert "do-not-leak" not in denied_message.content
    persisted = (await runtime.sessions.load(runtime.session_id)).messages[2]
    assert isinstance(persisted, ToolSessionMessage)
    assert persisted.status == "error"
    assert persisted.error is not None
    assert persisted.error.code == "tool_denied"
    first_request = provider.stream_requests[0]
    assert isinstance(first_request, ModelRequest)
    tool_guidance = first_request.system_prompt.split("<tool_guidance>\n", 1)[1].split(
        "</tool_guidance>", 1
    )[0]
    assert tool_guidance.splitlines() == [
        "- read_file: Read UTF-8 text lines from a file within the current Workspace.",
        "- list_files: List files and directories within the current Workspace.",
        "- search_files: Search UTF-8 text files within the current Workspace.",
    ]
    assert "input_schema" not in tool_guidance
    assert "properties" not in tool_guidance


@pytest.mark.asyncio
async def test_read_file_normalizes_binary_content_as_an_error_before_continuing(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (workspace / "binary.dat").write_bytes(b"prefix\x00private-payload")
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    tool_call = ModelToolCall(
        id="call_binary",
        name="read_file",
        arguments={"path": "binary.dat"},
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
                            message=AssistantModelMessage(content="The file is not readable text."),
                            usage=ModelUsage(input_tokens=10, output_tokens=4, total_tokens=14),
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
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_TOOL_UUID,
                TOOL_RESULT_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session.id,
            )
        ),
    )

    events = [event async for event in conversation.submit("Inspect the binary file.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    tool_message = second_request.messages[-1]
    assert tool_message.to_dict() == {
        "role": "tool",
        "tool_call_id": "call_binary",
        "name": "read_file",
        "content": "read_file could not complete the request.",
    }
    persisted = (await store.load(session.id)).messages[2]
    assert isinstance(persisted, ToolSessionMessage)
    assert persisted.status == "error"
    assert persisted.error is not None
    assert persisted.error.code == "tool_failed"


@pytest.mark.asyncio
async def test_multiple_calls_and_tool_rounds_persist_in_order_before_one_final_output(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (workspace / "alpha.txt").write_text("needle alpha\n", encoding="utf-8")
    (workspace / "beta.txt").write_text("beta\n", encoding="utf-8")
    list_call = ModelToolCall(
        id="call_list_many",
        name="list_files",
        arguments={"path": "."},
    )
    read_call = ModelToolCall(
        id="call_read_many",
        name="read_file",
        arguments={"path": "alpha.txt"},
    )
    search_call = ModelToolCall(
        id="call_search_next",
        name="search_files",
        arguments={"query": "needle"},
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="I will inspect both files.",
                                tool_calls=(list_call, read_call),
                            ),
                            usage=ModelUsage(input_tokens=9, output_tokens=4, total_tokens=13),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="I will confirm the match.",
                                tool_calls=(search_call,),
                            ),
                            usage=ModelUsage(input_tokens=18, output_tokens=4, total_tokens=22),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Only alpha contains needle."),
                            usage=ModelUsage(input_tokens=24, output_tokens=5, total_tokens=29),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation, store, session_id = _prepared_tool_conversation(
        home=home,
        workspace=workspace,
        provider=provider,
        clock=FakeClock(NOW),
        conversation_uuids=(
            TURN_UUID,
            USER_UUID,
            REQUEST_UUID,
            ASSISTANT_TOOL_UUID,
            TOOL_RESULT_UUID,
            EXTRA_ONE_UUID,
            FINAL_REQUEST_UUID,
            EXTRA_TWO_UUID,
            EXTRA_THREE_UUID,
            EXTRA_FOUR_UUID,
            FINAL_ASSISTANT_UUID,
        ),
    )

    events = [event async for event in conversation.submit("Inspect this workspace.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    validate_agent_event_sequence(events)
    assert len(provider.stream_requests) == 3
    third_request = provider.stream_requests[2]
    assert isinstance(third_request, ModelRequest)
    assert [message.role for message in third_request.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "tool",
    ]
    assert [
        message.to_dict().get("tool_call_id")
        for message in third_request.messages
        if message.role == "tool"
    ] == ["call_list_many", "call_read_many", "call_search_next"]
    reloaded = await store.load(session_id)
    assert [message.role for message in reloaded.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert reloaded.metadata.cumulative_usage.to_dict() == {
        "model_calls": 3,
        "input_tokens": 51,
        "output_tokens": 13,
        "total_tokens": 64,
    }


@pytest.mark.asyncio
async def test_read_file_denies_a_symlink_that_resolves_outside_workspace(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    outside = workspace.parent / "symlink-target"
    outside.mkdir()
    (outside / "secret.txt").write_text("symlink-secret", encoding="utf-8")
    link = workspace / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable on this host: {error}")
        try:
            subprocess.run(
                ("cmd", "/c", "mklink", "/J", str(link), str(outside)),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as junction_error:
            pytest.skip(f"directory links are unavailable on this host: {junction_error}")
    tool_call = ModelToolCall(
        id="call_symlink",
        name="read_file",
        arguments={"path": "escape/secret.txt"},
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(tool_calls=(tool_call,), content=""),
                            usage=ModelUsage(input_tokens=6, output_tokens=2, total_tokens=8),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Access denied."),
                            usage=ModelUsage(input_tokens=9, output_tokens=2, total_tokens=11),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation, store, session_id = _prepared_tool_conversation(
        home=home,
        workspace=workspace,
        provider=provider,
        clock=FakeClock(NOW),
        conversation_uuids=(
            TURN_UUID,
            USER_UUID,
            REQUEST_UUID,
            ASSISTANT_TOOL_UUID,
            TOOL_RESULT_UUID,
            FINAL_REQUEST_UUID,
            FINAL_ASSISTANT_UUID,
        ),
    )

    events = [event async for event in conversation.submit("Read the symlink target.")]

    assert events[-1].type == "turn_completed"
    persisted = (await store.load(session_id)).messages[2]
    assert isinstance(persisted, ToolSessionMessage)
    assert persisted.status == "error"
    assert persisted.error is not None
    assert persisted.error.code == "tool_denied"
    assert "symlink-secret" not in persisted.content
