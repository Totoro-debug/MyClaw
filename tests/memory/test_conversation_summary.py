import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.logging.session import session_log
from myclaw.memory.conversation_summary import (
    ConversationSummaryManager,
    SummaryModelSettings,
    WorkspaceJsonlSummaryStore,
)
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouter
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from myclaw.session.records import (
    AssistantSessionMessage,
    MetadataUpdate,
    UserSessionMessage,
)
from myclaw.session.session_store import JsonlSessionStore
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock, ScriptedFakeProvider
from tests.fixtures.log_capture import configured_process_logging

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 16, 0, 0, tzinfo=LOCAL_OFFSET)


def _state(workspace: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _usage(input_tokens: int = 4, output_tokens: int = 2) -> ModelUsage:
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _response(content: str, *, usage: ModelUsage | None = None) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=_usage() if usage is None else usage,
        finish_reason="stop",
    )


async def _append_user(sessions: JsonlSessionStore, session_id: str, content: str) -> None:
    await sessions.append_message(
        session_id,
        UserSessionMessage(id=str(uuid4()), created_at=NOW, content=content),
    )


async def _append_assistant(
    sessions: JsonlSessionStore,
    session_id: str,
    content: str,
) -> None:
    await sessions.append_message(
        session_id,
        AssistantSessionMessage(
            id=str(uuid4()),
            created_at=NOW,
            content=content,
            tool_calls=(),
            status="completed",
            error=None,
            usage=_usage(),
        ),
    )


@pytest.mark.asyncio
async def test_message_threshold_synchronously_summarizes_early_turns(
    agent_home: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    metadata = sessions.prepare()
    await _append_user(sessions, metadata.id, "First question.")
    await _append_assistant(sessions, metadata.id, "First answer.")
    await _append_user(sessions, metadata.id, "Second question.")
    await _append_assistant(sessions, metadata.id, "Second answer.")
    await _append_user(sessions, metadata.id, "Current question.")
    summary_usage = _usage(input_tokens=20, output_tokens=5)
    provider = ScriptedFakeProvider(
        completions=(_response("First turn summary.", usage=summary_usage),)
    )
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    router = ModelRouter(
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        clock=FakeClock(NOW),
    )
    state = _state(workspace)
    manager = ConversationSummaryManager(
        provider=router,
        sessions=sessions,
        summaries=WorkspaceJsonlSummaryStore(state),
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=10_000,
        chat_max_output=1_000,
        consolidation_message_threshold=4,
        chat_system_prompt="CHAT SYSTEM\nPRIVATE LONG-TERM MEMORY",
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    with configured_process_logging(), session_log(state, metadata.id):
        prepared = await manager.prepare(await sessions.load(metadata.id))

    assert [message.content for message in prepared.short_term_messages] == [
        "Second question.",
        "Second answer.",
        "Current question.",
    ]
    assert prepared.metadata.consolidation_cursor == 2
    assert prepared.metadata.cumulative_usage.to_dict() == {
        "model_calls": 3,
        "input_tokens": 28,
        "output_tokens": 9,
        "total_tokens": 37,
    }
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert request.route == "memory"
    assert request.stream is False
    assert request.tools == ()
    assert "PRIVATE LONG-TERM MEMORY" not in request.system_prompt
    assert len(request.messages) == 1
    summary_input = request.messages[0].content
    assert "First question." in summary_input
    assert "First answer." in summary_input
    assert "Second question." not in summary_input
    assert capsys.readouterr().err == ""
    assert not (state.logs_directory / f"{metadata.id}.log").exists()
    records = [
        json.loads(line)
        for line in (_state(workspace).memory_directory / "summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records == [
        {
            "index": 1,
            "timestamp": "2026-07-11T16:00:00.000+08:00",
            "content": "First turn summary.",
        }
    ]
    await router.close()


@pytest.mark.asyncio
async def test_summary_usage_preserves_an_auxiliary_call_that_finishes_during_generation(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    metadata = sessions.prepare()
    await _append_user(sessions, metadata.id, "First question.")
    await _append_assistant(sessions, metadata.id, "First answer.")
    await _append_user(sessions, metadata.id, "Second question.")
    await _append_assistant(sessions, metadata.id, "Second answer.")
    await _append_user(sessions, metadata.id, "Current question.")
    title_usage = _usage(input_tokens=7, output_tokens=3)
    summary_usage = _usage(input_tokens=20, output_tokens=5)

    class CompletingProvider(ScriptedFakeProvider):
        async def complete(self, request: object) -> ModelResponse:
            await sessions.update_metadata(
                metadata.id,
                MetadataUpdate(usage_delta=title_usage),
            )
            return await super().complete(request)

    provider = CompletingProvider(
        completions=(_response("First turn summary.", usage=summary_usage),)
    )
    manager = ConversationSummaryManager(
        provider=provider,
        sessions=sessions,
        summaries=WorkspaceJsonlSummaryStore(_state(workspace)),
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=10_000,
        chat_max_output=1_000,
        consolidation_message_threshold=4,
        chat_system_prompt="CHAT SYSTEM",
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    prepared = await manager.prepare(await sessions.load(metadata.id))

    assert prepared.metadata.cumulative_usage.to_dict() == {
        "model_calls": 4,
        "input_tokens": 35,
        "output_tokens": 12,
        "total_tokens": 47,
    }


@pytest.mark.asyncio
async def test_token_budget_summarizes_roughly_half_the_available_input(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    metadata = sessions.prepare()
    first_question = "Q1 " + "x" * 1_100
    first_answer = "A1 " + "y" * 1_100
    second_question = "Q2 " + "z" * 1_100
    second_answer = "A2 " + "w" * 1_100
    await _append_user(sessions, metadata.id, first_question)
    await _append_assistant(sessions, metadata.id, first_answer)
    await _append_user(sessions, metadata.id, second_question)
    await _append_assistant(sessions, metadata.id, second_answer)
    await _append_user(sessions, metadata.id, "Current question.")
    provider = ScriptedFakeProvider(completions=(_response("First turn summary."),))
    manager = ConversationSummaryManager(
        provider=provider,
        sessions=sessions,
        summaries=WorkspaceJsonlSummaryStore(_state(workspace)),
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=128,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=1_024,
        chat_max_output=128,
        consolidation_message_threshold=100,
        chat_system_prompt="CHAT SYSTEM",
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    prepared = await manager.prepare(await sessions.load(metadata.id))

    assert prepared.metadata.consolidation_cursor == 2
    assert [message.content for message in prepared.short_term_messages] == [
        second_question,
        second_answer,
        "Current question.",
    ]
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert first_question in request.messages[0].content
    assert first_answer in request.messages[0].content
    assert second_question not in request.messages[0].content


@pytest.mark.asyncio
async def test_repeated_consolidation_advances_cursor_and_summary_index_once_per_slice(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    metadata = sessions.prepare()
    for user, assistant in (
        ("Question one.", "Answer one."),
        ("Question two.", "Answer two."),
    ):
        await _append_user(sessions, metadata.id, user)
        await _append_assistant(sessions, metadata.id, assistant)
    await _append_user(sessions, metadata.id, "Question three.")
    provider = ScriptedFakeProvider(
        completions=(
            _response("Summary one."),
            _response("Summary two."),
        )
    )
    summaries = WorkspaceJsonlSummaryStore(_state(workspace))
    manager = ConversationSummaryManager(
        provider=provider,
        sessions=sessions,
        summaries=summaries,
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=100_000,
        chat_max_output=4_096,
        consolidation_message_threshold=4,
        chat_system_prompt="CHAT SYSTEM",
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    first = await manager.prepare(await sessions.load(metadata.id))
    await _append_assistant(sessions, metadata.id, "Answer three.")
    await _append_user(sessions, metadata.id, "Question four.")
    await _append_assistant(sessions, metadata.id, "Answer four.")
    await _append_user(sessions, metadata.id, "Question five.")
    second = await manager.prepare(await sessions.load(metadata.id))

    assert first.metadata.consolidation_cursor == 2
    assert second.metadata.consolidation_cursor == 4
    assert [
        json.loads(line)["index"]
        for line in summaries.path.read_text(encoding="utf-8").splitlines()
    ] == [1, 2]


@pytest.mark.asyncio
async def test_summary_persistence_failure_does_not_advance_the_session_cursor(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    metadata = sessions.prepare()
    await _append_user(sessions, metadata.id, "First question.")
    await _append_assistant(sessions, metadata.id, "First answer.")
    await _append_user(sessions, metadata.id, "Second question.")
    await _append_assistant(sessions, metadata.id, "Second answer.")
    await _append_user(sessions, metadata.id, "Current question.")
    summary_path = _state(workspace).memory_directory / "summary.jsonl"
    summary_path.mkdir()
    provider = ScriptedFakeProvider(completions=(_response("First turn summary."),))
    manager = ConversationSummaryManager(
        provider=provider,
        sessions=sessions,
        summaries=WorkspaceJsonlSummaryStore(_state(workspace)),
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=10_000,
        chat_max_output=1_000,
        consolidation_message_threshold=4,
        chat_system_prompt="CHAT SYSTEM",
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    with pytest.raises(ModelCallError) as raised:
        await manager.prepare(await sessions.load(metadata.id))

    assert raised.value.error.code == "persistence_error"
    reloaded = await sessions.load(metadata.id)
    assert reloaded.metadata.consolidation_cursor == 0
    assert reloaded.metadata.cumulative_usage.to_dict() == {
        "model_calls": 3,
        "input_tokens": 12,
        "output_tokens": 6,
        "total_tokens": 18,
    }


@pytest.mark.asyncio
async def test_oversized_chat_system_prompt_fails_without_summarizing_current_input(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    metadata = sessions.prepare()
    await _append_user(sessions, metadata.id, "Do not discard this current input.")
    provider = ScriptedFakeProvider()
    summaries = WorkspaceJsonlSummaryStore(_state(workspace))
    manager = ConversationSummaryManager(
        provider=provider,
        sessions=sessions,
        summaries=summaries,
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=128,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=1_024,
        chat_max_output=128,
        consolidation_message_threshold=100,
        chat_system_prompt="M" * 4_000,
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    with pytest.raises(ModelCallError) as raised:
        await manager.prepare(await sessions.load(metadata.id))

    assert raised.value.error.code == "memory_context_too_large"
    assert provider.complete_requests == []
    assert not summaries.path.exists()
    assert (await sessions.load(metadata.id)).metadata.consolidation_cursor == 0


@pytest.mark.asyncio
async def test_context_overflow_without_an_old_complete_turn_keeps_the_current_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    metadata = sessions.prepare()
    current_input = "Current only: " + "x" * 4_000
    await _append_user(sessions, metadata.id, current_input)
    provider = ScriptedFakeProvider()
    summaries = WorkspaceJsonlSummaryStore(_state(workspace))
    manager = ConversationSummaryManager(
        provider=provider,
        sessions=sessions,
        summaries=summaries,
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=128,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=1_024,
        chat_max_output=128,
        consolidation_message_threshold=100,
        chat_system_prompt="Small system prompt.",
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    with pytest.raises(ModelCallError) as raised:
        await manager.prepare(await sessions.load(metadata.id))

    assert raised.value.error.code == "model_context_overflow"
    assert provider.complete_requests == []
    assert not summaries.path.exists()
    reloaded = await sessions.load(metadata.id)
    assert reloaded.metadata.consolidation_cursor == 0
    assert [message.content for message in reloaded.short_term_messages] == [current_input]


@pytest.mark.asyncio
async def test_runtime_summarizes_before_chat_without_injecting_summary_into_chat(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "consolidation_message_threshold = 50",
            "consolidation_message_threshold = 4",
        ),
        encoding="utf-8",
    )
    configuration = ConfigLoader(home).load()

    class RuntimeProvider:
        def __init__(self) -> None:
            self.stream_requests: list[ModelRequest] = []
            self.complete_requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelCompleted]:
            self.stream_requests.append(request)
            answer = f"Answer {len(self.stream_requests)}."
            yield ModelCompleted(response=_response(answer))

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.complete_requests.append(request)
            if request.route == "memory":
                return _response("Summary of the first turn.")
            return _response("First conversation")

        async def close(self) -> None:
            return None

    provider: ModelProvider = RuntimeProvider()
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    for user_input in ("Question one.", "Question two.", "Question three."):
        _ = [event async for event in runtime.conversation.submit(user_input)]

    assert isinstance(provider, RuntimeProvider)
    memory_requests = [
        request for request in provider.complete_requests if request.route == "memory"
    ]
    assert len(memory_requests) == 1
    summary_request = memory_requests[0]
    assert summary_request.tools == ()
    assert "Long-term Memory" not in summary_request.system_prompt
    assert "Question one." in summary_request.messages[0].content
    assert "Question two." not in summary_request.messages[0].content
    assert len(provider.stream_requests) == 3
    third_chat = provider.stream_requests[2]
    third_chat_payload = json.dumps(third_chat.to_dict(), ensure_ascii=False)
    assert "Question one." not in third_chat_payload
    assert "Question two." in third_chat_payload
    assert "Question three." in third_chat_payload
    assert "Summary of the first turn." not in third_chat_payload
    reloaded = await runtime.sessions.load(runtime.session_id)
    assert reloaded.metadata.consolidation_cursor == 2


@pytest.mark.asyncio
async def test_empty_summary_response_is_a_model_failure_without_cursor_progress(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    metadata = sessions.prepare()
    await _append_user(sessions, metadata.id, "First question.")
    await _append_assistant(sessions, metadata.id, "First answer.")
    await _append_user(sessions, metadata.id, "Second question.")
    await _append_assistant(sessions, metadata.id, "Second answer.")
    await _append_user(sessions, metadata.id, "Current question.")
    provider = ScriptedFakeProvider(completions=(_response(""),))
    summaries = WorkspaceJsonlSummaryStore(_state(workspace))
    manager = ConversationSummaryManager(
        provider=provider,
        sessions=sessions,
        summaries=summaries,
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=10_000,
        chat_max_output=1_000,
        consolidation_message_threshold=4,
        chat_system_prompt="CHAT SYSTEM",
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    with pytest.raises(ModelCallError) as raised:
        await manager.prepare(await sessions.load(metadata.id))

    assert raised.value.error.code == "model_failed"
    assert not summaries.path.exists()
    reloaded = await sessions.load(metadata.id)
    assert reloaded.metadata.consolidation_cursor == 0
    assert reloaded.metadata.cumulative_usage.to_dict() == {
        "model_calls": 3,
        "input_tokens": 12,
        "output_tokens": 6,
        "total_tokens": 18,
    }


@pytest.mark.parametrize(
    ("messages", "expected_cursor", "last_summarized", "first_retained"),
    [
        (
            (
                ("user", "Question one."),
                ("assistant", "Answer one."),
                ("user", "Question two."),
                ("assistant", "Answer two."),
                ("user", "Question three."),
                ("assistant", "Answer three."),
                ("user", "Current question."),
            ),
            4,
            "Answer two.",
            "Question three.",
        ),
        (
            (
                ("user", "Question one."),
                ("assistant", "Answer one."),
                ("user", "Current question."),
                ("assistant", "Current answer."),
                ("assistant", "Current tool continuation."),
                ("assistant", "Current final continuation."),
            ),
            2,
            "Answer one.",
            "Current question.",
        ),
    ],
    ids=("advance-to-next-user", "fallback-to-previous-user"),
)
@pytest.mark.asyncio
async def test_cutoff_keeps_the_retained_suffix_at_a_user_boundary(
    agent_home: Path,
    workspace: Path,
    messages: tuple[tuple[str, str], ...],
    expected_cursor: int,
    last_summarized: str,
    first_retained: str,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    metadata = sessions.prepare()
    for role, content in messages:
        if role == "user":
            await _append_user(sessions, metadata.id, content)
        else:
            await _append_assistant(sessions, metadata.id, content)
    provider = ScriptedFakeProvider(completions=(_response("Aligned summary."),))
    manager = ConversationSummaryManager(
        provider=provider,
        sessions=sessions,
        summaries=WorkspaceJsonlSummaryStore(_state(workspace)),
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=100_000,
        chat_max_output=4_096,
        consolidation_message_threshold=6,
        chat_system_prompt="CHAT SYSTEM",
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    prepared = await manager.prepare(await sessions.load(metadata.id))

    assert prepared.metadata.consolidation_cursor == expected_cursor
    assert prepared.short_term_messages[0].role == "user"
    assert prepared.short_term_messages[0].content == first_retained
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert last_summarized in request.messages[0].content
    assert first_retained not in request.messages[0].content
