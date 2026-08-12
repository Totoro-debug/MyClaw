import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.events import TurnFailedPayload
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.memory.conversation_summary import (
    ConversationSummaryManager,
    SummaryModelSettings,
    WorkspaceJsonlSummaryStore,
)
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    UserModelMessage,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session import Session
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures import ScriptedFakeProvider, StreamScript

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 4, 16, 0, 0, tzinfo=LOCAL_OFFSET)


def _state(workspace: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _usage(input_tokens: int = 4, output_tokens: int = 2) -> dict[str, int]:
    return {
        "model_calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _model_usage(input_tokens: int = 4, output_tokens: int = 2) -> ModelUsage:
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _response(
    content: str,
    *,
    usage: ModelUsage | None = None,
) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=_model_usage(20, 5) if usage is None else usage,
        finish_reason="stop",
    )


def _manager(
    provider: ScriptedFakeProvider,
    summaries: WorkspaceJsonlSummaryStore,
    *,
    context_window: int = 10_000,
    max_output: int = 1_000,
    threshold: int = 4,
    system_prompt: str = "CHAT SYSTEM",
) -> ConversationSummaryManager:
    return ConversationSummaryManager(
        provider=provider,
        summaries=summaries,
        settings=SummaryModelSettings(
            model="memory-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        chat_context_window=context_window,
        chat_max_output=max_output,
        consolidation_message_threshold=threshold,
        chat_system_prompt=system_prompt,
        tools=(),
        now=lambda: NOW,
        new_uuid=uuid4,
    )


def _add_assistant(
    session: Session,
    content: str,
    *,
    input_tokens: int = 4,
    output_tokens: int = 2,
) -> None:
    session.add_message(
        "assistant",
        content,
        tool_calls=[],
        status="completed",
        error=None,
        token_usage=_usage(input_tokens, output_tokens),
    )


def _session_with_history(state: WorkspaceState) -> Session:
    session = Session.create(state)
    session.add_message("user", "First question.", future_field={"ignored": True})
    _add_assistant(session, "First answer.")
    session.add_message("user", "Second question.")
    _add_assistant(session, "Second answer.")
    session.add_message("user", "Current question.")
    return session


def _conversation(
    provider: ScriptedFakeProvider,
    session: Session,
    manager: ConversationSummaryManager,
) -> StreamingConversationPort:
    return StreamingConversationPort(
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="chat-model",
            max_output=512,
            temperature=0.0,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        history_preparer=manager.prepare,
    )


@pytest.mark.asyncio
async def test_message_threshold_summarizes_session_suffix_and_updates_public_state(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = _session_with_history(state)
    provider = ScriptedFakeProvider(completions=(_response("First turn summary."),))
    summaries = WorkspaceJsonlSummaryStore(state)

    prepared = await _manager(provider, summaries).prepare(session)

    assert prepared is session
    assert session.last_consolidated == 2
    assert [message["content"] for message in session.messages[session.last_consolidated :]] == [
        "Second question.",
        "Second answer.",
        "Current question.",
    ]
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 28,
        "output_tokens": 9,
        "total_tokens": 37,
    }
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert isinstance(request.messages[0], UserModelMessage)
    assert request.route == "memory"
    assert request.stream is False
    assert "First question." in request.messages[0].content
    assert "First answer." in request.messages[0].content
    assert "Second question." not in request.messages[0].content
    assert "future_field" not in request.messages[0].content
    assert (await summaries.after(0, 10))[0].content == "First turn summary."


@pytest.mark.asyncio
async def test_token_budget_summarizes_roughly_half_the_available_input(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    first_question = "Q1 " + "x" * 1_100
    first_answer = "A1 " + "y" * 1_100
    second_question = "Q2 " + "z" * 1_100
    second_answer = "A2 " + "w" * 1_100
    session.add_message("user", first_question)
    _add_assistant(session, first_answer)
    session.add_message("user", second_question)
    _add_assistant(session, second_answer)
    session.add_message("user", "Current question.")
    provider = ScriptedFakeProvider(completions=(_response("First turn summary."),))

    await _manager(
        provider,
        WorkspaceJsonlSummaryStore(state),
        context_window=1_024,
        max_output=128,
        threshold=100,
    ).prepare(session)

    assert session.last_consolidated == 2
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert isinstance(request.messages[0], UserModelMessage)
    assert first_question in request.messages[0].content
    assert first_answer in request.messages[0].content
    assert second_question not in request.messages[0].content


@pytest.mark.asyncio
async def test_repeated_summary_preparation_advances_last_consolidated_once_per_summary(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    for user, assistant in (
        ("Question one.", "Answer one."),
        ("Question two.", "Answer two."),
    ):
        session.add_message("user", user)
        _add_assistant(session, assistant)
    session.add_message("user", "Question three.")
    provider = ScriptedFakeProvider(
        completions=(
            _response("Summary one."),
            _response("Summary two."),
        )
    )
    summaries = WorkspaceJsonlSummaryStore(state)
    manager = _manager(provider, summaries)

    first = await manager.prepare(session)
    first_position = first.last_consolidated
    _add_assistant(session, "Answer three.")
    session.add_message("user", "Question four.")
    _add_assistant(session, "Answer four.")
    session.add_message("user", "Question five.")
    second = await manager.prepare(session)

    assert first is session
    assert second is session
    assert first_position == 2
    assert second.last_consolidated == 4
    assert [entry.index for entry in await summaries.after(0, 10)] == [1, 2]
    second_request = provider.complete_requests[1]
    assert isinstance(second_request, ModelRequest)
    assert isinstance(second_request.messages[0], UserModelMessage)
    assert "Question one." not in second_request.messages[0].content
    assert "Question two." in second_request.messages[0].content


@pytest.mark.asyncio
async def test_summary_persistence_failure_leaves_last_consolidated_unchanged(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = _session_with_history(state)
    summaries = WorkspaceJsonlSummaryStore(state)
    summaries.path.mkdir()
    provider = ScriptedFakeProvider(completions=(_response("First turn summary."),))

    with pytest.raises(ModelCallError) as raised:
        await _manager(provider, summaries).prepare(session)

    assert raised.value.error.code == "persistence_error"
    assert session.last_consolidated == 0
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 28,
        "output_tokens": 9,
        "total_tokens": 37,
    }


@pytest.mark.asyncio
async def test_oversized_system_prompt_fails_without_summary_or_last_consolidated_progress(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    session.add_message("user", "Do not discard this current input.")
    provider = ScriptedFakeProvider()
    summaries = WorkspaceJsonlSummaryStore(state)

    with pytest.raises(ModelCallError) as raised:
        await _manager(
            provider,
            summaries,
            context_window=1_024,
            max_output=128,
            threshold=100,
            system_prompt="M" * 4_000,
        ).prepare(session)

    assert raised.value.error.code == "memory_context_too_large"
    assert provider.complete_requests == []
    assert session.last_consolidated == 0
    assert not summaries.path.exists()


@pytest.mark.asyncio
async def test_context_overflow_without_old_complete_turn_keeps_current_message(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    current_input = "Current only: " + "x" * 4_000
    session.add_message("user", current_input)
    provider = ScriptedFakeProvider()
    summaries = WorkspaceJsonlSummaryStore(state)

    with pytest.raises(ModelCallError) as raised:
        await _manager(
            provider,
            summaries,
            context_window=1_024,
            max_output=128,
            threshold=100,
        ).prepare(session)

    assert raised.value.error.code == "model_context_overflow"
    assert provider.complete_requests == []
    assert session.last_consolidated == 0
    assert [message["content"] for message in session.messages] == [current_input]
    assert not summaries.path.exists()


@pytest.mark.asyncio
async def test_empty_summary_response_is_model_failure_without_last_consolidated_progress(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = _session_with_history(state)
    provider = ScriptedFakeProvider(completions=(_response(""),))
    summaries = WorkspaceJsonlSummaryStore(state)

    with pytest.raises(ModelCallError) as raised:
        await _manager(provider, summaries).prepare(session)

    assert raised.value.error.code == "model_failed"
    assert session.last_consolidated == 0
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 28,
        "output_tokens": 9,
        "total_tokens": 37,
    }
    assert not summaries.path.exists()


@pytest.mark.asyncio
async def test_summary_provider_failure_preserves_user_visible_model_error(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = _session_with_history(state)
    failure = ModelCallError(ErrorInfo(code="model_failed", message="PRIVATE FAILURE"))
    provider = ScriptedFakeProvider(completions=(failure,))
    summaries = WorkspaceJsonlSummaryStore(state)

    with pytest.raises(ModelCallError) as raised:
        await _manager(provider, summaries).prepare(session)

    assert raised.value.error.code == "model_failed"
    assert raised.value.error.message == "PRIVATE FAILURE"
    assert session.last_consolidated == 0
    assert not summaries.path.exists()


@pytest.mark.asyncio
async def test_active_conversation_prepares_summary_before_chat_request(
    workspace: Path,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    for user, assistant in (
        ("First question.", "First answer."),
        ("Second question.", "Second answer."),
    ):
        session.add_message("user", user)
        _add_assistant(session, assistant)
    provider = ScriptedFakeProvider(
        completions=(_response("First turn summary."),),
        streams=(StreamScript(events=(ModelCompleted(response=_response("Chat answer.")),)),),
    )
    summaries = WorkspaceJsonlSummaryStore(state)
    conversation = _conversation(provider, session, _manager(provider, summaries))

    events = [event async for event in conversation.submit("Current question.")]

    assert [event.type for event in events] == [
        "turn_started",
        "model_call_completed",
        "turn_completed",
    ]
    assert session.last_consolidated == 2
    request = provider.stream_requests[0]
    assert isinstance(request, ModelRequest)
    request_content = json.dumps(request.to_dict(), ensure_ascii=False)
    assert "First question." not in request_content
    assert "Second question." in request_content
    assert "Current question." in request_content


@pytest.mark.asyncio
async def test_summary_failure_is_not_rewritten_by_silent_session_persistence_failure(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace)
    session = _session_with_history(state)
    session.messages.pop()
    failure = ModelCallError(ErrorInfo(code="model_failed", message="Safe summary failure."))
    provider = ScriptedFakeProvider(completions=(failure,))
    summaries = WorkspaceJsonlSummaryStore(state)
    conversation = _conversation(provider, session, _manager(provider, summaries))
    persistence_attempts: list[Path] = []

    def fail_session_persistence(path: Path, _content: bytes) -> None:
        persistence_attempts.append(path)
        raise OSError("injected ordinary Session persistence failure")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_session_persistence)

    events = [event async for event in conversation.submit("Current question.")]
    await asyncio.sleep(0)

    assert [event.type for event in events] == ["turn_started", "turn_failed"]
    failed = events[-1].payload
    assert isinstance(failed, TurnFailedPayload)
    assert failed.error == failure.error
    assert session.last_consolidated == 0
    assert session.messages[-1]["error"] == {
        "code": "model_failed",
        "message": "Safe summary failure.",
    }
    assert persistence_attempts == [state.sessions_directory / f"{session.session_id}.jsonl"]
    assert provider.stream_requests == []
    assert not summaries.path.exists()


@pytest.mark.parametrize(
    ("messages", "expected_position", "last_summarized", "first_retained"),
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
async def test_cutoff_keeps_retained_suffix_at_user_boundary(
    workspace: Path,
    messages: tuple[tuple[str, str], ...],
    expected_position: int,
    last_summarized: str,
    first_retained: str,
) -> None:
    state = _state(workspace)
    session = Session.create(state)
    for role, content in messages:
        if role == "user":
            session.add_message(role, content)
        else:
            _add_assistant(session, content)
    provider = ScriptedFakeProvider(completions=(_response("Aligned summary."),))
    summaries = WorkspaceJsonlSummaryStore(state)

    await _manager(
        provider,
        summaries,
        context_window=100_000,
        max_output=4_096,
        threshold=6,
    ).prepare(session)

    assert session.last_consolidated == expected_position
    assert session.messages[session.last_consolidated]["content"] == first_retained
    request = provider.complete_requests[0]
    assert isinstance(request, ModelRequest)
    assert isinstance(request.messages[0], UserModelMessage)
    assert last_summarized in request.messages[0].content
    assert first_retained not in request.messages[0].content


def test_session_messages_remain_json_native(workspace: Path) -> None:
    session = _session_with_history(_state(workspace))

    assert json.loads(json.dumps(session.messages)) == session.messages
