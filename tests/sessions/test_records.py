from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from myclaw.provider.models import ModelUsage
from myclaw.session.records import (
    AssistantMessageStatus,
    AssistantSessionMessage,
    ConversationSession,
    CumulativeUsage,
    MetadataUpdate,
    SessionError,
    SessionMetadata,
    ToolSessionMessage,
    UserSessionMessage,
)
from myclaw.tools.artifacts import ArtifactReference
from myclaw.tools.models import (
    ModelToolCall,
    ToolResultStatus,
)

LOCAL_OFFSET = timezone(timedelta(hours=8))


def test_session_metadata_serializes_as_the_exact_first_jsonl_record() -> None:
    metadata = SessionMetadata(
        id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
        title="MyClaw implementation",
        created_at=datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET),
        updated_at=datetime(2026, 7, 11, 15, 31, 2, 456000, tzinfo=LOCAL_OFFSET),
        consolidation_cursor=0,
        cumulative_usage=CumulativeUsage(
            model_calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        ),
    )

    assert metadata.to_json_line() == (
        '{"record_type":"metadata","schema_version":1,'
        '"id":"20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",'
        '"title":"MyClaw implementation",'
        '"created_at":"2026-07-11T15:30:12.123+08:00",'
        '"updated_at":"2026-07-11T15:31:02.456+08:00",'
        '"consolidation_cursor":0,'
        '"cumulative_usage":{"model_calls":0,"input_tokens":0,'
        '"output_tokens":0,"total_tokens":0}}\n'
    )


def test_session_metadata_rejects_values_outside_the_persisted_contract() -> None:
    with pytest.raises(ValueError, match="model_calls"):
        CumulativeUsage(model_calls=-1, input_tokens=0, output_tokens=0, total_tokens=0)
    with pytest.raises(ValueError, match="total_tokens"):
        CumulativeUsage(model_calls=1, input_tokens=2, output_tokens=3, total_tokens=6)

    valid_usage = CumulativeUsage(model_calls=0, input_tokens=0, output_tokens=0, total_tokens=0)
    with pytest.raises(ValueError, match="Session ID"):
        SessionMetadata(
            id="20260711-153012-123456_123e4567-e89b-12d3-a456-426614174000",
            title="MyClaw implementation",
            created_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
            updated_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
            consolidation_cursor=0,
            cumulative_usage=valid_usage,
        )
    with pytest.raises(ValueError, match="consolidation_cursor"):
        SessionMetadata(
            id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
            title="MyClaw implementation",
            created_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
            updated_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
            consolidation_cursor=-1,
            cumulative_usage=valid_usage,
        )


@pytest.mark.parametrize("record_kind", ["metadata", "update"])
@pytest.mark.parametrize("title", ["", "  not   normalized  ", "x" * 61])
def test_session_metadata_and_updates_reject_noncontract_titles(
    record_kind: str,
    title: str,
) -> None:
    with pytest.raises(ValueError, match="title"):
        if record_kind == "metadata":
            SessionMetadata(
                id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
                title=title,
                created_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
                updated_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
                consolidation_cursor=0,
                cumulative_usage=CumulativeUsage(
                    model_calls=0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                ),
            )
        else:
            MetadataUpdate(title=title)


def test_untitled_session_is_a_valid_normalized_title() -> None:
    metadata = SessionMetadata(
        id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
        title="Untitled session",
        created_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
        updated_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
        consolidation_cursor=0,
        cumulative_usage=CumulativeUsage(
            model_calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        ),
    )

    assert replace(metadata, title="Untitled session").title == "Untitled session"


def test_user_session_message_serializes_as_the_exact_jsonl_record() -> None:
    message = UserSessionMessage(
        id="0f8fad5b-d9cb-469f-a165-70867728950e",
        created_at=datetime(2026, 7, 11, 15, 30, 12, 200000, tzinfo=LOCAL_OFFSET),
        content="Help me inspect this project.",
    )

    assert message.to_json_line() == (
        '{"record_type":"message","id":"0f8fad5b-d9cb-469f-a165-70867728950e",'
        '"created_at":"2026-07-11T15:30:12.200+08:00","role":"user",'
        '"content":"Help me inspect this project."}\n'
    )


def test_user_session_message_rejects_invalid_ids_times_and_content() -> None:
    with pytest.raises(ValueError, match="UUID4"):
        UserSessionMessage(
            id="123e4567-e89b-12d3-a456-426614174000",
            created_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
            content="Help me.",
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        UserSessionMessage(
            id="0f8fad5b-d9cb-469f-a165-70867728950e",
            created_at=datetime(2026, 7, 11),
            content="Help me.",
        )
    with pytest.raises(ValueError, match="content"):
        UserSessionMessage(
            id="0f8fad5b-d9cb-469f-a165-70867728950e",
            created_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
            content="   ",
        )


def test_assistant_session_message_serializes_as_the_exact_jsonl_record() -> None:
    message = AssistantSessionMessage(
        id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        created_at=datetime(2026, 7, 11, 15, 30, 13, tzinfo=LOCAL_OFFSET),
        content="I will inspect the files.",
        tool_calls=(ModelToolCall(id="call_123", name="list_files", arguments='{"path":"."}'),),
        status="completed",
        error=None,
        usage=ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144),
    )

    assert message.to_json_line() == (
        '{"record_type":"message","id":"7c9e6679-7425-40de-944b-e07fc1f90ae7",'
        '"created_at":"2026-07-11T15:30:13.000+08:00","role":"assistant",'
        '"content":"I will inspect the files.",'
        '"tool_calls":[{"id":"call_123","name":"list_files",'
        '"arguments":"{\\"path\\":\\".\\"}"}],"status":"completed","error":null,'
        '"usage":{"input_tokens":120,"output_tokens":24,"total_tokens":144}}\n'
    )


def test_interrupted_assistant_persists_the_exact_safe_error_projection() -> None:
    message = AssistantSessionMessage(
        id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        created_at=datetime(2026, 7, 11, 15, 30, 13, tzinfo=LOCAL_OFFSET),
        content="Partial answer",
        tool_calls=(),
        status="interrupted",
        error=SessionError(code="turn_cancelled", message="Turn interrupted by user."),
        usage=ModelUsage(input_tokens=120, output_tokens=8, total_tokens=128),
    )

    assert message.to_json_line() == (
        '{"record_type":"message","id":"7c9e6679-7425-40de-944b-e07fc1f90ae7",'
        '"created_at":"2026-07-11T15:30:13.000+08:00","role":"assistant",'
        '"content":"Partial answer","tool_calls":[],"status":"interrupted",'
        '"error":{"code":"turn_cancelled","message":"Turn interrupted by user."},'
        '"usage":{"input_tokens":120,"output_tokens":8,"total_tokens":128}}\n'
    )


def test_assistant_session_message_enforces_status_content_and_error_coherence() -> None:
    def message(
        *,
        content: str,
        status: AssistantMessageStatus,
        error: SessionError | None,
    ) -> AssistantSessionMessage:
        return AssistantSessionMessage(
            id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
            created_at=datetime(2026, 7, 11, 15, 30, 13, tzinfo=LOCAL_OFFSET),
            content=content,
            tool_calls=(),
            status=status,
            error=error,
            usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )

    error = SessionError(code="model_failed", message="Model call failed.")

    with pytest.raises(ValueError, match="completed assistant must not have an error"):
        message(content="Answer", status="completed", error=error)
    with pytest.raises(ValueError, match="non-completed assistant requires an error"):
        message(content="Partial", status="interrupted", error=None)
    with pytest.raises(ValueError, match="content or tool_calls"):
        message(content="", status="completed", error=None)


def test_tool_session_message_serializes_as_the_exact_jsonl_record() -> None:
    message = ToolSessionMessage(
        id="9b2c3a42-1d2e-4a1e-a827-61f36dc54713",
        created_at=datetime(2026, 7, 11, 15, 30, 13, 500000, tzinfo=LOCAL_OFFSET),
        tool_call_id="call_123",
        name="list_files",
        content="CONTEXT.md\ndocs/",
        status="success",
        artifact=None,
    )

    assert message.to_json_line() == (
        '{"record_type":"message","id":"9b2c3a42-1d2e-4a1e-a827-61f36dc54713",'
        '"created_at":"2026-07-11T15:30:13.500+08:00","role":"tool",'
        '"tool_call_id":"call_123","name":"list_files",'
        '"content":"CONTEXT.md\\ndocs/","status":"success","artifact":null}\n'
    )


def test_refused_tool_session_message_persists_the_exact_safe_error() -> None:
    message = ToolSessionMessage(
        id="9b2c3a42-1d2e-4a1e-a827-61f36dc54713",
        created_at=datetime(2026, 7, 11, 15, 30, 13, 500000, tzinfo=LOCAL_OFFSET),
        tool_call_id="call_123",
        name="write_file",
        content="Permission denied by user.",
        status="refused",
        artifact=None,
    )

    assert message.to_json_line() == (
        '{"record_type":"message","id":"9b2c3a42-1d2e-4a1e-a827-61f36dc54713",'
        '"created_at":"2026-07-11T15:30:13.500+08:00","role":"tool",'
        '"tool_call_id":"call_123","name":"write_file",'
        '"content":"Permission denied by user.","status":"refused","artifact":null}\n'
    )


def test_tool_artifact_reference_is_preserved_in_the_exact_session_shape() -> None:
    message = ToolSessionMessage(
        id="9b2c3a42-1d2e-4a1e-a827-61f36dc54713",
        created_at=datetime(2026, 7, 11, 15, 30, 13, 500000, tzinfo=LOCAL_OFFSET),
        tool_call_id="call_123",
        name="read_file",
        content=(
            "preview\n\n...[truncated; full result stored at "
            "artifacts/20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000/"
            "call_123.txt]"
        ),
        status="success",
        artifact=ArtifactReference(
            path=(
                "artifacts/20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000/call_123.txt"
            ),
            total_chars=73421,
            preview_chars=2000,
        ),
    )

    assert message.to_dict() == {
        "record_type": "message",
        "id": "9b2c3a42-1d2e-4a1e-a827-61f36dc54713",
        "created_at": "2026-07-11T15:30:13.500+08:00",
        "role": "tool",
        "tool_call_id": "call_123",
        "name": "read_file",
        "content": (
            "preview\n\n...[truncated; full result stored at "
            "artifacts/20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000/"
            "call_123.txt]"
        ),
        "status": "success",
        "artifact": {
            "path": (
                "artifacts/20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000/call_123.txt"
            ),
            "total_chars": 73421,
            "preview_chars": 2000,
        },
    }


def test_tool_session_message_enforces_identity() -> None:
    def message(
        *,
        id: str,
        status: ToolResultStatus,
    ) -> ToolSessionMessage:
        return ToolSessionMessage(
            id=id,
            created_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
            tool_call_id="call_123",
            name="read_file",
            content="result",
            status=status,
            artifact=None,
        )

    with pytest.raises(ValueError, match="UUID4"):
        message(
            id="123e4567-e89b-12d3-a456-426614174000",
            status="success",
        )


def test_conversation_session_exposes_the_suffix_after_the_consolidation_cursor() -> None:
    user = UserSessionMessage(
        id="0f8fad5b-d9cb-469f-a165-70867728950e",
        created_at=datetime(2026, 7, 11, 15, 30, 12, 200000, tzinfo=LOCAL_OFFSET),
        content="Help me inspect this project.",
    )
    assistant = AssistantSessionMessage(
        id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        created_at=datetime(2026, 7, 11, 15, 30, 13, tzinfo=LOCAL_OFFSET),
        content="Done.",
        tool_calls=(),
        status="completed",
        error=None,
        usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
    )
    session = ConversationSession(
        metadata=SessionMetadata(
            id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
            title="MyClaw implementation",
            created_at=datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET),
            updated_at=datetime(2026, 7, 11, 15, 31, 2, 456000, tzinfo=LOCAL_OFFSET),
            consolidation_cursor=1,
            cumulative_usage=CumulativeUsage(
                model_calls=1,
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
            ),
        ),
        messages=(user, assistant),
    )

    assert session.short_term_messages == (assistant,)


def test_conversation_session_rejects_a_cursor_past_the_message_boundary() -> None:
    metadata = SessionMetadata(
        id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
        title="MyClaw implementation",
        created_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
        updated_at=datetime(2026, 7, 11, tzinfo=LOCAL_OFFSET),
        consolidation_cursor=1,
        cumulative_usage=CumulativeUsage(
            model_calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        ),
    )

    with pytest.raises(ValueError, match="message boundary"):
        ConversationSession(metadata=metadata, messages=())
