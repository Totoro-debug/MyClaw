from pathlib import Path

import pytest

from myclaw.contracts import (
    ArtifactReference,
    ErrorInfo,
    PermissionDecision,
    ToolExecutionContext,
    ToolResult,
)

SESSION_ID = "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000"


def test_normalized_tool_result_serializes_the_exact_artifact_shape() -> None:
    result = ToolResult(
        tool_call_id="call_123",
        name="read_file",
        status="success",
        content="preview",
        error=None,
        artifact=ArtifactReference(
            path=(
                "artifacts/20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000/call_123.txt"
            ),
            total_chars=73421,
            preview_chars=2000,
        ),
    )

    assert result.to_dict() == {
        "tool_call_id": "call_123",
        "name": "read_file",
        "status": "success",
        "content": "preview",
        "error": None,
        "artifact": {
            "path": (
                "artifacts/20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000/call_123.txt"
            ),
            "total_chars": 73421,
            "preview_chars": 2000,
        },
    }


def test_permission_decisions_and_execution_lanes_are_frozen_types() -> None:
    assert tuple(decision.value for decision in PermissionDecision) == ("allow", "ask", "deny")

    context = ToolExecutionContext(
        lane="foreground",
        workspace=Path("D:/desktop/project"),
        agent_home=Path("D:/users/user/.myclaw"),
        session_id="20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",
    )

    assert context.lane == "foreground"
    assert context.workspace == Path("D:/desktop/project")
    assert context.agent_home == Path("D:/users/user/.myclaw")
    assert context.session_id == ("20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000")


def test_tool_records_reject_values_outside_the_normalized_contract() -> None:
    with pytest.raises(ValueError, match="artifact path contract"):
        ArtifactReference(path="C:/secret.txt", total_chars=10, preview_chars=10)
    with pytest.raises(ValueError, match="preview_chars"):
        ArtifactReference(
            path=f"artifacts/{SESSION_ID}/call.txt",
            total_chars=10,
            preview_chars=11,
        )

    error = ErrorInfo(code="tool_failed", message="Tool failed.")
    with pytest.raises(ValueError, match="success result must not have an error"):
        ToolResult(
            tool_call_id="call_123",
            name="read_file",
            status="success",
            content="",
            error=error,
            artifact=None,
        )
    with pytest.raises(ValueError, match="non-success result requires an error"):
        ToolResult(
            tool_call_id="call_123",
            name="read_file",
            status="error",
            content="Tool failed.",
            error=None,
            artifact=None,
        )
    with pytest.raises(ValueError, match="Session ID"):
        ToolExecutionContext(
            lane="foreground",
            workspace=Path("D:/desktop/project"),
            agent_home=Path("D:/users/user/.myclaw"),
            session_id="20260711-153012-123456_123e4567-e89b-12d3-a456-426614174000",
        )


@pytest.mark.parametrize(
    "path",
    [
        f"C:/agent/artifacts/{SESSION_ID}/call_123.txt",
        f"artifacts/../{SESSION_ID}/call_123.txt",
        f"artifacts\\{SESSION_ID}\\call_123.txt",
        f"results/{SESSION_ID}/call_123.txt",
        "artifacts/session-id/call_123.txt",
        f"artifacts/{SESSION_ID}/call:123.txt",
        f"artifacts/{SESSION_ID}/.txt",
    ],
)
def test_artifact_reference_rejects_paths_outside_the_exact_persisted_shape(path: str) -> None:
    with pytest.raises(ValueError):
        ArtifactReference(path=path, total_chars=10, preview_chars=10)
