from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.tools.tool_gateway import (
    ConfirmationChannel,
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
    ToolGateway,
)


def _gateway(workspace: Path, agent_home: Path, *, scheduled_agent: bool = False) -> ToolGateway:
    identity = Workspace.from_path(workspace)
    state = WorkspaceState(identity)
    state.initialize(agent_home_root=agent_home)
    return ToolGateway(
        workspace=identity,
        schedule_store=WorkspaceScheduleStore(state),
        scheduled_agent=scheduled_agent,
    )


def _names(gateway: ToolGateway) -> list[str]:
    return [definition["function"]["name"] for definition in gateway.schemas]


def test_fixed_catalog_order_and_detached_definitions(
    workspace: Path,
    agent_home: Path,
) -> None:
    gateway = _gateway(workspace, agent_home)

    assert _names(gateway) == [
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "glob",
        "grep",
        "exec",
        "web_search",
        "web_fetch",
        "schedule",
    ]
    definitions = gateway.schemas
    assert isinstance(definitions, list)
    function = cast(dict[str, object], definitions[0]["function"])
    function["name"] = "changed"
    parameters = cast(dict[str, object], function["parameters"])
    properties = cast(dict[str, object], parameters["properties"])
    path = cast(dict[str, object], properties["path"])
    path["description"] = "changed"
    assert _names(gateway)[0] == "read_file"
    current_function = cast(dict[str, object], gateway.schemas[0]["function"])
    current_parameters = cast(dict[str, object], current_function["parameters"])
    current_properties = cast(dict[str, object], current_parameters["properties"])
    current_path = cast(dict[str, object], current_properties["path"])
    assert current_path["description"] != "changed"
    assert not hasattr(gateway, "register_tools")
    assert not hasattr(gateway, "for_run")
    assert not any(name in vars(gateway) for name in ("workspace", "schedule_store"))


@pytest.mark.asyncio
async def test_foreground_and_scheduled_catalogs_always_include_web_and_exec(
    workspace: Path,
    agent_home: Path,
) -> None:
    foreground = _gateway(workspace, agent_home)
    scheduled = _gateway(workspace, agent_home, scheduled_agent=True)

    assert _names(foreground) == _names(scheduled)
    refused = await scheduled.call(
        ModelToolCall(
            id="call_scheduled_add",
            name="schedule",
            arguments=json.dumps(
                {
                    "action": "add",
                    "message": "should be refused",
                    "every_seconds": 60,
                }
            ),
        )
    )

    assert refused.status == "refused"
    assert refused.content == "Schedule add is unavailable in scheduled Agent context."
    assert refused.confirmation is None


@pytest.mark.asyncio
async def test_fixed_gateway_calls_core_tool_and_returns_unified_result(
    workspace: Path,
    agent_home: Path,
) -> None:
    (workspace / "note.txt").write_text("hello\r\nworld\r\n", encoding="utf-8", newline="")
    result = await _gateway(workspace, agent_home).call(
        ModelToolCall(
            id="call_read",
            name="read_file",
            arguments='{"path":"note.txt","offset":1,"limit":1}',
        )
    )

    assert result.to_dict() == {
        "tool_call_id": "call_read",
        "name": "read_file",
        "status": "success",
        "content": "hello\r\n",
        "artifact": None,
    }


@pytest.mark.asyncio
async def test_confirmation_cancellation_propagates_and_invalidates_the_request(
    workspace: Path,
    agent_home: Path,
) -> None:
    outside = (workspace.parent / "cancelled-read.txt").resolve()
    outside.write_text("outside", encoding="utf-8")
    channel = ConfirmationChannel()
    task = asyncio.create_task(
        _gateway(workspace, agent_home).call(
            ModelToolCall(
                id="call_cancelled",
                name="read_file",
                arguments=json.dumps({"path": str(outside)}),
            ),
            confirmation=channel,
        )
    )
    request = await channel.next_request()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ValueError, match="late or unknown"):
        channel.respond_to_confirmation(request.confirmation_id, "approved")


@pytest.mark.asyncio
async def test_exec_confirmation_preserves_the_exact_normalized_operation(
    workspace: Path,
    agent_home: Path,
) -> None:
    command = f'printf "{"x" * 300}" && rm -rf "build output"'
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await _gateway(workspace, agent_home).call(
        ModelToolCall(
            id="call_long_exec",
            name="exec",
            arguments=json.dumps({"command": command, "cwd": ".", "timeout": 45}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details == {"command": command, "cwd": ".", "timeout": 45}


@pytest.mark.asyncio
async def test_unexpected_core_tool_failure_is_redacted(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (workspace / "note.txt").write_text("hello", encoding="utf-8")

    target = workspace / "note.txt"
    original_read_bytes = Path.read_bytes

    def fail_target_read(path: Path) -> bytes:
        if path == target:
            raise RuntimeError("secret implementation detail")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_target_read)
    result = await _gateway(workspace, agent_home).call(
        ModelToolCall(
            id="call_failure",
            name="read_file",
            arguments='{"path":"note.txt"}',
        )
    )

    assert result.status == "error"
    assert result.content == "read_file could not complete the request."
    assert "secret implementation detail" not in result.content
