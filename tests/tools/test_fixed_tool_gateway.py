from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.schedule.service import ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.tools.core.schedule import ScheduleTool
from myclaw.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
    ToolGateway,
)


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 7, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


def _gateway(
    workspace: Path,
    agent_home: Path,
    *,
    skill_root: Path | None = None,
) -> ToolGateway:
    identity = Workspace.from_path(workspace)
    state = WorkspaceState(identity)
    state.initialize(agent_home_root=agent_home)
    return ToolGateway(
        workspace=identity,
        schedule_service=ScheduleService(
            store=WorkspaceScheduleStore(state),
            clock=_Clock(),
        ),
        skill_root=skill_root,
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
async def test_fixed_gateway_reads_skill_root_without_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    skill_file = agent_home / "skills" / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_bytes(b"---\nname: review\n---\nbody\n")
    requests: list[ConfirmationRequest] = []

    async def unexpected_confirmation(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    gateway = _gateway(workspace, agent_home, skill_root=agent_home / "skills")
    result = await gateway.call(
        ModelToolCall(
            id="call_skill_read",
            name="read_file",
            arguments=json.dumps({"path": str(skill_file)}),
        ),
        confirmation=unexpected_confirmation,
    )

    assert (result.status, result.content) == ("success", "---\nname: review\n---\nbody\n")
    assert requests == []
    assert len(gateway.schemas) == 10


@pytest.mark.asyncio
async def test_foreground_and_scheduled_catalogs_always_include_web_and_exec(
    workspace: Path,
    agent_home: Path,
) -> None:
    foreground = _gateway(workspace, agent_home)
    scheduled = _gateway(workspace, agent_home)

    assert _names(foreground) == _names(scheduled)
    token = ScheduleTool._in_schedule_job.set(True)
    try:
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
    finally:
        ScheduleTool._in_schedule_job.reset(token)

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
