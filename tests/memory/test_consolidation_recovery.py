from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.memory.conversation_summary import WorkspaceJsonlSummaryStore
from myclaw.provider.models import AssistantModelMessage, ModelCompleted, ModelResponse, ModelUsage
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 4, 16, 0, 0, tzinfo=LOCAL_OFFSET)


def _state(workspace: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_foreground_runtime_does_not_recover_pending_journals(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state = _state(workspace)
    summaries = WorkspaceJsonlSummaryStore(state)
    summaries.pending_directory.mkdir()
    journal = summaries.pending_directory / "invalid.json"
    journal.write_text("not a production recovery input", encoding="utf-8")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(ModelCompleted(response=_response("Chat answer.")),),
            ),
        )
    )

    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    try:
        events = [event async for event in runtime.conversation.submit("Continue the chat.")]
    finally:
        await runtime.close()

    assert [event.type for event in events] == ["turn_started", "turn_completed"]
    assert journal.read_text(encoding="utf-8") == "not a production recovery input"
    assert provider.complete_requests == []
