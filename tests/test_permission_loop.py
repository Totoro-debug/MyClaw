from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.runtime import prepare_runtime
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
)
from myclaw.tools.tool_gateway import ModelToolCall
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import (
    ScriptedFakeProvider,
    StreamScript,
)
from tests.runtime_bus import collect_foreground_outbound

NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=timezone(timedelta(hours=8)))


@pytest.mark.asyncio
async def test_foreground_mutations_execute_without_a_permission_pause(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    target = workspace / "notes.txt"
    target.write_text("before", encoding="utf-8")
    calls = (
        ModelToolCall(
            id="call_write",
            name="write_file",
            arguments='{"path":"created.txt","content":"must not be written"}',
        ),
        ModelToolCall(
            id="call_edit",
            name="edit_file",
            arguments='{"path":"notes.txt","old_text":"before","new_text":"after"}',
        ),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=calls),
                            usage=ModelUsage(
                                input_tokens=8,
                                output_tokens=2,
                                total_tokens=10,
                            ),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Mutations were refused."),
                            usage=ModelUsage(
                                input_tokens=12,
                                output_tokens=3,
                                total_tokens=15,
                            ),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    runtime = prepare_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    confirmations: list[object] = []
    runtime.control.bind_confirmation_callback(confirmations.append)
    try:
        await runtime.start()
        messages = await collect_foreground_outbound(runtime, "Change the files.")
    finally:
        await runtime.close()

    assert confirmations == []
    assert (workspace / "created.txt").read_text(encoding="utf-8") == "must not be written"
    assert target.read_text(encoding="utf-8") == "after"
    tool_messages = [message for message in runtime.session.messages if message["role"] == "tool"]
    assert [message["status"] for message in tool_messages] == ["success", "success"]
    follow_up = provider.stream_requests[1]
    model_results = [message for message in follow_up.messages if message["role"] == "tool"]
    assert [(message["name"], message["content"]) for message in model_results] == [
        ("write_file", "File written successfully."),
        ("edit_file", "File edited successfully."),
    ]
    assert messages[-1].metadata == {"_streamed": True}
