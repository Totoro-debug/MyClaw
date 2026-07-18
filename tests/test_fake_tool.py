import pytest

from myclaw.tools.models import ToolDefinition
from tests.fixtures.tool import FakeTool, FakeToolCall


@pytest.mark.asyncio
async def test_fake_tool_returns_a_scripted_result_and_records_the_call() -> None:
    definition = ToolDefinition(
        name="read_file",
        description="Read a UTF-8 file.",
        input_schema={"type": "object"},
    )
    context = {"lane": "foreground", "workspace": "workspace"}
    arguments = {"path": "CONTEXT.md"}
    tool = FakeTool(definition=definition, outcomes=["file contents"])

    result = await tool.execute(arguments, context)

    assert tool.definition is definition
    assert result == "file contents"
    assert tool.calls == [FakeToolCall(arguments=arguments, context=context)]


@pytest.mark.asyncio
async def test_fake_tool_raises_a_scripted_failure_after_recording_the_call() -> None:
    failure = OSError("disk unavailable")
    tool = FakeTool(
        definition=ToolDefinition(
            name="read_file",
            description="Read a UTF-8 file.",
            input_schema={"type": "object"},
        ),
        outcomes=[failure],
    )

    with pytest.raises(OSError, match="disk unavailable"):
        await tool.execute({"path": "missing.txt"}, "foreground context")

    assert tool.calls == [
        FakeToolCall(
            arguments={"path": "missing.txt"},
            context="foreground context",
        )
    ]
