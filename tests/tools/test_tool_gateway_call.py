from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, cast

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.runtime_log import install_runtime_logging
from myclaw.tools.base import BaseTool
from myclaw.tools.errors import ToolError
from myclaw.tools.models import ModelToolCall
from myclaw.tools.schema import OpenAIToolSchema, ToolParam
from myclaw.tools.tool_gateway import ToolGateway


def _call(name: str, arguments: str, *, call_id: str = "call_1") -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments=arguments)


class _PrepareTool(BaseTool):
    name = "prepare"
    description = "Exercise Tool argument preparation."
    required = ("text", "count", "enabled")

    text: Annotated[str, ToolParam(min_length=1)]
    count: Annotated[int, ToolParam(minimum=-10, maximum=10)]
    enabled: bool
    suffix: str | None = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool, str | None]] = []

    async def execute(
        self,
        *,
        text: str,
        count: int,
        enabled: bool,
        suffix: str | None,
    ) -> str:
        self.calls.append((text, count, enabled, suffix))
        return f"{text}:{count}:{enabled}:{suffix}"


@pytest.mark.asyncio
async def test_registered_catalog_preserves_order_and_returns_defensive_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _PrepareTool()

    class SecondTool(BaseTool):
        name = "second"
        description = "Second registered Tool."

        async def execute(self) -> str:
            return "second"

    second = SecondTool()
    schema_calls: list[str] = []
    original = BaseTool.to_schema

    def counted(tool: BaseTool) -> OpenAIToolSchema:
        schema_calls.append(tool.name)
        return original(tool)

    monkeypatch.setattr(BaseTool, "to_schema", counted)
    gateway = ToolGateway()
    gateway.register_tools((first, second))

    assert schema_calls == ["prepare", "second"]
    schemas = gateway.schemas
    assert [cast(dict[str, object], schema["function"])["name"] for schema in schemas] == [
        "prepare",
        "second",
    ]
    cast(dict[str, object], schemas[0]["function"])["name"] = "mutated"
    assert cast(dict[str, object], gateway.schemas[0]["function"])["name"] == "prepare"
    assert schema_calls == ["prepare", "second"]

    result = await gateway.call(_call("second", "{}"))
    assert result.status == "success"
    assert result.content == "second"
    with pytest.raises(RuntimeError, match="already been registered"):
        gateway.register_tools(())


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{", "[]", "null", '"text"', "1"])
async def test_call_rejects_malformed_or_non_object_json(arguments: str) -> None:
    gateway = ToolGateway()
    gateway.register_tools((_PrepareTool(),))

    result = await gateway.call(_call("prepare", arguments))

    assert result.status == "error"
    assert result.content == "Tool arguments could not be parsed."


@pytest.mark.asyncio
async def test_call_projects_unknown_fields_converts_safe_values_and_fills_defaults() -> None:
    tool = _PrepareTool()
    gateway = ToolGateway()
    gateway.register_tools((tool,))

    result = await gateway.call(
        _call(
            "prepare",
            '{"text":"hello","count":"-2","enabled":"TrUe","unknown":{"x":1}}',
        )
    )

    assert result.status == "success"
    assert result.content == "hello:-2:True:None"
    assert tool.calls == [("hello", -2, True, None)]


@pytest.mark.asyncio
async def test_call_accepts_finite_integral_floats_and_explicit_null() -> None:
    tool = _PrepareTool()
    gateway = ToolGateway()
    gateway.register_tools((tool,))

    result = await gateway.call(
        _call("prepare", '{"text":"hello","count":2.0,"enabled":false,"suffix":null}')
    )

    assert result.status == "success"
    assert tool.calls == [("hello", 2, False, None)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        '{"count":1,"enabled":true}',
        '{"text":"","count":1,"enabled":true}',
        '{"text":"hello","count":true,"enabled":true}',
        '{"text":"hello","count":1.5,"enabled":true}',
        '{"text":"hello","count":" 1","enabled":true}',
        '{"text":"hello","count":11,"enabled":true}',
        '{"text":1,"count":1,"enabled":true}',
        '{"text":"hello","count":1,"enabled":"yes"}',
    ],
)
async def test_call_rejects_missing_constrained_or_unsafe_values(arguments: str) -> None:
    tool = _PrepareTool()
    gateway = ToolGateway()
    gateway.register_tools((tool,))

    result = await gateway.call(_call("prepare", arguments))

    assert result.status == "error"
    assert result.content == "Invalid arguments for prepare."
    assert tool.calls == []


@pytest.mark.asyncio
async def test_call_returns_stable_unavailable_tool_error_after_parsing() -> None:
    gateway = ToolGateway()
    gateway.register_tools(())

    result = await gateway.call(_call("missing", "{}"))

    assert result.status == "error"
    assert result.content == "The requested tool is not available."


@pytest.mark.asyncio
async def test_refusal_skips_execution_and_retry() -> None:
    class RefusingTool(BaseTool):
        name = "refusing"
        description = "Always refuse."
        required = ("action",)
        max_retries = 5
        action: str

        def __init__(self) -> None:
            self.calls = 0

        def refusal_reason(self, *, action: str) -> str | None:
            return f"{action} is refused."

        async def execute(self, *, action: str) -> str:
            self.calls += 1
            return action

    tool = RefusingTool()
    sleeps: list[float] = []
    gateway = ToolGateway(sleep=_sleep_recorder(sleeps))
    gateway.register_tools((tool,))

    result = await gateway.call(_call("refusing", '{"action":"write"}'))

    assert result.status == "refused"
    assert result.content == "write is refused."
    assert tool.calls == 0
    assert sleeps == []


@pytest.mark.asyncio
async def test_invalid_unavailable_and_refused_calls_do_not_create_runtime_logs(
    agent_home: Path,
) -> None:
    class RefusingTool(BaseTool):
        name = "refusing"
        description = "Always refuse."
        required = ("action",)
        action: str

        def refusal_reason(self, *, action: str) -> str:
            return f"{action} is refused."

        async def execute(self, *, action: str) -> str:
            raise AssertionError(action)

    gateway = ToolGateway()
    gateway.register_tools((_PrepareTool(), RefusingTool()))
    lifetime = install_runtime_logging(AgentHome(agent_home))

    with lifetime.session("foreground-session-51"):
        malformed = await gateway.call(_call("prepare", "{"))
        invalid = await gateway.call(_call("prepare", '{}'))
        unavailable = await gateway.call(_call("missing", '{}'))
        refused = await gateway.call(_call("refusing", '{"action":"write"}'))
    lifetime.close()

    assert [result.status for result in (malformed, invalid, unavailable, refused)] == [
        "error",
        "error",
        "error",
        "refused",
    ]
    assert not (agent_home / "logs").exists()


@pytest.mark.asyncio
async def test_call_normalizes_expected_unexpected_and_non_string_failures() -> None:
    class FailingTool(BaseTool):
        name = "failing"
        description = "Fail in a selected mode."
        required = ("mode",)
        mode: str

        async def execute(self, *, mode: str) -> str:
            if mode == "expected":
                raise ToolError("The expected operation failed.")
            if mode == "unexpected":
                raise RuntimeError("secret implementation detail")
            if mode == "non-string":
                return cast(str, 42)
            return "ok"

    gateway = ToolGateway()
    gateway.register_tools((FailingTool(),))

    expected = await gateway.call(_call("failing", '{"mode":"expected"}'))
    unexpected = await gateway.call(_call("failing", '{"mode":"unexpected"}'))
    non_string = await gateway.call(_call("failing", '{"mode":"non-string"}'))

    assert (expected.status, expected.content) == ("error", "The expected operation failed.")
    assert (unexpected.status, unexpected.content) == (
        "error",
        "failing could not complete the request.",
    )
    assert "secret" not in unexpected.content
    assert (non_string.status, non_string.content) == (
        "error",
        "failing could not complete the request.",
    )


@pytest.mark.asyncio
async def test_retries_are_extra_attempts_with_bounded_exponential_delays() -> None:
    class RetryTool(BaseTool):
        name = "retry"
        description = "Fail every attempt."
        max_retries = 5

        def __init__(self) -> None:
            self.calls = 0

        async def execute(self) -> str:
            self.calls += 1
            if self.calls == 6:
                raise ToolError("The final attempt failed safely.")
            raise RuntimeError(f"failure {self.calls}")

    tool = RetryTool()
    sleeps: list[float] = []
    gateway = ToolGateway(sleep=_sleep_recorder(sleeps))
    gateway.register_tools((tool,))

    result = await gateway.call(_call("retry", "{}"))

    assert tool.calls == 6
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert result.status == "error"
    assert result.content == "The final attempt failed safely."


@pytest.mark.asyncio
async def test_retryable_execution_failures_log_retries_and_one_terminal_error(
    agent_home: Path,
) -> None:
    class RetryTool(BaseTool):
        name = "retry"
        description = "Fail every attempt without exposing input or boundary details."
        required = ("payload",)
        max_retries = 2
        payload: str

        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, *, payload: str) -> str:
            self.calls += 1
            try:
                raise OSError(
                    "RAW_RESPONSE_BODY_51 https://user:credential@example.test/private"
                )
            except OSError as cause:
                raise ToolError("The retry Tool failed safely.") from cause

    tool = RetryTool()
    sleeps: list[float] = []
    gateway = ToolGateway(sleep=_sleep_recorder(sleeps))
    gateway.register_tools((tool,))
    lifetime = install_runtime_logging(AgentHome(agent_home))

    with lifetime.session("scheduled-session-51"):
        result = await gateway.call(
            _call("retry", '{"payload":"RAW_TOOL_ARGUMENT_51"}')
        )
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert (result.status, result.content, result.artifact) == (
        "error",
        "The retry Tool failed safely.",
        None,
    )
    assert tool.calls == 3
    assert sleeps == [1.0, 2.0]
    assert content.count(" WARNING ") == 2
    assert content.count(" ERROR ") == 1
    assert (
        "session=scheduled-session-51 myclaw.tools.tool_gateway: "
        "Tool execution failed name=retry attempt=1/3 type=ToolError"
    ) in content
    assert "name=retry attempt=2/3 type=ToolError" in content
    assert "name=retry attempt=3/3 type=ToolError" in content
    assert "Traceback (most recent call last):" in content
    assert content.count("ToolError: [REDACTED]") == 3
    assert content.count("OSError: [REDACTED]") == 3
    assert content.count(
        "The above exception was the direct cause of the following exception:"
    ) == 3
    assert "RAW_TOOL_ARGUMENT_51" not in content
    assert "RAW_RESPONSE_BODY_51" not in content
    assert "credential" not in content
    assert "The retry Tool failed safely." not in content


@pytest.mark.asyncio
async def test_retry_can_recover_from_an_ordinary_execution_failure() -> None:
    class RecoveringTool(BaseTool):
        name = "recovering"
        description = "Recover on a retry."
        max_retries = 2

        def __init__(self) -> None:
            self.calls = 0

        async def execute(self) -> str:
            self.calls += 1
            if self.calls < 3:
                raise ToolError("temporary")
            return "recovered"

    tool = RecoveringTool()
    sleeps: list[float] = []
    gateway = ToolGateway(sleep=_sleep_recorder(sleeps))
    gateway.register_tools((tool,))

    result = await gateway.call(_call("recovering", "{}"))

    assert result.status == "success"
    assert result.content == "recovered"
    assert tool.calls == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_cancellation_propagates_without_retry() -> None:
    class CancelledTool(BaseTool):
        name = "cancelled"
        description = "Propagate cancellation."
        max_retries = 5

        def __init__(self) -> None:
            self.calls = 0

        async def execute(self) -> str:
            self.calls += 1
            raise asyncio.CancelledError

    tool = CancelledTool()
    sleeps: list[float] = []
    gateway = ToolGateway(sleep=_sleep_recorder(sleeps))
    gateway.register_tools((tool,))

    with pytest.raises(asyncio.CancelledError):
        await gateway.call(_call("cancelled", "{}"))
    assert tool.calls == 1
    assert sleeps == []


def _sleep_recorder(delays: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(delay: float) -> None:
        delays.append(delay)

    return sleep
