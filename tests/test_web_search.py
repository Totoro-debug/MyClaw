import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent_home import AgentHome
from myclaw.config import ConfigLoader
from myclaw.contracts import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    ToolCompletedPayload,
    ToolExecutionContext,
    ToolModelMessage,
    ToolSessionMessage,
)
from myclaw.runtime import prepare_repl_runtime
from myclaw.tool_gateway import ToolGateway
from myclaw.web_search import DuckDuckGoSearchBoundary, WebSearchResult, WebSearchTool
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript
from tests.test_config import VALID_CONFIG

SESSION_UUIDS = (
    "550e8400-e29b-41d4-a716-446655440000",
    "0f8fad5b-d9cb-469f-a165-70867728950e",
    "7c9e6679-7425-40de-944b-e07fc1f90ae7",
    "9b2c3a42-1d2e-4a1e-a827-61f36dc54713",
    "a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34",
    "6fa459ea-ee8a-4ca4-894e-db77e160355e",
    "16fd2706-8baf-433b-82eb-8c7fada847da",
    "886313e1-3b8a-4a2d-9f7f-77611a4b6f4e",
)
NOW = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)


class FakeWebSearchBoundary:
    def __init__(self, results: tuple[WebSearchResult, ...]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> tuple[WebSearchResult, ...]:
        self.calls.append((query, max_results))
        return self._results


class FailingWebSearchBoundary:
    def __init__(self, detail: str) -> None:
        self._detail = detail
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, max_results: int) -> tuple[WebSearchResult, ...]:
        self.calls.append((query, max_results))
        raise ConnectionError(self._detail)


def gateway_context(agent_home: Path, workspace: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        lane="foreground",
        workspace=workspace,
        agent_home=agent_home,
        session_id="20260712-120000-000000_550e8400-e29b-41d4-a716-446655440000",
    )


@pytest.mark.asyncio
async def test_tool_gateway_returns_provider_neutral_web_search_results(
    agent_home: Path,
    workspace: Path,
) -> None:
    search = FakeWebSearchBoundary(
        (
            WebSearchResult(
                title="MyClaw runtime",
                url="https://example.com/myclaw",
                snippet="A local-first Personal Agent runtime.",
            ),
            WebSearchResult(
                title="Tool Gateway guide",
                url="https://docs.example/gateway",
                snippet="Normalized tools for model calls.",
            ),
        )
    )
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        tools=(WebSearchTool(search),),
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_search",
            name="web_search",
            arguments={"query": "MyClaw agent runtime", "max_results": 2},
        )
    )

    assert result.status == "success"
    assert result.error is None
    assert json.loads(result.content) == [
        {
            "title": "MyClaw runtime",
            "url": "https://example.com/myclaw",
            "snippet": "A local-first Personal Agent runtime.",
        },
        {
            "title": "Tool Gateway guide",
            "url": "https://docs.example/gateway",
            "snippet": "Normalized tools for model calls.",
        },
    ]
    assert search.calls == [("MyClaw agent runtime", 2)]


@pytest.mark.asyncio
async def test_duckduckgo_boundary_maps_sdk_fields_before_the_gateway_returns_them(
    agent_home: Path,
    workspace: Path,
) -> None:
    calls: list[tuple[str, int]] = []

    def text_search(query: str, *, max_results: int) -> list[dict[str, object]]:
        calls.append((query, max_results))
        return [
            {
                "title": "DuckDuckGo result",
                "href": "https://example.com/result",
                "body": "Provider-shaped result body.",
                "provider_metadata": "must not escape",
            }
        ]

    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        tools=(WebSearchTool(DuckDuckGoSearchBoundary(text_search=text_search)),),
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_duckduckgo",
            name="web_search",
            arguments={"query": "provider mapping", "max_results": 1},
        )
    )

    assert result.status == "success"
    assert json.loads(result.content) == [
        {
            "title": "DuckDuckGo result",
            "url": "https://example.com/result",
            "snippet": "Provider-shaped result body.",
        }
    ]
    assert calls == [("provider mapping", 1)]


@pytest.mark.asyncio
async def test_tool_gateway_returns_an_empty_array_when_web_search_finds_nothing(
    agent_home: Path,
    workspace: Path,
) -> None:
    search = FakeWebSearchBoundary(())
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        web_search=search,
    )

    assert [definition.name for definition in gateway.definitions] == [
        "read_file",
        "list_files",
        "search_files",
        "write_file",
        "edit_file",
        "web_search",
    ]

    result = await gateway.execute(
        ModelToolCall(
            id="call_empty_search",
            name="web_search",
            arguments={"query": "no matching page"},
        )
    )

    assert result.status == "success"
    assert result.content == "[]"
    assert search.calls == [("no matching page", 5)]


@pytest.mark.asyncio
async def test_conversation_returns_one_safe_tool_error_for_a_web_search_network_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace("[tools.web]\nenabled = false", "[tools.web]\nenabled = true"),
        encoding="utf-8",
    )
    search = FailingWebSearchBoundary("private upstream response and network address")
    tool_call = ModelToolCall(
        id="call_web_failure",
        name="web_search",
        arguments={"query": "MyClaw runtime", "max_results": 3},
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=(tool_call,)),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="Search is temporarily unavailable."
                            ),
                            usage=ModelUsage(input_tokens=12, output_tokens=4, total_tokens=16),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    clock = FakeClock(NOW)
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=clock.now,
        new_uuid=iter(map(UUID, SESSION_UUIDS)).__next__,
        web_search=search,
    )

    events = [event async for event in runtime.conversation.submit("Search the public web.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    completed = events[2].payload
    assert isinstance(completed, ToolCompletedPayload)
    assert completed.status == "error"
    assert search.calls == [("MyClaw runtime", 3)]
    assert "private upstream" not in str([event.to_dict() for event in events])
    assert len(provider.stream_requests) == 2
    first_request = provider.stream_requests[0]
    second_request = provider.stream_requests[1]
    assert isinstance(first_request, ModelRequest)
    assert isinstance(second_request, ModelRequest)
    assert "web_search" in [definition.name for definition in first_request.tools]
    tool_message = second_request.messages[-1]
    assert isinstance(tool_message, ToolModelMessage)
    assert tool_message.content == "web_search could not complete the request."
    persisted = (await runtime.sessions.load(runtime.session_id)).messages[2]
    assert isinstance(persisted, ToolSessionMessage)
    assert persisted.error is not None
    assert persisted.error.code == "tool_failed"


@pytest.mark.asyncio
async def test_conversation_receives_only_provider_neutral_web_search_results(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace("[tools.web]\nenabled = false", "[tools.web]\nenabled = true"),
        encoding="utf-8",
    )
    calls: list[tuple[str, int]] = []

    def text_search(query: str, *, max_results: int) -> list[dict[str, object]]:
        calls.append((query, max_results))
        return [
            {
                "title": "MyClaw reference",
                "href": "https://example.com/reference",
                "body": "Public runtime documentation.",
                "provider_metadata": "must not reach the conversation",
            }
        ]

    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="",
                                tool_calls=(
                                    ModelToolCall(
                                        id="call_web_success",
                                        name="web_search",
                                        arguments={"query": "MyClaw docs", "max_results": 1},
                                    ),
                                ),
                            ),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="I found the reference."),
                            usage=ModelUsage(input_tokens=14, output_tokens=4, total_tokens=18),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=FakeClock(NOW).now,
        new_uuid=iter(map(UUID, SESSION_UUIDS)).__next__,
        web_search=DuckDuckGoSearchBoundary(text_search=text_search),
    )

    events = [event async for event in runtime.conversation.submit("Find the docs.")]

    assert events[-1].type == "turn_completed"
    assert calls == [("MyClaw docs", 1)]
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    tool_message = second_request.messages[-1]
    assert isinstance(tool_message, ToolModelMessage)
    expected = [
        {
            "title": "MyClaw reference",
            "url": "https://example.com/reference",
            "snippet": "Public runtime documentation.",
        }
    ]
    assert json.loads(tool_message.content) == expected
    assert "provider_metadata" not in tool_message.content
    persisted = (await runtime.sessions.load(runtime.session_id)).messages[2]
    assert isinstance(persisted, ToolSessionMessage)
    assert persisted.status == "success"
    assert json.loads(persisted.content) == expected


@pytest.mark.asyncio
async def test_conversation_catalog_omits_web_search_when_web_tools_are_disabled(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    search = FakeWebSearchBoundary(())
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="No web tools needed."),
                            usage=ModelUsage(input_tokens=5, output_tokens=3, total_tokens=8),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=FakeClock(NOW).now,
        new_uuid=iter(map(UUID, SESSION_UUIDS[:5])).__next__,
        web_search=search,
    )

    events = [event async for event in runtime.conversation.submit("Answer locally.")]

    assert events[-1].type == "turn_completed"
    request = provider.stream_requests[0]
    assert isinstance(request, ModelRequest)
    assert [definition.name for definition in request.tools] == [
        "read_file",
        "list_files",
        "search_files",
        "write_file",
        "edit_file",
        "shell",
        "create_scheduled_work",
    ]
    assert search.calls == []


@pytest.mark.asyncio
async def test_conversation_catalog_includes_builtin_web_search_when_enabled(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace("[tools.web]\nenabled = false", "[tools.web]\nenabled = true"),
        encoding="utf-8",
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Web search is available."),
                            usage=ModelUsage(input_tokens=6, output_tokens=3, total_tokens=9),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=FakeClock(NOW).now,
        new_uuid=iter(map(UUID, SESSION_UUIDS[:5])).__next__,
    )

    events = [event async for event in runtime.conversation.submit("What tools are available?")]

    assert events[-1].type == "turn_completed"
    request = provider.stream_requests[0]
    assert isinstance(request, ModelRequest)
    assert [definition.name for definition in request.tools] == [
        "read_file",
        "list_files",
        "search_files",
        "write_file",
        "edit_file",
        "web_search",
        "web_fetch",
        "shell",
        "create_scheduled_work",
    ]
