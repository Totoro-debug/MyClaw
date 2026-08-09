import asyncio
import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from myclaw.agent.events import ToolCompletedPayload
from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolModelMessage,
)
from myclaw.tools.tool_gateway import ModelToolCall, ToolGateway
from myclaw.tools.web.web_search import (
    AsyncioDuckDuckGoSearchProcessSpawner,
    DuckDuckGoSearchBoundary,
    WebSearchResult,
    WebSearchTool,
)
from myclaw.tools.web.web_search import main as run_web_search_worker
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import FakeClock, ScriptedFakeProvider, StreamScript

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
WINDOWS_NEW_GROUP_NO_WINDOW = 0x08000200


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


class ByteOutput:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class CompletedSearchProcess:
    returncode = 0

    def __init__(self, records: list[dict[str, object]]) -> None:
        self._stdout = json.dumps(records, ensure_ascii=False).encode("utf-8")

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, b""

    def terminate(self) -> None:
        raise AssertionError("completed search process must not be terminated")

    async def wait(self) -> int:
        return self.returncode


class RecordingSearchProcessSpawner:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self._records = records
        self.calls: list[tuple[str, int]] = []

    async def spawn(self, query: str, max_results: int) -> CompletedSearchProcess:
        self.calls.append((query, max_results))
        return CompletedSearchProcess(self._records)


def test_duckduckgo_worker_writes_non_ascii_results_as_utf8_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDuckDuckGo:
        def __enter__(self) -> "FakeDuckDuckGo":
            return self

        def __exit__(self, *errors: object) -> None:
            del errors

        def text(
            self,
            query: str,
            *,
            max_results: int,
            backend: str,
        ) -> list[dict[str, object]]:
            assert (query, max_results, backend) == ("运行时资源", 1, "duckduckgo")
            return [
                {
                    "title": "运行时关闭",
                    "href": "https://example.com/runtime",
                    "body": "取消后等待资源释放。",
                }
            ]

    output = ByteOutput()
    monkeypatch.setattr("myclaw.tools.web.web_search.DDGS", FakeDuckDuckGo)
    monkeypatch.setattr("myclaw.tools.web.web_search.sys.stdout", output)

    assert run_web_search_worker(["运行时资源", "1"]) == 0
    assert json.loads(output.buffer.getvalue().decode("utf-8")) == [
        {
            "title": "运行时关闭",
            "href": "https://example.com/runtime",
            "body": "取消后等待资源释放。",
        }
    ]


def test_duckduckgo_boundary_rejects_synchronous_sdk_execution() -> None:
    constructor: Callable[..., DuckDuckGoSearchBoundary] = DuckDuckGoSearchBoundary

    with pytest.raises(TypeError, match="text_search"):
        constructor(text_search=lambda _query, *, max_results: [])


@pytest.mark.asyncio
async def test_default_duckduckgo_process_isolated_from_foreground_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options: dict[str, object] = {}

    async def create_process(*command: str, **kwargs: object) -> asyncio.subprocess.Process:
        del command
        options.update(kwargs)
        return cast(asyncio.subprocess.Process, object())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    await AsyncioDuckDuckGoSearchProcessSpawner().spawn("runtime shutdown", 3)

    assert options == {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "creationflags": WINDOWS_NEW_GROUP_NO_WINDOW,
    }


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
    gateway = ToolGateway()
    gateway.register_tools((WebSearchTool(search=search),))

    result = await gateway.call(
        ModelToolCall(
            id="call_search",
            name="web_search",
            arguments='{"query":"MyClaw agent runtime","max_results":2}',
        )
    )

    assert result.status == "success"
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
    spawner = RecordingSearchProcessSpawner(
        [
            {
                "title": "DuckDuckGo 运行时结果",
                "href": "https://example.com/result",
                "body": "进程返回的结果摘要。",
                "provider_metadata": "must not escape",
            }
        ]
    )
    gateway = ToolGateway()
    gateway.register_tools(
        (WebSearchTool(search=DuckDuckGoSearchBoundary(process_spawner=spawner)),)
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_duckduckgo",
            name="web_search",
            arguments='{"query":"provider mapping","max_results":1}',
        )
    )

    assert result.status == "success"
    assert json.loads(result.content) == [
        {
            "title": "DuckDuckGo 运行时结果",
            "url": "https://example.com/result",
            "snippet": "进程返回的结果摘要。",
        }
    ]
    assert spawner.calls == [("provider mapping", 1)]


@pytest.mark.asyncio
async def test_duckduckgo_search_cancellation_terminates_and_waits_for_its_process() -> None:
    class BlockingSearchProcess:
        def __init__(self) -> None:
            self.communicate_started = asyncio.Event()
            self._stopped = asyncio.Event()
            self.terminated = False
            self.waited = False
            self.returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            self.communicate_started.set()
            await self._stopped.wait()
            return b"[]", b""

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -1
            self._stopped.set()

        async def wait(self) -> int:
            self.waited = True
            await self._stopped.wait()
            return -1

    class OneProcessSpawner:
        def __init__(self, process: BlockingSearchProcess) -> None:
            self._process = process

        async def spawn(self, query: str, max_results: int) -> BlockingSearchProcess:
            assert (query, max_results) == ("blocking search", 3)
            return self._process

    process = BlockingSearchProcess()
    search = DuckDuckGoSearchBoundary(process_spawner=OneProcessSpawner(process))
    searching = asyncio.create_task(search.search("blocking search", 3))
    await process.communicate_started.wait()

    searching.cancel()
    with pytest.raises(asyncio.CancelledError):
        await searching

    assert process.terminated
    assert process.waited


@pytest.mark.asyncio
async def test_duckduckgo_search_cancellation_joins_spawn_before_reaping_process() -> None:
    class CreatedSearchProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.waited = False
            self.returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            raise AssertionError("communication must not start after cancelled spawn")

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -1

        async def wait(self) -> int:
            self.waited = True
            return -1

    class DelayedSpawner:
        def __init__(self, process: CreatedSearchProcess) -> None:
            self._process = process
            self.created = asyncio.Event()
            self.release = asyncio.Event()

        async def spawn(self, query: str, max_results: int) -> CreatedSearchProcess:
            del query, max_results
            self.created.set()
            await self.release.wait()
            return self._process

    process = CreatedSearchProcess()
    spawner = DelayedSpawner(process)
    search = DuckDuckGoSearchBoundary(process_spawner=spawner)
    searching = asyncio.create_task(search.search("cancel during spawn", 2))
    await spawner.created.wait()

    searching.cancel()
    searching.cancel()
    await asyncio.sleep(0)
    try:
        assert not searching.done()
    finally:
        spawner.release.set()
        if searching.done():
            process.terminate()
            await process.wait()
        else:
            with pytest.raises(asyncio.CancelledError):
                await searching

    assert process.terminated
    assert process.waited


@pytest.mark.asyncio
async def test_duckduckgo_search_cancellation_reaps_a_same_tick_spawn() -> None:
    class CreatedSearchProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.waited = False
            self.returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            raise AssertionError("communication must not start after cancelled spawn")

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -1

        async def wait(self) -> int:
            self.waited = True
            return -1

    class SameTickSpawner:
        def __init__(self, process: CreatedSearchProcess) -> None:
            self._process = process
            self.created = asyncio.Event()

        async def spawn(self, query: str, max_results: int) -> CreatedSearchProcess:
            assert (query, max_results) == ("same tick cancellation", 1)
            self.created.set()
            return self._process

    process = CreatedSearchProcess()
    spawner = SameTickSpawner(process)
    search = DuckDuckGoSearchBoundary(process_spawner=spawner)
    searching = asyncio.create_task(search.search("same tick cancellation", 1))
    await spawner.created.wait()

    searching.cancel()
    with pytest.raises(asyncio.CancelledError):
        await searching

    assert process.terminated
    assert process.waited


@pytest.mark.asyncio
async def test_duckduckgo_search_preserves_a_process_spawn_failure() -> None:
    class FailingSpawner:
        async def spawn(self, query: str, max_results: int) -> CompletedSearchProcess:
            assert (query, max_results) == ("spawn failure", 2)
            raise LookupError("search process could not start")

    search = DuckDuckGoSearchBoundary(process_spawner=FailingSpawner())

    with pytest.raises(LookupError, match="search process could not start") as raised:
        await search.search("spawn failure", 2)

    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_tool_gateway_returns_an_empty_array_when_web_search_finds_nothing(
    agent_home: Path,
    workspace: Path,
) -> None:
    search = FakeWebSearchBoundary(())
    gateway = ToolGateway()
    gateway.register_tools((WebSearchTool(search=search),))

    assert [schema["function"]["name"] for schema in gateway.schemas] == ["web_search"]

    result = await gateway.call(
        ModelToolCall(
            id="call_empty_search",
            name="web_search",
            arguments='{"query":"no matching page"}',
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
        arguments='{"query":"MyClaw runtime","max_results":3}',
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
    assert search.calls == [("MyClaw runtime", 3)] * 3
    assert "private upstream" not in repr(events)
    assert len(provider.stream_requests) == 2
    first_request = provider.stream_requests[0]
    second_request = provider.stream_requests[1]
    assert isinstance(first_request, ModelRequest)
    assert isinstance(second_request, ModelRequest)
    assert "web_search" in [schema["function"]["name"] for schema in first_request.tools]
    tool_message = second_request.messages[-1]
    assert isinstance(tool_message, ToolModelMessage)
    assert tool_message.content == "web_search could not complete the request."
    persisted = runtime.session.messages[2]
    assert persisted["role"] == "tool"


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
    spawner = RecordingSearchProcessSpawner(
        [
            {
                "title": "MyClaw reference",
                "href": "https://example.com/reference",
                "body": "Public runtime documentation.",
                "provider_metadata": "must not reach the conversation",
            }
        ]
    )
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
                                        arguments='{"query":"MyClaw docs","max_results":1}',
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
        web_search=DuckDuckGoSearchBoundary(process_spawner=spawner),
    )

    events = [event async for event in runtime.conversation.submit("Find the docs.")]

    assert events[-1].type == "turn_completed"
    assert spawner.calls == [("MyClaw docs", 1)]
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
    persisted = runtime.session.messages[2]
    assert persisted["role"] == "tool"
    assert persisted["status"] == "success"
    assert json.loads(persisted["content"]) == expected


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
    assert [schema["function"]["name"] for schema in request.tools] == [
        "read_file",
        "list_dir",
        "glob",
        "search_files",
        "write_file",
        "edit_file",
        "shell",
        "schedule",
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
    assert [schema["function"]["name"] for schema in request.tools] == [
        "read_file",
        "list_dir",
        "glob",
        "search_files",
        "write_file",
        "edit_file",
        "web_search",
        "web_fetch",
        "shell",
        "schedule",
    ]
