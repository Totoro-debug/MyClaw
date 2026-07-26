import asyncio
from collections.abc import Awaitable, Callable

import pytest

from myclaw.tools.models import ModelToolCall
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.tools.web.web_fetch import WebFetchRejected, WebFetchTool
from myclaw.tools.web.web_search import WebSearchResult, WebSearchTool


class ScriptedSearch:
    def __init__(self, outcomes: list[BaseException | tuple[WebSearchResult, ...]]) -> None:
        self._outcomes = iter(outcomes)
        self.calls = 0

    async def search(self, query: str, max_results: int) -> tuple[WebSearchResult, ...]:
        del query, max_results
        self.calls += 1
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ScriptedFetch:
    def __init__(self, outcomes: list[BaseException | str]) -> None:
        self._outcomes = iter(outcomes)
        self.calls = 0

    async def fetch(self, url: str) -> str:
        del url
        self.calls += 1
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _recording_sleep(waits: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(delay: float) -> None:
        waits.append(delay)

    return sleep


def test_web_tools_export_exact_schemas_and_two_extra_retries() -> None:
    search = WebSearchTool(search=ScriptedSearch([()]))
    fetch = WebFetchTool(fetcher=ScriptedFetch(["unused"]))

    assert search.to_schema() == {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web and return normalized result summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Public web search query.",
                        "minLength": 1,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum normalized results.",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }
    assert fetch.to_schema() == {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch readable text from a public HTTP or HTTPS URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Public HTTP or HTTPS URL.",
                        "minLength": 1,
                        "format": "uri",
                    }
                },
                "required": ["url"],
            },
        },
    }
    assert search.max_retries == 2
    assert fetch.max_retries == 2


@pytest.mark.asyncio
async def test_web_search_waits_one_then_two_seconds_before_final_success() -> None:
    waits: list[float] = []
    expected = WebSearchResult(
        title="Result",
        url="https://example.com/result",
        snippet="Public snippet.",
    )
    search = ScriptedSearch([OSError("first"), RuntimeError("second"), (expected,)])
    gateway = ToolGateway(sleep=_recording_sleep(waits))
    gateway.register_tools((WebSearchTool(search=search),))

    result = await gateway.call(
        ModelToolCall(
            id="call_search_retry",
            name="web_search",
            arguments='{"query":"runtime"}',
        )
    )

    assert result.status == "success"
    assert search.calls == 3
    assert waits == [1.0, 2.0]


@pytest.mark.asyncio
async def test_web_fetch_expected_rejection_is_safe_after_retry_exhaustion() -> None:
    waits: list[float] = []
    fetch = ScriptedFetch(
        [
            WebFetchRejected("private first detail"),
            WebFetchRejected("private second detail"),
            WebFetchRejected("private final detail"),
        ]
    )
    gateway = ToolGateway(sleep=_recording_sleep(waits))
    gateway.register_tools((WebFetchTool(fetcher=fetch),))

    result = await gateway.call(
        ModelToolCall(
            id="call_fetch_rejected",
            name="web_fetch",
            arguments='{"url":"https://public.example/page"}',
        )
    )

    assert result.status == "error"
    assert result.content == "WebFetch rejected an unsafe or unverifiable request."
    assert "private" not in result.content
    assert fetch.calls == 3
    assert waits == [1.0, 2.0]


@pytest.mark.asyncio
async def test_web_fetch_unexpected_failure_is_generic_after_retry_exhaustion() -> None:
    waits: list[float] = []
    fetch = ScriptedFetch([OSError("private detail") for _ in range(3)])
    gateway = ToolGateway(sleep=_recording_sleep(waits))
    gateway.register_tools((WebFetchTool(fetcher=fetch),))

    result = await gateway.call(
        ModelToolCall(
            id="call_fetch_failure",
            name="web_fetch",
            arguments='{"url":"https://public.example/page"}',
        )
    )

    assert result.status == "error"
    assert result.content == "web_fetch could not complete the request."
    assert fetch.calls == 3
    assert waits == [1.0, 2.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ("web_search", "web_fetch"))
async def test_web_tool_cancellation_propagates_without_retry(name: str) -> None:
    waits: list[float] = []
    gateway = ToolGateway(sleep=_recording_sleep(waits))
    counter: ScriptedSearch | ScriptedFetch
    if name == "web_search":
        search = ScriptedSearch([asyncio.CancelledError()])
        gateway.register_tools((WebSearchTool(search=search),))
        arguments = '{"query":"runtime"}'
        counter = search
    else:
        fetch = ScriptedFetch([asyncio.CancelledError()])
        gateway.register_tools((WebFetchTool(fetcher=fetch),))
        arguments = '{"url":"https://public.example/page"}'
        counter = fetch

    with pytest.raises(asyncio.CancelledError):
        await gateway.call(
            ModelToolCall(
                id="call_cancelled_web",
                name=name,
                arguments=arguments,
            )
        )

    assert counter.calls == 1
    assert waits == []
