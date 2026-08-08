import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from myclaw.tools.tool_gateway import ModelToolCall, ToolGateway
from myclaw.tools.web.web_fetch import WebFetchRejected, WebFetchTool
from myclaw.tools.web.web_search import WebSearchResult, WebSearchTool
from tests.fixtures.diagnostic_capture import capture_diagnostics


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
async def test_web_search_logs_retry_warnings_without_query_or_upstream_body(
    agent_home: Path,
) -> None:
    waits: list[float] = []
    expected = WebSearchResult(
        title="Result",
        url="https://example.com/result",
        snippet="Public snippet.",
    )
    search = ScriptedSearch(
        [
            OSError("RAW_SEARCH_RESPONSE_BODY_51"),
            RuntimeError("RAW_SEARCH_RESPONSE_BODY_52"),
            (expected,),
        ]
    )
    gateway = ToolGateway(sleep=_recording_sleep(waits))
    gateway.register_tools((WebSearchTool(search=search),))
    capture = capture_diagnostics()

    with capture.session("scheduled-web-search-session-51"):
        result = await gateway.call(
            ModelToolCall(
                id="call_search_retry",
                name="web_search",
                arguments='{"query":"RAW_WEB_QUERY_51"}',
            )
        )
    capture.close()

    assert result.status == "success"
    assert search.calls == 3
    assert waits == [1.0, 2.0]
    content = capture.text
    event_text = capture.event_text
    assert content.count(" WARNING ") == 2
    assert " ERROR " not in content
    assert "name=web_search attempt=1/3 type=OSError" in content
    assert "name=web_search attempt=2/3 type=RuntimeError" in content
    assert "RAW_WEB_QUERY_51" not in event_text
    assert "RAW_SEARCH_RESPONSE_BODY_51" not in event_text
    assert "RAW_SEARCH_RESPONSE_BODY_52" not in event_text
    assert "https://example.com/result" not in event_text
    assert "Public snippet." not in event_text
    assert "RAW_SEARCH_RESPONSE_BODY_51" in content
    assert "RAW_SEARCH_RESPONSE_BODY_52" in content


@pytest.mark.asyncio
async def test_web_fetch_failure_log_excludes_credential_url_and_response_body(
    agent_home: Path,
) -> None:
    waits: list[float] = []
    fetch = ScriptedFetch(
        [
            WebFetchRejected("RAW_FETCH_RESPONSE_BODY_51"),
            WebFetchRejected("RAW_FETCH_RESPONSE_BODY_52"),
            WebFetchRejected("RAW_FETCH_RESPONSE_BODY_53"),
        ]
    )
    gateway = ToolGateway(sleep=_recording_sleep(waits))
    gateway.register_tools((WebFetchTool(fetcher=fetch),))
    capture = capture_diagnostics()

    with capture.session("foreground-web-fetch-session-51"):
        result = await gateway.call(
            ModelToolCall(
                id="call_fetch_rejected",
                name="web_fetch",
                arguments=(
                    '{"url":"https://user:URL_CREDENTIAL_51@public.example/'
                    'RAW_FETCH_PATH_51?token=RAW_FETCH_QUERY_51"}'
                ),
            )
        )
    capture.close()

    assert result.status == "error"
    assert result.content == "WebFetch rejected an unsafe or unverifiable request."
    assert fetch.calls == 3
    assert waits == [1.0, 2.0]
    content = capture.text
    event_text = capture.event_text
    assert content.count(" WARNING ") == 2
    assert content.count(" ERROR ") == 1
    assert "name=web_fetch attempt=1/3 type=ToolError" in content
    assert "name=web_fetch attempt=3/3 type=ToolError" in content
    assert "URL_CREDENTIAL_51" not in event_text
    assert "RAW_FETCH_PATH_51" not in event_text
    assert "RAW_FETCH_QUERY_51" not in event_text
    assert "RAW_FETCH_RESPONSE_BODY_51" not in event_text
    assert "RAW_FETCH_RESPONSE_BODY_52" not in event_text
    assert "RAW_FETCH_RESPONSE_BODY_53" not in event_text
    assert "RAW_FETCH_RESPONSE_BODY_51" in content
    assert "RAW_FETCH_RESPONSE_BODY_52" in content
    assert "RAW_FETCH_RESPONSE_BODY_53" in content


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
