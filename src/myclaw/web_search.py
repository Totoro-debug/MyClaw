"""Provider-neutral WebSearch tool boundary and result normalization."""

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from myclaw.contracts import JsonObject, ToolDefinition, ToolExecutionContext


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


class WebSearchBoundary(Protocol):
    async def search(self, query: str, max_results: int) -> tuple[WebSearchResult, ...]: ...


class DuckDuckGoTextSearch(Protocol):
    def __call__(self, query: str, *, max_results: int) -> list[dict[str, object]]: ...


def _duckduckgo_text_search(query: str, *, max_results: int) -> list[dict[str, object]]:
    try:
        with DDGS() as client:
            results = client.text(
                query,
                max_results=max_results,
                backend="duckduckgo",
            )
    except DDGSException as exc:
        if str(exc) == "No results found.":
            return []
        raise
    return results


class DuckDuckGoSearchBoundary:
    """Adapt DuckDuckGo SDK records to the provider-neutral search boundary."""

    def __init__(self, *, text_search: DuckDuckGoTextSearch = _duckduckgo_text_search) -> None:
        self._text_search = text_search

    async def search(self, query: str, max_results: int) -> tuple[WebSearchResult, ...]:
        raw_results = await asyncio.to_thread(
            self._text_search,
            query,
            max_results=max_results,
        )
        results: list[WebSearchResult] = []
        for raw in raw_results[:max_results]:
            title = raw.get("title")
            url = raw.get("href")
            snippet = raw.get("body")
            if (
                not isinstance(title, str)
                or not isinstance(url, str)
                or not isinstance(snippet, str)
            ):
                raise ValueError("DuckDuckGo returned a malformed search result")
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        return tuple(results)


class WebSearchTool:
    """Expose credential-free web search through the Tool protocol."""

    _definition = ToolDefinition(
        name="web_search",
        description="Search the public web and return normalized result summaries.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    def __init__(self, search: WebSearchBoundary) -> None:
        self._search = search

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str:
        del context
        query = arguments["query"]
        max_results = arguments.get("max_results", 5)
        if not isinstance(query, str) or not isinstance(max_results, int):
            raise ValueError("invalid web_search arguments")
        results = await self._search.search(query, max_results)
        return json.dumps(
            [result.to_dict() for result in results],
            ensure_ascii=False,
            separators=(",", ":"),
        )
