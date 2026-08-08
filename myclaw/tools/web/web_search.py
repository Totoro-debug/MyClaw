"""Provider-neutral WebSearch tool boundary and result normalization."""

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Annotated, Final, Protocol, cast

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from myclaw.tools.base import BaseTool, ToolParam

_SEARCH_CREATION_FLAGS: Final = 0x08000200


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


class DuckDuckGoSearchProcess(Protocol):
    @property
    def returncode(self) -> int | None: ...

    async def communicate(self) -> tuple[bytes, bytes]: ...

    def terminate(self) -> None: ...

    async def wait(self) -> int: ...


class DuckDuckGoSearchProcessSpawner(Protocol):
    async def spawn(self, query: str, max_results: int) -> DuckDuckGoSearchProcess: ...


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


class _AsyncioSearchProcess:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        stdout, stderr = await self._process.communicate()
        return stdout or b"", stderr or b""

    def terminate(self) -> None:
        self._process.terminate()

    async def wait(self) -> int:
        return await self._process.wait()


class AsyncioDuckDuckGoSearchProcessSpawner:
    async def spawn(self, query: str, max_results: int) -> DuckDuckGoSearchProcess:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "myclaw.tools.web.web_search",
            query,
            str(max_results),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=_SEARCH_CREATION_FLAGS,
        )
        return _AsyncioSearchProcess(process)


class DuckDuckGoSearchBoundary:
    """Adapt DuckDuckGo SDK records to the provider-neutral search boundary."""

    def __init__(
        self,
        *,
        process_spawner: DuckDuckGoSearchProcessSpawner | None = None,
    ) -> None:
        self._process_spawner = (
            AsyncioDuckDuckGoSearchProcessSpawner() if process_spawner is None else process_spawner
        )

    async def search(self, query: str, max_results: int) -> tuple[WebSearchResult, ...]:
        raw_results = await self._search_in_process(query, max_results)
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

    async def _search_in_process(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, object]]:
        spawn = asyncio.create_task(self._process_spawner.spawn(query, max_results))
        try:
            process = await asyncio.shield(spawn)
        except BaseException as primary_error:
            if spawn.done() and (spawn.cancelled() or spawn.exception() is not None):
                raise
            try:
                process = await _join_spawn(spawn)
                await _stop_process(process, None)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        communication = asyncio.create_task(process.communicate())
        try:
            stdout, _stderr = await asyncio.shield(communication)
        except BaseException as primary_error:
            try:
                await _stop_process(process, communication)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        if process.returncode != 0:
            raise RuntimeError("DuckDuckGo search process failed")
        try:
            decoded = cast(object, json.loads(stdout.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("DuckDuckGo search process returned invalid data") from error
        if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
            raise RuntimeError("DuckDuckGo search process returned invalid data")
        return cast(list[dict[str, object]], decoded)


async def _stop_process(
    process: DuckDuckGoSearchProcess,
    communication: asyncio.Task[tuple[bytes, bytes]] | None,
) -> None:
    stop = asyncio.create_task(_terminate_process(process, communication))
    await _await_cleanup(stop)


async def _terminate_process(
    process: DuckDuckGoSearchProcess,
    communication: asyncio.Task[tuple[bytes, bytes]] | None,
) -> None:
    failures: list[BaseException] = []
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    except BaseException as error:
        failures.append(error)
    try:
        await process.wait()
    except BaseException as error:
        failures.append(error)
    if communication is not None:
        if not communication.done():
            communication.cancel()
        await asyncio.gather(communication, return_exceptions=True)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("WebSearch process shutdown failed", failures)


async def _join_spawn(
    task: asyncio.Task[DuckDuckGoSearchProcess],
) -> DuckDuckGoSearchProcess:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                break
        except BaseException:
            break
    return task.result()


async def _await_cleanup(task: asyncio.Task[None]) -> None:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if task.cancelled():
                break
            if cancellation is None:
                cancellation = error
        except BaseException:
            break
    try:
        task.result()
    except BaseException as cleanup_error:
        if cancellation is not None:
            raise cancellation from cleanup_error
        raise
    if cancellation is not None:
        raise cancellation


class WebSearchTool(BaseTool):
    """Expose credential-free web search through the Tool protocol."""

    name = "web_search"
    description = "Search the public web and return normalized result summaries."
    required = ("query",)
    max_retries = 2

    query: Annotated[str, ToolParam(description="Public web search query.", min_length=1)]
    max_results: Annotated[
        int,
        ToolParam(description="Maximum normalized results.", minimum=1, maximum=10),
    ] = 5

    def __init__(self, *, search: WebSearchBoundary) -> None:
        self._search = search

    async def execute(self, *, query: str, max_results: int) -> str:
        results = await self._search.search(query, max_results)
        return json.dumps(
            [result.to_dict() for result in results],
            ensure_ascii=False,
            separators=(",", ":"),
        )


def main(arguments: list[str] | None = None) -> int:
    """Run the synchronous DuckDuckGo SDK in an isolated process."""
    values = sys.argv[1:] if arguments is None else arguments
    if len(values) != 2:
        return 2
    query, raw_max_results = values
    try:
        max_results = int(raw_max_results)
        results = _duckduckgo_text_search(query, max_results=max_results)
    except ValueError:
        return 2
    except Exception:
        return 1
    payload = json.dumps(results, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
