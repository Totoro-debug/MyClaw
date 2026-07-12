import asyncio
from collections.abc import AsyncIterator, Mapping
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
from myclaw.web_fetch import (
    AioHttpWebFetchClient,
    HTTPResponseBoundary,
    PublicWebFetchBoundary,
    WebFetchRejected,
    WebFetchTool,
)
from myclaw.web_search import WebSearchResult
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


class FakeDNSResolver:
    def __init__(self, answers: tuple[str, ...]) -> None:
        self._answers = answers
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        self.calls.append((hostname, port))
        return self._answers


class RoutedDNSResolver:
    def __init__(self, answers: dict[str, tuple[str, ...]]) -> None:
        self._answers = answers
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        self.calls.append((hostname, port))
        return self._answers[hostname]


class NeverHTTPClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get(
        self,
        url: str,
        *,
        allowed_ips: frozenset[str],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        del allowed_ips, connect_timeout_seconds, total_timeout_seconds
        self.calls.append(url)
        raise AssertionError("HTTP must not run for a non-public DNS answer")


class FakeHTTPResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        peer_ip: str | None,
        chunks: tuple[bytes, ...] = (),
    ) -> None:
        self.status_code = status_code
        self.headers: Mapping[str, str] = {} if headers is None else headers
        self.peer_ip = peer_ip
        self._chunks = chunks
        self.closed = False
        self.iterated = False

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        self.iterated = True
        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        self.closed = True


class FakeHTTPClient:
    def __init__(self, responses: tuple[HTTPResponseBoundary, ...]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, frozenset[str], float, float]] = []

    async def get(
        self,
        url: str,
        *,
        allowed_ips: frozenset[str],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        self.calls.append((url, allowed_ips, connect_timeout_seconds, total_timeout_seconds))
        return next(self._responses)


class HangingHTTPClient:
    def __init__(self) -> None:
        self.started = False

    async def get(
        self,
        url: str,
        *,
        allowed_ips: frozenset[str],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        del url, allowed_ips, connect_timeout_seconds, total_timeout_seconds
        self.started = True
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class CancellationResistantCloseResponse:
    status_code = 200
    headers: Mapping[str, str] = {"content-type": "text/plain"}
    peer_ip: str | None = "93.184.216.34"

    def __init__(self) -> None:
        self.body_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.closed = False

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        self.body_started.set()
        await asyncio.Event().wait()
        if False:
            yield b""

    async def close(self) -> None:
        self.close_started.set()
        await self.release_close.wait()
        self.closed = True


class FakeWebFetchBoundary:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[str] = []

    async def fetch(self, url: str) -> str:
        self.calls.append(url)
        return self._content


class FailingWebFetchBoundary:
    def __init__(self, detail: str) -> None:
        self._detail = detail
        self.calls: list[str] = []

    async def fetch(self, url: str) -> str:
        self.calls.append(url)
        raise WebFetchRejected(self._detail)


class EmptyWebSearchBoundary:
    async def search(
        self,
        query: str,
        max_results: int,
    ) -> tuple[WebSearchResult, ...]:
        del query, max_results
        return ()


def gateway_context(agent_home: Path, workspace: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        lane="foreground",
        workspace=workspace,
        agent_home=agent_home,
        session_id="20260712-120000-000000_550e8400-e29b-41d4-a716-446655440000",
    )


@pytest.mark.asyncio
async def test_web_fetch_rejects_when_any_dns_answer_is_not_public(
    workspace: Path,
) -> None:
    del workspace
    resolver = FakeDNSResolver(("93.184.216.34", "10.0.0.7"))
    http = NeverHTTPClient()
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/page")

    assert resolver.calls == [("public.example", 443)]
    assert http.calls == []


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_response_close_before_propagating() -> None:
    response = CancellationResistantCloseResponse()
    fetcher = PublicWebFetchBoundary(
        resolver=FakeDNSResolver(("93.184.216.34",)),
        http_client=FakeHTTPClient((response,)),
    )
    fetch = asyncio.create_task(fetcher.fetch("https://public.example/page"))
    await response.body_started.wait()

    fetch.cancel()
    await response.close_started.wait()
    fetch.cancel()
    await asyncio.sleep(0)

    assert not fetch.done()
    response.release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await fetch
    assert response.closed


@pytest.mark.asyncio
async def test_http_client_cancellation_waits_for_pre_response_session_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingSession:
        def __init__(self, **options: object) -> None:
            del options
            self.get_started = asyncio.Event()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()
            self.closed = False

        async def get(self, url: str, *, allow_redirects: bool) -> object:
            del url, allow_redirects
            self.get_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.closed = True

    session = BlockingSession()
    monkeypatch.setattr("myclaw.web_fetch.ClientSession", lambda **_options: session)
    client = AioHttpWebFetchClient()
    request = asyncio.create_task(
        client.get(
            "https://public.example/page",
            allowed_ips=frozenset({"93.184.216.34"}),
            connect_timeout_seconds=1,
            total_timeout_seconds=2,
        )
    )
    await session.get_started.wait()

    request.cancel()
    await session.close_started.wait()
    request.cancel()
    await asyncio.sleep(0)

    assert not request.done()
    session.release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert session.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    (
        "127.0.0.1",
        "10.0.0.1",
        "169.254.1.1",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "100.64.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::",
        "ff02::1",
        "2001:db8::1",
    ),
)
async def test_web_fetch_rejects_every_non_public_address_category(address: str) -> None:
    resolver = FakeDNSResolver((address,))
    http = NeverHTTPClient()
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/page")

    assert http.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "ftp://public.example/page",
        "https://user:password@public.example/page",
    ),
)
async def test_web_fetch_rejects_invalid_scheme_or_userinfo_before_dns(url: str) -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    http = NeverHTTPClient()
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch(url)

    assert resolver.calls == []
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_rejects_a_malformed_port_before_dns() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    http = NeverHTTPClient()
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example:70000/page")

    assert resolver.calls == []
    assert http.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hostname", ("localhost", "service.localhost", "LOCALHOST."))
async def test_web_fetch_rejects_localhost_names_before_dns(hostname: str) -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    http = NeverHTTPClient()
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch(f"http://{hostname}/status")

    assert resolver.calls == []
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_refuses_when_the_actual_peer_cannot_be_verified() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    response = FakeHTTPResponse(peer_ip=None)
    http = FakeHTTPClient((response,))
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/page")

    assert http.calls == [
        (
            "https://public.example/page",
            frozenset({"93.184.216.34"}),
            10.0,
            30.0,
        )
    ]
    assert response.closed is True


@pytest.mark.asyncio
async def test_web_fetch_rejects_a_peer_outside_the_validated_dns_set() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    response = FakeHTTPResponse(peer_ip="8.8.8.8")
    http = FakeHTTPClient((response,))
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/page")

    assert resolver.calls == [("public.example", 443)]
    assert response.closed is True


@pytest.mark.asyncio
async def test_web_fetch_returns_text_from_a_validated_public_peer() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    response = FakeHTTPResponse(
        headers={"Content-Type": "text/plain; charset=utf-8"},
        peer_ip="93.184.216.34",
        chunks=(b"Public ", b"text."),
    )
    http = FakeHTTPClient((response,))
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    content = await fetcher.fetch("https://public.example/page")

    assert content == "Public text."
    assert response.closed is True


@pytest.mark.asyncio
async def test_web_fetch_revalidates_a_redirect_before_the_next_request() -> None:
    resolver = RoutedDNSResolver(
        {
            "public.example": ("93.184.216.34",),
            "internal.example": ("192.168.1.9",),
        }
    )
    redirect = FakeHTTPResponse(
        status_code=302,
        headers={"Location": "http://internal.example/admin"},
        peer_ip="93.184.216.34",
    )
    http = FakeHTTPClient((redirect,))
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/start")

    assert resolver.calls == [
        ("public.example", 443),
        ("internal.example", 80),
    ]
    assert len(http.calls) == 1
    assert redirect.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redirect_target",
    (
        "ftp://public.example/file",
        "https://user:password@public.example/file",
    ),
)
async def test_web_fetch_rechecks_scheme_and_userinfo_on_every_redirect(
    redirect_target: str,
) -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    redirect = FakeHTTPResponse(
        status_code=302,
        headers={"Location": redirect_target},
        peer_ip="93.184.216.34",
    )
    http = FakeHTTPClient((redirect,))
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/start")

    assert resolver.calls == [("public.example", 443)]
    assert len(http.calls) == 1
    assert redirect.closed is True


@pytest.mark.asyncio
async def test_web_fetch_follows_at_most_five_redirects() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    redirects = tuple(
        FakeHTTPResponse(
            status_code=302,
            headers={"Location": f"/hop-{index + 1}"},
            peer_ip="93.184.216.34",
        )
        for index in range(6)
    )
    http = FakeHTTPClient(redirects)
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/hop-0")

    assert len(http.calls) == 6
    assert resolver.calls == [("public.example", 443)] * 6
    assert all(response.closed for response in redirects)


@pytest.mark.asyncio
async def test_web_fetch_rejects_a_body_larger_than_ten_mebibytes() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    one_mebibyte = b"x" * (1024 * 1024)
    response = FakeHTTPResponse(
        headers={"Content-Type": "text/plain"},
        peer_ip="93.184.216.34",
        chunks=(one_mebibyte,) * 10 + (b"!",),
    )
    http = FakeHTTPClient((response,))
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/large")

    assert response.closed is True


@pytest.mark.asyncio
async def test_web_fetch_rejects_an_oversized_content_length_before_body() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    response = FakeHTTPResponse(
        headers={
            "Content-Type": "text/plain",
            "Content-Length": "10485761",
        },
        peer_ip="93.184.216.34",
        chunks=(),
    )
    fetcher = PublicWebFetchBoundary(
        resolver=resolver,
        http_client=FakeHTTPClient((response,)),
    )

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/declared-large")

    assert response.iterated is False
    assert response.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    (
        {},
        {"Content-Type": "application/octet-stream"},
    ),
)
async def test_web_fetch_rejects_unsupported_or_unverifiable_media_before_body(
    headers: dict[str, str],
) -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    response = FakeHTTPResponse(
        headers=headers,
        peer_ip="93.184.216.34",
        chunks=(b"\x00\xffprivate binary",),
    )
    http = FakeHTTPClient((response,))
    fetcher = PublicWebFetchBoundary(resolver=resolver, http_client=http)

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/download")

    assert response.iterated is False
    assert response.closed is True


@pytest.mark.asyncio
async def test_web_fetch_preserves_other_textual_media_as_text() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    response = FakeHTTPResponse(
        headers={"Content-Type": "application/json; charset=utf-8"},
        peer_ip="93.184.216.34",
        chunks=(b'{"ok":', b"true}"),
    )
    fetcher = PublicWebFetchBoundary(
        resolver=resolver,
        http_client=FakeHTTPClient((response,)),
    )

    content = await fetcher.fetch("https://public.example/data.json")

    assert content == '{"ok":true}'


@pytest.mark.asyncio
async def test_web_fetch_decodes_text_with_the_declared_charset() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    response = FakeHTTPResponse(
        headers={"Content-Type": "text/plain; charset=iso-8859-1"},
        peer_ip="93.184.216.34",
        chunks=(b"caf\xe9",),
    )
    fetcher = PublicWebFetchBoundary(
        resolver=resolver,
        http_client=FakeHTTPClient((response,)),
    )

    content = await fetcher.fetch("https://public.example/text")

    assert content == "caf\u00e9"


@pytest.mark.asyncio
async def test_web_fetch_converts_html_to_readable_text() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    response = FakeHTTPResponse(
        headers={"Content-Type": "text/html; charset=utf-8"},
        peer_ip="93.184.216.34",
        chunks=(
            b"<html><head><title>Example</title>",
            b"<style>private style</style><script>private script</script></head>",
            b"<body><h1>Public page</h1><p>Hello <strong>world</strong>.</p></body></html>",
        ),
    )
    fetcher = PublicWebFetchBoundary(
        resolver=resolver,
        http_client=FakeHTTPClient((response,)),
    )

    content = await fetcher.fetch("https://public.example/page")

    assert content == "Example\nPublic page\nHello world."
    assert "private" not in content
    assert "<" not in content


@pytest.mark.asyncio
async def test_web_fetch_enforces_one_total_timeout_for_the_whole_fetch() -> None:
    resolver = FakeDNSResolver(("93.184.216.34",))
    http = HangingHTTPClient()
    fetcher = PublicWebFetchBoundary(
        resolver=resolver,
        http_client=http,
        total_timeout_seconds=0.01,
    )

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/slow")

    assert http.started is True


@pytest.mark.asyncio
async def test_tool_gateway_returns_provider_neutral_web_fetch_text(
    agent_home: Path,
    workspace: Path,
) -> None:
    fetch = FakeWebFetchBoundary("Public page\nReadable content.")
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        tools=(WebFetchTool(fetch),),
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_fetch",
            name="web_fetch",
            arguments={"url": "https://public.example/page"},
        )
    )

    assert result.status == "success"
    assert result.error is None
    assert result.content == "Public page\nReadable content."
    assert fetch.calls == ["https://public.example/page"]


def test_tool_gateway_places_web_fetch_next_to_web_search_in_the_catalog(
    agent_home: Path,
    workspace: Path,
) -> None:
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        web_search=EmptyWebSearchBoundary(),
        web_fetch=FakeWebFetchBoundary("unused"),
    )

    assert [definition.name for definition in gateway.definitions] == [
        "read_file",
        "list_files",
        "search_files",
        "write_file",
        "edit_file",
        "web_search",
        "web_fetch",
    ]


@pytest.mark.asyncio
async def test_disabled_web_tools_omit_both_search_and_fetch_from_conversation(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    fetch = FakeWebFetchBoundary("must not be fetched")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Answered locally."),
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
        new_uuid=iter(map(UUID, SESSION_UUIDS)).__next__,
        web_search=EmptyWebSearchBoundary(),
        web_fetch=fetch,
    )

    events = [event async for event in runtime.conversation.submit("Answer locally.")]

    assert events[-1].type == "turn_completed"
    request = provider.stream_requests[0]
    assert isinstance(request, ModelRequest)
    names = [definition.name for definition in request.tools]
    assert "web_search" not in names
    assert "web_fetch" not in names
    assert fetch.calls == []


@pytest.mark.asyncio
async def test_conversation_returns_a_safe_error_for_web_fetch_failure(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace("[tools.web]\nenabled = false", "[tools.web]\nenabled = true"),
        encoding="utf-8",
    )
    fetch = FailingWebFetchBoundary("rebound peer 127.0.0.1 and private upstream detail")
    tool_call = ModelToolCall(
        id="call_fetch_failure",
        name="web_fetch",
        arguments={"url": "https://public.example/page"},
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
                            message=AssistantModelMessage(content="Fetch is unavailable."),
                            usage=ModelUsage(input_tokens=10, output_tokens=4, total_tokens=14),
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
        web_fetch=fetch,
    )

    events = [event async for event in runtime.conversation.submit("Fetch the page.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    completed = events[2].payload
    assert isinstance(completed, ToolCompletedPayload)
    assert completed.status == "error"
    assert fetch.calls == ["https://public.example/page"]
    assert "127.0.0.1" not in str([event.to_dict() for event in events])
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    tool_message = second_request.messages[-1]
    assert isinstance(tool_message, ToolModelMessage)
    assert tool_message.content == "web_fetch could not complete the request."
    persisted = (await runtime.sessions.load(runtime.session_id)).messages[2]
    assert isinstance(persisted, ToolSessionMessage)
    assert persisted.error is not None
    assert persisted.error.code == "tool_failed"
    assert "private upstream" not in persisted.content
