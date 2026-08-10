from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import ClassVar, Self, cast

import pytest
from aiohttp import ClientTimeout

from myclaw.tools.core.web_fetch import (
    HTTPClientBoundary,
    HTTPResponseBoundary,
    JinaReaderBoundary,
    JinaReaderClient,
    WebFetchTool,
)
from myclaw.tools.network_safety import DNSResolver
from myclaw.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationRequester,
    ModelToolCall,
)
from tests.fixtures import SingleToolGateway


def _call(arguments: dict[str, object], *, call_id: str = "call_fetch") -> ModelToolCall:
    return ModelToolCall(
        id=call_id,
        name="web_fetch",
        arguments=json.dumps(arguments),
    )


class FakeResolver:
    def __init__(self, answers: dict[str, tuple[str, ...]] | tuple[str, ...]) -> None:
        self._answers = answers
        self.calls: list[tuple[str, int]] = []
        self.failure: BaseException | None = None

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        self.calls.append((hostname, port))
        if self.failure is not None:
            raise self.failure
        if isinstance(self._answers, dict):
            return self._answers[hostname]
        return self._answers


class FakeJina:
    outcomes: ClassVar[list[str | BaseException]] = []
    calls: ClassVar[list[tuple[str, str]]] = []

    async def fetch(self, url: str, *, output_format: str) -> str:
        self.calls.append((url, output_format))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeJinaHTTPResponse:
    def __init__(self, *, status: int, content: str = "") -> None:
        self.status = status
        self._content = content
        self.entered = False
        self.exited = False
        self.text_calls: list[tuple[str, str]] = []

    async def __aenter__(self) -> Self:
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.exited = True

    async def text(self, *, encoding: str, errors: str) -> str:
        self.text_calls.append((encoding, errors))
        return self._content


class FakeJinaHTTPSession:
    responses: ClassVar[list[FakeJinaHTTPResponse]] = []
    options: ClassVar[list[dict[str, object]]] = []
    calls: ClassVar[list[tuple[str, dict[str, str], bool]]] = []

    def __init__(self, **options: object) -> None:
        self.options.append(options)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        allow_redirects: bool,
    ) -> FakeJinaHTTPResponse:
        self.calls.append((url, headers, allow_redirects))
        return self.responses.pop(0)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (),
    ) -> None:
        self.status_code = status_code
        self.headers = {} if headers is None else headers
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
        self.calls: list[tuple[str, float, float]] = []

    async def get(
        self,
        url: str,
        *,
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        self.calls.append((url, connect_timeout_seconds, total_timeout_seconds))
        response = next(self._responses)
        return response


def _gateway(
    *,
    resolver: DNSResolver,
    jina: JinaReaderBoundary | None = None,
    http: HTTPClientBoundary | None = None,
    confirmation: ConfirmationRequester | None = None,
) -> SingleToolGateway:
    tool = WebFetchTool(resolver=resolver, jina_reader=jina, http_client=http)
    return SingleToolGateway((tool,), confirmation=confirmation)


@pytest.fixture(autouse=True)
def reset_jina() -> None:
    FakeJina.outcomes = []
    FakeJina.calls = []
    FakeJinaHTTPSession.responses = []
    FakeJinaHTTPSession.options = []
    FakeJinaHTTPSession.calls = []


def test_web_fetch_schema_declares_format_and_max_chars() -> None:
    schema = WebFetchTool().to_schema()

    assert schema == {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch readable content from an HTTP or HTTPS URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "HTTP or HTTPS URL to fetch.",
                        "minLength": 1,
                        "format": "uri",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: markdown or text.",
                        "minLength": 1,
                        "default": "markdown",
                    },
                    "maxChars": {
                        "type": "integer",
                        "description": "Maximum returned characters.",
                        "minimum": 1,
                        "default": 50000,
                    },
                },
                "required": ["url"],
            },
        },
    }


@pytest.mark.asyncio
async def test_web_fetch_uses_jina_first_for_public_targets() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = ["# Public page"]
    http = FakeHTTPClient(())

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "  https://public.example/page  "})
    )

    assert result.status == "success"
    assert result.content == "# Public page"
    assert resolver.calls == [("public.example", 443)]
    assert jina.calls == [("https://public.example/page", "markdown")]
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_falls_back_to_direct_text_after_jina_timeout() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = [TimeoutError()]
    response = FakeResponse(
        headers={"Content-Type": "text/plain; charset=utf-8"},
        chunks=(b"Direct ", b"content"),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/page", "format": "text"})
    )

    assert result.status == "success"
    assert result.content == "Direct content"
    assert jina.calls == [("https://public.example/page", "text")]
    assert http.calls == [("https://public.example/page", 10.0, 30.0)]
    assert response.closed


@pytest.mark.asyncio
async def test_web_fetch_cancellation_propagates_from_jina() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class HangingJina:
        async def fetch(self, url: str, *, output_format: str) -> str:
            del url, output_format
            started.set()
            await release.wait()
            return "unreachable"

    resolver = FakeResolver(("93.184.216.34",))
    task = asyncio.create_task(
        _gateway(resolver=resolver, jina=HangingJina(), http=FakeHTTPClient(())).call(
            _call({"url": "https://public.example/page"})
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    (
        {"url": ""},
        {"url": "  \t"},
        {"url": "ftp://public.example/page"},
        {"url": "https://user:password@public.example/page"},
        {"url": "https://public.example:bad/page"},
        {"url": "https://public.example:0/page"},
        {"url": "https:///page"},
        {"url": "https://public.example/page", "format": "html"},
        {"url": "https://public.example/page", "maxChars": 0},
    ),
)
async def test_web_fetch_rejects_invalid_parameters_before_dns(
    arguments: dict[str, object],
) -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = ["must not run"]
    http = FakeHTTPClient(())

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(_call(arguments))

    assert result.status == "error"
    assert resolver.calls == []
    assert jina.calls == []
    assert http.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (TimeoutError(), RuntimeError("transport")))
async def test_web_fetch_falls_back_after_jina_failure(failure: BaseException) -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = [failure]
    response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"fallback",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/page"})
    )

    assert result.status == "success"
    assert result.content == "fallback"
    assert len(jina.calls) == 1
    assert len(http.calls) == 1


@pytest.mark.asyncio
async def test_web_fetch_falls_back_after_empty_or_non_success_jina_content() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = [""]
    response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"direct",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/page"})
    )

    assert result.status == "success"
    assert result.content == "direct"
    assert response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "content", "output_format"),
    (
        (302, "redirect", "markdown"),
        (429, "rate limited", "markdown"),
        (503, "unavailable", "text"),
        (200, "  \n", "markdown"),
    ),
)
async def test_web_fetch_falls_back_after_jina_http_response(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    content: str,
    output_format: str,
) -> None:
    jina_response = FakeJinaHTTPResponse(status=status, content=content)
    FakeJinaHTTPSession.responses = [jina_response]
    monkeypatch.setattr(
        "myclaw.tools.core.web_fetch.ClientSession",
        FakeJinaHTTPSession,
    )
    resolver = FakeResolver(("93.184.216.34",))
    direct_response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"direct",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, direct_response),))

    result = await _gateway(
        resolver=resolver,
        jina=JinaReaderClient(),
        http=http,
    ).call(
        _call(
            {
                "url": "https://public.example/page",
                "format": output_format,
            }
        )
    )

    assert result.status == "success"
    assert result.content == "direct"
    assert FakeJinaHTTPSession.calls == [
        (
            "https://r.jina.ai/https://public.example/page",
            {
                "Accept": "text/plain",
                "X-Respond-With": output_format,
            },
            False,
        )
    ]
    timeout = FakeJinaHTTPSession.options[0]["timeout"]
    assert isinstance(timeout, ClientTimeout)
    assert timeout.total is None
    assert timeout.connect == 10.0
    assert timeout.sock_connect == 10.0
    assert jina_response.entered
    assert jina_response.exited
    assert jina_response.text_calls == ([("utf-8", "replace")] if status == 200 else [])
    assert direct_response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ("markdown", "text"))
async def test_web_fetch_requests_the_selected_jina_output_format(
    monkeypatch: pytest.MonkeyPatch,
    output_format: str,
) -> None:
    jina_response = FakeJinaHTTPResponse(status=200, content="Jina content")
    FakeJinaHTTPSession.responses = [jina_response]
    monkeypatch.setattr(
        "myclaw.tools.core.web_fetch.ClientSession",
        FakeJinaHTTPSession,
    )
    http = FakeHTTPClient(())

    result = await _gateway(
        resolver=FakeResolver(("93.184.216.34",)),
        jina=JinaReaderClient(),
        http=http,
    ).call(
        _call(
            {
                "url": "https://public.example/page",
                "format": output_format,
            }
        )
    )

    assert result.status == "success"
    assert result.content == "Jina content"
    assert FakeJinaHTTPSession.calls[0][1:] == (
        {
            "Accept": "text/plain",
            "X-Respond-With": output_format,
        },
        False,
    )
    assert "Authorization" not in FakeJinaHTTPSession.calls[0][1]
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_approved_private_target_skips_jina() -> None:
    resolver = FakeResolver(("127.0.0.1",))
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("must not run")]
    response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"private",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(
        resolver=resolver,
        jina=jina,
        http=http,
        confirmation=approve,
    ).call(_call({"url": "http://private.example/status"}))

    assert result.status == "success"
    assert result.content == "private"
    assert jina.calls == []
    assert len(requests) == 1
    assert "private" in requests[0].reason


@pytest.mark.asyncio
async def test_web_fetch_dns_failure_requests_confirmation_and_skips_jina() -> None:
    resolver = FakeResolver(())
    resolver.failure = OSError("DNS unavailable")
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("must not run")]
    response = FakeResponse(headers={"content-type": "text/plain"}, chunks=(b"approved",))
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(
        resolver=resolver,
        jina=jina,
        http=http,
        confirmation=approve,
    ).call(_call({"url": "https://missing.example/page"}))

    assert result.status == "success"
    assert result.content == "approved"
    assert jina.calls == []
    assert "DNS" in requests[0].reason


@pytest.mark.asyncio
async def test_web_fetch_maps_ipv4_mapped_ipv6_before_requesting_confirmation() -> None:
    resolver = FakeResolver(("::ffff:10.0.0.7",))
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("must not run")]
    response = FakeResponse(headers={"content-type": "text/plain"}, chunks=(b"approved",))
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(
        resolver=resolver,
        jina=jina,
        http=http,
        confirmation=approve,
    ).call(_call({"url": "https://mapped.example/page"}))

    assert result.status == "success"
    assert result.content == "approved"
    assert len(requests) == 1
    assert "private" in requests[0].reason


@pytest.mark.asyncio
async def test_web_fetch_refuses_private_target_without_confirmation_channel() -> None:
    resolver = FakeResolver(("192.168.1.7",))
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("must not run")]
    http = FakeHTTPClient(())

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://private.example/page"})
    )

    assert result.status == "refused"
    assert jina.calls == []
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_follows_public_redirects_and_rechecks_each_target() -> None:
    resolver = FakeResolver(
        {
            "public.example": ("93.184.216.34",),
            "next.example": ("8.8.8.8",),
        }
    )
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("Jina unavailable")]
    redirect = FakeResponse(
        status_code=302,
        headers={"location": "https://next.example/final"},
    )
    final = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"redirected",),
    )
    http = FakeHTTPClient(
        (
            cast(HTTPResponseBoundary, redirect),
            cast(HTTPResponseBoundary, final),
        )
    )

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/start"})
    )

    assert result.status == "success"
    assert result.content == "redirected"
    assert resolver.calls == [
        ("public.example", 443),
        ("next.example", 443),
    ]
    assert [call[0] for call in http.calls] == [
        "https://public.example/start",
        "https://next.example/final",
    ]
    assert redirect.closed
    assert final.closed


@pytest.mark.asyncio
async def test_web_fetch_stops_at_a_newly_unsafe_redirect_for_a_separate_call() -> None:
    resolver = FakeResolver(
        {
            "public.example": ("93.184.216.34",),
            "internal.example": ("10.0.0.7",),
        }
    )
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("Jina unavailable")]
    redirect = FakeResponse(
        status_code=302,
        headers={"location": "http://internal.example/admin"},
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, redirect),))

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/start"})
    )

    assert result.status == "error"
    assert "separate confirmed invocation" in result.content
    assert "http://internal.example/admin" in result.content
    assert resolver.calls == [
        ("public.example", 443),
        ("internal.example", 80),
    ]
    assert len(http.calls) == 1
    assert redirect.closed


@pytest.mark.asyncio
async def test_web_fetch_follows_at_most_five_redirects() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("Jina unavailable")]
    redirects = tuple(
        FakeResponse(
            status_code=302,
            headers={"location": f"/hop-{index + 1}"},
        )
        for index in range(6)
    )
    http = FakeHTTPClient(tuple(cast(HTTPResponseBoundary, response) for response in redirects))

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/hop-0"})
    )

    assert result.status == "error"
    assert "redirect limit" in result.content
    assert len(http.calls) == 6
    assert all(response.closed for response in redirects)
    assert len(resolver.calls) == 6


@pytest.mark.asyncio
async def test_web_fetch_accepts_textual_media_and_declared_charset() -> None:
    cases = (
        ("application/json", b'{"ok":true}', '{"ok":true}'),
        ("application/xml", b"<ok>true</ok>", "<ok>true</ok>"),
        ("application/javascript", b"const ok = true;", "const ok = true;"),
        ("text/plain; charset=iso-8859-1", b"caf\xe9", "café"),
    )
    for content_type, body, expected in cases:
        resolver = FakeResolver(("93.184.216.34",))
        jina = FakeJina()
        FakeJina.outcomes = [RuntimeError("Jina unavailable")]
        response = FakeResponse(headers={"content-type": content_type}, chunks=(body,))
        http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

        result = await _gateway(resolver=resolver, jina=jina, http=http).call(
            _call({"url": "https://public.example/data"})
        )

        assert result.status == "success"
        assert result.content == expected
        assert response.closed


@pytest.mark.asyncio
async def test_web_fetch_decodes_missing_content_type_with_utf8_replacement() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("Jina unavailable")]
    response = FakeResponse(chunks=(b"valid\xff",))
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/data"})
    )

    assert result.status == "success"
    assert result.content == "valid�"


@pytest.mark.asyncio
async def test_web_fetch_rejects_explicit_binary_media() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("Jina unavailable")]
    response = FakeResponse(
        headers={"content-type": "application/octet-stream"},
        chunks=(b"binary",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/download"})
    )

    assert result.status == "error"
    assert "media type" in result.content
    assert response.closed
    assert not response.iterated


@pytest.mark.asyncio
async def test_web_fetch_extracts_readable_html_without_ignored_elements() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("Jina unavailable")]
    response = FakeResponse(
        headers={"content-type": "text/html; charset=utf-8"},
        chunks=(
            b"<html><head><title>Example</title><style>hidden style</style>",
            b"<script>hidden script</script></head><body><h1>Public page</h1>",
            b"<p>Hello &amp; <strong>world</strong>.</p><template>hidden template</template>",
            b"</body></html>",
        ),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/page", "format": "text"})
    )

    assert result.status == "success"
    assert result.content == "Example\nPublic page\nHello & world."
    assert "hidden" not in result.content
    assert "<" not in result.content


@pytest.mark.asyncio
async def test_web_fetch_applies_final_shared_prefix_truncation_to_jina_output() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = ["abcdefghijklmnopqrstuvwxyz"]
    http = FakeHTTPClient(())

    result = await _gateway(resolver=resolver, jina=jina, http=http).call(
        _call({"url": "https://public.example/page", "maxChars": 20})
    )

    assert result.status == "success"
    assert result.content == "abcd\n\n...[truncated]"
    assert len(result.content) == 20
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_whole_call_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowJina:
        async def fetch(self, url: str, *, output_format: str) -> str:
            del url, output_format
            started.set()
            await release.wait()
            return "never"

    monkeypatch.setattr("myclaw.tools.core.web_fetch.TOTAL_TIMEOUT_SECONDS", 0.01)
    resolver = FakeResolver(("93.184.216.34",))
    result = await _gateway(
        resolver=resolver,
        jina=SlowJina(),
        http=FakeHTTPClient(()),
    ).call(_call({"url": "https://public.example/slow"}))

    assert result.status == "error"
    assert result.content == "Web Fetch timed out after 0.01 seconds."
    assert started.is_set()


@pytest.mark.asyncio
async def test_web_fetch_cancellation_propagates_from_direct_client() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class HangingHTTP:
        async def get(
            self,
            url: str,
            *,
            connect_timeout_seconds: float,
            total_timeout_seconds: float,
        ) -> HTTPResponseBoundary:
            del url, connect_timeout_seconds, total_timeout_seconds
            started.set()
            await release.wait()
            raise AssertionError("unreachable")

    resolver = FakeResolver(("93.184.216.34",))
    jina = FakeJina()
    FakeJina.outcomes = [RuntimeError("Jina unavailable")]
    task = asyncio.create_task(
        _gateway(resolver=resolver, jina=jina, http=HangingHTTP()).call(
            _call({"url": "https://public.example/slow"})
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
