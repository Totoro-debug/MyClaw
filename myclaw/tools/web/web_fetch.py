"""Public-only WebFetch boundary with SSRF-resistant address validation."""

import asyncio
import socket
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from email.message import Message
from html.parser import HTMLParser
from typing import Annotated, Any, Protocol
from urllib.parse import urljoin, urlsplit

from aiohttp import ClientResponse, ClientSession, ClientTimeout, TCPConnector
from aiohttp.abc import AbstractResolver, ResolveResult

from myclaw.tools.base import BaseTool, ToolError, ToolParam, normalize_public_ip

CONNECT_TIMEOUT_SECONDS = 10.0
TOTAL_TIMEOUT_SECONDS = 30.0
MAX_REDIRECTS = 5
MAX_BODY_BYTES = 10 * 1024 * 1024
_TEXTUAL_APPLICATION_MEDIA_TYPES = frozenset(
    {
        "application/ecmascript",
        "application/javascript",
        "application/json",
        "application/sql",
        "application/x-www-form-urlencoded",
        "application/xml",
        "application/yaml",
    }
)
_HTML_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})
_HTML_IGNORED_ELEMENTS = frozenset({"noscript", "script", "style", "template"})
_HTML_BLOCK_ELEMENTS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "title",
        "tr",
    }
)


class WebFetchRejected(RuntimeError):
    """Raised when WebFetch cannot prove that a request is safe."""


class DNSResolverBoundary(Protocol):
    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]: ...


class HTTPResponseBoundary(Protocol):
    status_code: int
    headers: Mapping[str, str]
    peer_ip: str | None

    def iter_bytes(self) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


class HTTPClientBoundary(Protocol):
    async def get(
        self,
        url: str,
        *,
        allowed_ips: frozenset[str],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary: ...


class WebFetchBoundary(Protocol):
    async def fetch(self, url: str) -> str: ...


class SocketDNSResolver:
    """Resolve all TCP addresses through the event loop's system DNS boundary."""

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        records = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        return tuple(dict.fromkeys(record[4][0] for record in records))


class _PinnedResolver(AbstractResolver):
    def __init__(self, allowed_ips: frozenset[str]) -> None:
        self._allowed_ips = allowed_ips

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        del family
        return [
            ResolveResult(
                hostname=host,
                host=address,
                port=port,
                family=(socket.AF_INET6 if ":" in address else socket.AF_INET),
                proto=socket.IPPROTO_TCP,
                flags=socket.AI_NUMERICHOST,
            )
            for address in sorted(self._allowed_ips)
        ]

    async def close(self) -> None:
        return None


class _PeerVerifyingConnector(TCPConnector):
    """Reject an unverified TCP peer before aiohttp can send the HTTP request."""

    def __init__(self, *, allowed_ips: frozenset[str], resolver: AbstractResolver) -> None:
        self._allowed_ips = allowed_ips
        self.peer_ip: str | None = None
        super().__init__(
            resolver=resolver,
            use_dns_cache=False,
            force_close=True,
            limit=1,
        )

    async def _wrap_create_connection(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[asyncio.Transport, Any]:
        transport, protocol = await super()._wrap_create_connection(*args, **kwargs)
        peer = transport.get_extra_info("peername")
        peer_value = peer[0] if isinstance(peer, tuple) and peer else None
        if not isinstance(peer_value, str):
            transport.close()
            raise WebFetchRejected("WebFetch could not verify the connected peer")
        try:
            normalized_peer = _normalize_public_ip(peer_value)
        except ValueError as exc:
            transport.close()
            raise WebFetchRejected("WebFetch could not verify the connected peer") from exc
        if normalized_peer not in self._allowed_ips:
            transport.close()
            raise WebFetchRejected("WebFetch connected peer was not validated")
        self.peer_ip = normalized_peer
        return transport, protocol


class _AioHttpResponse:
    def __init__(
        self,
        *,
        session: ClientSession,
        response: ClientResponse,
        peer_ip: str | None,
    ) -> None:
        self._session = session
        self._response = response
        self.status_code = response.status
        self.headers: Mapping[str, str] = response.headers
        self.peer_ip = peer_ip

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.content.iter_chunked(64 * 1024):
            yield chunk

    async def close(self) -> None:
        failures: list[BaseException] = []
        try:
            self._response.close()
        except BaseException as error:
            failures.append(error)
        try:
            await self._session.close()
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("HTTP response shutdown failed", failures)


class AioHttpWebFetchClient:
    """Perform one no-redirect HTTP GET pinned to a prevalidated address set."""

    async def get(
        self,
        url: str,
        *,
        allowed_ips: frozenset[str],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        resolver = _PinnedResolver(allowed_ips)
        connector = _PeerVerifyingConnector(
            allowed_ips=allowed_ips,
            resolver=resolver,
        )
        session = ClientSession(
            connector=connector,
            timeout=ClientTimeout(
                total=total_timeout_seconds,
                connect=connect_timeout_seconds,
                sock_connect=connect_timeout_seconds,
            ),
            auto_decompress=True,
            trust_env=False,
        )
        try:
            response = await session.get(url, allow_redirects=False)
        except BaseException as primary_error:
            try:
                await _close_session(session)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        return _AioHttpResponse(
            session=session,
            response=response,
            peer_ip=connector.peer_ip,
        )


class WebFetchTool(BaseTool):
    """Expose public-only HTTP fetching through the Tool protocol."""

    name = "web_fetch"
    description = "Fetch readable text from a public HTTP or HTTPS URL."
    required = ("url",)
    max_retries = 2

    url: Annotated[
        str,
        ToolParam(description="Public HTTP or HTTPS URL.", min_length=1, format="uri"),
    ]

    def __init__(self, *, fetcher: WebFetchBoundary) -> None:
        self._fetcher = fetcher

    async def execute(self, *, url: str) -> str:
        try:
            return await self._fetcher.fetch(url)
        except WebFetchRejected as error:
            raise ToolError("WebFetch rejected an unsafe or unverifiable request.") from error


class PublicWebFetchBoundary:
    """Fetch only targets whose complete DNS answer set is public."""

    def __init__(
        self,
        *,
        resolver: DNSResolverBoundary,
        http_client: HTTPClientBoundary,
        total_timeout_seconds: float = TOTAL_TIMEOUT_SECONDS,
    ) -> None:
        self._resolver = resolver
        self._http_client = http_client
        self._total_timeout_seconds = total_timeout_seconds

    async def fetch(self, url: str) -> str:
        try:
            async with asyncio.timeout(self._total_timeout_seconds):
                return await self._fetch(url)
        except TimeoutError as exc:
            raise WebFetchRejected("WebFetch timed out") from exc

    async def _fetch(self, url: str) -> str:
        current_url = url
        redirects_followed = 0
        while True:
            try:
                parsed = urlsplit(current_url)
            except ValueError as exc:
                raise WebFetchRejected("WebFetch URL could not be verified") from exc
            scheme = parsed.scheme.lower()
            if scheme not in {"http", "https"}:
                raise WebFetchRejected("WebFetch supports only HTTP and HTTPS")
            if parsed.username is not None or parsed.password is not None:
                raise WebFetchRejected("WebFetch does not allow URL userinfo")
            hostname = parsed.hostname
            if hostname is None:
                raise WebFetchRejected("WebFetch requires a hostname")
            normalized_hostname = hostname.rstrip(".").lower()
            if normalized_hostname == "localhost" or normalized_hostname.endswith(".localhost"):
                raise WebFetchRejected("WebFetch target is not public")
            try:
                requested_port = parsed.port
            except ValueError as exc:
                raise WebFetchRejected("WebFetch URL could not be verified") from exc
            if requested_port == 0:
                raise WebFetchRejected("WebFetch URL could not be verified")
            port = (
                requested_port if requested_port is not None else (443 if scheme == "https" else 80)
            )
            answers = await self._resolver.resolve(hostname, port)
            if not answers:
                raise WebFetchRejected("WebFetch could not verify a public target")
            try:
                allowed_ips = frozenset(_normalize_public_ip(answer) for answer in answers)
            except ValueError as exc:
                raise WebFetchRejected("WebFetch could not verify a public target") from exc
            response = await self._http_client.get(
                current_url,
                allowed_ips=allowed_ips,
                connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
                total_timeout_seconds=self._total_timeout_seconds,
            )
            async with _closing_response(response):
                if response.peer_ip is None:
                    raise WebFetchRejected("WebFetch could not verify the connected peer")
                try:
                    peer_ip = _normalize_public_ip(response.peer_ip)
                except ValueError as exc:
                    raise WebFetchRejected("WebFetch could not verify the connected peer") from exc
                if peer_ip not in allowed_ips:
                    raise WebFetchRejected("WebFetch connected peer was not validated")
                location = _header(response.headers, "location")
                if 300 <= response.status_code < 400 and location:
                    if redirects_followed == MAX_REDIRECTS:
                        raise WebFetchRejected("WebFetch redirect limit exceeded")
                    redirects_followed += 1
                    current_url = urljoin(current_url, location)
                    continue
                content_type = _header(response.headers, "content-type")
                media_type = (
                    ("" if content_type is None else content_type.split(";", maxsplit=1)[0])
                    .strip()
                    .lower()
                )
                if not _is_textual_media_type(media_type):
                    raise WebFetchRejected("WebFetch response media type is unsupported")
                content_length = _header(response.headers, "content-length")
                if content_length is not None:
                    normalized_length = content_length.strip()
                    if not normalized_length.isascii() or not normalized_length.isdecimal():
                        raise WebFetchRejected(
                            "WebFetch response content length could not be verified"
                        )
                    if int(normalized_length) > MAX_BODY_BYTES:
                        raise WebFetchRejected("WebFetch response body is too large")
                body = bytearray()
                async for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > MAX_BODY_BYTES:
                        raise WebFetchRejected("WebFetch response body is too large")
                    body.extend(chunk)
                charset = _charset(content_type)
                try:
                    text = bytes(body).decode(charset)
                except (LookupError, UnicodeDecodeError) as exc:
                    raise WebFetchRejected("WebFetch response text could not be decoded") from exc
                if media_type in _HTML_MEDIA_TYPES:
                    return _readable_html(text)
                return text


@asynccontextmanager
async def _closing_response(
    response: HTTPResponseBoundary,
) -> AsyncIterator[HTTPResponseBoundary]:
    try:
        yield response
    except BaseException as primary_error:
        try:
            await _close_response(response)
        except BaseException as cleanup_error:
            raise primary_error from cleanup_error
        raise
    else:
        await _close_response(response)


async def _close_response(response: HTTPResponseBoundary) -> None:
    close_task = asyncio.create_task(response.close())
    await _await_cleanup(close_task)


async def _close_session(session: ClientSession) -> None:
    close_task = asyncio.create_task(session.close())
    await _await_cleanup(close_task)


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


def _header(headers: Mapping[str, str], name: str) -> str | None:
    normalized_name = name.lower()
    return next(
        (value for key, value in headers.items() if key.lower() == normalized_name),
        None,
    )


def _normalize_public_ip(value: str) -> str:
    try:
        return normalize_public_ip(value)
    except ValueError as error:
        raise ValueError("address is not globally routable") from error


def _is_textual_media_type(media_type: str) -> bool:
    return (
        media_type.startswith("text/")
        or media_type in _TEXTUAL_APPLICATION_MEDIA_TYPES
        or media_type.endswith("+json")
        or media_type.endswith("+xml")
    )


def _charset(content_type: str | None) -> str:
    if content_type is None:
        return "utf-8"
    message = Message()
    message["content-type"] = content_type
    return message.get_content_charset() or "utf-8"


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_element: str | None = None
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        normalized = tag.lower()
        if self._ignored_element is not None:
            if normalized == self._ignored_element:
                self._ignored_depth += 1
            return
        if normalized in _HTML_IGNORED_ELEMENTS:
            self._ignored_element = normalized
            self._ignored_depth += 1
        elif normalized in _HTML_BLOCK_ELEMENTS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if self._ignored_element is not None:
            if normalized == self._ignored_element:
                self._ignored_depth -= 1
                if self._ignored_depth == 0:
                    self._ignored_element = None
            return
        if normalized in _HTML_BLOCK_ELEMENTS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self._parts.append(data)

    def readable_text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def _readable_html(content: str) -> str:
    parser = _ReadableHTMLParser()
    parser.feed(content)
    parser.close()
    return parser.readable_text()
