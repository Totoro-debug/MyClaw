"""Web Fetch Core Catalog Tool."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from email.message import Message
from html import unescape
from typing import Annotated, Protocol
from urllib.parse import SplitResult, urljoin, urlsplit

from aiohttp import ClientResponse, ClientSession, ClientTimeout

from myclaw.tools.base import BaseTool, ToolError, ToolParam, truncate_text
from myclaw.tools.network_safety import DNSResolver, SocketDNSResolver, assess_target

CONNECT_TIMEOUT_SECONDS = 10.0
TOTAL_TIMEOUT_SECONDS = 30.0
MAX_REDIRECTS = 5
DEFAULT_MAX_CHARS = 50000

_JINA_READER_URL = "https://r.jina.ai/"
_TEXTUAL_APPLICATION_MEDIA_TYPES = frozenset(
    {
        "application/ecmascript",
        "application/javascript",
        "application/json",
        "application/sql",
        "application/x-javascript",
        "application/x-yaml",
        "application/xml",
        "application/yaml",
    }
)
_HTML_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})
_HTML_IGNORED_ELEMENTS = ("noscript", "script", "style", "template")
_HTML_BLOCK_TAG_PATTERN = re.compile(
    r"</?(?:address|article|aside|blockquote|br|div|footer|h[1-6]|header|li|main|nav|p|"
    r"pre|section|table|td|th|title|tr)\b[^>]*>",
    re.IGNORECASE,
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]*>")


class JinaReaderBoundary(Protocol):
    async def fetch(self, url: str, *, output_format: str) -> str: ...


class HTTPResponseBoundary(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_bytes(self) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


class HTTPClientBoundary(Protocol):
    async def get(
        self,
        url: str,
        *,
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary: ...


class _AioHttpResponse:
    def __init__(self, *, session: ClientSession, response: ClientResponse) -> None:
        self._session = session
        self._response = response
        self.status_code = response.status
        self.headers: Mapping[str, str] = response.headers

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.content.iter_chunked(64 * 1024):
            yield chunk

    async def close(self) -> None:
        self._response.close()
        await self._session.close()


class AioHttpWebFetchClient:
    """Perform one no-redirect GET with the caller's bounded timeout."""

    async def get(
        self,
        url: str,
        *,
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        session = ClientSession(
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
        except BaseException:
            await _close_session(session)
            raise
        return _AioHttpResponse(session=session, response=response)


class JinaReaderClient:
    """Fetch one public target through anonymous Jina Reader."""

    async def fetch(self, url: str, *, output_format: str) -> str:
        timeout = ClientTimeout(
            total=None,
            connect=CONNECT_TIMEOUT_SECONDS,
            sock_connect=CONNECT_TIMEOUT_SECONDS,
        )
        async with ClientSession(
            timeout=timeout,
            auto_decompress=True,
            trust_env=False,
        ) as session:
            async with session.get(
                f"{_JINA_READER_URL}{url}",
                headers={
                    "Accept": "text/plain",
                    "X-Respond-With": output_format,
                },
                allow_redirects=False,
            ) as response:
                if not 200 <= response.status < 300:
                    return ""
                content = await response.text(encoding="utf-8", errors="replace")
                return content if content.strip() else ""


@dataclass(frozen=True, slots=True)
class _TargetEvaluation:
    safety_reason: str | None


class WebFetchTool(BaseTool):
    """Fetch readable web content with a Jina-first public path."""

    name = "web_fetch"
    description = "Fetch readable content from an HTTP or HTTPS URL."
    required = ("url",)

    url: Annotated[
        str,
        ToolParam(description="HTTP or HTTPS URL to fetch.", min_length=1, format="uri"),
    ]
    format: Annotated[
        str,
        ToolParam(description="Output format: markdown or text.", min_length=1),
    ] = "markdown"
    maxChars: Annotated[
        int,
        ToolParam(description="Maximum returned characters.", minimum=1),
    ] = DEFAULT_MAX_CHARS

    def __init__(
        self,
        *,
        resolver: DNSResolver | None = None,
        jina_reader: JinaReaderBoundary | None = None,
        http_client: HTTPClientBoundary | None = None,
    ) -> None:
        self._resolver = SocketDNSResolver() if resolver is None else resolver
        self._jina_reader = JinaReaderClient() if jina_reader is None else jina_reader
        self._http_client = AioHttpWebFetchClient() if http_client is None else http_client
        self._evaluations: dict[str, _TargetEvaluation] = {}

    def validate_arguments(  # type: ignore[override]
        self,
        *,
        url: str,
        format: str,
        maxChars: int,
    ) -> str | None:
        del maxChars
        try:
            _parse_url(url.strip())
        except ValueError as error:
            return f"Web Fetch URL is invalid: {error}"
        if format not in {"markdown", "text"}:
            return "Web Fetch format must be either markdown or text."
        return None

    async def check_safety(  # type: ignore[override]
        self,
        *,
        url: str,
        format: str,
        maxChars: int,
    ) -> str | None:
        del format, maxChars
        normalized_url = url.strip()
        evaluation = await self._evaluate_target(normalized_url)
        self._evaluations[normalized_url] = evaluation
        return evaluation.safety_reason

    async def execute(self, *, url: str, format: str, maxChars: int) -> str:
        normalized_url = url.strip()
        evaluation = self._evaluations.pop(normalized_url, None)
        if evaluation is None:
            evaluation = await self._evaluate_target(normalized_url)

        try:
            async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
                if evaluation.safety_reason is None:
                    try:
                        jina_content = await self._jina_reader.fetch(
                            normalized_url,
                            output_format=format,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        jina_content = ""
                    if isinstance(jina_content, str) and jina_content.strip():
                        return truncate_text(jina_content, limit=maxChars)

                content = await self._fetch_direct(normalized_url)
                return truncate_text(content, limit=maxChars)
        except asyncio.CancelledError:
            raise
        except TimeoutError as error:
            raise ToolError(
                f"Web Fetch timed out after {TOTAL_TIMEOUT_SECONDS:g} seconds."
            ) from error

    async def _evaluate_target(self, url: str) -> _TargetEvaluation:
        try:
            parsed = _parse_url(url)
        except ValueError as error:
            raise ToolError(f"Web Fetch URL is invalid: {error}") from error

        hostname = parsed.hostname
        if hostname is None:
            raise ToolError("Web Fetch URL must contain a hostname.")
        port = _effective_port(parsed)
        assessment = await assess_target(hostname, port, self._resolver)
        reasons = {
            "literal_non_global": (
                "Web Fetch target uses a private or non-global address and requires confirmation."
            ),
            "dns_failure": ("Web Fetch target DNS resolution failed and requires confirmation."),
            "dns_empty": (
                "Web Fetch target DNS resolution returned no addresses and requires confirmation."
            ),
            "dns_non_global": (
                "Web Fetch target resolves to a private or non-global address and requires "
                "confirmation."
            ),
        }
        return _TargetEvaluation(
            safety_reason=None if assessment.risk is None else reasons[assessment.risk]
        )

    async def _fetch_direct(self, url: str) -> str:
        current_url = url
        redirects_followed = 0
        while True:
            response = await self._http_client.get(
                current_url,
                connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
                total_timeout_seconds=TOTAL_TIMEOUT_SECONDS,
            )
            try:
                location = _header(response.headers, "location")
                if 300 <= response.status_code < 400 and location:
                    if redirects_followed >= MAX_REDIRECTS:
                        raise ToolError("Web Fetch redirect limit exceeded.")
                    next_url = urljoin(current_url, location)
                    next_evaluation = await self._evaluate_target(next_url)
                    if next_evaluation.safety_reason is not None:
                        raise ToolError(
                            "Web Fetch redirect target requires a separate confirmed invocation: "
                            f"{next_url}"
                        )
                    redirects_followed += 1
                    current_url = next_url
                    continue

                content_type = _header(response.headers, "content-type")
                if not _is_textual_media_type(_media_type(content_type)):
                    raise ToolError("Web Fetch response media type is unsupported.")
                body = await _read_body(response)
                return _decode_response(
                    body,
                    content_type=content_type,
                )
            finally:
                await response.close()


async def _read_body(response: HTTPResponseBoundary) -> bytes:
    chunks = [chunk async for chunk in response.iter_bytes()]
    return b"".join(chunks)


async def _close_session(session: ClientSession) -> None:
    try:
        await session.close()
    except BaseException:
        return None


def _parse_url(url: str) -> SplitResult:
    if not url:
        raise ValueError("URL must not be blank")
    if any(character.isspace() or ord(character) < 32 for character in url):
        raise ValueError("URL must not contain whitespace or control characters")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("URL is malformed") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("URL scheme must be HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL userinfo is not supported")
    if not parsed.netloc or hostname is None:
        raise ValueError("URL must contain a hostname")
    if port == 0:
        raise ValueError("URL port must not be zero")
    return parsed


def _effective_port(parsed: SplitResult) -> int:
    port = parsed.port
    if port is not None:
        return port
    return 443 if parsed.scheme.lower() == "https" else 80


def _header(headers: Mapping[str, str], name: str) -> str | None:
    normalized_name = name.lower()
    return next(
        (value for key, value in headers.items() if key.lower() == normalized_name),
        None,
    )


def _media_type(content_type: str | None) -> str:
    if content_type is None:
        return ""
    return content_type.split(";", maxsplit=1)[0].strip().lower()


def _is_textual_media_type(media_type: str) -> bool:
    return (
        not media_type
        or media_type.startswith("text/")
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


def _decode_response(
    body: bytes,
    *,
    content_type: str | None,
) -> str:
    media_type = _media_type(content_type)
    if not _is_textual_media_type(media_type):
        raise ToolError("Web Fetch response media type is unsupported.")
    try:
        charset = _charset(content_type)
    except (LookupError, ValueError):
        charset = "utf-8"
    try:
        content = body.decode(charset, errors="replace")
    except LookupError:
        content = body.decode("utf-8", errors="replace")
    if media_type in _HTML_MEDIA_TYPES:
        return _readable_html(content)
    return content


def _readable_html(content: str) -> str:
    for tag in _HTML_IGNORED_ELEMENTS:
        content = re.sub(
            rf"<{tag}\b[^>]*>.*?(?:</{tag}\s*>|$)",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
    content = _HTML_BLOCK_TAG_PATTERN.sub("\n", content)
    content = _HTML_TAG_PATTERN.sub("", content)
    content = unescape(content)
    lines = (" ".join(line.split()) for line in content.splitlines())
    return "\n".join(line for line in lines if line)


__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_CHARS",
    "MAX_REDIRECTS",
    "TOTAL_TIMEOUT_SECONDS",
    "AioHttpWebFetchClient",
    "HTTPClientBoundary",
    "HTTPResponseBoundary",
    "JinaReaderBoundary",
    "JinaReaderClient",
    "WebFetchTool",
]
