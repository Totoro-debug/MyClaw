import os
import shutil
import subprocess
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.tools.models import (
    ModelToolCall,
    ToolExecutionContext,
)
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.tools.web.web_fetch import (
    HTTPResponseBoundary,
    PublicWebFetchBoundary,
    WebFetchRejected,
)

SCHEMA_INVALID_API_KEY_ALIAS_CONFIG = """[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
API-Key = "sk-schema-alias-secret"
models = ["model-id"]

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

MALFORMED_DOTTED_API_KEY_CONFIG = """models.providers.primary.API-Key = "sk-dotted-secret"
models.providers.primary.protocol = "anthropic"
broken = [
"""

MALFORMED_QUOTED_DOTTED_API_KEY_CONFIG = """"models"."providers"."primary"."API-Key" = "sk-quoted-secret"
broken = [
"""

MALFORMED_INLINE_API_KEY_CONFIG = """models = { providers = { primary = { api_key = "sk-inline-secret" } } }
broken = [
"""

MALFORMED_INLINE_MULTILINE_API_KEY_CONFIG = '''models = { providers = { primary = { api_key = """sk-inline-line-one
sk-inline-line-two""" } } }
broken = [
'''

MALFORMED_INLINE_ARRAY_API_KEY_CONFIG = """models = { providers = { primary = { api_key = ["sk-inline-array-secret"] } } }
broken = [
"""

MALFORMED_MULTILINE_ARRAY_API_KEY_CONFIG = """models.providers.primary.api_key = [
  "sk-multiline-array-secret",
]
broken = [
"""

MALFORMED_MULTILINE_API_KEY_CONFIG = '''models.providers.primary.api_key = """sk-line-one
sk-line-two"""
models.providers.primary.protocol = "anthropic"
broken = [
'''

SCHEMA_INVALID_NESTED_API_KEY_CONFIG = """[diagnostics]
API_Key = "sk-nested-secret"
message = "keep this diagnostic"
"""

SCHEMA_INVALID_NON_STRING_API_KEY_CONFIG = """[diagnostics]
api_key = ["sk-array-secret"]
message = "keep this diagnostic"
"""

SCHEMA_INVALID_ARRAY_TABLE_API_KEY_CONFIG = """[[diagnostics]]
api_key = "sk-array-table-secret"
message = "keep this diagnostic"
"""


class StaticAddressResolver:
    def __init__(self, address: str) -> None:
        self._address = address

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        del hostname, port
        return (self._address,)


class NeverDNSResolver:
    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        del hostname, port
        raise AssertionError("DNS must not run for an invalid URL port")


class NeverHTTPClient:
    async def get(
        self,
        url: str,
        *,
        allowed_ips: frozenset[str],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        del url, allowed_ips, connect_timeout_seconds, total_timeout_seconds
        raise AssertionError("HTTP must not run for a non-public address")


class StaticHTTPResponse:
    status_code = 200
    headers: Mapping[str, str] = {"Content-Type": "text/html; charset=utf-8"}
    peer_ip: str | None = "93.184.216.34"

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.closed = False

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        yield self._body

    async def close(self) -> None:
        self.closed = True


class OneHTTPClient:
    def __init__(self, response: StaticHTTPResponse) -> None:
        self._response = response

    async def get(
        self,
        url: str,
        *,
        allowed_ips: frozenset[str],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        del url, allowed_ips, connect_timeout_seconds, total_timeout_seconds
        return self._response


class SecretFailingWebSearch:
    async def search(self, query: str, max_results: int) -> tuple[()]:
        del query, max_results
        raise ConnectionError("api_key=sk-search-adapter-secret\nTraceback: private upstream")


class SecretFailingWebFetch:
    async def fetch(self, url: str) -> str:
        del url
        raise RuntimeError("api_key=sk-fetch-adapter-secret\nTraceback: private response")


def gateway_context(agent_home: Path, workspace: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        lane="foreground",
        workspace=workspace,
        agent_home=agent_home,
        session_id="20260712-120000-000000_550e8400-e29b-41d4-a716-446655440000",
    )


def test_schema_invalid_config_view_redacts_api_key_alias(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(SCHEMA_INVALID_API_KEY_ALIAS_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_invalid"
    assert "sk-schema-alias-secret" not in view.redacted_content
    assert 'API-Key = "***REDACTED***"' in view.redacted_content


def test_schema_invalid_config_view_redacts_api_key_in_unknown_table(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(SCHEMA_INVALID_NESTED_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_invalid"
    assert "sk-nested-secret" not in view.redacted_content
    assert 'API_Key = "***REDACTED***"' in view.redacted_content
    assert 'message = "keep this diagnostic"' in view.redacted_content


def test_schema_invalid_config_view_redacts_non_string_api_key_value(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(SCHEMA_INVALID_NON_STRING_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_invalid"
    assert "sk-array-secret" not in view.redacted_content
    assert 'api_key = "***REDACTED***"' in view.redacted_content
    assert 'message = "keep this diagnostic"' in view.redacted_content


def test_schema_invalid_config_view_redacts_api_key_in_array_table(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(SCHEMA_INVALID_ARRAY_TABLE_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_invalid"
    assert "sk-array-table-secret" not in view.redacted_content
    assert 'api_key = "***REDACTED***"' in view.redacted_content
    assert 'message = "keep this diagnostic"' in view.redacted_content


def test_malformed_config_view_redacts_dotted_api_key_assignment(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MALFORMED_DOTTED_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "sk-dotted-secret" not in view.redacted_content
    assert 'models.providers.primary.API-Key = "***REDACTED***"' in view.redacted_content


def test_malformed_config_view_redacts_quoted_dotted_api_key_assignment(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MALFORMED_QUOTED_DOTTED_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "sk-quoted-secret" not in view.redacted_content
    assert '"models"."providers"."primary"."API-Key" = "***REDACTED***"' in (view.redacted_content)


def test_malformed_config_view_redacts_escaped_basic_quoted_api_key(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        r""""api\u005fkey" = "sk-escaped-key-secret"
broken = [
""",
        encoding="utf-8",
    )

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "sk-escaped-key-secret" not in view.redacted_content
    assert r'''"api\u005fkey" = "***REDACTED***"''' in view.redacted_content


def test_malformed_config_view_redacts_fully_escaped_basic_quoted_api_key(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        r""""\u0041\U00000050\u0049\u002d\U0000004b\u0045\U00000059" = "sk-fully-escaped-secret"
broken = [
""",
        encoding="utf-8",
    )

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "sk-fully-escaped-secret" not in view.redacted_content


@pytest.mark.parametrize(
    ("assignment", "secret"),
    (
        ("'API-Key' = \"sk-literal-quoted-secret\"", "sk-literal-quoted-secret"),
        (
            r'''"models".'providers'.primary."api\u005fkey" = "sk-dotted-escaped-secret"''',
            "sk-dotted-escaped-secret",
        ),
    ),
)
def test_malformed_config_view_redacts_equivalent_quoted_and_dotted_api_keys(
    agent_home: Path,
    assignment: str,
    secret: str,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(f"{assignment}\nbroken = [\n", encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert secret not in view.redacted_content


def test_valid_config_view_redacts_escaped_basic_quoted_api_key(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(
        r""""api\u005fkey" = "sk-valid-escaped-secret"
""",
        encoding="utf-8",
    )

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_invalid"
    assert "sk-valid-escaped-secret" not in view.redacted_content


def test_malformed_config_view_redacts_inline_api_key_assignment(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MALFORMED_INLINE_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "sk-inline-secret" not in view.redacted_content
    assert 'api_key = "***REDACTED***"' in view.redacted_content
    assert "models = { providers = { primary =" in view.redacted_content


def test_malformed_config_view_redacts_inline_multiline_api_key_assignment(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MALFORMED_INLINE_MULTILINE_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "sk-inline-line-one" not in view.redacted_content
    assert "sk-inline-line-two" not in view.redacted_content
    assert 'api_key = "***REDACTED***"' in view.redacted_content
    assert "models = { providers = { primary =" in view.redacted_content


def test_malformed_config_view_fails_closed_for_inline_non_string_api_key(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MALFORMED_INLINE_ARRAY_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "sk-inline-array-secret" not in view.redacted_content
    assert 'api_key = "***REDACTED***"' in view.redacted_content
    assert "models = { providers = { primary =" in view.redacted_content


def test_malformed_config_view_fails_closed_for_multiline_array_api_key(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MALFORMED_MULTILINE_ARRAY_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "sk-multiline-array-secret" not in view.redacted_content
    assert 'models.providers.primary.api_key = "***REDACTED***"' in view.redacted_content


def test_malformed_config_view_redacts_multiline_api_key_value(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MALFORMED_MULTILINE_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "sk-line-one" not in view.redacted_content
    assert "sk-line-two" not in view.redacted_content
    assert 'models.providers.primary.api_key = "***REDACTED***"' in view.redacted_content
    assert 'models.providers.primary.protocol = "anthropic"' in view.redacted_content


def test_installed_config_command_hides_invalid_utf8_and_traceback(agent_home: Path) -> None:
    executable = shutil.which("myclaw")
    assert executable is not None
    agent_home.mkdir(parents=True)
    (agent_home / "config.toml").write_bytes(
        b'api_key = "sk-invalid-utf8-secret"\ninvalid = "\xff"\n'
    )
    environment = os.environ.copy()
    environment["HOME"] = str(agent_home.parent)
    environment["USERPROFILE"] = str(agent_home.parent)

    result = subprocess.run(
        [executable, "config"],
        capture_output=True,
        check=False,
        cwd=agent_home.parent,
        env=environment,
        text=True,
    )

    visible = result.stdout + result.stderr
    assert result.returncode == 1
    assert "persistence_error" in result.stdout
    assert f"Path: {agent_home / 'config.toml'}" in result.stdout
    assert "sk-invalid-utf8-secret" not in visible
    assert "Traceback" not in visible


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ("::7f00:1", "64:ff9b::7f00:1"))
async def test_web_fetch_rejects_ipv6_address_embedding_non_public_ipv4(address: str) -> None:
    fetcher = PublicWebFetchBoundary(
        resolver=StaticAddressResolver(address),
        http_client=NeverHTTPClient(),
    )

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/status")


@pytest.mark.asyncio
async def test_web_fetch_rejects_deprecated_ipv6_site_local_address() -> None:
    fetcher = PublicWebFetchBoundary(
        resolver=StaticAddressResolver("fec0::1"),
        http_client=NeverHTTPClient(),
    )

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example/status")


@pytest.mark.asyncio
async def test_web_fetch_rejects_zero_port_before_dns() -> None:
    fetcher = PublicWebFetchBoundary(
        resolver=NeverDNSResolver(),
        http_client=NeverHTTPClient(),
    )

    with pytest.raises(WebFetchRejected):
        await fetcher.fetch("https://public.example:0/status")


@pytest.mark.asyncio
async def test_web_fetch_does_not_expose_template_data_after_mismatched_end_tag() -> None:
    response = StaticHTTPResponse(
        b"<template></style>sk-raw-template-secret</template><p>Visible text.</p>"
    )
    fetcher = PublicWebFetchBoundary(
        resolver=StaticAddressResolver("93.184.216.34"),
        http_client=OneHTTPClient(response),
    )

    content = await fetcher.fetch("https://public.example/page")

    assert content == "Visible text."
    assert response.closed


@pytest.mark.asyncio
async def test_web_search_gateway_hides_secret_adapter_failure_and_raw_query(
    agent_home: Path,
    workspace: Path,
) -> None:
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        web_search=SecretFailingWebSearch(),
    )

    result = await gateway.call(
        ModelToolCall(
            id="call-secret-search",
            name="web_search",
            arguments='{"query":"sk-raw-query-secret"}',
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_failed"
    assert result.content == "web_search could not complete the request."
    assert "sk-search-adapter-secret" not in result.content
    assert "sk-raw-query-secret" not in result.content
    assert "Traceback" not in result.content


@pytest.mark.asyncio
async def test_web_fetch_gateway_hides_secret_adapter_failure_and_raw_url(
    agent_home: Path,
    workspace: Path,
) -> None:
    gateway = ToolGateway(
        context=gateway_context(agent_home, workspace),
        web_fetch=SecretFailingWebFetch(),
    )

    result = await gateway.call(
        ModelToolCall(
            id="call-secret-fetch",
            name="web_fetch",
            arguments='{"url":"https://public.example/?api_key=sk-raw-url-secret"}',
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_failed"
    assert result.content == "web_fetch could not complete the request."
    assert "sk-fetch-adapter-secret" not in result.content
    assert "sk-raw-url-secret" not in result.content
    assert "Traceback" not in result.content
