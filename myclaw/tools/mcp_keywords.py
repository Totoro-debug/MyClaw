"""Prepare and persist discovery keywords for discovered MCP Tools."""

from __future__ import annotations

import asyncio
import json
import re
import tomllib
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

import tomlkit

from myclaw.config.config import (
    MCPServerConfiguration,
    UserConfiguration,
    _parse_configuration,
)
from myclaw.templates import render_template
from myclaw.tools.mcp import MCPTool
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

_MAX_KEYWORD_PREPARATION_CONCURRENCY = 4

type KeywordIdentity = tuple[str, str]
type KeywordConfiguration = Mapping[str, MCPServerConfiguration] | UserConfiguration
type GeneratedKeywordValues = (
    Mapping[KeywordIdentity, Sequence[str]] | Mapping[str, Mapping[str, Sequence[str]]]
)


class KeywordModelRouter(Protocol):
    """The direct chat completion seam used for startup metadata."""

    async def complete(
        self,
        route: Literal["chat"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class _KeywordCacheKey:
    server_name: str
    remote_name: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _PreparedKeywords:
    identity: KeywordIdentity
    model_name: str
    keywords: tuple[str, ...]
    generated: bool


class MCPKeywordPreparer:
    """Prepare one complete MCP snapshot without affecting Session state."""

    def __init__(
        self,
        router: KeywordModelRouter | None = None,
        config_path: Path | None = None,
        *,
        model_router: KeywordModelRouter | None = None,
        cache: MutableMapping[_KeywordCacheKey, tuple[str, ...]] | None = None,
    ) -> None:
        if router is not None and model_router is not None:
            raise TypeError("MCP Keyword Preparer accepts only one Model Router")
        selected_router = router if router is not None else model_router
        if selected_router is None:
            raise TypeError("MCP Keyword Preparer requires a Model Router")
        if config_path is not None and not isinstance(config_path, Path):
            raise TypeError("MCP Keyword Preparer requires a Path configuration path")
        self._router = selected_router
        self._config_path = config_path
        self._cache = {} if cache is None else cache

    async def prepare(
        self,
        tools: Sequence[MCPTool],
        configuration: KeywordConfiguration | Mapping[str, object],
    ) -> Mapping[str, tuple[str, ...]]:
        """Return prepared keywords keyed by each Tool's model-facing name."""
        servers = _configuration_servers(configuration)
        candidates: list[tuple[str, str, str, str, dict[str, Any]]] = []
        for tool in tools:
            metadata = _tool_metadata_or_none(tool)
            if metadata is not None:
                candidates.append(metadata)
        if not candidates:
            return MappingProxyType({})

        semaphore = asyncio.Semaphore(_MAX_KEYWORD_PREPARATION_CONCURRENCY)

        async def prepare_one(
            metadata: tuple[str, str, str, str, dict[str, Any]],
        ) -> _PreparedKeywords:
            async with semaphore:
                server_name, remote_name, model_name, description, parameters = metadata
                identity = (server_name, remote_name)
                configured = _configured_keywords(servers.get(server_name), remote_name)
                if configured:
                    return _PreparedKeywords(identity, model_name, configured, generated=False)

                cache_key = _cache_key(
                    server_name=server_name,
                    remote_name=remote_name,
                    description=description,
                    parameters=parameters,
                )
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return _PreparedKeywords(identity, model_name, cached, generated=False)

                try:
                    generated = await self._generate(
                        remote_name=remote_name,
                        description=description,
                        parameters=parameters,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    generated = (remote_name,)
                    persist = False
                else:
                    persist = True
                self._cache[cache_key] = generated
                return _PreparedKeywords(identity, model_name, generated, generated=persist)

        tasks = tuple(asyncio.create_task(prepare_one(metadata)) for metadata in candidates)
        try:
            prepared = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        generated_values = {item.identity: item.keywords for item in prepared if item.generated}
        effective_values: Mapping[KeywordIdentity, tuple[str, ...]] = MappingProxyType({})
        if generated_values and self._config_path is not None:
            try:
                effective_values = fill_mcp_tool_keywords(self._config_path, generated_values)
            except Exception:
                effective_values = MappingProxyType({})

        result: dict[str, tuple[str, ...]] = {}
        for metadata, item in zip(candidates, prepared, strict=True):
            effective = effective_values.get(item.identity, item.keywords)
            result[item.model_name] = effective
            if item.generated:
                server_name, remote_name, _model_name, description, parameters = metadata
                self._cache[
                    _cache_key(
                        server_name=server_name,
                        remote_name=remote_name,
                        description=description,
                        parameters=parameters,
                    )
                ] = effective
        return MappingProxyType(result)

    async def _generate(
        self,
        *,
        remote_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> tuple[str, ...]:
        payload = json.dumps(
            {
                "name": remote_name,
                "description": description,
                "input_schema": deepcopy(parameters),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        response = await self._router.complete(
            "chat",
            messages=[
                {
                    "role": "system",
                    "content": render_template("mcp-keyword-system-prompt.md"),
                },
                {"role": "user", "content": payload},
            ],
            tools=(),
        )
        content = response.message.content
        if not isinstance(content, str):
            raise ValueError("MCP keyword generation returned non-text content")
        decoded = json.loads(content)
        if not isinstance(decoded, list):
            raise ValueError("MCP keyword generation returned a non-array")
        keywords = _normalize_keywords(decoded, english=True)
        if not keywords:
            raise ValueError("MCP keyword generation returned no keywords")
        return keywords


def fill_mcp_tool_keywords(
    config_path: Path,
    generated: GeneratedKeywordValues,
) -> Mapping[KeywordIdentity, tuple[str, ...]]:
    """Fill still-empty MCP keyword entries while preserving the latest TOML."""
    if not isinstance(config_path, Path):
        raise TypeError("MCP keyword configuration path must be a Path")
    assignments = _normalize_assignments(generated)
    if not assignments:
        return MappingProxyType({})

    content = config_path.read_text(encoding="utf-8")
    source_document = tomlkit.parse(content)
    parsed_document = tomllib.loads(content)
    _parse_configuration(cast(dict[str, object], parsed_document), diagnostics=[])

    mcp = source_document.get("mcp")
    if mcp is None:
        return MappingProxyType({})
    if not isinstance(mcp, MutableMapping):
        raise TypeError("mcp must be a table")
    servers = mcp.get("servers")
    if servers is None:
        return MappingProxyType({})
    if not isinstance(servers, MutableMapping):
        raise TypeError("mcp.servers must be a table")

    effective: dict[KeywordIdentity, tuple[str, ...]] = {}
    changed = False
    for (server_name, remote_name), keywords in assignments.items():
        server = servers.get(server_name)
        if server is None:
            continue
        if not isinstance(server, MutableMapping):
            raise TypeError(f"mcp.servers.{server_name} must be a table")
        keyword_table = server.get("tool_keywords")
        if keyword_table is None:
            keyword_table = tomlkit.table()
            server["tool_keywords"] = keyword_table
            changed = True
        if not isinstance(keyword_table, MutableMapping):
            raise TypeError(f"mcp.servers.{server_name}.tool_keywords must be a table")

        existing = keyword_table.get(remote_name)
        if existing is not None:
            existing_keywords = _normalize_keywords(existing, english=False)
            if existing_keywords:
                effective[(server_name, remote_name)] = existing_keywords
                continue
        keyword_table[remote_name] = list(keywords)
        effective[(server_name, remote_name)] = keywords
        changed = True

    if not changed:
        return MappingProxyType(effective)

    candidate_content = tomlkit.dumps(source_document)
    candidate = tomllib.loads(candidate_content)
    _parse_configuration(cast(dict[str, object], candidate), diagnostics=[])
    HOST_FILESYSTEM.atomic_replace_text(config_path, candidate_content)
    return MappingProxyType(effective)


def _configuration_servers(
    configuration: KeywordConfiguration | Mapping[str, object],
) -> Mapping[str, object]:
    if isinstance(configuration, UserConfiguration):
        return configuration.mcp
    if isinstance(configuration, Mapping):
        return configuration
    raise TypeError("MCP keyword preparation requires MCP Server configuration")


def _tool_metadata_or_none(
    tool: MCPTool,
) -> tuple[str, str, str, str, dict[str, Any]] | None:
    try:
        return _tool_metadata(tool)
    except (AttributeError, TypeError, ValueError):
        return None


def _tool_metadata(tool: MCPTool) -> tuple[str, str, str, str, dict[str, Any]]:
    server_name = tool.server_name
    remote_name = tool.remote_name
    model_name = tool.name
    description = tool.description
    parameters = tool.parameters
    if not all(
        isinstance(value, str) and value
        for value in (server_name, remote_name, model_name, description)
    ):
        raise ValueError("MCP Tool metadata must contain non-empty strings")
    if not isinstance(parameters, dict):
        raise TypeError("MCP Tool parameters must be a dictionary")
    return server_name, remote_name, model_name, description, deepcopy(parameters)


def _configured_keywords(server: object | None, remote_name: str) -> tuple[str, ...]:
    if server is None:
        return ()
    configured = getattr(server, "tool_keywords", {})
    if not isinstance(configured, Mapping):
        return ()
    raw = configured.get(remote_name)
    if raw is None:
        return ()
    try:
        return _normalize_keywords(raw, english=False)
    except (TypeError, ValueError):
        return ()


def _cache_key(
    *,
    server_name: str,
    remote_name: str,
    description: str,
    parameters: dict[str, Any],
) -> _KeywordCacheKey:
    try:
        fingerprint_source = json.dumps(
            {"description": description, "parameters": parameters},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        fingerprint_source = repr((description, parameters))
    fingerprint = sha256(fingerprint_source.encode("utf-8")).hexdigest()
    return _KeywordCacheKey(server_name, remote_name, fingerprint)


def _normalize_keywords(value: object, *, english: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("MCP Tool keywords must be an array of strings")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("MCP Tool keywords must be an array of strings")
        keyword = item.strip()
        if not keyword:
            continue
        if english and (
            not keyword.isascii()
            or re.search(r"[A-Za-z]", keyword) is None
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in keyword)
        ):
            raise ValueError("MCP Tool keywords must contain English terms")
        if keyword not in normalized:
            normalized.append(keyword)
    return tuple(normalized)


def _normalize_assignments(
    generated: GeneratedKeywordValues,
) -> dict[KeywordIdentity, tuple[str, ...]]:
    if not isinstance(generated, Mapping):
        raise TypeError("Generated MCP keywords must be a mapping")
    assignments: dict[KeywordIdentity, tuple[str, ...]] = {}
    for key, value in generated.items():
        if isinstance(key, tuple):
            if len(key) != 2 or not all(isinstance(item, str) and item for item in key):
                raise TypeError("Generated MCP keyword identities must name a Server and Tool")
            keywords = _normalize_keywords(value, english=True)
            if keywords:
                assignments[(key[0], key[1])] = keywords
            continue
        if isinstance(key, str) and isinstance(value, Mapping):
            for remote_name, remote_keywords in value.items():
                if not isinstance(remote_name, str) or not remote_name:
                    raise TypeError("Generated MCP Tool names must be non-empty strings")
                keywords = _normalize_keywords(remote_keywords, english=True)
                if keywords:
                    assignments[(key, remote_name)] = keywords
            continue
        raise TypeError("Generated MCP keywords must be keyed by Server and Tool")
    return assignments


__all__ = [
    "KeywordIdentity",
    "MCPKeywordPreparer",
    "fill_mcp_tool_keywords",
]
