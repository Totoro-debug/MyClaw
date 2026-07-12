"""User Configuration generation and loading."""

import re
import tomllib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, NoReturn, cast
from urllib.parse import urlsplit

import tomlkit
from croniter import croniter  # type: ignore[import-untyped]

from myclaw.agent_home import AgentHome
from myclaw.atomic_files import atomic_create_text
from myclaw.contracts.errors import ErrorInfo
from myclaw.contracts.management import ConfigView

DEFAULT_CONFIG_TEMPLATE: Final = """[runtime]
max_tool_result_chars = 50000

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[tools.web]
enabled = true

[tools.shell]
enabled = true

[models.providers.anthropic-default]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = ""
models = []

[models.providers.openai-local]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []
"""

type ReasoningEffort = Literal["low", "medium", "high"]

_PROVIDER_ID_PATTERN: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ROUTE_NAMES: Final = frozenset({"default", "chat", "memory", "cron"})
_API_KEY_LINE_PATTERN: Final = re.compile(
    r"^(?P<prefix>\s*api[-_]?key\s*=\s*)(?P<value>.*)$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_REDACTED_API_KEY: Final = "***REDACTED***"


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    max_tool_result_chars: int


@dataclass(frozen=True, slots=True)
class MemoryConfiguration:
    consolidation_message_threshold: int
    batch_size: int
    schedule: str


@dataclass(frozen=True, slots=True)
class ToolConfiguration:
    enabled: bool


@dataclass(frozen=True, slots=True)
class ToolsConfiguration:
    web: ToolConfiguration
    shell: ToolConfiguration


@dataclass(frozen=True, slots=True)
class ProviderConfiguration:
    provider_id: str
    protocol: str
    base_url: str
    api_key: str
    models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RouteConfiguration:
    provider_id: str
    model: str
    context_window: int
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout: int


@dataclass(frozen=True, slots=True)
class ModelsConfiguration:
    providers: Mapping[str, ProviderConfiguration]
    routes: Mapping[str, RouteConfiguration]


@dataclass(frozen=True, slots=True)
class ResolvedModelRoute:
    requested_route: str
    selected_route: str
    provider: ProviderConfiguration
    route: RouteConfiguration
    used_default: bool


@dataclass(frozen=True, slots=True)
class UserConfiguration:
    runtime: RuntimeConfiguration
    memory: MemoryConfiguration
    tools: ToolsConfiguration
    models: ModelsConfiguration

    def resolve_route(self, requested_route: str) -> ResolvedModelRoute:
        """Resolve a Model Route, falling back to a usable default when permitted."""
        if requested_route not in _ROUTE_NAMES:
            _invalid("models.routes", "was requested with an unsupported route name")

        candidate = _usable_route(self.models, requested_route)
        selected_route = requested_route
        if candidate is None and requested_route != "default":
            candidate = _usable_route(self.models, "default")
            selected_route = "default"
        if candidate is None:
            raise ConfigError(ErrorInfo("route_unavailable", "Default Model Route is unavailable."))
        provider, route = candidate
        return ResolvedModelRoute(
            requested_route=requested_route,
            selected_route=selected_route,
            provider=provider,
            route=route,
            used_default=selected_route != requested_route,
        )


class ConfigError(Exception):
    """A safe User Configuration error suitable for a CLI or Management view."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


def _invalid(field: str, rule: str) -> NoReturn:
    raise ConfigError(ErrorInfo("config_invalid", f"Configuration field '{field}' {rule}."))


def _table(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _invalid(field, "must be a table")
    return cast(dict[str, object], value)


def _required(table: Mapping[str, object], key: str, field: str) -> object:
    if key not in table:
        _invalid(field, "is required")
    return table[key]


def _reject_unknown(table: Mapping[str, object], allowed: set[str], prefix: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        field = f"{prefix}.{unknown[0]}" if prefix else unknown[0]
        _invalid(field, "is not recognized")


def _string(value: object, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        _invalid(field, "must be a string")
    if nonempty and (not value or value != value.strip()):
        _invalid(field, "must be a nonempty string without surrounding whitespace")
    return value


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        _invalid(field, f"must be an integer from {minimum} to {maximum}")
    return value


def _number(value: object, field: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not minimum <= value <= maximum
    ):
        _invalid(field, f"must be a finite number from {minimum:g} to {maximum:g}")
    return float(value)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _invalid(field, "must be a boolean")
    return value


def _has_absolute_http_url(value: str) -> bool:
    if not value or any(character.isspace() for character in value):
        return False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and parsed.hostname is not None


def _usable_route(
    models: ModelsConfiguration, route_name: str
) -> tuple[ProviderConfiguration, RouteConfiguration] | None:
    route = models.routes.get(route_name)
    if route is None:
        return None
    provider = models.providers.get(route.provider_id)
    if provider is None or provider.protocol not in {"anthropic", "openai-compatible"}:
        return None
    if (
        not _has_absolute_http_url(provider.base_url)
        or not provider.api_key.strip()
        or not provider.models
        or route.model not in provider.models
    ):
        return None
    return provider, route


def _redact_parsed_content(content: str) -> str:
    source_document = tomlkit.parse(content)
    models = source_document.get("models")
    if not isinstance(models, MutableMapping):
        return content
    providers = models.get("providers")
    if not isinstance(providers, MutableMapping):
        return content
    for provider in providers.values():
        if not isinstance(provider, MutableMapping):
            continue
        api_key = provider.get("api_key")
        if isinstance(api_key, str) and api_key:
            provider["api_key"] = _REDACTED_API_KEY
    return tomlkit.dumps(source_document)


def _redact_unparsed_content(content: str) -> str:
    def redact_line(match: re.Match[str]) -> str:
        return f'{match.group("prefix")}"{_REDACTED_API_KEY}"'

    return _API_KEY_LINE_PATTERN.sub(redact_line, content)


def _parse_runtime(document: Mapping[str, object]) -> RuntimeConfiguration:
    table = _table(document.get("runtime", {}), "runtime")
    _reject_unknown(table, {"max_tool_result_chars"}, "runtime")
    return RuntimeConfiguration(
        max_tool_result_chars=_integer(
            table.get("max_tool_result_chars", 50_000),
            "runtime.max_tool_result_chars",
            1000,
            1_000_000,
        )
    )


def _parse_memory(document: Mapping[str, object]) -> MemoryConfiguration:
    table = _table(document.get("memory", {}), "memory")
    _reject_unknown(
        table,
        {"consolidation_message_threshold", "batch_size", "schedule"},
        "memory",
    )
    schedule = _string(table.get("schedule", "0 * * * *"), "memory.schedule")
    if len(schedule.split()) != 5 or not croniter.is_valid(schedule):
        _invalid("memory.schedule", "must be a valid five-field cron expression")
    return MemoryConfiguration(
        consolidation_message_threshold=_integer(
            table.get("consolidation_message_threshold", 40),
            "memory.consolidation_message_threshold",
            4,
            10_000,
        ),
        batch_size=_integer(
            table.get("batch_size", 10),
            "memory.batch_size",
            1,
            1000,
        ),
        schedule=schedule,
    )


def _parse_tools(document: Mapping[str, object]) -> ToolsConfiguration:
    table = _table(document.get("tools", {}), "tools")
    _reject_unknown(table, {"web", "shell"}, "tools")

    def parse_tool(name: str) -> ToolConfiguration:
        field = f"tools.{name}"
        tool = _table(table.get(name, {}), field)
        _reject_unknown(tool, {"enabled"}, field)
        return ToolConfiguration(enabled=_boolean(tool.get("enabled", True), f"{field}.enabled"))

    return ToolsConfiguration(web=parse_tool("web"), shell=parse_tool("shell"))


def _parse_provider(provider_id: str, value: object) -> ProviderConfiguration:
    prefix = f"models.providers.{provider_id}"
    if not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
        _invalid(prefix, "must use a lowercase kebab-case provider ID")
    table = _table(value, prefix)
    _reject_unknown(table, {"protocol", "base_url", "api_key", "models"}, prefix)
    models_value = _required(table, "models", f"{prefix}.models")
    if not isinstance(models_value, list):
        _invalid(f"{prefix}.models", "must be an array of unique nonempty model IDs")
    model_items = cast(list[object], models_value)
    models: list[str] = []
    for model_value in model_items:
        model = _string(model_value, f"{prefix}.models", nonempty=True)
        if model in models:
            _invalid(f"{prefix}.models", "must contain unique model IDs")
        models.append(model)
    return ProviderConfiguration(
        provider_id=provider_id,
        protocol=_string(_required(table, "protocol", f"{prefix}.protocol"), f"{prefix}.protocol"),
        base_url=_string(_required(table, "base_url", f"{prefix}.base_url"), f"{prefix}.base_url"),
        api_key=_string(_required(table, "api_key", f"{prefix}.api_key"), f"{prefix}.api_key"),
        models=tuple(models),
    )


def _parse_route(route_name: str, value: object) -> RouteConfiguration:
    prefix = f"models.routes.{route_name}"
    if route_name not in _ROUTE_NAMES:
        _invalid(prefix, "is not a supported Model Route")
    table = _table(value, prefix)
    _reject_unknown(
        table,
        {
            "provider_id",
            "model",
            "context_window",
            "max_output",
            "temperature",
            "reasoning_effort",
            "timeout",
        },
        prefix,
    )
    provider_id = _string(
        _required(table, "provider_id", f"{prefix}.provider_id"),
        f"{prefix}.provider_id",
        nonempty=True,
    )
    if not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
        _invalid(f"{prefix}.provider_id", "must be a lowercase kebab-case provider ID")
    context_window = _integer(
        _required(table, "context_window", f"{prefix}.context_window"),
        f"{prefix}.context_window",
        1024,
        10_000_000,
    )
    max_output = _integer(
        _required(table, "max_output", f"{prefix}.max_output"),
        f"{prefix}.max_output",
        1,
        9_999_999,
    )
    if max_output >= context_window:
        _invalid(f"{prefix}.max_output", "must be less than context_window")
    reasoning_value = table.get("reasoning_effort")
    reasoning_effort: ReasoningEffort | None = None
    if reasoning_value is not None:
        reasoning = _string(reasoning_value, f"{prefix}.reasoning_effort")
        if reasoning not in {"low", "medium", "high"}:
            _invalid(f"{prefix}.reasoning_effort", "must be low, medium, or high")
        reasoning_effort = cast(ReasoningEffort, reasoning)
    return RouteConfiguration(
        provider_id=provider_id,
        model=_string(
            _required(table, "model", f"{prefix}.model"),
            f"{prefix}.model",
            nonempty=True,
        ),
        context_window=context_window,
        max_output=max_output,
        temperature=_number(
            _required(table, "temperature", f"{prefix}.temperature"),
            f"{prefix}.temperature",
            0,
            2,
        ),
        reasoning_effort=reasoning_effort,
        timeout=_integer(
            _required(table, "timeout", f"{prefix}.timeout"),
            f"{prefix}.timeout",
            1,
            600,
        ),
    )


def _parse_models(document: Mapping[str, object]) -> ModelsConfiguration:
    models_value = document.get("models", {})
    table = _table(models_value, "models")
    _reject_unknown(table, {"providers", "routes"}, "models")
    provider_tables = _table(table.get("providers", {}), "models.providers")
    route_tables = _table(table.get("routes", {}), "models.routes")
    providers = {
        provider_id: _parse_provider(provider_id, provider)
        for provider_id, provider in provider_tables.items()
    }
    routes = {
        route_name: _parse_route(route_name, route) for route_name, route in route_tables.items()
    }
    return ModelsConfiguration(
        providers=MappingProxyType(providers),
        routes=MappingProxyType(routes),
    )


def _parse_configuration(document: dict[str, object]) -> UserConfiguration:
    _reject_unknown(document, {"runtime", "memory", "tools", "models"}, "")
    return UserConfiguration(
        runtime=_parse_runtime(document),
        memory=_parse_memory(document),
        tools=_parse_tools(document),
        models=_parse_models(document),
    )


class ConfigLoader:
    """Access User Configuration beneath an injected fixed Agent Home."""

    def __init__(self, agent_home: AgentHome) -> None:
        self.agent_home = agent_home

    @property
    def path(self) -> Path:
        return self.agent_home.path / "config.toml"

    def ensure_default(self) -> bool:
        """Create the accepted default template when missing."""
        self.agent_home.initialize()
        return atomic_create_text(self.path, DEFAULT_CONFIG_TEMPLATE)

    def load(self) -> UserConfiguration:
        """Load User Configuration as immutable typed values."""
        try:
            loaded: object = tomllib.loads(self.path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
            raise ConfigError(
                ErrorInfo(
                    "config_parse_error",
                    "User Configuration TOML could not be parsed.",
                )
            ) from error
        document = _table(loaded, "configuration")
        return _parse_configuration(document)

    def load_for_startup(self) -> UserConfiguration:
        """Generate missing configuration or return a startup-usable configuration."""
        try:
            if self.ensure_default():
                raise ConfigError(
                    ErrorInfo(
                        "config_missing",
                        "A default User Configuration was created; edit it before starting MyClaw.",
                    )
                )
            configuration = self.load()
            configuration.resolve_route("default")
            return configuration
        except OSError as error:
            raise ConfigError(
                ErrorInfo(
                    "persistence_error",
                    "User Configuration could not be read or written.",
                )
            ) from error

    def view(self) -> ConfigView:
        """Return complete User Configuration text with plaintext API keys redacted."""
        content = self.path.read_text(encoding="utf-8")
        try:
            loaded: object = tomllib.loads(content)
        except tomllib.TOMLDecodeError:
            return ConfigView(
                path=self.path,
                redacted_content=_redact_unparsed_content(content),
                error=ErrorInfo(
                    "config_parse_error",
                    "User Configuration TOML could not be parsed.",
                ),
            )
        document = _table(loaded, "configuration")
        error: ErrorInfo | None = None
        try:
            _parse_configuration(document)
        except ConfigError as config_error:
            error = config_error.error
        return ConfigView(
            path=self.path,
            redacted_content=_redact_parsed_content(content),
            error=error,
        )
