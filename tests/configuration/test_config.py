import os
from pathlib import Path

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader

EXPECTED_DEFAULT_CONFIG = """[runtime]
max_tool_result_chars = 4096
max_iterations = 50

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[models.providers.openai-local]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []

# Replace provider_id, model, and model limits with values supported by your provider.
# Remove any purpose-specific route to fall back to default.
[models.routes.default]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.chat]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.memory]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.schedule]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120
"""

VALID_CONFIG = """[runtime]
max_tool_result_chars = 60000

[memory]
consolidation_message_threshold = 50
batch_size = 12
schedule = "15 * * * *"

[models.providers.anthropic-default]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "sk-ant-secret"
models = ["claude-model"]

[models.routes.default]
provider_id = "anthropic-default"
model = "claude-model"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120
"""

MINIMAL_VALID_CONFIG = """[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "minimal-secret"
models = ["small-model"]

[models.routes.default]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30
"""

PROJECTED_CONFIG = """unexpected = true

[runtime]
max_tool_result_chars = 60000
ignored_runtime_field = true

[models]
ignored_models_field = true

[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://models.example/v1"
api_key = "minimal-secret"
models = ["small-model"]
ignored_provider_field = true

[models.routes.default]
provider_id = "primary"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30
ignored_route_field = true

[models.routes.cron]
this_legacy_route_is_ignored = true

[models.routes.future]
this_undefined_route_is_ignored = true
"""

SCHEDULE_ROUTE = """

[models.providers.schedule-provider]
protocol = "openai-compatible"
base_url = "https://schedule.example/v1"
api_key = "schedule-secret"
models = ["schedule-model"]

[models.routes.schedule]
provider_id = "schedule-provider"
model = "schedule-model"
context_window = 100000
max_output = 4096
temperature = 0.1
reasoning_effort = "high"
timeout = 90
"""

REDACTION_CONFIG = """# User Configuration
[runtime]
max_tool_result_chars = 50000

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "plaintext-primary-key"
models = ["model-id"]

[models.providers.empty-template]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0.2
timeout = 60
"""

EXPECTED_REDACTED_CONFIG = """# User Configuration
[runtime]
max_tool_result_chars = 50000

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[models.providers.primary]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "***REDACTED***"
models = ["model-id"]

[models.providers.empty-template]
protocol = "openai-compatible"
base_url = ""
api_key = ""
models = []

[models.routes.default]
provider_id = "primary"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0.2
timeout = 60
"""

OVERLAPPING_API_KEY_CONFIG = """# model-id must remain visible outside api_key
[models.providers.model-id]
protocol = "openai-compatible"
base_url = "https://model-id.example/v1/model-id"
api_key = "model-id"
models = ["model-id"]

[models.routes.default]
provider_id = "model-id"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

EXPECTED_REDACTED_OVERLAPPING_CONFIG = """# model-id must remain visible outside api_key
[models.providers.model-id]
protocol = "openai-compatible"
base_url = "https://model-id.example/v1/model-id"
api_key = "***REDACTED***"
models = ["model-id"]

[models.routes.default]
provider_id = "model-id"
model = "model-id"
context_window = 4096
max_output = 512
temperature = 0
timeout = 60
"""

MULTILINE_DOTTED_API_KEY_CONFIG = '''models.providers.primary.protocol = "anthropic"
models.providers.primary.base_url = "https://api.anthropic.com"
models.providers.primary.api_key = """line-one-secret
line-two-secret"""
models.providers.primary.models = ["model-id"]
models.routes.default.provider_id = "primary"
models.routes.default.model = "model-id"
models.routes.default.context_window = 4096
models.routes.default.max_output = 512
models.routes.default.temperature = 0
models.routes.default.timeout = 60
'''

EXPECTED_REDACTED_MULTILINE_DOTTED_CONFIG = """models.providers.primary.protocol = "anthropic"
models.providers.primary.base_url = "https://api.anthropic.com"
models.providers.primary.api_key = "***REDACTED***"
models.providers.primary.models = ["model-id"]
models.routes.default.provider_id = "primary"
models.routes.default.model = "model-id"
models.routes.default.context_window = 4096
models.routes.default.max_output = 512
models.routes.default.temperature = 0
models.routes.default.timeout = 60
"""

MALFORMED_CONFIG = """[runtime
max_tool_result_chars = 50000
api_key = "first-plaintext-key"
  API-Key = 'second-plaintext-key' # remove this whole value
not_api_key = "not-a-provider-key"
broken = [
"""

EXPECTED_REDACTED_MALFORMED_CONFIG = """[runtime
max_tool_result_chars = 50000
api_key = "***REDACTED***"
  API-Key = "***REDACTED***"
not_api_key = "not-a-provider-key"
broken = [
"""

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


def test_missing_configuration_is_created_exactly_once(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))

    assert loader.ensure_default() is True
    assert (agent_home / "config.toml").read_text(encoding="utf-8") == EXPECTED_DEFAULT_CONFIG

    existing = b"# Keep this existing configuration byte-for-byte.\n"
    (agent_home / "config.toml").write_bytes(existing)

    assert loader.ensure_default() is False
    assert (agent_home / "config.toml").read_bytes() == existing


def test_generated_configuration_scaffolds_one_provider_and_all_model_routes(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))

    assert loader.ensure_default() is True

    models = loader.load().models
    assert set(models.providers) == {"openai-local"}
    provider = models.providers["openai-local"]
    assert (provider.protocol, provider.base_url, provider.api_key, provider.models) == (
        "openai-compatible",
        "",
        "",
        (),
    )

    routes = models.routes
    assert set(routes) == {"default", "chat", "memory", "schedule"}
    for route in routes.values():
        assert (
            route.provider_id,
            route.model,
            route.context_window,
            route.max_output,
            route.temperature,
            route.reasoning_effort,
            route.timeout,
        ) == (
            "openai-local",
            "replace-with-a-model-id",
            200000,
            8192,
            0.2,
            "medium",
            120,
        )


def test_configuration_projects_undefined_fields_and_legacy_routes(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(PROJECTED_CONFIG, encoding="utf-8")

    configuration = loader.load_for_startup()

    assert set(configuration.models.routes) == {"default"}
    assert set(configuration.models.providers) == {"primary"}
    assert configuration.runtime.max_tool_result_chars == 60000


def test_startup_gate_does_not_resolve_provider_or_model_usability(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content = VALID_CONFIG.replace('protocol = "anthropic"', 'protocol = "future-protocol"')
    loader.path.write_text(content, encoding="utf-8")

    configuration = loader.load_for_startup()

    assert configuration.models.routes["default"].provider_id == "anthropic-default"


def test_schedule_route_resolves_directly_and_falls_back_to_default_when_unusable(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(VALID_CONFIG + SCHEDULE_ROUTE, encoding="utf-8")
    configuration = loader.load()

    resolved = configuration.resolve_route("schedule")

    assert (
        resolved.requested_route,
        resolved.selected_route,
        resolved.provider.provider_id,
        resolved.route.model,
        resolved.used_default,
    ) == ("schedule", "schedule", "schedule-provider", "schedule-model", False)

    loader.path.write_text(
        (VALID_CONFIG + SCHEDULE_ROUTE).replace(
            'protocol = "openai-compatible"', 'protocol = "future-protocol"', 1
        ),
        encoding="utf-8",
    )
    configuration = loader.load()

    fallback = configuration.resolve_route("schedule")

    assert (
        fallback.requested_route,
        fallback.selected_route,
        fallback.provider.provider_id,
        fallback.route.model,
        fallback.used_default,
    ) == ("schedule", "default", "anthropic-default", "claude-model", True)


def test_failed_startup_generation_leaves_no_partial_configuration(
    agent_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))

    def fail_publish(source: Path | str, destination: Path | str) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(os, "link", fail_publish)

    with pytest.raises(ConfigError) as raised:
        loader.load_for_startup()

    assert raised.value.error.code == "persistence_error"
    assert not loader.path.exists()
    assert not tuple(agent_home.glob(".config.toml.*.tmp"))


def test_valid_configuration_loads_as_typed_values(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(VALID_CONFIG, encoding="utf-8")

    configuration = loader.load()

    assert (
        configuration.runtime.max_tool_result_chars,
        configuration.memory.consolidation_message_threshold,
        configuration.memory.batch_size,
        configuration.memory.schedule,
        configuration.models.providers["anthropic-default"].models,
        configuration.models.routes["default"].reasoning_effort,
        configuration.models.routes["default"].timeout,
    ) == (60000, 50, 12, "15 * * * *", ("claude-model",), "medium", 120)


def test_omitted_defaulted_configuration_fields_use_accepted_defaults(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")

    configuration = loader.load()

    assert (
        configuration.runtime.max_tool_result_chars,
        configuration.memory.consolidation_message_threshold,
        configuration.memory.batch_size,
        configuration.memory.schedule,
        configuration.models.routes["default"].reasoning_effort,
    ) == (4096, 40, 10, "0 * * * *", None)


def test_config_view_redacts_nonempty_provider_keys_and_preserves_complete_content(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(REDACTION_CONFIG, encoding="utf-8")

    view = loader.view()

    assert (view.path, view.redacted_content, view.error) == (
        loader.path,
        EXPECTED_REDACTED_CONFIG,
        None,
    )


def test_config_view_redacts_only_the_api_key_when_its_text_is_reused(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(OVERLAPPING_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.error is None
    assert view.redacted_content == EXPECTED_REDACTED_OVERLAPPING_CONFIG


def test_config_view_redacts_multiline_api_key_in_valid_dotted_toml(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MULTILINE_DOTTED_API_KEY_CONFIG, encoding="utf-8")

    view = loader.view()

    if "line-one-secret" in view.redacted_content or "line-two-secret" in view.redacted_content:
        pytest.fail("ConfigView leaked a multiline plaintext provider API key", pytrace=False)
    assert view.error is None
    assert view.redacted_content == EXPECTED_REDACTED_MULTILINE_DOTTED_CONFIG


@pytest.mark.parametrize(
    ("content", "secrets"),
    (
        pytest.param(
            SCHEMA_INVALID_API_KEY_ALIAS_CONFIG,
            ("sk-schema-alias-secret",),
            id="schema-alias",
        ),
        pytest.param(
            '[diagnostics]\nAPI_Key = "sk-nested-secret"\nmessage = "keep"\n',
            ("sk-nested-secret",),
            id="unknown-table",
        ),
        pytest.param(
            '[diagnostics]\napi_key = ["sk-array-secret"]\nmessage = "keep"\n',
            ("sk-array-secret",),
            id="non-string-value",
        ),
        pytest.param(
            '[[diagnostics]]\napi_key = "sk-array-table-secret"\nmessage = "keep"\n',
            ("sk-array-table-secret",),
            id="array-table",
        ),
        pytest.param(
            r""""api\u005fkey" = "sk-valid-escaped-secret"
""",
            ("sk-valid-escaped-secret",),
            id="escaped-key",
        ),
    ),
)
def test_config_view_redacts_structured_api_key_variants(
    agent_home: Path,
    content: str,
    secrets: tuple[str, ...],
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_invalid"
    assert all(secret not in view.redacted_content for secret in secrets)
    assert "***REDACTED***" in view.redacted_content


@pytest.mark.parametrize(
    ("content", "secrets"),
    (
        pytest.param(
            MALFORMED_DOTTED_API_KEY_CONFIG,
            ("sk-dotted-secret",),
            id="dotted",
        ),
        pytest.param(
            MALFORMED_QUOTED_DOTTED_API_KEY_CONFIG,
            ("sk-quoted-secret",),
            id="quoted-dotted",
        ),
        pytest.param(
            r""""api\u005fkey" = "sk-escaped-key-secret"
broken = [
""",
            ("sk-escaped-key-secret",),
            id="escaped-key",
        ),
        pytest.param(
            r""""\u0041\U00000050\u0049\u002d\U0000004b\u0045\U00000059" = "sk-fully-escaped-secret"
broken = [
""",
            ("sk-fully-escaped-secret",),
            id="fully-escaped-key",
        ),
        pytest.param(
            MALFORMED_INLINE_API_KEY_CONFIG,
            ("sk-inline-secret",),
            id="inline",
        ),
        pytest.param(
            MALFORMED_INLINE_MULTILINE_API_KEY_CONFIG,
            ("sk-inline-line-one", "sk-inline-line-two"),
            id="inline-multiline",
        ),
        pytest.param(
            MALFORMED_INLINE_ARRAY_API_KEY_CONFIG,
            ("sk-inline-array-secret",),
            id="inline-array",
        ),
        pytest.param(
            MALFORMED_MULTILINE_ARRAY_API_KEY_CONFIG,
            ("sk-multiline-array-secret",),
            id="multiline-array",
        ),
        pytest.param(
            MALFORMED_MULTILINE_API_KEY_CONFIG,
            ("sk-line-one", "sk-line-two"),
            id="multiline-string",
        ),
    ),
)
def test_config_view_redacts_malformed_api_key_variants(
    agent_home: Path,
    content: str,
    secrets: tuple[str, ...],
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert all(secret not in view.redacted_content for secret in secrets)
    assert "***REDACTED***" in view.redacted_content


def test_config_view_returns_safe_parse_error_and_conservatively_redacted_raw_text(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(MALFORMED_CONFIG, encoding="utf-8")

    view = loader.view()

    assert view.path == loader.path
    assert view.redacted_content == EXPECTED_REDACTED_MALFORMED_CONFIG
    assert view.error is not None
    assert view.error.code == "config_parse_error"
    assert "first-plaintext-key" not in view.error.message
    assert "second-plaintext-key" not in view.error.message


def test_config_view_keeps_undefined_configuration_inspectable(agent_home: Path) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content = REDACTION_CONFIG.replace(
        "max_tool_result_chars = 50000",
        "max_tool_result_chars = 50000\nmisspelled_setting = true",
    )
    loader.path.write_text(content, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_invalid"
    assert "runtime.misspelled_setting" in view.error.message
    assert "misspelled_setting = true" in view.redacted_content
    assert "plaintext-primary-key" not in view.redacted_content


@pytest.mark.parametrize(
    ("content", "field"),
    (
        (VALID_CONFIG + "\n[unexpected]\nvalue = true\n", "unexpected"),
        (
            VALID_CONFIG.replace(
                "max_tool_result_chars = 60000",
                "max_tool_result_chars = 60000\nunknown = true",
            ),
            "runtime.unknown",
        ),
        (
            VALID_CONFIG.replace("batch_size = 12", "batch_size = 12\nunknown = true"),
            "memory.unknown",
        ),
        (VALID_CONFIG + "\n[tools.web]\nenabled = true\n", "tools"),
        (
            VALID_CONFIG.replace(
                "[models.providers.anthropic-default]",
                "[models]\nunknown = true\n\n[models.providers.anthropic-default]",
            ),
            "models.unknown",
        ),
        (
            VALID_CONFIG.replace(
                'models = ["claude-model"]',
                'models = ["claude-model"]\nunknown = true',
            ),
            "models.providers.anthropic-default.unknown",
        ),
        (
            VALID_CONFIG.replace("timeout = 120", "timeout = 120\nunknown = true"),
            "models.routes.default.unknown",
        ),
        (
            VALID_CONFIG + "\n[models.routes.cron]\nlegacy = true\n",
            "models.routes.cron",
        ),
        (
            VALID_CONFIG + "\n[models.routes.future]\nfuture = true\n",
            "models.routes.future",
        ),
    ),
)
def test_config_view_reports_undefined_fields(
    agent_home: Path,
    content: str,
    field: str,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")

    view = loader.view()

    assert view.error is not None
    assert view.error.code == "config_invalid"
    assert field in view.error.message
    assert "sk-ant-secret" not in view.redacted_content


@pytest.mark.parametrize(
    ("content", "field"),
    [
        (
            VALID_CONFIG.replace("max_tool_result_chars = 60000", "max_tool_result_chars = true"),
            "runtime.max_tool_result_chars",
        ),
        (
            VALID_CONFIG.replace("max_tool_result_chars = 60000", "max_tool_result_chars = 999"),
            "runtime.max_tool_result_chars",
        ),
        (
            VALID_CONFIG.replace(
                "consolidation_message_threshold = 50",
                "consolidation_message_threshold = 3",
            ),
            "memory.consolidation_message_threshold",
        ),
        (VALID_CONFIG.replace("batch_size = 12", "batch_size = 1001"), "memory.batch_size"),
        (
            VALID_CONFIG.replace('schedule = "15 * * * *"', 'schedule = "99 99 99 99 99"'),
            "memory.schedule",
        ),
        (
            VALID_CONFIG.replace('protocol = "anthropic"', "protocol = 1"),
            "models.providers.anthropic-default.protocol",
        ),
        (
            VALID_CONFIG.replace('base_url = "https://api.anthropic.com"', "base_url = 1"),
            "models.providers.anthropic-default.base_url",
        ),
        (
            VALID_CONFIG.replace('base_url = "https://api.anthropic.com"\n', ""),
            "models.providers.anthropic-default.base_url",
        ),
        (
            VALID_CONFIG.replace('api_key = "sk-ant-secret"', "api_key = 1"),
            "models.providers.anthropic-default.api_key",
        ),
        (
            VALID_CONFIG.replace(
                'models = ["claude-model"]', 'models = ["claude-model", "claude-model"]'
            ),
            "models.providers.anthropic-default.models",
        ),
        (
            VALID_CONFIG.replace('models = ["claude-model"]', 'models = [""]'),
            "models.providers.anthropic-default.models",
        ),
        (
            VALID_CONFIG.replace('models = ["claude-model"]', "models = [1]"),
            "models.providers.anthropic-default.models",
        ),
        (
            VALID_CONFIG.replace('provider_id = "anthropic-default"\n', ""),
            "models.routes.default.provider_id",
        ),
        (
            VALID_CONFIG.replace('provider_id = "anthropic-default"', 'provider_id = "Bad_ID"'),
            "models.routes.default.provider_id",
        ),
        (
            VALID_CONFIG.replace('model = "claude-model"\n', ""),
            "models.routes.default.model",
        ),
        (
            VALID_CONFIG.replace('model = "claude-model"', 'model = ""'),
            "models.routes.default.model",
        ),
        (
            VALID_CONFIG.replace("context_window = 200000", "context_window = 1023"),
            "models.routes.default.context_window",
        ),
        (
            VALID_CONFIG.replace("context_window = 200000", "context_window = true"),
            "models.routes.default.context_window",
        ),
        (
            VALID_CONFIG.replace("max_output = 8192", "max_output = 0"),
            "models.routes.default.max_output",
        ),
        (
            VALID_CONFIG.replace("max_output = 8192", "max_output = 200000"),
            "models.routes.default.max_output",
        ),
        (
            VALID_CONFIG.replace("temperature = 0.2", "temperature = nan"),
            "models.routes.default.temperature",
        ),
        (
            VALID_CONFIG.replace("temperature = 0.2", 'temperature = "0.2"'),
            "models.routes.default.temperature",
        ),
        (
            VALID_CONFIG.replace('reasoning_effort = "medium"', 'reasoning_effort = "extreme"'),
            "models.routes.default.reasoning_effort",
        ),
        (
            VALID_CONFIG.replace("timeout = 120", "timeout = 601"),
            "models.routes.default.timeout",
        ),
    ],
)
def test_configuration_rejects_schema_violations(
    agent_home: Path, content: str, field: str
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        loader.load()

    assert raised.value.error.code == "config_invalid"
    assert field in raised.value.error.message
    assert "sk-ant-secret" not in str(raised.value)


UNUSABLE_CHAT_PROVIDER = """

[models.providers.chat-provider]
protocol = "anthropic"
base_url = "not-an-absolute-url"
api_key = "chat-secret"
models = ["chat-model"]

[models.routes.chat]
provider_id = "chat-provider"
model = "chat-model"
context_window = 100000
max_output = 4096
temperature = 0.1
timeout = 90
"""

UNKNOWN_PROTOCOL_CHAT_PROVIDER = UNUSABLE_CHAT_PROVIDER.replace(
    'protocol = "anthropic"', 'protocol = "future-protocol"'
).replace('base_url = "not-an-absolute-url"', 'base_url = "https://future.example/v1"')

MISSING_PROVIDER_CHAT_ROUTE = """

[models.routes.chat]
provider_id = "missing-provider"
model = "chat-model"
context_window = 100000
max_output = 4096
temperature = 0.1
timeout = 90
"""

OUTSIDE_CATALOG_CHAT_ROUTE = MISSING_PROVIDER_CHAT_ROUTE.replace(
    'provider_id = "missing-provider"', 'provider_id = "anthropic-default"'
)


@pytest.mark.parametrize(
    ("content", "requested_route", "expected_selected_route"),
    [
        (VALID_CONFIG, "chat", "default"),
        (VALID_CONFIG + MISSING_PROVIDER_CHAT_ROUTE, "chat", "default"),
        (VALID_CONFIG + OUTSIDE_CATALOG_CHAT_ROUTE, "chat", "default"),
        (VALID_CONFIG + UNUSABLE_CHAT_PROVIDER, "chat", "default"),
        (VALID_CONFIG + UNKNOWN_PROTOCOL_CHAT_PROVIDER, "chat", "default"),
        (EXPECTED_DEFAULT_CONFIG, "default", None),
        (VALID_CONFIG.replace('protocol = "anthropic"', 'protocol = "future"'), "default", None),
        (
            VALID_CONFIG.replace(
                'base_url = "https://api.anthropic.com"', 'base_url = "not-an-absolute-url"'
            ),
            "default",
            None,
        ),
        (
            VALID_CONFIG.replace('base_url = "https://api.anthropic.com"', 'base_url = ""'),
            "default",
            None,
        ),
        (VALID_CONFIG.replace('api_key = "sk-ant-secret"', 'api_key = ""'), "default", None),
        (VALID_CONFIG.replace('models = ["claude-model"]', "models = []"), "default", None),
        (
            VALID_CONFIG.replace(
                'provider_id = "anthropic-default"', 'provider_id = "missing-provider"'
            ),
            "default",
            None,
        ),
        (
            VALID_CONFIG.replace('model = "claude-model"', 'model = "outside-catalog"'),
            "default",
            None,
        ),
    ],
)
def test_model_route_resolution_uses_only_a_usable_default(
    agent_home: Path,
    content: str,
    requested_route: str,
    expected_selected_route: str | None,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")
    configuration = loader.load()

    if expected_selected_route is None:
        with pytest.raises(ConfigError) as raised:
            configuration.resolve_route(requested_route)
        assert raised.value.error.code == "route_unavailable"
        assert "sk-ant-secret" not in str(raised.value)
    else:
        resolved = configuration.resolve_route(requested_route)
        assert (
            resolved.requested_route,
            resolved.selected_route,
            resolved.provider.provider_id,
            resolved.route.model,
            resolved.used_default,
        ) == (requested_route, "default", "anthropic-default", "claude-model", True)


def test_missing_default_model_route_names_the_required_configuration_table(
    agent_home: Path,
) -> None:
    loader = ConfigLoader(AgentHome(agent_home))
    loader.ensure_default()
    content_without_routes = VALID_CONFIG.partition("\n[models.routes.default]")[0] + "\n"
    loader.path.write_text(content_without_routes, encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        loader.load_for_startup()

    assert raised.value.error.code == "route_unavailable"
    assert raised.value.error.message == (
        "Default Model Route is missing. Add [models.routes.default] to User Configuration."
    )
