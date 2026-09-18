from pathlib import Path

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader

VALID_CONFIG = """[runtime]
max_tool_result_chars = 4096
max_iterations = 50
enable_skill_always_load = false

[memory]
compaction_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[models.providers.primary]
protocol = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "secret"
models = ["model"]

[models.routes.default]
provider_id = "primary"
model = "model"
context_window = 100000
max_output = 2048
temperature = 0.2
timeout = 30
"""


def _loader(tmp_path: Path, content: str = VALID_CONFIG) -> ConfigLoader:
    loader = ConfigLoader(AgentHome(tmp_path / "agent-home"))
    loader.ensure_default()
    loader.path.write_text(content, encoding="utf-8")
    return loader


@pytest.mark.parametrize("value", (0.5, 0.75, 0.9, 0.95))
def test_compact_ratio_accepts_supported_values_without_diagnostic(
    tmp_path: Path,
    value: float,
) -> None:
    loader = _loader(
        tmp_path, VALID_CONFIG.replace("[runtime]\n", f"[runtime]\ncompact_ratio = {value}\n")
    )

    configuration = loader.load()

    assert configuration.runtime.compact_ratio == value
    assert loader.diagnostics == ()


@pytest.mark.parametrize(
    "value",
    ("true", '"0.75"', "nan", "inf", "0.49", "0.96"),
)
def test_invalid_compact_ratio_falls_back_once_with_safe_diagnostic(
    tmp_path: Path,
    value: str,
) -> None:
    loader = _loader(
        tmp_path, VALID_CONFIG.replace("[runtime]\n", f"[runtime]\ncompact_ratio = {value}\n")
    )

    configuration = loader.load()

    assert configuration.runtime.compact_ratio == 0.9
    assert len(loader.diagnostics) == 1
    assert "runtime.compact_ratio" in loader.diagnostics[0].message
    assert "0.9" in loader.diagnostics[0].message


def test_missing_compact_ratio_uses_default_and_one_diagnostic(tmp_path: Path) -> None:
    loader = _loader(tmp_path, VALID_CONFIG.replace("compact_ratio = 0.9\n", ""))

    configuration = loader.load()

    assert configuration.runtime.compact_ratio == 0.9
    assert len(loader.diagnostics) == 1


def test_config_view_exposes_effective_ratio_before_raw_content(tmp_path: Path) -> None:
    loader = _loader(
        tmp_path, VALID_CONFIG.replace("[runtime]\n", "[runtime]\ncompact_ratio = 0.75\n")
    )

    view = loader.view()

    assert view.error is None
    assert view.effective_compact_ratio == 0.75
    assert view.redacted_content.index("compact_ratio = 0.75") >= 0


def test_malformed_config_has_no_effective_ratio(tmp_path: Path) -> None:
    loader = _loader(tmp_path, "[runtime\ncompact_ratio = 0.75\n")

    view = loader.view()

    assert view.error is not None
    assert view.effective_compact_ratio is None


def test_removed_threshold_is_not_a_memory_configuration_field(tmp_path: Path) -> None:
    loader = _loader(tmp_path)

    configuration = loader.load()

    assert not hasattr(configuration.memory, "compaction_message_threshold")


def test_runtime_configuration_still_rejects_other_known_invalid_fields(tmp_path: Path) -> None:
    loader = _loader(tmp_path, VALID_CONFIG.replace("max_iterations = 50", "max_iterations = 49"))

    with pytest.raises(ConfigError):
        loader.load()
