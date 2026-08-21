from pathlib import Path

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader, RuntimeConfiguration


def _config(max_iterations: str | None = None) -> str:
    runtime = "max_tool_result_chars = 4096"
    if max_iterations is not None:
        runtime += f"\nmax_iterations = {max_iterations}"
    return f"""[runtime]
{runtime}

[memory]
consolidation_message_threshold = 40
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


def _load(tmp_path: Path, max_iterations: str | None = None) -> int:
    loader = ConfigLoader(AgentHome(tmp_path))
    loader.ensure_default()
    loader.path.write_text(_config(max_iterations), encoding="utf-8")
    return loader.load().runtime.max_iterations


def test_max_iterations_defaults_to_fifty_when_omitted(tmp_path: Path) -> None:
    assert _load(tmp_path) == 50


def test_runtime_configuration_direct_construction_keeps_fifty_default() -> None:
    assert RuntimeConfiguration(max_tool_result_chars=4096).max_iterations == 50


@pytest.mark.parametrize("value", ("50", "51", "1000000000"))
def test_max_iterations_accepts_fifty_and_larger_integers(tmp_path: Path, value: str) -> None:
    assert _load(tmp_path, value) == int(value)


def test_config_view_recognizes_max_iterations_as_a_defined_runtime_field(tmp_path: Path) -> None:
    loader = ConfigLoader(AgentHome(tmp_path))
    loader.ensure_default()
    loader.path.write_text(_config("50"), encoding="utf-8")

    assert loader.view().error is None


@pytest.mark.parametrize("value", ("49", "0", "-1", "true", "50.0", '"50"'))
def test_max_iterations_rejects_values_below_fifty_and_non_integers(
    tmp_path: Path, value: str
) -> None:
    loader = ConfigLoader(AgentHome(tmp_path))
    loader.ensure_default()
    loader.path.write_text(_config(value), encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        loader.load()

    assert raised.value.error.code == "config_invalid"
    assert "runtime.max_iterations" in raised.value.error.message
