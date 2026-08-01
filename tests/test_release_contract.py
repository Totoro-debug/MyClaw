import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_metadata_builds_one_host_neutral_wheel() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["scripts"]["myclaw"] == "myclaw.terminal.cli:app"
    assert "Operating System :: OS Independent" in project["classifiers"]
    assert "Operating System :: Microsoft :: Windows" not in project["classifiers"]
    setup_path = ROOT / "setup.cfg"
    setup = setup_path.read_text(encoding="utf-8") if setup_path.exists() else ""
    assert "plat_name" not in setup


def test_release_workflow_builds_and_smokes_one_universal_wheel_on_windows() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-validation.yml").read_text(
        encoding="utf-8"
    )

    assert "runs-on: windows-latest" in workflow
    assert "python -m build --wheel" in workflow
    assert "py3-none-any" in workflow
    assert "py3-none-win_amd64" not in workflow
    assert "Expected exactly one release artifact" in workflow
    assert "PYTHONNOUSERSITE" in workflow
    assert "Remove-Item Env:PYTHONPATH" in workflow
    assert "python -m build\n" not in workflow
    assert "ubuntu-latest" not in workflow


def test_active_code_has_no_platform_support_gate() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "myclaw").rglob("*.py"))
    )
    for residue in ("UnsupportedPlatformError", "unsupported_platform", "SUPPORTED_PLATFORM_TAG"):
        assert residue not in production
    assert not (ROOT / "myclaw" / "platform_support.py").exists()
    assert not (ROOT / "myclaw" / "terminal" / "entrypoint.py").exists()
