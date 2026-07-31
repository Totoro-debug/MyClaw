import configparser
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_metadata_targets_only_windows_x64() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wheel = configparser.ConfigParser()
    wheel.read(ROOT / "setup.cfg", encoding="utf-8")

    assert project["scripts"]["myclaw"] == "myclaw.terminal.entrypoint:cli_entrypoint"
    assert "Operating System :: Microsoft :: Windows" in project["classifiers"]
    assert wheel["bdist_wheel"]["plat_name"] == "win_amd64"


def test_release_workflow_builds_and_smokes_one_windows_x64_wheel() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-validation.yml").read_text(
        encoding="utf-8"
    )

    assert "runs-on: windows-latest" in workflow
    assert "python -m build --wheel" in workflow
    assert "py3-none-win_amd64" in workflow
    assert "Expected exactly one release artifact" in workflow
    assert "PYTHONNOUSERSITE" in workflow
    assert "Remove-Item Env:PYTHONPATH" in workflow
    assert "python -m build\n" not in workflow
    assert "ubuntu-latest" not in workflow


def test_active_code_and_support_surfaces_have_no_retired_platform_contract() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "myclaw").rglob("*.py"))
    )
    for residue in (
        "os.name",
        "sys.platform",
        "fcntl",
        "start_new_session",
        "killpg",
        "PurePosixPath",
        "S_ISDIR",
        "S_ISREG",
        "S_ISLNK",
    ):
        assert residue not in production

    support_files = (
        ROOT / "README.md",
        ROOT / "docs" / "myclaw-runtime-contracts.md",
        ROOT / "docs" / "release-readiness.md",
        ROOT / "docs" / "release" / "windows-validation.md",
        ROOT / "docs" / "security-fault-review.md",
    )
    support = "\n".join(path.read_text(encoding="utf-8").lower() for path in support_files)
    for retired_claim in (
        "posix",
        "ubuntu-latest",
        "py3-none-any",
        "mypy --platform linux",
        "cross-platform",
    ):
        assert retired_claim not in support
    assert not (ROOT / "docs" / "release" / "posix-validation.md").exists()
