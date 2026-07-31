from pathlib import Path, PureWindowsPath

import pytest

from myclaw.agent.workspace import Workspace


def test_windows_drive_workspace_has_the_accepted_identity() -> None:
    workspace = Workspace.from_path(PureWindowsPath(r"D:\desktop\project\Demo-one"))

    assert workspace.path == PureWindowsPath(r"D:\desktop\project\Demo-one")


def test_unc_workspace_has_the_accepted_identity() -> None:
    workspace = Workspace.from_path(PureWindowsPath(r"\\server\share\Demo-one"))

    assert workspace.path == PureWindowsPath(r"\\server\share\Demo-one")


def test_native_workspace_is_absolutized_and_lexically_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "Project"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    workspace = Workspace.from_path(Path("Project") / "discarded" / "..")

    assert workspace.path == PureWindowsPath(project)


def test_workspace_does_not_expose_a_legacy_slug() -> None:
    workspace = Workspace.from_path(PureWindowsPath(r"D:\desktop\project"))

    assert not hasattr(workspace, "slug")


def test_windows_workspace_is_lexically_normalized() -> None:
    workspace = Workspace.from_path(PureWindowsPath(r"D:\desktop\project\discarded\..\current"))

    assert workspace.path == PureWindowsPath(r"D:\desktop\project\current")


def test_relative_pure_windows_workspace_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute"):
        Workspace.from_path(PureWindowsPath(r"project\subdirectory"))
