import os
from pathlib import Path, PurePath, PureWindowsPath

import pytest

from myclaw.agent.workspace import Workspace

windows_only = pytest.mark.skipif(os.name != "nt", reason="requires native Windows paths")


@windows_only
def test_windows_drive_workspace_has_the_accepted_identity() -> None:
    workspace = Workspace.from_path(PureWindowsPath(r"D:\desktop\project\Demo-one"))

    assert workspace.path == Path(r"D:\desktop\project\Demo-one")


@windows_only
def test_unc_workspace_has_the_accepted_identity() -> None:
    workspace = Workspace.from_path(PureWindowsPath(r"\\server\share\Demo-one"))

    assert workspace.path == Path(r"\\server\share\Demo-one")


def test_native_workspace_is_absolutized_and_lexically_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "Project"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    workspace = Workspace.from_path(Path("Project") / "discarded" / "..")

    assert workspace.path == project.absolute()


def test_native_pure_workspace_path_is_lexically_normalized(tmp_path: Path) -> None:
    workspace = Workspace.from_path(PurePath(tmp_path / "discarded" / ".."))

    assert workspace.path == tmp_path


def test_workspace_identity_uses_the_current_hosts_native_path_type(
    tmp_path: Path,
) -> None:
    workspace = Workspace.from_path(tmp_path)

    assert workspace.path == tmp_path.absolute()
    assert type(workspace.path) is type(Path())


def test_workspace_does_not_expose_a_legacy_slug(tmp_path: Path) -> None:
    workspace = Workspace.from_path(tmp_path)

    assert not hasattr(workspace, "slug")


@windows_only
def test_windows_workspace_is_lexically_normalized() -> None:
    workspace = Workspace.from_path(PureWindowsPath(r"D:\desktop\project\discarded\..\current"))

    assert workspace.path == Path(r"D:\desktop\project\current")


@windows_only
def test_relative_pure_windows_workspace_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute"):
        Workspace.from_path(PureWindowsPath(r"project\subdirectory"))
