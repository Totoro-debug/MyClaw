from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from myclaw.workspace import Workspace


def test_windows_drive_workspace_has_the_accepted_identity_and_slug() -> None:
    workspace = Workspace.from_path(PureWindowsPath(r"D:\desktop\project\Demo-one"))

    assert (workspace.path, workspace.slug) == (
        PureWindowsPath(r"D:\desktop\project\Demo-one"),
        "d-desktop-project-demo_one",
    )


def test_posix_workspace_omits_the_root_from_its_slug() -> None:
    workspace = Workspace.from_path(PurePosixPath("/home/Alice/Demo-one"))

    assert (workspace.path, workspace.slug) == (
        PurePosixPath("/home/Alice/Demo-one"),
        "home-alice-demo_one",
    )


def test_unc_workspace_has_the_accepted_identity_and_slug() -> None:
    workspace = Workspace.from_path(PureWindowsPath(r"\\server\share\Demo-one"))

    assert (workspace.path, workspace.slug) == (
        PureWindowsPath(r"\\server\share\Demo-one"),
        "unc-server-share-demo_one",
    )


def test_native_workspace_is_absolutized_and_lexically_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "Project"
    project.mkdir()
    monkeypatch.chdir(tmp_path)

    workspace = Workspace.from_path(Path("Project") / "discarded" / "..")

    assert (workspace.path, isinstance(workspace.path, Path)) == (project, True)


def test_workspace_slug_uses_unicode_lowercase() -> None:
    workspace = Workspace.from_path(PurePosixPath("/\u00c4LICE/PRO-J\u00c9CT/\u0130"))

    assert workspace.slug == "\u00e4lice-pro_j\u00e9ct-i\u0307"
