import subprocess
from pathlib import Path

import pytest

from myclaw.utils.host_filesystem import WINDOWS_HOST_FILESYSTEM


def test_windows_host_filesystem_prepares_local_and_unc_io_paths(tmp_path: Path) -> None:
    local = tmp_path / "state.txt"
    unc = Path(r"\\server\share\state.txt")

    assert WINDOWS_HOST_FILESYSTEM.path_for_io(local) == Path(f"\\\\?\\{local.absolute()}")
    assert WINDOWS_HOST_FILESYSTEM.path_for_io(unc) == Path(r"\\?\UNC\server\share\state.txt")


def test_windows_host_filesystem_accepts_an_owned_directory(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    child = owned / "child"
    child.mkdir(parents=True)

    assert WINDOWS_HOST_FILESYSTEM.require_owned_directory(child, within=owned) == child.resolve(
        strict=True
    )


def test_windows_host_filesystem_rejects_redirected_or_external_directory(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = owned / "junction"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        check=True,
        text=True,
    )

    for candidate in (junction, outside):
        with pytest.raises(PermissionError):
            WINDOWS_HOST_FILESYSTEM.require_owned_directory(candidate, within=owned)


def test_windows_host_filesystem_accepts_an_owned_regular_file(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    state = owned / "state.json"
    state.write_bytes(b"{}")

    assert WINDOWS_HOST_FILESYSTEM.require_owned_regular_file(state, within=owned) == state.resolve(
        strict=True
    )


def test_host_filesystem_create_only_publication_preserves_existing_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state.txt"

    assert WINDOWS_HOST_FILESYSTEM.atomic_create_text(target, "first\n") is True
    assert WINDOWS_HOST_FILESYSTEM.atomic_create_text(target, "replacement\n") is False
    assert target.read_bytes() == b"first\n"
    assert tuple(tmp_path.iterdir()) == (target,)


def test_host_filesystem_atomic_replace_publishes_exact_utf8_content(tmp_path: Path) -> None:
    target = tmp_path / "state.txt"
    target.write_bytes(b"old")

    WINDOWS_HOST_FILESYSTEM.atomic_replace_text(
        target, "User: \u5f20\u4e09\nPreference: caf\u00e9\n"
    )

    assert target.read_bytes() == (b"User: \xe5\xbc\xa0\xe4\xb8\x89\nPreference: caf\xc3\xa9\n")


def test_windows_host_filesystem_rejects_an_open_file_with_mismatched_path(
    tmp_path: Path,
) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    opened_path = owned / "opened.log"
    opened_path.write_bytes(b"opened")
    current_path = owned / "current.log"
    current_path.write_bytes(b"current")

    with opened_path.open("rb", buffering=0) as stream:
        with pytest.raises(PermissionError):
            WINDOWS_HOST_FILESYSTEM.require_opened_owned_regular_file(
                stream.fileno(), current_path, within=owned
            )
