import asyncio
import os
from pathlib import Path

import pytest

from myclaw.atomic_files import atomic_replace_bytes, atomic_replace_text


def test_failed_atomic_bytes_replace_preserves_official_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old-state")

    def fail_replace(source: Path | str, destination: Path | str) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        atomic_replace_bytes(target, b"new-state")

    assert (target.read_bytes(), sorted(path.name for path in tmp_path.iterdir())) == (
        b"old-state",
        ["state.bin"],
    )


def test_atomic_text_replace_writes_exact_utf8_bytes(tmp_path: Path) -> None:
    target = tmp_path / "state.txt"

    atomic_replace_text(target, "User: \u5f20\u4e09\nPreference: caf\u00e9\n")

    assert target.read_bytes() == (b"User: \xe5\xbc\xa0\xe4\xb8\x89\nPreference: caf\xc3\xa9\n")


def test_cancelled_atomic_bytes_replace_preserves_official_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old-state")

    def cancel_fsync(descriptor: int) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(os, "fsync", cancel_fsync)

    with pytest.raises(asyncio.CancelledError):
        atomic_replace_bytes(target, b"new-state")

    assert (target.read_bytes(), sorted(path.name for path in tmp_path.iterdir())) == (
        b"old-state",
        ["state.bin"],
    )
