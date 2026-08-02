from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from loguru import logger

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.logging.session import session_log
from myclaw.session.identifiers import make_session_id


def _session_id() -> str:
    return make_session_id(
        __import__("datetime").datetime(2026, 8, 2, 12, 0, tzinfo=__import__("datetime").timezone.utc),
        uuid4(),
    )


def _state(tmp_path: Path) -> WorkspaceState:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return WorkspaceState(Workspace.from_path(workspace))


def test_session_log_directory_is_lazy_and_filters_records(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    assert state.logs_directory == state.path / "logs"
    assert not state.logs_directory.exists()

    with session_log(state, session_id):
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")

    active = state.logs_directory / f"{session_id}.log"
    content = active.read_text(encoding="utf-8")
    assert "warning" in content
    assert "error" in content
    assert "debug" not in content
    assert "info" not in content


def test_session_log_contexts_do_not_cross_write(tmp_path: Path) -> None:
    state = _state(tmp_path)
    first, second = _session_id(), _session_id()

    with session_log(state, first):
        logger.warning("first")
    with session_log(state, second):
        logger.warning("second")

    assert "first" in (state.logs_directory / f"{first}.log").read_text(encoding="utf-8")
    assert "second" not in (state.logs_directory / f"{first}.log").read_text(encoding="utf-8")
    assert "second" in (state.logs_directory / f"{second}.log").read_text(encoding="utf-8")


def test_session_log_does_not_create_a_file_without_a_warning(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    with session_log(state, session_id):
        logger.info("ordinary progress")

    assert state.logs_directory.exists()
    assert not (state.logs_directory / f"{session_id}.log").exists()


def test_session_log_retains_an_ordinary_exception_traceback(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    with session_log(state, session_id):
        try:
            raise RuntimeError("technical detail")
        except RuntimeError as error:
            logger.opt(exception=error).error("operation failed")

    content = (state.logs_directory / f"{session_id}.log").read_text(encoding="utf-8")
    assert "operation failed" in content
    assert "Traceback (most recent call last)" in content
    assert "RuntimeError: technical detail" in content


def test_session_log_rejects_invalid_session_id_before_creating_state(tmp_path: Path) -> None:
    state = _state(tmp_path)

    with pytest.raises(ValueError, match="valid Session ID"):
        with session_log(state, "invalid"):
            pass

    assert not state.logs_directory.exists()


def test_session_log_directory_failure_isolated_and_retried(
    tmp_path: Path, monkeypatch: Any
) -> None:
    state = _state(tmp_path)
    session_id = _session_id()
    original = Path.mkdir
    calls = 0

    def fail_once(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal calls
        if path == state.logs_directory and calls == 0:
            calls += 1
            raise OSError("blocked")
        return original(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", fail_once)
    with session_log(state, session_id):
        logger.warning("first")
    assert not state.logs_directory.exists()

    with session_log(state, session_id):
        logger.warning("second")
    assert "second" in (state.logs_directory / f"{session_id}.log").read_text(encoding="utf-8")


def test_session_log_rotates_and_retains_at_most_one_history_file(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    with session_log(state, session_id):
        for index in range(3):
            logger.warning("%s", "x" * 5_000_000 + str(index))

    files = list(state.logs_directory.glob(f"{session_id}*.log"))
    assert (state.logs_directory / f"{session_id}.log").exists()
    assert len(files) <= 2
