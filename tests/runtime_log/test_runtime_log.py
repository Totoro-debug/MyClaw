import errno
import logging
import os
import re
import threading
import time
import warnings
from pathlib import Path

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.runtime_log import install_runtime_logging


def test_runtime_log_lifetime_is_lazy_and_only_persists_myclaw_warnings(
    agent_home: Path,
) -> None:
    root = logging.getLogger()
    original_root_state = (root.level, tuple(root.handlers))

    lifetime = install_runtime_logging(AgentHome(agent_home))
    assert not (agent_home / "logs").exists()

    logging.getLogger("myclaw.agent.turn").info("ordinary progress")
    logging.getLogger("myclaw.agent.turn").warning("recoverable failure")
    logging.getLogger("third_party").error("SDK failure")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        warnings.warn("deprecated SDK", stacklevel=1)
    assert len(caught_warnings) == 1
    lifetime.close()

    logs = agent_home / "logs"
    assert (logs / "run.log.cursor").read_bytes() == b"0\n"
    content = (logs / "run.log.0").read_text(encoding="utf-8")
    assert "WARNING" in content
    assert "myclaw.agent.turn: recoverable failure" in content
    assert "ordinary progress" not in content
    assert "SDK failure" not in content
    assert "deprecated SDK" not in content
    assert (root.level, tuple(root.handlers)) == original_root_state


def test_runtime_log_formats_correlated_exception_records_as_safe_plain_text(
    agent_home: Path,
) -> None:
    diagnostic_path = (agent_home / "sessions" / "broken.jsonl").resolve()
    lifetime = install_runtime_logging(AgentHome(agent_home))

    with lifetime.session("session-123"):
        try:
            try:
                raise OSError(f"read failed at {diagnostic_path}\r\x00\x1b")
            except OSError as cause:
                raise RuntimeError("outer\nmessage") from cause
        except RuntimeError:
            logging.getLogger("myclaw.session.store").exception("persistence\r\nfailure")
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    lines = content.splitlines()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2} ERROR "
        r"pid=\d+ session=session-123 myclaw\.session\.store: persistence\\r\\nfailure",
        lines[0],
    )
    assert all(line.startswith("    ") for line in lines[1:])
    assert "OSError" in content
    assert "RuntimeError" in content
    assert str(diagnostic_path) in content
    assert "outer\\nmessage" in content
    assert "\\r\\x00\\x1b" in content
    assert "\x00" not in content
    assert "\x1b" not in content
    assert content.endswith("\n")
    assert not content.endswith("\n\n")


def test_runtime_log_tracebacks_do_not_persist_source_line_content(
    agent_home: Path,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sensitive_module.py"
    source_secret = "UNIQUE_SOURCE_LINE_SECRET_49"
    source_path.write_text(
        "def fail():\n"
        "    try:\n"
        "        raise OSError('inner safe message')\n"
        "    except OSError as error:\n"
        f"        raise RuntimeError('outer safe message') from error  # {source_secret}\n"
        "\n"
        "fail()\n",
        encoding="utf-8",
    )
    lifetime = install_runtime_logging(AgentHome(agent_home))

    try:
        exec(compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec"), {})
    except RuntimeError:
        logging.getLogger("myclaw.runtime").exception("dynamic module failed")
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert str(source_path) in content
    assert "line 5, in fail" in content
    assert "OSError: inner safe message" in content
    assert "RuntimeError: outer safe message" in content
    assert "direct cause" in content
    assert source_secret not in content
    assert "raise RuntimeError('outer safe message')" not in content


def test_runtime_log_renders_nested_exception_group_leaves(agent_home: Path) -> None:
    lifetime = install_runtime_logging(AgentHome(agent_home))
    grouped = ExceptionGroup(
        "outer group",
        (
            ExceptionGroup(
                "nested group",
                (
                    ValueError("first\nleaf"),
                    LookupError("second\r\x1b leaf"),
                ),
            ),
            OSError("third leaf"),
        ),
    )

    try:
        raise grouped
    except ExceptionGroup:
        logging.getLogger("myclaw.background").exception("grouped failure")
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert "ExceptionGroup: outer group" in content
    assert "ExceptionGroup: nested group" in content
    assert "ValueError: first\\nleaf" in content
    assert "LookupError: second\\r\\x1b leaf" in content
    assert "OSError: third leaf" in content
    assert "\x1b" not in content
    assert all(line.startswith("    ") for line in content.splitlines()[1:])


def test_runtime_log_normalizes_all_accepted_levels_to_warning_or_error(
    agent_home: Path,
) -> None:
    lifetime = install_runtime_logging(AgentHome(agent_home))
    logger = logging.getLogger("myclaw.levels")
    logger.warning("standard warning")
    logger.log(35, "high warning")
    logger.error("standard error")
    logger.critical("critical error")
    logger.log(45, "high error")
    lifetime.close()

    lines = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8").splitlines()
    assert [line.split()[1] for line in lines] == [
        "WARNING",
        "WARNING",
        "ERROR",
        "ERROR",
        "ERROR",
    ]
    assert "CRITICAL" not in "\n".join(lines)
    assert "Level 35" not in "\n".join(lines)
    assert "Level 45" not in "\n".join(lines)


def test_runtime_log_redacts_generic_credentials_and_configured_api_keys(
    agent_home: Path,
) -> None:
    lifetime = install_runtime_logging(AgentHome(agent_home))
    logger = logging.getLogger("myclaw.provider")

    logger.error(
        "transport failed Authorization: Bearer bearer-secret "
        "X-API-Key=header-secret Cookie: session=cookie-secret"
    )
    lifetime.add_api_keys(("configured-secret.[]",))
    try:
        raise RuntimeError("provider rejected configured-secret.[]")
    except RuntimeError:
        logger.exception("configured provider failure")
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert "[REDACTED]" in content
    assert "bearer-secret" not in content
    assert "header-secret" not in content
    assert "cookie-secret" not in content
    assert "configured-secret.[]" not in content


def test_runtime_log_submission_drops_the_oldest_of_1024_pending_records(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_blocked = threading.Event()
    release_writer = threading.Event()

    def blocked_fsync(descriptor: int) -> None:
        del descriptor
        writer_blocked.set()
        assert release_writer.wait(timeout=5)

    monkeypatch.setattr(os, "fsync", blocked_fsync)
    lifetime = install_runtime_logging(AgentHome(agent_home))
    logger = logging.getLogger("myclaw.queue")
    logger.warning("active-record")
    assert writer_blocked.wait(timeout=5)

    for index in range(1025):
        logger.warning("pending-%04d", index)

    release_writer.set()
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert "active-record" in content
    assert "pending-0000" not in content
    assert "pending-0001" in content
    assert "pending-1024" in content
    assert content.index("pending-0001") < content.index("pending-1024")


def test_runtime_log_first_use_creates_complete_private_state(agent_home: Path) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True, mode=0o777)
    if os.name == "posix":
        logs.chmod(0o777)
        (logs / "run.log.0").write_bytes(b"")
        (logs / "run.log.1").write_bytes(b"")
        (logs / "run.log.cursor").write_bytes(b"0\n")
        (logs / "run.log.lock").write_bytes(b"")
        for path in logs.iterdir():
            path.chmod(0o666)

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.storage").warning("first record")
    lifetime.close()

    expected_files = {"run.log.0", "run.log.1", "run.log.cursor", "run.log.lock"}
    assert {path.name for path in logs.iterdir()} == expected_files
    assert (logs / "run.log.cursor").read_bytes() == b"0\n"
    assert "first record" in (logs / "run.log.0").read_text(encoding="utf-8")
    assert (logs / "run.log.1").read_bytes() == b""
    if os.name == "posix":
        assert logs.stat().st_mode & 0o777 == 0o700
        assert all((logs / name).stat().st_mode & 0o777 == 0o600 for name in expected_files)


def test_runtime_log_appends_to_the_slot_selected_by_the_canonical_cursor(
    agent_home: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"older slot\n")
    (logs / "run.log.1").write_bytes(b"")
    (logs / "run.log.cursor").write_bytes(b"1\n")
    (logs / "run.log.lock").write_bytes(b"")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.storage").error("selected slot")
    lifetime.close()

    assert (logs / "run.log.0").read_bytes() == b"older slot\n"
    assert "selected slot" in (logs / "run.log.1").read_text(encoding="utf-8")


def test_runtime_log_rejects_hard_linked_state_without_affecting_the_caller(
    agent_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_text("outside remains unchanged\n", encoding="utf-8")
    os.link(outside, logs / "run.log.0")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logger = logging.getLogger("myclaw.storage")
    logger.error("first failed append")
    logger.error("second failed append")
    lifetime.close()

    assert outside.read_text(encoding="utf-8") == "outside remains unchanged\n"
    assert capsys.readouterr().err == "Runtime Log failure: PermissionError\n"


def test_runtime_log_close_abandons_records_after_ten_second_deadline(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_blocked = threading.Event()
    release_writer = threading.Event()

    def blocked_fsync(descriptor: int) -> None:
        del descriptor
        writer_blocked.set()
        release_writer.wait(timeout=30)

    monkeypatch.setattr(os, "fsync", blocked_fsync)
    lifetime = install_runtime_logging(AgentHome(agent_home))
    logger = logging.getLogger("myclaw.shutdown")
    logger.error("blocked record")
    assert writer_blocked.wait(timeout=5)
    logger.error("must be abandoned")

    started = time.monotonic()
    lifetime.close()
    elapsed = time.monotonic() - started
    assert 9 <= elapsed <= 10.5

    release_writer.set()
    time.sleep(0.5)
    slot = agent_home / "logs" / "run.log.0"
    content = slot.read_text(encoding="utf-8") if slot.exists() else ""
    assert "must be abandoned" not in content


def test_runtime_log_tolerates_filesystems_without_fsync(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unsupported_fsync(descriptor: int) -> None:
        del descriptor
        raise OSError(errno.EINVAL, "unsupported")

    monkeypatch.setattr(os, "fsync", unsupported_fsync)
    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.storage").error("durability best effort")
    lifetime.close()

    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert "durability best effort" in content
    assert capsys.readouterr().err == ""


def test_runtime_log_rejects_reparse_or_symlink_directory_outside_agent_home(
    agent_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent_home.mkdir(parents=True)
    outside = tmp_path / "outside-logs"
    outside.mkdir()
    try:
        (agent_home / "logs").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {type(error).__name__}")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.storage").error("must not escape Agent Home")
    lifetime.close()

    assert list(outside.iterdir()) == []
    assert capsys.readouterr().err == "Runtime Log failure: PermissionError\n"


def test_runtime_log_success_resets_the_stderr_failure_period(
    agent_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"outside\n")
    slot = logs / "run.log.0"
    os.link(outside, slot)
    lifetime = install_runtime_logging(AgentHome(agent_home))
    logger = logging.getLogger("myclaw.storage")

    logger.error("first failure period")
    deadline = time.monotonic() + 5
    first_diagnostic = ""
    while time.monotonic() < deadline and not first_diagnostic:
        first_diagnostic = capsys.readouterr().err
        time.sleep(0.01)
    assert first_diagnostic == "Runtime Log failure: PermissionError\n"

    slot.unlink()
    logger.error("successful append")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if slot.exists() and "successful append" in slot.read_text(encoding="utf-8"):
            break
        time.sleep(0.01)
    else:
        pytest.fail("Runtime Log writer did not complete the successful append", pytrace=False)

    slot.unlink()
    os.link(outside, slot)
    logger.error("second failure period")
    lifetime.close()

    assert capsys.readouterr().err == "Runtime Log failure: PermissionError\n"
