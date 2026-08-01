import errno
import logging
import os
import re
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import IO, Any

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.runtime_log import install_runtime_logging
from myclaw.utils.atomic_files import atomic_replace_bytes

_SLOT_TARGET_BYTES = 10_485_760


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
    capsys: pytest.CaptureFixture[str],
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
    assert capsys.readouterr().err == ""


def test_runtime_log_first_use_creates_complete_private_state(agent_home: Path) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True, mode=0o777)

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.storage").warning("first record")
    lifetime.close()

    expected_files = {"run.log.0", "run.log.1", "run.log.cursor", "run.log.lock"}
    assert {path.name for path in logs.iterdir()} == expected_files
    assert (logs / "run.log.cursor").read_bytes() == b"0\n"
    content = (logs / "run.log.0").read_text(encoding="utf-8")
    assert "first record" in content
    assert "cursor was recovered" not in content
    assert content.count(" WARNING ") == 1
    assert (logs / "run.log.1").read_bytes() == b""


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


def test_runtime_log_rotates_after_a_complete_record_crosses_the_fixed_target(
    agent_home: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    prefix = b"x" * (_SLOT_TARGET_BYTES - 1)
    (logs / "run.log.0").write_bytes(prefix)
    (logs / "run.log.1").write_bytes(b"stale slot content\n")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").warning("crossing record")
    lifetime.close()

    slot_zero = (logs / "run.log.0").read_bytes()
    assert slot_zero.startswith(prefix)
    assert slot_zero.endswith(b"myclaw.rotation: crossing record\n")
    assert len(slot_zero) > _SLOT_TARGET_BYTES
    assert (logs / "run.log.1").read_bytes() == b""
    assert (logs / "run.log.cursor").read_bytes() == b"1\n"


def test_runtime_log_completes_an_interrupted_rotation_before_appending(
    agent_home: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    full_slot = b"x" * _SLOT_TARGET_BYTES
    (logs / "run.log.0").write_bytes(full_slot)
    (logs / "run.log.1").write_bytes(b"stale slot content\n")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").warning("record after interruption")
    lifetime.close()

    assert (logs / "run.log.0").read_bytes() == full_slot
    slot_one = (logs / "run.log.1").read_text(encoding="utf-8")
    assert "myclaw.rotation: record after interruption" in slot_one
    assert "stale slot content" not in slot_one
    assert (logs / "run.log.cursor").read_bytes() == b"1\n"


def test_runtime_log_keeps_the_utf8_crossing_record_as_the_only_overshoot(
    agent_home: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    prefix = b"x" * (_SLOT_TARGET_BYTES - 100)
    (logs / "run.log.0").write_bytes(prefix)
    (logs / "run.log.1").write_bytes(b"")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").warning("crossing utf8 %s", "\u754c" * 100)
    lifetime.close()

    slot_zero = (logs / "run.log.0").read_bytes()
    crossing_record = slot_zero[len(prefix) :]
    assert len(slot_zero) >= _SLOT_TARGET_BYTES
    assert len(slot_zero) < _SLOT_TARGET_BYTES + len(crossing_record)
    assert crossing_record.decode("utf-8").endswith(
        f"myclaw.rotation: crossing utf8 {'\u754c' * 100}\n"
    )
    assert (logs / "run.log.cursor").read_bytes() == b"1\n"


def test_runtime_log_restart_continues_from_the_durable_cursor(agent_home: Path) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"x" * (_SLOT_TARGET_BYTES - 1))
    (logs / "run.log.1").write_bytes(b"old cycle\n")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")

    first_lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").warning("record before restart")
    first_lifetime.close()

    second_lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").warning("record after restart")
    second_lifetime.close()

    assert b"record before restart" in (logs / "run.log.0").read_bytes()
    slot_one = (logs / "run.log.1").read_text(encoding="utf-8")
    assert "old cycle" not in slot_one
    assert "record after restart" in slot_one
    assert (logs / "run.log.cursor").read_bytes() == b"1\n"


def test_runtime_log_repeatedly_alternates_between_the_two_slots(agent_home: Path) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"x" * (_SLOT_TARGET_BYTES - 1))
    (logs / "run.log.1").write_bytes(b"old cycle one\n")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")
    huge_message = "y" * _SLOT_TARGET_BYTES

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logger = logging.getLogger("myclaw.rotation")
    logger.warning("cross slot zero")
    logger.warning("%s", huge_message)
    logger.warning("after returning to slot zero")
    lifetime.close()

    slot_zero = (logs / "run.log.0").read_text(encoding="utf-8")
    slot_one = (logs / "run.log.1").read_bytes()
    assert "after returning to slot zero" in slot_zero
    assert "cross slot zero" not in slot_zero
    assert b"old cycle one" not in slot_one
    assert slot_one.endswith(huge_message.encode("utf-8") + b"\n")
    assert (logs / "run.log.cursor").read_bytes() == b"0\n"


def test_runtime_log_recovers_a_missing_cursor_before_the_triggering_record(
    agent_home: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"older slot zero\n")
    (logs / "run.log.1").write_bytes(b"older slot one\n")
    (logs / "run.log.lock").write_bytes(b"")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").error("triggering record")
    lifetime.close()

    slot_zero = (logs / "run.log.0").read_text(encoding="utf-8")
    recovery = "myclaw.runtime_log: Runtime Log cursor was recovered to slot 0"
    trigger = "myclaw.rotation: triggering record"
    assert slot_zero.count(recovery) == 1
    assert slot_zero.index(recovery) < slot_zero.index(trigger)
    assert (logs / "run.log.1").read_bytes() == b"older slot one\n"
    assert (logs / "run.log.cursor").read_bytes() == b"0\n"


def test_runtime_log_recovers_a_malformed_cursor_without_persisting_its_bytes(
    agent_home: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    invalid_cursor = b"1\nBearer invalid-cursor-secret\x00"
    (logs / "run.log.0").write_bytes(b"")
    (logs / "run.log.1").write_bytes(b"older slot one\n")
    (logs / "run.log.cursor").write_bytes(invalid_cursor)
    (logs / "run.log.lock").write_bytes(b"")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").error("triggering malformed recovery")
    lifetime.close()

    content = (logs / "run.log.0").read_text(encoding="utf-8")
    recovery = "myclaw.runtime_log: Runtime Log cursor was recovered to slot 0"
    trigger = "myclaw.rotation: triggering malformed recovery"
    assert content.count(recovery) == 1
    assert content.index(recovery) < content.index(trigger)
    assert "invalid-cursor-secret" not in content
    assert "Bearer" not in content
    assert (logs / "run.log.1").read_bytes() == b"older slot one\n"
    assert (logs / "run.log.cursor").read_bytes() == b"0\n"


def test_runtime_log_missing_cursor_recovery_rotates_a_full_initial_slot(
    agent_home: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    full_slot = b"x" * _SLOT_TARGET_BYTES
    (logs / "run.log.0").write_bytes(full_slot)
    (logs / "run.log.1").write_bytes(b"older slot one\n")
    (logs / "run.log.lock").write_bytes(b"")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").error("trigger after full recovery")
    lifetime.close()

    assert (logs / "run.log.0").read_bytes() == full_slot
    slot_one = (logs / "run.log.1").read_text(encoding="utf-8")
    recovery = "myclaw.runtime_log: Runtime Log cursor was recovered to slot 0"
    trigger = "myclaw.rotation: trigger after full recovery"
    assert slot_one.count(recovery) == 1
    assert slot_one.index(recovery) < slot_one.index(trigger)
    assert "older slot one" not in slot_one
    assert (logs / "run.log.cursor").read_bytes() == b"1\n"


def test_runtime_log_recovers_when_target_reset_is_interrupted(agent_home: Path) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    prefix = b"x" * (_SLOT_TARGET_BYTES - 1)
    (logs / "run.log.0").write_bytes(prefix)
    (logs / "run.log.1").write_bytes(b"old target cycle\n")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")
    interrupted = False

    def interrupt_target_reset(path: Path, content: bytes) -> None:
        nonlocal interrupted
        if not interrupted and path == logs / "run.log.1" and content == b"":
            interrupted = True
            raise OSError("simulated interruption before target reset")
        atomic_replace_bytes(path, content)

    interrupted_lifetime = install_runtime_logging(
        AgentHome(agent_home), replace_bytes=interrupt_target_reset
    )
    logging.getLogger("myclaw.rotation").warning("record committed before reset")
    interrupted_lifetime.close()

    assert interrupted
    committed_slot = (logs / "run.log.0").read_bytes()
    assert committed_slot.startswith(prefix)
    assert committed_slot.endswith(b"myclaw.rotation: record committed before reset\n")
    assert (logs / "run.log.1").read_bytes() == b"old target cycle\n"
    assert (logs / "run.log.cursor").read_bytes() == b"0\n"

    recovered_lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").error("record after reset recovery")
    recovered_lifetime.close()

    assert (logs / "run.log.0").read_bytes() == committed_slot
    recovered_slot = (logs / "run.log.1").read_text(encoding="utf-8")
    assert "old target cycle" not in recovered_slot
    assert "record after reset recovery" in recovered_slot
    assert (logs / "run.log.cursor").read_bytes() == b"1\n"


def test_runtime_log_recovers_when_cursor_publication_is_interrupted(
    agent_home: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    prefix = b"x" * (_SLOT_TARGET_BYTES - 1)
    (logs / "run.log.0").write_bytes(prefix)
    (logs / "run.log.1").write_bytes(b"old target cycle\n")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")
    interrupted = False

    def interrupt_cursor_publication(path: Path, content: bytes) -> None:
        nonlocal interrupted
        if not interrupted and path == logs / "run.log.cursor" and content == b"1\n":
            interrupted = True
            raise OSError("simulated interruption before cursor publication")
        atomic_replace_bytes(path, content)

    interrupted_lifetime = install_runtime_logging(
        AgentHome(agent_home), replace_bytes=interrupt_cursor_publication
    )
    logging.getLogger("myclaw.rotation").warning("record committed before cursor publish")
    interrupted_lifetime.close()

    assert interrupted
    committed_slot = (logs / "run.log.0").read_bytes()
    assert committed_slot.startswith(prefix)
    assert committed_slot.endswith(b"myclaw.rotation: record committed before cursor publish\n")
    assert (logs / "run.log.1").read_bytes() == b""
    assert (logs / "run.log.cursor").read_bytes() == b"0\n"

    recovered_lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.rotation").error("record after cursor recovery")
    recovered_lifetime.close()

    assert (logs / "run.log.0").read_bytes() == committed_slot
    recovered_slot = (logs / "run.log.1").read_text(encoding="utf-8")
    assert "record after cursor recovery" in recovered_slot
    assert (logs / "run.log.cursor").read_bytes() == b"1\n"


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


def test_runtime_log_rejects_an_unsafe_inactive_slot(
    agent_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    active = logs / "run.log.0"
    active.write_bytes(b"")
    outside = tmp_path / "outside.log"
    outside.write_text("outside remains unchanged\n", encoding="utf-8")
    os.link(outside, logs / "run.log.1")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.storage").error("must fail before the active append")
    lifetime.close()

    assert active.read_bytes() == b""
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


def test_runtime_log_rejects_a_junction_directory_outside_agent_home(
    agent_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent_home.mkdir(parents=True)
    outside = tmp_path / "outside-logs"
    outside.mkdir()
    subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(agent_home / "logs"), str(outside)),
        check=True,
        capture_output=True,
        text=True,
    )

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


def test_runtime_log_lock_deadline_falls_back_once_without_rotation(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"older slot zero\n")
    prefix = b"x" * (_SLOT_TARGET_BYTES - 1)
    (logs / "run.log.1").write_bytes(prefix)
    (logs / "run.log.cursor").write_bytes(b"1\n")
    (logs / "run.log.lock").write_bytes(b"")
    elapsed = 0.0
    attempts = 0
    released = False
    synchronized: list[int] = []

    class DeadlineOnlyLock:
        def try_acquire(self, descriptor: int) -> bool:
            nonlocal attempts
            del descriptor
            attempts += 1
            return elapsed >= 1.0

        def release(self, descriptor: int) -> None:
            nonlocal released
            del descriptor
            released = True

    def monotonic() -> float:
        return elapsed

    def sleep(delay: float) -> None:
        nonlocal elapsed
        elapsed += delay

    monkeypatch.setattr(os, "fsync", synchronized.append)
    lifetime = install_runtime_logging(
        AgentHome(agent_home),
        lock_system=DeadlineOnlyLock(),
        monotonic=monotonic,
        sleep=sleep,
    )
    logging.getLogger("myclaw.lock").warning("deadline fallback")
    lifetime.close()

    assert elapsed == pytest.approx(1.0)
    assert attempts > 1
    assert released is False
    assert len(synchronized) == 1
    assert (logs / "run.log.0").read_bytes() == b"older slot zero\n"
    slot_one = (logs / "run.log.1").read_bytes()
    assert slot_one.startswith(prefix)
    assert slot_one.endswith(b"myclaw.lock: deadline fallback\n")
    assert len(slot_one) > _SLOT_TARGET_BYTES
    assert (logs / "run.log.cursor").read_bytes() == b"1\n"
    assert capsys.readouterr().err == "Runtime Log failure: TimeoutError\n"


def test_runtime_log_reselects_the_cursor_and_completes_rotation_under_one_lock(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"old slot zero\n")
    prefix = b"x" * (_SLOT_TARGET_BYTES - 1)
    (logs / "run.log.1").write_bytes(prefix)
    cursor = logs / "run.log.cursor"
    cursor.write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")
    held = False
    replacements: list[tuple[Path, bytes]] = []
    synchronized = 0

    class TrackingLock:
        def try_acquire(self, descriptor: int) -> bool:
            nonlocal held
            del descriptor
            cursor.write_bytes(b"1\n")
            held = True
            return True

        def release(self, descriptor: int) -> None:
            nonlocal held
            del descriptor
            assert held
            held = False

    def replace_while_locked(path: Path, content: bytes) -> None:
        assert held
        replacements.append((path, content))
        atomic_replace_bytes(path, content)

    def fsync_while_locked(descriptor: int) -> None:
        nonlocal synchronized
        del descriptor
        assert held
        synchronized += 1

    monkeypatch.setattr(os, "fsync", fsync_while_locked)
    lifetime = install_runtime_logging(
        AgentHome(agent_home),
        replace_bytes=replace_while_locked,
        lock_system=TrackingLock(),
    )
    logging.getLogger("myclaw.lock").warning("locked rotation")
    lifetime.close()

    assert held is False
    assert synchronized == 3
    assert replacements == [
        (logs / "run.log.0", b""),
        (logs / "run.log.cursor", b"0\n"),
    ]
    slot_one = (logs / "run.log.1").read_bytes()
    assert slot_one.startswith(prefix)
    assert slot_one.endswith(b"myclaw.lock: locked rotation\n")
    assert (logs / "run.log.0").read_bytes() == b""
    assert cursor.read_bytes() == b"0\n"


def test_runtime_log_unreadable_prelock_cursor_falls_back_to_the_initial_slot(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"")
    (logs / "run.log.1").write_bytes(b"older slot one\n")
    invalid_cursor = b"not-a-canonical-cursor\n"
    (logs / "run.log.cursor").write_bytes(invalid_cursor)
    (logs / "run.log.lock").write_bytes(b"")
    elapsed = 0.0

    class UnavailableLock:
        def try_acquire(self, descriptor: int) -> bool:
            del descriptor
            return False

        def release(self, descriptor: int) -> None:
            del descriptor
            pytest.fail("an unavailable lock must not be released")

    def monotonic() -> float:
        return elapsed

    def sleep(delay: float) -> None:
        nonlocal elapsed
        elapsed += delay

    lifetime = install_runtime_logging(
        AgentHome(agent_home),
        lock_system=UnavailableLock(),
        monotonic=monotonic,
        sleep=sleep,
    )
    logging.getLogger("myclaw.lock").error("invalid cursor fallback")
    lifetime.close()

    assert "invalid cursor fallback" in (logs / "run.log.0").read_text(encoding="utf-8")
    assert (logs / "run.log.1").read_bytes() == b"older slot one\n"
    assert (logs / "run.log.cursor").read_bytes() == invalid_cursor
    assert capsys.readouterr().err == "Runtime Log failure: TimeoutError\n"


def test_runtime_log_lock_failures_follow_stderr_failure_periods(
    agent_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"")
    (logs / "run.log.1").write_bytes(b"")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")
    outcomes: list[Exception | bool] = [
        OSError("first lock failure"),
        OSError("same failure period"),
        True,
        OSError("second lock failure period"),
    ]

    class SequencedLock:
        def try_acquire(self, descriptor: int) -> bool:
            del descriptor
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def release(self, descriptor: int) -> None:
            del descriptor

    lifetime = install_runtime_logging(AgentHome(agent_home), lock_system=SequencedLock())
    logger = logging.getLogger("myclaw.lock")
    logger.error("first fallback")
    logger.error("second fallback")
    logger.warning("locked success")
    logger.error("third fallback")
    lifetime.close()

    content = (logs / "run.log.0").read_text(encoding="utf-8")
    assert all(
        marker in content
        for marker in ("first fallback", "second fallback", "locked success", "third fallback")
    )
    assert capsys.readouterr().err == (
        "Runtime Log failure: OSError\nRuntime Log failure: OSError\n"
    )


def test_runtime_log_fallback_failures_are_isolated_and_reported_once(
    agent_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"outside remains unchanged\n")
    os.link(outside, logs / "run.log.0")
    (logs / "run.log.1").write_bytes(b"")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")

    class FailingLock:
        def try_acquire(self, descriptor: int) -> bool:
            del descriptor
            raise OSError("lock system failed")

        def release(self, descriptor: int) -> None:
            del descriptor
            pytest.fail("a failed lock must not be released")

    lifetime = install_runtime_logging(AgentHome(agent_home), lock_system=FailingLock())
    logger = logging.getLogger("myclaw.lock")
    logger.error("first failed fallback")
    logger.error("second failed fallback")
    lifetime.close()

    assert outside.read_bytes() == b"outside remains unchanged\n"
    assert capsys.readouterr().err == "Runtime Log failure: PermissionError\n"


def test_runtime_log_revalidates_the_opened_slot_after_locked_cursor_selection(
    agent_home: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"x" * _SLOT_TARGET_BYTES)
    target = logs / "run.log.1"
    target.write_bytes(b"old target\n")
    cursor = logs / "run.log.cursor"
    cursor.write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"outside remains unchanged\n")

    def replace_then_alias_target(path: Path, content: bytes) -> None:
        atomic_replace_bytes(path, content)
        if path == cursor and content == b"1\n":
            target.unlink()
            os.link(outside, target)

    lifetime = install_runtime_logging(
        AgentHome(agent_home), replace_bytes=replace_then_alias_target
    )
    logging.getLogger("myclaw.lock").error("must not follow replacement")
    lifetime.close()

    assert outside.read_bytes() == b"outside remains unchanged\n"
    assert capsys.readouterr().err == "Runtime Log failure: PermissionError\n"


def test_runtime_log_rejects_an_in_home_alias_swap_after_open(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    target = logs / "run.log.0"
    target.write_bytes(b"original slot\n")
    (logs / "run.log.1").write_bytes(b"")
    (logs / "run.log.cursor").write_bytes(b"0\n")
    (logs / "run.log.lock").write_bytes(b"")
    victim = agent_home / "internal-state.txt"
    victim.write_bytes(b"internal state must remain unchanged\n")
    original_victim = victim.read_bytes()
    original_open = Path.open
    original_stat = Path.stat
    original_resolve = Path.resolve
    alias_opened = False

    def open_alias(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        nonlocal alias_opened
        if path == target and mode == "ab":
            alias_opened = True
            path = victim
        return original_open(path, mode, buffering, encoding, errors, newline)

    def stat_alias(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if alias_opened and path == target:
            path = victim
        return original_stat(path, follow_symlinks=follow_symlinks)

    def resolve_alias(path: Path, strict: bool = False) -> Path:
        if alias_opened and path == target:
            path = victim
        return original_resolve(path, strict=strict)

    def lstat_actual(path: Path) -> os.stat_result:
        return os.lstat(path)

    monkeypatch.setattr(Path, "open", open_alias)
    monkeypatch.setattr(Path, "stat", stat_alias)
    monkeypatch.setattr(Path, "lstat", lstat_actual)
    monkeypatch.setattr(Path, "resolve", resolve_alias)

    lifetime = install_runtime_logging(AgentHome(agent_home))
    logging.getLogger("myclaw.lock").error("must not reach another Agent Home file")
    lifetime.close()

    assert victim.read_bytes() == original_victim
    assert target.read_bytes() == b"original slot\n"
    assert capsys.readouterr().err == "Runtime Log failure: PermissionError\n"
