import msvcrt
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SLOT_TARGET_BYTES = 10_485_760
_APPEND_SCRIPT = """
import logging
import sys
import time
from pathlib import Path

from myclaw.config.agent_home import AgentHome
from myclaw.runtime_log import install_runtime_logging

lifetime = install_runtime_logging(AgentHome(Path(sys.argv[1])))
token = sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
payload_size = int(sys.argv[4]) if len(sys.argv) > 4 else 0
if len(sys.argv) > 5:
    gate = Path(sys.argv[5])
    gate.with_name(f"{gate.name}.{token}.ready").write_bytes(b"")
    while not gate.exists():
        time.sleep(0.005)
logger = logging.getLogger("myclaw.process")
for index in range(count):
    logger.warning("[%s:%04d]%s", token, index, "x" * payload_size)
lifetime.close()
"""


def _lock_control_file(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)


def _unlock_control_file(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


def _run_concurrent_writers(
    agent_home: Path,
    gate: Path,
    tokens: tuple[str, ...],
    *,
    count: int,
    payload_size: int,
) -> list[tuple[int, str]]:
    root = Path(__file__).resolve().parents[2]
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _APPEND_SCRIPT,
                str(agent_home),
                token,
                str(count),
                str(payload_size),
                str(gate),
            ],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for token in tokens
    ]
    deadline = time.monotonic() + 10
    ready = tuple(gate.with_name(f"{gate.name}.{token}.ready") for token in tokens)
    while time.monotonic() < deadline and not all(path.exists() for path in ready):
        time.sleep(0.01)
    if not all(path.exists() for path in ready):
        for process in processes:
            process.kill()
        pytest.fail("Runtime Log subprocess writers did not reach the start barrier", pytrace=False)
    gate.write_bytes(b"")
    results: list[tuple[int, str]] = []
    for process in processes:
        _, stderr = process.communicate(timeout=30)
        results.append((process.returncode, stderr))
    return results


def test_runtime_log_subprocess_uses_one_second_unlocked_fallback(agent_home: Path) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    (logs / "run.log.0").write_bytes(b"older slot zero\n")
    (logs / "run.log.1").write_bytes(b"")
    (logs / "run.log.cursor").write_bytes(b"1\n")
    lock_path = logs / "run.log.lock"
    lock_path.write_bytes(b"")

    with lock_path.open("r+b", buffering=0) as lock_stream:
        _lock_control_file(lock_stream.fileno())
        try:
            started = time.monotonic()
            result = subprocess.run(
                [sys.executable, "-c", _APPEND_SCRIPT, str(agent_home), "fallback-process"],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            elapsed = time.monotonic() - started
        finally:
            _unlock_control_file(lock_stream.fileno())

    assert result.returncode == 0, result.stderr
    assert 0.9 <= elapsed < 5
    assert result.stderr == "Runtime Log failure: TimeoutError\n"
    assert (logs / "run.log.0").read_bytes() == b"older slot zero\n"
    assert "fallback-process" in (logs / "run.log.1").read_text(encoding="utf-8")
    assert (logs / "run.log.cursor").read_bytes() == b"1\n"


def test_runtime_log_subprocesses_append_only_complete_records(
    agent_home: Path,
    tmp_path: Path,
) -> None:
    tokens = ("process-a", "process-b", "process-c", "process-d")
    count = 75

    results = _run_concurrent_writers(
        agent_home,
        tmp_path / "complete-records.start",
        tokens,
        count=count,
        payload_size=256,
    )

    assert results == [(0, "")] * len(tokens)
    logs = agent_home / "logs"
    content = (logs / "run.log.0").read_text(encoding="utf-8")
    expected = {f"[{token}:{index:04d}]" + "x" * 256 for token in tokens for index in range(count)}
    records: set[str] = set()
    for line in content.splitlines():
        match = re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2} "
            r"WARNING pid=\d+ session=- myclaw\.process: (?P<body>.+)",
            line,
        )
        assert match is not None, line
        records.add(match.group("body"))
    assert records == expected
    assert len(content.splitlines()) == len(expected)
    assert (logs / "run.log.1").read_bytes() == b""
    assert (logs / "run.log.cursor").read_bytes() == b"0\n"


def test_runtime_log_subprocesses_serialize_alternating_rotations(
    agent_home: Path,
    tmp_path: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    prefix = b"x" * (_SLOT_TARGET_BYTES - 1)
    slot_zero = logs / "run.log.0"
    slot_one = logs / "run.log.1"
    cursor = logs / "run.log.cursor"
    lock_path = logs / "run.log.lock"
    slot_zero.write_bytes(prefix)
    slot_one.write_bytes(b"stale slot one\n")
    cursor.write_bytes(b"0\n")
    lock_path.write_bytes(b"")
    lock_status = lock_path.stat()
    lock_identity = (lock_status.st_dev, lock_status.st_ino)

    first_tokens = ("zero-crossing-a", "zero-crossing-b")
    first_results = _run_concurrent_writers(
        agent_home,
        tmp_path / "rotate-zero.start",
        first_tokens,
        count=1,
        payload_size=64,
    )

    assert first_results == [(0, ""), (0, "")]
    first_cycle = slot_zero.read_text(encoding="utf-8") + slot_one.read_text(encoding="utf-8")
    assert "stale slot one" not in first_cycle
    assert all(first_cycle.count(f"[{token}:0000]") == 1 for token in first_tokens)
    assert cursor.read_bytes() == b"1\n"

    slot_one.write_bytes(prefix)
    second_tokens = ("one-crossing-a", "one-crossing-b")
    second_results = _run_concurrent_writers(
        agent_home,
        tmp_path / "rotate-one.start",
        second_tokens,
        count=1,
        payload_size=64,
    )

    assert second_results == [(0, ""), (0, "")]
    second_cycle = slot_zero.read_text(encoding="utf-8") + slot_one.read_text(encoding="utf-8")
    assert all(second_cycle.count(f"[{token}:0000]") == 1 for token in second_tokens)
    assert all(token not in second_cycle for token in first_tokens)
    assert cursor.read_bytes() == b"0\n"
    final_lock_status = lock_path.stat()
    assert (final_lock_status.st_dev, final_lock_status.st_ino) == lock_identity


def test_runtime_log_fallback_record_may_be_removed_by_a_later_rotation(
    agent_home: Path,
) -> None:
    logs = agent_home / "logs"
    logs.mkdir(parents=True)
    slot_zero = logs / "run.log.0"
    slot_one = logs / "run.log.1"
    cursor = logs / "run.log.cursor"
    lock_path = logs / "run.log.lock"
    slot_zero.write_bytes(b"")
    slot_one.write_bytes(b"")
    cursor.write_bytes(b"0\n")
    lock_path.write_bytes(b"")
    root = Path(__file__).resolve().parents[2]

    with lock_path.open("r+b", buffering=0) as lock_stream:
        _lock_control_file(lock_stream.fileno())
        try:
            fallback = subprocess.run(
                [sys.executable, "-c", _APPEND_SCRIPT, str(agent_home), "at-risk-fallback"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        finally:
            _unlock_control_file(lock_stream.fileno())

    assert fallback.returncode == 0, fallback.stderr
    assert fallback.stderr == "Runtime Log failure: TimeoutError\n"
    assert "at-risk-fallback" in slot_zero.read_text(encoding="utf-8")

    slot_one.write_bytes(b"x" * _SLOT_TARGET_BYTES)
    cursor.write_bytes(b"1\n")
    rotation = subprocess.run(
        [sys.executable, "-c", _APPEND_SCRIPT, str(agent_home), "after-risk-rotation"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert rotation.returncode == 0, rotation.stderr
    assert rotation.stderr == ""
    rotated = slot_zero.read_text(encoding="utf-8")
    assert "after-risk-rotation" in rotated
    assert "at-risk-fallback" not in rotated
    assert cursor.read_bytes() == b"0\n"
