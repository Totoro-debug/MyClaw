"""Process-local lifetime for the Agent Home Runtime Log."""

from __future__ import annotations

import errno
import logging
import os
import queue
import re
import sys
import threading
import traceback
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Final

from myclaw.config.agent_home import AgentHome
from myclaw.utils.atomic_files import atomic_create_bytes, atomic_replace_bytes

_LOGGER_NAME: Final = "myclaw"
_SLOT_TARGET_BYTES: Final = 10_485_760
_SESSION_ID: ContextVar[str] = ContextVar("myclaw_runtime_log_session", default="-")
_REDACTED: Final = "[REDACTED]"
_AUTHORIZATION: Final = re.compile(
    r"(?i)\b((?:proxy-)?authorization\s*:\s*(?:bearer|basic))\s+\S+"
)
_BEARER: Final = re.compile(r"(?i)\b(bearer)\s+\S+")
_API_KEY: Final = re.compile(r"(?i)\b((?:x-)?api[-_ ]?key)\s*([:=])\s*([^\s,;]+)")
_COOKIE: Final = re.compile(r"(?i)\b((?:set-)?cookie)\s*([:=])\s*[^\n]+")
_WINDOWS_REPARSE_POINT: Final = 0x400
_UNSUPPORTED_FSYNC_ERRNOS: Final = frozenset(
    {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
)

type AtomicReplaceBytes = Callable[[Path, bytes], None]


@dataclass(frozen=True, slots=True)
class _PendingRecord:
    record: logging.LogRecord
    session_id: str


class _DropOldestQueue(queue.Queue[_PendingRecord]):
    def put_latest(self, pending: _PendingRecord) -> None:
        """Replace the oldest item atomically when this queue is full."""
        with self.mutex:
            if self.maxsize > 0 and self._qsize() >= self.maxsize:
                self._get()
                self.unfinished_tasks -= 1
            self._put(pending)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class _RuntimeLogHandler(logging.Handler):
    def __init__(self, records: _DropOldestQueue) -> None:
        super().__init__(logging.WARNING)
        self._records = records
        self._accepting = True

    def emit(self, record: logging.LogRecord) -> None:
        if self._accepting:
            self._records.put_latest(_PendingRecord(record, _SESSION_ID.get()))

    def stop_accepting(self) -> None:
        self._accepting = False


class RuntimeLogLifetime:
    """An installed dedicated logger handler and its owned writer lifetime."""

    def __init__(
        self,
        agent_home: AgentHome,
        *,
        replace_bytes: AtomicReplaceBytes = atomic_replace_bytes,
    ) -> None:
        self._agent_home = agent_home
        self._replace_bytes = replace_bytes
        self._records = _DropOldestQueue(maxsize=1024)
        self._handler = _RuntimeLogHandler(self._records)
        self._logger = logging.getLogger(_LOGGER_NAME)
        self._previous_level = self._logger.level
        self._previous_propagate = self._logger.propagate
        self._closed = False
        self._closing = threading.Event()
        self._abandon = threading.Event()
        self._api_keys: tuple[str, ...] = ()
        self._api_key_lock = threading.Lock()
        self._failure_reported = False
        self._writer = threading.Thread(
            target=self._write_records,
            name="myclaw-runtime-log",
            daemon=True,
        )
        self._logger.setLevel(logging.WARNING)
        self._logger.propagate = False
        self._logger.addHandler(self._handler)
        self._writer.start()

    def close(self) -> None:
        """Stop accepting records and drain the writer."""
        if self._closed:
            return
        self._closed = True
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._previous_level)
        self._logger.propagate = self._previous_propagate
        self._handler.stop_accepting()
        self._closing.set()
        self._writer.join(timeout=10)
        if self._writer.is_alive():
            self._abandon.set()
        self._handler.close()

    def __enter__(self) -> RuntimeLogLifetime:
        return self

    def __exit__(self, *errors: object) -> None:
        del errors
        self.close()

    def add_api_keys(self, api_keys: Iterable[str]) -> None:
        """Add exact configured Provider API-key values to the writer redactor."""
        additions = tuple(key for key in api_keys if key)
        if not additions:
            return
        with self._api_key_lock:
            self._api_keys = tuple(
                sorted(set((*self._api_keys, *additions)), key=len, reverse=True)
            )

    @contextmanager
    def session(self, session_id: str) -> Iterator[None]:
        """Correlate records submitted in this context with an owning Session."""
        token = _SESSION_ID.set(session_id)
        try:
            yield
        finally:
            _SESSION_ID.reset(token)

    def _write_records(self) -> None:
        while True:
            if self._abandon.is_set():
                return
            try:
                pending = self._records.get(timeout=0.1)
            except queue.Empty:
                if self._closing.is_set():
                    return
                continue
            if self._abandon.is_set():
                self._records.task_done()
                return
            try:
                self._append(pending)
            except Exception as error:
                self._report_failure(error)
            else:
                self._failure_reported = False
            finally:
                self._records.task_done()

    def _append(self, pending: _PendingRecord) -> None:
        slot, cursor_recovered = self._prepare_storage()
        if cursor_recovered:
            recovery = logging.LogRecord(
                name="myclaw.runtime_log",
                level=logging.WARNING,
                pathname=__file__,
                lineno=0,
                msg="Runtime Log cursor was recovered to slot 0",
                args=(),
                exc_info=None,
            )
            self._append_encoded(slot, self._encode(_PendingRecord(recovery, "-")))
            slot, _ = self._prepare_storage()
        self._append_encoded(slot, self._encode(pending))

    def _encode(self, pending: _PendingRecord) -> bytes:
        record = pending.record
        timestamp = datetime.fromtimestamp(record.created).astimezone().isoformat(
            timespec="milliseconds"
        )
        header = (
            f"{timestamp} {_severity(record.levelno)} pid={record.process} "
            f"session={_visible(pending.session_id)} {_visible(record.name)}: "
            f"{_visible(record.getMessage())}"
        )
        continuation = _exception_lines(record)
        line = header
        if continuation:
            line += "\n" + "\n".join(f"    {part}" for part in continuation)
        line += "\n"
        return self._redact(line).encode("utf-8")

    def _append_encoded(self, slot: Path, encoded: bytes) -> None:
        with slot.open("ab", buffering=0) as stream:
            written = stream.write(encoded)
            if written != len(encoded):
                raise OSError("Runtime Log append did not write the complete record")
            stream.flush()
            _fsync_file(stream.fileno())
            slot_size = os.fstat(stream.fileno()).st_size
        if slot_size >= _SLOT_TARGET_BYTES:
            active_slot = int(slot.name.rsplit(".", 1)[1])
            self._rotate(slot.parent, active_slot)

    def _prepare_storage(self) -> tuple[Path, bool]:
        agent_home = self._agent_home.path
        agent_home.mkdir(parents=True, exist_ok=True)
        agent_home_root = agent_home.resolve(strict=True)
        logs = agent_home / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        _validate_log_directory(logs, agent_home_root)
        if os.name == "posix":
            logs.chmod(0o700)
        slot_existed = any(
            (logs / name).exists() or (logs / name).is_symlink()
            for name in ("run.log.0", "run.log.1")
        )
        for name, content in {
            "run.log.0": b"",
            "run.log.1": b"",
            "run.log.lock": b"",
        }.items():
            path = logs / name
            if path.exists() or path.is_symlink():
                _validate_log_file(path, agent_home_root)
            else:
                atomic_create_bytes(path, content)
            _validate_log_file(path, agent_home_root)
            if os.name == "posix":
                path.chmod(0o600)
        cursor_path = logs / "run.log.cursor"
        cursor_recovered = False
        if cursor_path.exists() or cursor_path.is_symlink():
            _validate_log_file(cursor_path, agent_home_root)
        elif slot_existed:
            self._replace_bytes(cursor_path, b"0\n")
            cursor_recovered = True
        else:
            atomic_create_bytes(cursor_path, b"0\n")
        _validate_log_file(cursor_path, agent_home_root)
        if os.name == "posix":
            cursor_path.chmod(0o600)
        cursor = cursor_path.read_bytes()
        if cursor not in {b"0\n", b"1\n"}:
            self._replace_bytes(cursor_path, b"0\n")
            cursor = b"0\n"
            cursor_recovered = True
        if cursor == b"0\n":
            active_slot = 0
        else:
            active_slot = 1
        slot = logs / f"run.log.{active_slot}"
        if slot.stat().st_size >= _SLOT_TARGET_BYTES:
            active_slot = self._rotate(logs, active_slot)
            slot = logs / f"run.log.{active_slot}"
        return slot, cursor_recovered

    def _rotate(self, logs: Path, active_slot: int) -> int:
        next_slot = 1 - active_slot
        self._replace_bytes(logs / f"run.log.{next_slot}", b"")
        self._replace_bytes(logs / "run.log.cursor", f"{next_slot}\n".encode())
        return next_slot

    def _report_failure(self, error: Exception) -> None:
        if self._failure_reported:
            return
        self._failure_reported = True
        try:
            sys.stderr.write(f"Runtime Log failure: {type(error).__name__}\n")
            sys.stderr.flush()
        except Exception:
            pass

    def _redact(self, content: str) -> str:
        with self._api_key_lock:
            api_keys = self._api_keys
        for api_key in api_keys:
            content = content.replace(api_key, _REDACTED)
        content = _AUTHORIZATION.sub(lambda match: f"{match.group(1)} {_REDACTED}", content)
        content = _BEARER.sub(lambda match: f"{match.group(1)} {_REDACTED}", content)
        content = _API_KEY.sub(
            lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}", content
        )
        return _COOKIE.sub(
            lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}", content
        )


def install_runtime_logging(
    agent_home: AgentHome,
    *,
    replace_bytes: AtomicReplaceBytes = atomic_replace_bytes,
) -> RuntimeLogLifetime:
    """Install one process-local Runtime Log lifetime for an Agent Home."""
    return RuntimeLogLifetime(agent_home, replace_bytes=replace_bytes)


def _severity(level: int) -> str:
    return "ERROR" if level >= logging.ERROR else "WARNING"


def _visible(value: object) -> str:
    text = str(value)
    rendered: list[str] = []
    for character in text:
        codepoint = ord(character)
        if character == "\n":
            rendered.append("\\n")
        elif character == "\r":
            rendered.append("\\r")
        elif character == "\t":
            rendered.append("\\t")
        elif codepoint < 32 or 127 <= codepoint < 160:
            rendered.append(f"\\x{codepoint:02x}")
        else:
            rendered.append(character)
    return "".join(rendered)


def _exception_lines(record: logging.LogRecord) -> list[str]:
    if record.exc_info is None or record.exc_info[1] is None:
        return []
    return _render_exception(record.exc_info[1])


def _render_exception(error: BaseException) -> list[str]:
    lines: list[str] = []
    if error.__cause__ is not None:
        lines.extend(_render_exception(error.__cause__))
        lines.extend(
            (
                "",
                "The above exception was the direct cause of the following exception:",
                "",
            )
        )
    elif error.__context__ is not None and not error.__suppress_context__:
        lines.extend(_render_exception(error.__context__))
        lines.extend(
            (
                "",
                "During handling of the above exception, another exception occurred:",
                "",
            )
        )

    extracted = traceback.extract_tb(error.__traceback__)
    if extracted:
        lines.append("Traceback (most recent call last):")
        for frame in extracted:
            lines.append(
                _visible(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
            )
    qualified_type = type(error).__qualname__
    if type(error).__module__ not in {"builtins", "__main__"}:
        qualified_type = f"{type(error).__module__}.{qualified_type}"
    message = _visible(error)
    lines.append(f"{qualified_type}: {message}" if message else qualified_type)
    if isinstance(error, BaseExceptionGroup):
        total = len(error.exceptions)
        for index, nested in enumerate(error.exceptions, start=1):
            lines.append(f"+- sub-exception {index} of {total}:")
            lines.extend(f"  {line}" for line in _render_exception(nested))
    return lines


def _validate_log_directory(path: Path, agent_home_root: Path) -> None:
    try:
        status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PermissionError("Runtime Log directory is unavailable") from error
    if (
        path.is_symlink()
        or _is_reparse(status)
        or not S_ISDIR(status.st_mode)
        or not resolved.is_relative_to(agent_home_root)
    ):
        raise PermissionError("Runtime Log directory must remain unaliased inside Agent Home")


def _validate_log_file(path: Path, agent_home_root: Path) -> None:
    try:
        status = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PermissionError("Runtime Log file is unavailable") from error
    if (
        path.is_symlink()
        or _is_reparse(status)
        or not S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or not resolved.is_relative_to(agent_home_root)
    ):
        raise PermissionError("Runtime Log files must be unaliased inside Agent Home")


def _is_reparse(status: os.stat_result) -> bool:
    attributes = getattr(status, "st_file_attributes", 0)
    return bool(attributes & _WINDOWS_REPARSE_POINT)


def _fsync_file(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
