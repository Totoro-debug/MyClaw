"""Small Loguru capture used by unit tests that assert diagnostic content."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from loguru import logger

from myclaw.config.agent_home import AgentHome
from myclaw.logging.process import configure_process_logging

if TYPE_CHECKING:
    from loguru import Message, Record


class LogCapture:
    """Capture WARNING+ records without recreating the removed runtime logger."""

    def __init__(self, agent_home: AgentHome) -> None:
        self._path = agent_home.path / "logs" / "run.log.0"
        self._handler_id = logger.add(
            self._write,
            level="WARNING",
            format=self._format,
            catch=False,
            backtrace=False,
            diagnose=False,
        )

    @contextmanager
    def session(self, session_id: str) -> Iterator[None]:
        with logger.contextualize(session_id=session_id):
            yield

    def add_api_keys(self, values: Iterable[str]) -> None:
        del values

    def close(self) -> None:
        logger.remove(self._handler_id)

    def _write(self, message: Message) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(str(message))
            stream.write(_render_exception(message.record.get("exception")))

    @staticmethod
    def _format(record: Record) -> str:
        extra = record["extra"]
        assert isinstance(extra, dict)
        session_id = extra.get("session_id", "-")
        return (
            "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} {level} pid={process} "
            f"session={session_id} "
            "{name}: {message}\n"
        )


def install_log_capture(agent_home: AgentHome) -> LogCapture:
    return LogCapture(agent_home)


@contextmanager
def configured_process_logging() -> Iterator[None]:
    configure_process_logging()
    try:
        yield
    finally:
        logger.remove()


def _render_exception(exception: object) -> str:
    if not isinstance(exception, tuple) or len(exception) < 2:
        return ""
    error = exception[1]
    if not isinstance(error, BaseException):
        return ""

    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__
        if current is None and not chain[-1].__suppress_context__:
            current = chain[-1].__context__
    chain.reverse()

    rendered = "Traceback (most recent call last):\n"
    for index, item in enumerate(chain):
        if index:
            rendered += (
                "\nThe above exception was the direct cause of the following exception:\n\n"
            )
        rendered += f"{type(item).__name__}: [REDACTED]\n"
    return rendered
