"""Process-level logging contract before any Conversation Session exists."""

import sys
from collections.abc import Iterator

import pytest
from loguru import logger

from myclaw.logging.process import configure_process_logging
from myclaw.terminal.process_entry import run


@pytest.fixture(autouse=True)
def _remove_process_logging_handlers() -> Iterator[None]:
    logger.remove()
    logger.add(sys.stderr)
    yield
    logger.remove()


def test_process_logging_silences_records_below_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_process_logging()

    logger.debug("debug detail")
    logger.info("ordinary progress")
    logger.warning("recoverable condition")

    assert capsys.readouterr().err == ""


def test_process_logging_emits_one_basic_error_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_process_logging()

    try:
        raise RuntimeError("technical failure")
    except RuntimeError as error:
        logger.opt(exception=error).error("Startup failed")

    assert capsys.readouterr().err == "Startup failed\n"


def test_process_logging_emits_critical_messages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_process_logging()

    logger.critical("Process cannot continue")

    assert capsys.readouterr().err == "Process cannot continue\n"


def test_process_logging_configuration_is_repeatable_without_duplicate_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_process_logging()
    configure_process_logging()

    logger.error("One diagnostic")

    assert capsys.readouterr().err == "One diagnostic\n"


def test_process_entry_configures_logging_on_eager_help_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["myclaw", "--help"])

    with pytest.raises(SystemExit) as exited:
        run()
    logger.warning("must remain silent")

    assert exited.value.code == 0
    assert capsys.readouterr().err == ""
