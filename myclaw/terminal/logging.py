"""Loguru configuration for diagnostics without Session ownership."""

import sys

from loguru import logger


def _basic_terminal_format(_record: object) -> str:
    return "{message}\n"


def configure_process_logging() -> None:
    """Install the non-persistent logging baseline for this process."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="ERROR",
        format=_basic_terminal_format,
        colorize=False,
        backtrace=False,
        diagnose=False,
    )
