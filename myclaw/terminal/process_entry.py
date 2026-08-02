"""Minimal process entry that configures logging before application imports."""

from myclaw.logging.process import configure_process_logging


def run() -> None:
    """Configure process diagnostics, then invoke the command-line application."""
    configure_process_logging()

    from myclaw.terminal.cli import app

    app()
