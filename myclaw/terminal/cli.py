"""Command-line entry point for MyClaw."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import typer
from rich.console import Console

from myclaw.agent.runtime import (
    prepare_repl_runtime,
)
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader, UserConfiguration
from myclaw.errors import ErrorInfo
from myclaw.provider.factory import create_provider
from myclaw.runtime_log import (
    RuntimeLogLifetime,
    install_runtime_logging,
    log_sanitized_exception,
)
from myclaw.terminal.interrupts import ForegroundInterruptController
from myclaw.terminal.repl import ConsoleProgressiveWriter, ConsoleReplInput

app = typer.Typer(
    add_completion=False,
    help="MyClaw Personal Agent runtime.",
    rich_markup_mode="rich",
)
console = Console()
logger = logging.getLogger(__name__)


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _print_error(error: ErrorInfo, path: object) -> None:
    console.print(
        f"{error.code}: {error.message}",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )
    console.print(f"Path: {path}", markup=False, highlight=False, soft_wrap=True)


def _configure_runtime_log(
    runtime_log: RuntimeLogLifetime, configuration: object
) -> None:
    if not isinstance(configuration, UserConfiguration):
        return
    runtime_log.add_api_keys(
        provider.api_key for provider in configuration.models.providers.values()
    )


@app.callback(invoke_without_command=True)
def main(context: typer.Context) -> None:
    """Start the MyClaw Personal Agent."""
    if context.invoked_subcommand is not None:
        return
    agent_home = AgentHome.production()
    runtime_log = install_runtime_logging(agent_home)
    loader = ConfigLoader(agent_home)
    try:
        try:
            configuration = loader.load_for_startup()
        except ConfigError as config_error:
            log_sanitized_exception(
                logger,
                logging.ERROR,
                "Startup failed "
                f"code={config_error.error.code} type={type(config_error).__name__}",
                config_error,
            )
            _print_error(config_error.error, loader.path)
            exit_code = 1 if config_error.error.code == "persistence_error" else 2
            raise typer.Exit(code=exit_code) from None
        except OSError as error:
            log_sanitized_exception(
                logger,
                logging.ERROR,
                "Startup failed code=persistence_error",
                error,
            )
            _print_error(
                ErrorInfo("persistence_error", "User Configuration could not be read or written."),
                loader.path,
            )
            raise typer.Exit(code=1) from None
        _configure_runtime_log(runtime_log, configuration)
        console.print("MyClaw Personal Agent configuration gate passed.")
        runtime = prepare_repl_runtime(
            agent_home=loader.agent_home,
            workspace=Path.cwd(),
            configuration=configuration,
            provider_factory=create_provider,
            now=_local_now,
            new_uuid=uuid4,
            runtime_log=runtime_log,
        )
        with asyncio.Runner() as runner:
            interrupts = ForegroundInterruptController(
                loop=runner.get_loop(),
                cancel_foreground=runtime.conversation.cancel_active_turn,
            )
            interrupts.install()
            try:
                try:
                    runner.run(
                        runtime.run(
                            input_reader=ConsoleReplInput(console),
                            writer=ConsoleProgressiveWriter(console),
                        )
                    )
                except BaseException as primary_error:
                    try:
                        runner.run(interrupts.close())
                    except BaseException as cleanup_error:
                        log_sanitized_exception(
                            logger,
                            logging.ERROR,
                            "Interrupt controller cleanup failed "
                            f"type={type(cleanup_error).__name__}",
                            cleanup_error,
                        )
                        raise primary_error from cleanup_error
                    raise
                else:
                    try:
                        runner.run(interrupts.close())
                    except BaseException as cleanup_error:
                        log_sanitized_exception(
                            logger,
                            logging.ERROR,
                            "Interrupt controller cleanup failed "
                            f"type={type(cleanup_error).__name__}",
                            cleanup_error,
                        )
                        raise
            finally:
                interrupts.restore()
    finally:
        runtime_log.close()


@app.command("config")
def config_command() -> None:
    """Display User Configuration with plaintext API keys redacted."""
    agent_home = AgentHome.production()
    runtime_log = install_runtime_logging(agent_home)
    loader = ConfigLoader(agent_home)
    try:
        try:
            loader.ensure_default()
            view = loader.view()
            configuration = loader.load() if view.error is None else None
        except (OSError, UnicodeError) as error:
            log_sanitized_exception(
                logger,
                logging.ERROR,
                "Configuration command failed code=persistence_error",
                error,
            )
            _print_error(
                ErrorInfo("persistence_error", "User Configuration could not be read or written."),
                loader.path,
            )
            raise typer.Exit(code=1) from None

        if view.error is not None:
            logger.error("Configuration command failed code=%s", view.error.code)
            _print_error(view.error, view.path)
        else:
            _configure_runtime_log(runtime_log, configuration)
            console.print(
                f"Path: {view.path}",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )
        console.print(
            view.redacted_content,
            markup=False,
            highlight=False,
            soft_wrap=True,
            end="" if view.redacted_content.endswith("\n") else "\n",
        )
        if view.error is not None:
            raise typer.Exit(code=2)
    finally:
        runtime_log.close()
