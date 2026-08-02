"""Command-line entry point for MyClaw."""

import asyncio
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import typer
from loguru import logger
from rich.console import Console

from myclaw.agent.runtime import (
    prepare_repl_runtime,
)
from myclaw.agent.workspace_state import WorkspaceStateError
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.provider.factory import create_provider
from myclaw.terminal.interrupts import ForegroundInterruptController
from myclaw.terminal.repl import ConsoleProgressiveWriter, ConsoleReplInput

app = typer.Typer(
    add_completion=False,
    help="MyClaw Personal Agent runtime.",
    rich_markup_mode="rich",
)
console = Console()


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _print_error_info(error: ErrorInfo) -> None:
    console.print(
        f"{error.code}: {error.message}",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def _print_error(error: ErrorInfo, path: object) -> None:
    _print_error_info(error)
    console.print(f"Path: {path}", markup=False, highlight=False, soft_wrap=True)


def _log_interrupt_cleanup_failure(error: BaseException) -> None:
    logger.opt(exception=error).error(
        "Interrupt controller cleanup failed type={}",
        type(error).__name__,
    )


@app.callback(invoke_without_command=True)
def main(context: typer.Context) -> None:
    """Start the MyClaw Personal Agent."""
    if context.invoked_subcommand is not None:
        return
    agent_home = AgentHome.production()
    loader = ConfigLoader(agent_home)
    try:
        configuration = loader.load_for_startup()
    except ConfigError as config_error:
        _print_error(config_error.error, loader.path)
        exit_code = 1 if config_error.error.code == "persistence_error" else 2
        raise typer.Exit(code=exit_code) from None
    except OSError:
        _print_error(
            ErrorInfo("persistence_error", "User Configuration could not be read or written."),
            loader.path,
        )
        raise typer.Exit(code=1) from None
    console.print("MyClaw Personal Agent configuration gate passed.")
    try:
        runtime = prepare_repl_runtime(
            agent_home=loader.agent_home,
            workspace=Path.cwd(),
            configuration=configuration,
            provider_factory=create_provider,
            now=_local_now,
            new_uuid=uuid4,
        )
    except WorkspaceStateError as workspace_state_error:
        _print_error(workspace_state_error.error, workspace_state_error.path)
        raise typer.Exit(code=1) from None
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
                    _log_interrupt_cleanup_failure(cleanup_error)
                    raise primary_error from cleanup_error
                raise
            else:
                try:
                    runner.run(interrupts.close())
                except BaseException as cleanup_error:
                    _log_interrupt_cleanup_failure(cleanup_error)
                    raise
        finally:
            interrupts.restore()


@app.command("config")
def config_command() -> None:
    """Display User Configuration with plaintext API keys redacted."""
    agent_home = AgentHome.production()
    loader = ConfigLoader(agent_home)
    try:
        loader.ensure_default()
        view = loader.view()
    except (OSError, UnicodeError):
        _print_error(
            ErrorInfo("persistence_error", "User Configuration could not be read or written."),
            loader.path,
        )
        raise typer.Exit(code=1) from None

    if view.error is not None:
        _print_error(view.error, view.path)
    else:
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
