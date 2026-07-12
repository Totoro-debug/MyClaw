"""Command-line entry point for MyClaw."""

import typer
from rich.console import Console

from myclaw.agent_home import AgentHome
from myclaw.config import ConfigError, ConfigLoader
from myclaw.contracts.errors import ErrorInfo

app = typer.Typer(
    add_completion=False,
    help="MyClaw Personal Agent runtime.",
    rich_markup_mode="rich",
)
console = Console()


def _print_error(error: ErrorInfo, path: object) -> None:
    console.print(
        f"{error.code}: {error.message}",
        markup=False,
        highlight=False,
        soft_wrap=True,
    )
    console.print(f"Path: {path}", markup=False, highlight=False, soft_wrap=True)


@app.callback(invoke_without_command=True)
def main(context: typer.Context) -> None:
    """Start the MyClaw Personal Agent."""
    if context.invoked_subcommand is not None:
        return
    loader = ConfigLoader(AgentHome.production())
    try:
        loader.load_for_startup()
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


@app.command("config")
def config_command() -> None:
    """Display User Configuration with plaintext API keys redacted."""
    loader = ConfigLoader(AgentHome.production())
    try:
        loader.ensure_default()
        view = loader.view()
    except OSError:
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
