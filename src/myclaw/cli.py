"""Command-line entry point for MyClaw."""

import typer
from rich.console import Console

app = typer.Typer(
    add_completion=False,
    help="MyClaw Personal Agent runtime.",
    rich_markup_mode="rich",
)
console = Console()


@app.callback(invoke_without_command=True)
def main() -> None:
    """Start the MyClaw Personal Agent."""
    console.print("MyClaw Personal Agent")
