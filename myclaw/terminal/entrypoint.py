"""Platform-gated console entry point with lazy Runtime imports."""

from myclaw.platform_support import UnsupportedPlatformError, require_supported_platform


def _dispatch_cli() -> None:
    from myclaw.terminal.cli import app

    app()


def cli_entrypoint() -> int:
    """Reject unsupported hosts before importing Windows Runtime modules."""
    try:
        require_supported_platform()
    except UnsupportedPlatformError as error:
        print(f"{error.error.code}: {error.error.message}")
        return 2
    _dispatch_cli()
    return 0
