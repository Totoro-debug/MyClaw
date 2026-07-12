"""Concrete read-only views exposed through the Management Port."""

from myclaw.agent_home import AgentHome
from myclaw.config import ConfigLoader
from myclaw.contracts.errors import ErrorInfo
from myclaw.contracts.management import ConfigView


class ManagementError(Exception):
    """A safe persistence error suitable for a Management Command."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


class ManagementViewService:
    """Read configuration and Long-term Memory from Agent Home."""

    def __init__(self, agent_home: AgentHome) -> None:
        self._config = ConfigLoader(agent_home)
        self._long_term_memory = agent_home.path / "memory" / "memory.md"

    async def config_view(self) -> ConfigView:
        """Return complete redacted User Configuration content."""
        try:
            self._config.ensure_default()
            return self._config.view()
        except (OSError, UnicodeError):
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "User Configuration could not be read or written.",
                )
            ) from None

    async def memory_view(self) -> str:
        """Return the complete current Long-term Memory file."""
        try:
            return self._long_term_memory.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise ManagementError(
                ErrorInfo("persistence_error", "Long-term Memory could not be read.")
            ) from None
