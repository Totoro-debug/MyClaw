"""Non-persisted Memory Task result values."""

from dataclasses import dataclass

from myclaw.errors import ErrorInfo
from myclaw.utils.validation import require_nonnegative_int


@dataclass(frozen=True, slots=True)
class MemoryTaskResult:
    """Observable summary returned by a manual Memory Task run."""

    status: str
    processed_count: int
    memory_updated: bool
    cursor: int
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        require_nonnegative_int(self.processed_count, field="processed_count")
        require_nonnegative_int(self.cursor, field="cursor")
        if not isinstance(self.memory_updated, bool):
            msg = "memory_updated must be a boolean"
            raise ValueError(msg)
