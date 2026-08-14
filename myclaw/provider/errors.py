"""Normalized Model Provider failures."""

from myclaw.errors import ErrorInfo


class EmptyModelResponseError(ValueError):
    """A successful Model Response contained neither text nor Tool calls."""


class ModelCallError(Exception):
    """A normalized Model Provider failure handled by the Model Router."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)
