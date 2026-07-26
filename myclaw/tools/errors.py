"""Public-safe failures raised by concrete Tools."""


class ToolError(Exception):
    """An expected Tool failure whose message is safe to return to the model."""

    def __init__(self, message: str) -> None:
        if not isinstance(message, str):
            msg = "ToolError message must be a string"
            raise TypeError(msg)
        self.message = message
        super().__init__(message)
