"""Traceback-preserving diagnostics without exception payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

__all__ = ["exception_logger"]

_REDACTED = "[REDACTED]"


def exception_logger(error: BaseException) -> Logger:
    """Bind a traceback-preserving, message-redacted exception to Loguru."""
    return logger.opt(exception=_redacted_exception(error, seen=set()))


def _redacted_exception(error: BaseException, *, seen: set[int]) -> BaseException:
    identity = id(error)
    if identity in seen:
        return RuntimeError(_REDACTED)
    seen.add(identity)

    redacted = (
        _redacted_exception_group(error, seen=seen)
        if isinstance(error, BaseExceptionGroup)
        else _empty_exception_of_same_type(error)
    )
    redacted.__traceback__ = error.__traceback__
    redacted.__suppress_context__ = error.__suppress_context__
    if error.__cause__ is not None:
        redacted.__cause__ = _redacted_exception(error.__cause__, seen=seen)
    elif error.__context__ is not None and not error.__suppress_context__:
        redacted.__context__ = _redacted_exception(error.__context__, seen=seen)
    return redacted


def _redacted_exception_group(
    error: BaseExceptionGroup[BaseException],
    *,
    seen: set[int],
) -> BaseException:
    children = tuple(_redacted_exception(child, seen=seen) for child in error.exceptions)
    try:
        return type(error)(_REDACTED, children)
    except Exception:
        return BaseExceptionGroup(_REDACTED, children)


def _empty_exception_of_same_type(error: BaseException) -> BaseException:
    try:
        redacted = type(error).__new__(type(error))
        redacted.args = (_REDACTED,)
        str(redacted)
    except Exception:
        return RuntimeError(f"{type(error).__name__}: {_REDACTED}")
    return redacted
