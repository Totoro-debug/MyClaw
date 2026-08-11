"""Foreground-only SIGINT coordination retained for the headless Runtime seam."""

import asyncio
import signal
from collections.abc import Awaitable, Callable
from types import FrameType
from typing import Protocol, cast

SignalHandler = Callable[[int, FrameType | None], None]
SignalDisposition = SignalHandler | int


class SignalSetter(Protocol):
    def __call__(self, signum: int, handler: SignalDisposition) -> SignalDisposition: ...


_set_signal = cast(SignalSetter, signal.signal)


class ForegroundInterruptController:
    """Route every SIGINT to the active foreground turn without touching Runtime lifetime."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        cancel_foreground: Callable[[], Awaitable[None]],
        set_signal: SignalSetter = _set_signal,
    ) -> None:
        self._loop = loop
        self._cancel_foreground = cancel_foreground
        self._set_signal = set_signal
        self._previous: SignalDisposition | None = None
        self._pending: set[asyncio.Task[None]] = set()
        self._failures: list[BaseException] = []
        self._closing = False

    def install(self) -> None:
        if self._previous is not None:
            raise RuntimeError("SIGINT handler is already installed")
        self._previous = self._set_signal(signal.SIGINT, self._handle)

    def restore(self) -> None:
        previous = self._previous
        if previous is None:
            return
        self._previous = None
        self._set_signal(signal.SIGINT, previous)

    async def close(self) -> None:
        self._closing = True
        pending = tuple(self._pending)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if len(self._failures) == 1:
            raise self._failures[0]
        if self._failures:
            raise BaseExceptionGroup("Foreground interrupt cleanup failed", self._failures)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        del signum, frame
        self._loop.call_soon_threadsafe(self._schedule_cancel)

    def _schedule_cancel(self) -> None:
        if self._closing or self._pending:
            return
        task = self._loop.create_task(self._run_cancel())
        self._pending.add(task)
        task.add_done_callback(self._cancel_finished)

    async def _run_cancel(self) -> None:
        await self._cancel_foreground()

    def _cancel_finished(self, task: asyncio.Task[None]) -> None:
        self._pending.discard(task)
        if task.cancelled():
            return
        failure = task.exception()
        if failure is not None:
            self._failures.append(failure)
