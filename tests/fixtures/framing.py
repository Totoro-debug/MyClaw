import asyncio

from myclaw.agent.blackboard import Blackboard, FramingResult


class DeterministicTaskFramingEvaluator:
    """Resolve to an empty Blackboard with a neutral usage delta."""

    def __init__(self) -> None:
        self.calls = 0

    async def frame(
        self,
        *,
        previous: Blackboard | None,
        last_assistant_content: str,
        current_user_input: str,
    ) -> FramingResult:
        del previous, last_assistant_content, current_user_input
        self.calls += 1
        return FramingResult(
            blackboard=None,
            usage_delta={
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            status="resolved",
        )


class BlockingTaskFramingEvaluator:
    """Block until cancellation and expose deterministic lifecycle handshakes."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def frame(
        self,
        *,
        previous: Blackboard | None,
        last_assistant_content: str,
        current_user_input: str,
    ) -> FramingResult:
        del previous, last_assistant_content, current_user_input
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")
