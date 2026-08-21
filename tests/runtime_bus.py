from __future__ import annotations

from myclaw.agent.message_bus import InboundMessage, OutboundMessage
from myclaw.agent.runtime import PreparedRuntime


async def collect_foreground_outbound(
    runtime: PreparedRuntime,
    content: str,
) -> tuple[OutboundMessage, ...]:
    """Submit one already-started foreground turn through the public bus seam."""

    await runtime.bus.put_inbound(InboundMessage(content=content))
    return await collect_outbound_until_terminal(runtime)


async def collect_outbound_until_terminal(
    runtime: PreparedRuntime,
) -> tuple[OutboundMessage, ...]:
    """Drain one foreground run until its single whole-run marker."""

    messages: list[OutboundMessage] = []
    while True:
        message = await runtime.bus.get_outbound()
        messages.append(message)
        if message.metadata.get("_streamed") is True:
            return tuple(messages)


def terminal_reason(message: OutboundMessage) -> str | None:
    """Return the finish reason for a terminal Outbound message, if present."""

    if message.metadata.get("_streamed") is not True:
        return None
    return (
        message.metadata.get("finish_reason") if message.type == "system_control" else "completed"
    )
