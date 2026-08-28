"""Reusable AgentLoop public-seam helpers."""

from myclaw.agent.loop import AgentLoop
from myclaw.agent.message_bus import InboundMessage, OutboundMessage


async def collect_foreground_outbound(
    loop: AgentLoop,
    content: str,
) -> tuple[OutboundMessage, ...]:
    """Submit one foreground input and collect through its terminal marker."""
    await loop.bus.put_inbound(InboundMessage(content=content))
    messages: list[OutboundMessage] = []
    while True:
        message = await loop.bus.get_outbound()
        messages.append(message)
        if message.metadata.get("_streamed") is True:
            return tuple(messages)
