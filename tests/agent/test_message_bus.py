import asyncio
from io import StringIO

import pytest
from loguru import logger

from myclaw.agent.message_bus import (
    InboundMessage,
    MessageBus,
    OutboundMessage,
)


def test_message_metadata_defaults_are_independent_and_outbound_types_are_supported() -> None:
    first_inbound = InboundMessage(content="first")
    second_inbound = InboundMessage(content="second")
    first_inbound.metadata["source"] = "test"

    assert second_inbound.metadata == {}

    outbound = [
        OutboundMessage(type="model_reasoning", content="thinking"),
        OutboundMessage(type="model_response", content="answer"),
        OutboundMessage(type="tool_call", content="read_file"),
        OutboundMessage(type="system_control", content="cancelled"),
    ]
    assert [message.type for message in outbound] == [
        "model_reasoning",
        "model_response",
        "tool_call",
        "system_control",
    ]


@pytest.mark.asyncio
async def test_inbound_snapshot_put_get_and_drain_publish_post_mutation_snapshots() -> None:
    bus = MessageBus()
    observed: list[tuple[InboundMessage, ...]] = []

    def observe(snapshot: tuple[InboundMessage, ...]) -> None:
        # Lock release is a required callback ordering contract with no public probe.
        assert not bus._inbound_condition.locked()
        observed.append(snapshot)

    bus.set_inbound_changed_callback(observe)
    first = InboundMessage(content="first")
    second = InboundMessage(content="second")

    assert await bus.inbound_snapshot() == ()
    assert observed == []

    await bus.put_inbound(first)
    assert list(observed) == [(first,)]
    assert await bus.inbound_snapshot() == (first,)
    assert list(observed) == [(first,)]

    await bus.put_inbound(second)
    assert list(observed[-1]) == [first, second]
    assert await bus.get_inbound() is first
    assert list(observed[-1]) == [second]

    assert await bus.drain_inbound() == (second,)
    assert list(observed[-1]) == []
    assert await bus.inbound_snapshot() == ()


@pytest.mark.asyncio
async def test_empty_drain_still_reports_the_post_operation_snapshot() -> None:
    bus = MessageBus()
    observed: list[tuple[InboundMessage, ...]] = []
    bus.set_inbound_changed_callback(observed.append)

    assert await bus.drain_inbound() == ()
    assert observed == [()]


@pytest.mark.asyncio
async def test_inbound_callback_can_be_replaced_or_cleared() -> None:
    bus = MessageBus()
    first_observed: list[tuple[InboundMessage, ...]] = []
    second_observed: list[tuple[InboundMessage, ...]] = []
    first = InboundMessage(content="first")
    second = InboundMessage(content="second")

    bus.set_inbound_changed_callback(first_observed.append)
    bus.set_inbound_changed_callback(second_observed.append)
    await bus.put_inbound(first)
    bus.set_inbound_changed_callback(None)
    await bus.put_inbound(second)

    assert first_observed == []
    assert second_observed == [(first,)]


@pytest.mark.asyncio
async def test_callback_failure_is_logged_and_does_not_roll_back_queue_state() -> None:
    bus = MessageBus()

    def fail(_snapshot: tuple[InboundMessage, ...]) -> None:
        raise RuntimeError("callback failure")

    bus.set_inbound_changed_callback(fail)
    message = InboundMessage(content="survives")
    output = StringIO()
    handler_id = logger.add(output, format="{message}", level="ERROR")

    try:
        await bus.put_inbound(message)
    finally:
        logger.remove(handler_id)

    assert await bus.inbound_snapshot() == (message,)
    assert "Inbound changed callback failed" in output.getvalue()


@pytest.mark.asyncio
async def test_blocked_inbound_get_resumes_when_a_message_arrives() -> None:
    bus = MessageBus()
    get_task = asyncio.create_task(bus.get_inbound())
    await asyncio.sleep(0)
    assert not get_task.done()

    message = InboundMessage(content="released")
    await bus.put_inbound(message)

    assert await get_task is message


@pytest.mark.asyncio
async def test_concurrent_inbound_puts_are_consumed_in_their_actual_admission_order() -> None:
    bus = MessageBus()
    snapshots: list[tuple[InboundMessage, ...]] = []
    bus.set_inbound_changed_callback(snapshots.append)
    all_ready = asyncio.Event()
    start = asyncio.Event()
    ready_count = 0

    async def put_message(message: InboundMessage) -> None:
        nonlocal ready_count
        ready_count += 1
        if ready_count == 10:
            all_ready.set()
        await start.wait()
        await bus.put_inbound(message)

    messages = [InboundMessage(content=str(index)) for index in range(10)]
    tasks = [asyncio.create_task(put_message(message)) for message in messages]
    await all_ready.wait()
    start.set()
    await asyncio.gather(*tasks)

    admitted = [snapshot[-1] for snapshot in snapshots]
    drained = await bus.drain_inbound()

    assert len(admitted) == 10
    assert len({message.content for message in admitted}) == 10
    assert drained == tuple(admitted)
    assert len({message.content for message in drained}) == 10


@pytest.mark.asyncio
async def test_outbound_messages_are_an_unbounded_single_consumer_fifo() -> None:
    bus = MessageBus()
    messages = [OutboundMessage(type="model_response", content=str(index)) for index in range(100)]

    for message in messages:
        await bus.put_outbound(message)

    for message in messages:
        assert await bus.get_outbound() is message
