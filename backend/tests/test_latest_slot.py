"""Tests for the drop-to-latest slot.

PLAN.md section 4 calls this the most important structural decision in the runtime: it
is what keeps alert latency bounded when inference falls behind. Its overwrite semantics
are therefore tested explicitly rather than assumed.
"""

from __future__ import annotations

import threading
import time

from perimeter.capture.latest_slot import LatestSlot, SlotStats


def test_publish_then_get():
    slot: LatestSlot[str] = LatestSlot()
    slot.publish("a")
    assert slot.get(timeout=0.1) == "a"


def test_slot_is_emptied_by_get():
    slot: LatestSlot[str] = LatestSlot()
    slot.publish("a")
    slot.get(timeout=0.1)
    assert slot.get_nowait() is None


def test_publish_overwrites_and_counts_the_drop():
    """The whole point: a slow consumer sees the newest frame, not a backlog."""
    slot: LatestSlot[int] = LatestSlot()
    for i in range(5):
        slot.publish(i)

    assert slot.get_nowait() == 4, "consumer must receive the freshest item"
    stats = slot.stats()
    assert stats.published == 5
    assert stats.dropped == 4
    assert stats.consumed == 1
    assert stats.drop_rate == 0.8


def test_no_drops_when_consumer_keeps_up():
    slot: LatestSlot[int] = LatestSlot()
    for i in range(5):
        slot.publish(i)
        assert slot.get_nowait() == i
    assert slot.stats().dropped == 0


def test_get_times_out_when_empty():
    slot: LatestSlot[int] = LatestSlot()
    start = time.monotonic()
    assert slot.get(timeout=0.05) is None
    assert time.monotonic() - start >= 0.04


def test_get_blocks_until_published():
    slot: LatestSlot[str] = LatestSlot()
    received: list[str | None] = []

    consumer = threading.Thread(target=lambda: received.append(slot.get(timeout=2.0)))
    consumer.start()
    time.sleep(0.05)
    slot.publish("late")
    consumer.join(timeout=2.0)

    assert received == ["late"]


def test_close_wakes_waiting_consumer():
    """Shutdown must not hang on a consumer parked in get()."""
    slot: LatestSlot[str] = LatestSlot()
    received: list[str | None] = []

    consumer = threading.Thread(target=lambda: received.append(slot.get(timeout=5.0)))
    consumer.start()
    time.sleep(0.05)

    start = time.monotonic()
    slot.close()
    consumer.join(timeout=2.0)

    assert not consumer.is_alive(), "close() must unblock the consumer"
    assert received == [None]
    assert time.monotonic() - start < 1.0
    assert slot.closed


# -- blocking mode (the Video test page's non-realtime capture only) -----------------


def test_blocking_publish_waits_instead_of_dropping():
    slot: LatestSlot[int] = LatestSlot(blocking=True)
    slot.publish(1)
    order: list[str] = []

    def publish_second():
        slot.publish(2)  # must wait for the first item to be consumed
        order.append("published")

    publisher = threading.Thread(target=publish_second)
    publisher.start()
    time.sleep(0.1)
    assert not order, "publish(2) must still be waiting - the slot is not empty yet"

    order.append("about to consume")
    assert slot.get_nowait() == 1
    publisher.join(timeout=2.0)

    assert order == ["about to consume", "published"]
    assert slot.get_nowait() == 2
    assert slot.stats().dropped == 0


def test_blocking_mode_never_drops_a_burst_of_publishes():
    """The whole point, mirrored against the non-blocking test above: nothing published
    is ever silently discarded, only delayed until the consumer catches up."""
    slot: LatestSlot[int] = LatestSlot(blocking=True)
    received: list[int] = []

    def consume():
        while len(received) < 5:
            item = slot.get(timeout=2.0)
            if item is not None:
                received.append(item)

    consumer = threading.Thread(target=consume)
    consumer.start()
    for i in range(5):
        slot.publish(i)
    consumer.join(timeout=2.0)

    assert received == [0, 1, 2, 3, 4]
    assert slot.stats() == SlotStats(published=5, consumed=5, dropped=0)


def test_close_wakes_a_publisher_blocked_waiting_for_room():
    """Shutdown must not hang on a producer parked in publish() either."""
    slot: LatestSlot[str] = LatestSlot(blocking=True)
    slot.publish("a")  # fills the slot; a second publish must block
    done = threading.Event()

    def publish_second():
        slot.publish("b")
        done.set()

    publisher = threading.Thread(target=publish_second)
    publisher.start()
    time.sleep(0.05)
    assert not done.is_set()

    start = time.monotonic()
    slot.close()
    publisher.join(timeout=2.0)

    assert not publisher.is_alive(), "close() must unblock a waiting publisher"
    assert time.monotonic() - start < 1.0
    # "b" was never actually published - close() made publish() give up and return.
    assert slot.get_nowait() == "a"


def test_concurrent_publishers_do_not_lose_the_invariant():
    slot: LatestSlot[int] = LatestSlot()
    stop = threading.Event()

    def produce():
        while not stop.is_set():
            slot.publish(1)

    threads = [threading.Thread(target=produce) for _ in range(4)]
    for t in threads:
        t.start()

    consumed = 0
    for _ in range(50):
        if slot.get(timeout=0.05) is not None:
            consumed += 1
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    stats = slot.stats()
    # Every published item is either consumed or counted as dropped, minus at most one
    # still sitting in the slot.
    assert stats.published - stats.dropped - stats.consumed in (0, 1)
    assert consumed > 0
