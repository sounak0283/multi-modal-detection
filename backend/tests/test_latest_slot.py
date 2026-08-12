"""Tests for the drop-to-latest slot.

PLAN.md section 4 calls this the most important structural decision in the runtime: it
is what keeps alert latency bounded when inference falls behind. Its overwrite semantics
are therefore tested explicitly rather than assumed.
"""

from __future__ import annotations

import threading
import time

from fsbd.capture.latest_slot import LatestSlot


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
