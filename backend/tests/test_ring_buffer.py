"""Tests for the evidence pre-roll ring buffer (Expansion Plan Phase C)."""

from __future__ import annotations

from perimeter.capture.ring_buffer import JpegRingBuffer


def test_append_and_snapshot_returns_oldest_first():
    buf = JpegRingBuffer(seconds=10)
    buf.append(1.0, b"a")
    buf.append(2.0, b"b")
    buf.append(3.0, b"c")

    assert buf.snapshot() == [(1.0, b"a"), (2.0, b"b"), (3.0, b"c")]
    assert len(buf) == 3


def test_frames_older_than_the_window_are_evicted():
    buf = JpegRingBuffer(seconds=5)
    buf.append(0.0, b"old")
    buf.append(10.0, b"new")  # 10s later - older than the 5s window relative to "now"

    assert buf.snapshot() == [(10.0, b"new")]


def test_eviction_is_relative_to_the_latest_append_not_wall_clock():
    """The buffer has no wall-clock timer of its own - it only ever looks stale relative
    to whatever timestamp it was last told about."""
    buf = JpegRingBuffer(seconds=2)
    buf.append(100.0, b"a")
    buf.append(100.5, b"b")
    buf.append(101.0, b"c")

    assert buf.snapshot() == [(100.0, b"a"), (100.5, b"b"), (101.0, b"c")]


def test_empty_buffer_snapshot_and_closest_to():
    buf = JpegRingBuffer(seconds=15)
    assert buf.snapshot() == []
    assert buf.closest_to(5.0) is None
    assert len(buf) == 0


def test_closest_to_picks_the_nearest_timestamp():
    buf = JpegRingBuffer(seconds=15)
    buf.append(1.0, b"a")
    buf.append(5.0, b"b")
    buf.append(9.0, b"c")

    assert buf.closest_to(0.0) == b"a"
    assert buf.closest_to(5.2) == b"b"
    assert buf.closest_to(100.0) == b"c"
    assert buf.closest_to(3.0) == b"a"  # tied at 2.0 away from both a and b - picks a


def test_appends_out_of_time_order_do_not_evict_everything():
    """Frames from a single camera should always arrive in ts order in practice, but the
    eviction rule (drop anything older than `latest_ts - seconds`) must not misbehave if
    one somehow arrives late."""
    buf = JpegRingBuffer(seconds=10)
    buf.append(5.0, b"a")
    buf.append(3.0, b"b")  # earlier than the previous append

    assert buf.snapshot() == [(5.0, b"a"), (3.0, b"b")]
