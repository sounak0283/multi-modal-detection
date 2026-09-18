"""Tests for EvidenceWriter (Expansion Plan Phase C).

Every test here injects a fake `mux` function instead of the real `mux_clip` (which opens
a real `cv2.VideoWriter`) - deliberately, so EvidenceWriter's own orchestration logic
(post-roll timing, per-camera lookup, graceful degradation, snapshot-vs-clip
independence) is tested without depending on real codec timing. `test_camera_api.py`'s
unreachable-RTSP-host probes are known to leave OpenCV's FFmpeg backend in a state that
stalls the *next* cv2 video open in the same test process - confirmed, while building
this file, to affect `cv2.VideoWriter` too (not just `cv2.VideoCapture`) and to run well
over test_camera_api.py's own documented ~30s figure. See the bottom of this file for
why `mux_clip`'s real codec path is deliberately not exercised by a test at all.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from perimeter.capture.ring_buffer import JpegRingBuffer
from perimeter.evidence.store import LocalEvidenceStore
from perimeter.evidence.writer import EvidenceWriter, mux_clip


class FakeDatabase:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def update_evidence_paths(self, event_id, snapshot_path=None, clip_path=None):
        with self.lock:
            self.calls.append(
                {"event_id": event_id, "snapshot_path": snapshot_path, "clip_path": clip_path}
            )


def jpeg(colour: int = 0) -> bytes:
    image = np.full((60, 80, 3), colour, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def filled_ring_buffer(n: int = 6, start_ts: float = 100.0, step: float = 0.5) -> JpegRingBuffer:
    buf = JpegRingBuffer(seconds=60)
    for i in range(n):
        buf.append(start_ts + i * step, jpeg(colour=i * 10 % 256))
    return buf


def fake_mux(frames: list[tuple[float, bytes]], event_id: str, tmp_dir: Path) -> Path | None:
    """Same contract as `mux_clip` (None if fewer than 2 frames, else a Path under
    `tmp_dir`), but instant and with no real codec involved."""
    if len(frames) < 2:
        return None
    path = tmp_dir / f"{event_id}.mp4"
    path.write_bytes(b"fake-clip-bytes")
    return path


def make_writer(
    tmp_path, database, ring_buffers, post_roll_seconds=0.0, mux=fake_mux
) -> EvidenceWriter:
    store = LocalEvidenceStore(
        root=tmp_path / "evidence", snapshot_dir="snapshots", clip_dir="clips"
    )
    return EvidenceWriter(
        database=database,
        store=store,
        ring_buffer_lookup=lambda cam_id: ring_buffers.get(cam_id),
        post_roll_seconds=post_roll_seconds,
        mux=mux,
    )


def alert(event_id="evt1", camera_id="cam_01", ts=101.5) -> dict:
    return {"id": event_id, "camera_id": camera_id, "ts": ts, "kind": "boundary"}


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# -- happy path ---------------------------------------------------------------


def test_produces_a_clip_and_snapshot_and_updates_the_database(tmp_path):
    database = FakeDatabase()
    ring_buffers = {"cam_01": filled_ring_buffer()}
    writer = make_writer(tmp_path, database, ring_buffers)
    writer.start()
    try:
        writer.on_alert(alert())
        assert wait_for(lambda: database.calls)
    finally:
        writer.stop()

    call = database.calls[0]
    assert call["event_id"] == "evt1"
    assert call["snapshot_path"] is not None
    assert call["clip_path"] is not None
    assert (tmp_path / "evidence" / call["snapshot_path"]).is_file()
    assert (tmp_path / "evidence" / call["clip_path"]).is_file()


def test_snapshot_picked_is_closest_to_the_alert_timestamp(tmp_path):
    database = FakeDatabase()
    ring_buffers = {"cam_01": filled_ring_buffer(n=6, start_ts=100.0, step=1.0)}
    writer = make_writer(tmp_path, database, ring_buffers)
    writer.start()
    try:
        writer.on_alert(alert(ts=102.9))  # closest to frame at ts=103.0
        assert wait_for(lambda: database.calls)
    finally:
        writer.stop()

    stored = (tmp_path / "evidence" / database.calls[0]["snapshot_path"]).read_bytes()
    expected = ring_buffers["cam_01"].closest_to(102.9)
    assert stored == expected


# -- graceful degradation ------------------------------------------------------


def test_unknown_camera_is_skipped_without_crashing(tmp_path):
    database = FakeDatabase()
    writer = make_writer(tmp_path, database, ring_buffers={})  # no ring buffer for any camera
    writer.start()
    try:
        writer.on_alert(alert(camera_id="cam_gone"))
        time.sleep(0.3)  # give the worker a chance to (not) do anything
    finally:
        writer.stop()

    assert database.calls == []


def test_empty_ring_buffer_is_skipped_without_crashing(tmp_path):
    database = FakeDatabase()
    ring_buffers = {"cam_01": JpegRingBuffer(seconds=15)}  # empty
    writer = make_writer(tmp_path, database, ring_buffers)
    writer.start()
    try:
        writer.on_alert(alert())
        time.sleep(0.3)
    finally:
        writer.stop()

    assert database.calls == []


def test_a_single_buffered_frame_produces_a_snapshot_but_no_clip(tmp_path):
    """A clip needs >= 2 frames to derive a frame rate from; one frame is still enough
    for a snapshot."""
    database = FakeDatabase()
    buf = JpegRingBuffer(seconds=15)
    buf.append(100.0, jpeg())
    writer = make_writer(tmp_path, database, {"cam_01": buf})
    writer.start()
    try:
        writer.on_alert(alert(ts=100.0))
        assert wait_for(lambda: database.calls)
    finally:
        writer.stop()

    call = database.calls[0]
    assert call["snapshot_path"] is not None
    assert call["clip_path"] is None


def test_a_clip_failure_does_not_cost_the_snapshot(tmp_path):
    def broken_mux(frames, event_id, tmp_dir):
        raise RuntimeError("no usable codec")

    database = FakeDatabase()
    writer = make_writer(tmp_path, database, {"cam_01": filled_ring_buffer()}, mux=broken_mux)
    writer.start()
    try:
        writer.on_alert(alert())
        assert wait_for(lambda: database.calls)
    finally:
        writer.stop()

    call = database.calls[0]
    assert call["snapshot_path"] is not None
    assert call["clip_path"] is None


def test_payload_without_an_id_is_never_enqueued(tmp_path):
    """AlertBus only calls sinks after persistence assigns payload['id'] - but if
    persistence itself failed, there is nothing to attach evidence to yet."""
    database = FakeDatabase()
    writer = make_writer(tmp_path, database, {"cam_01": filled_ring_buffer()})
    writer.on_alert({"camera_id": "cam_01", "ts": 100.0})  # no "id"

    assert writer._queue.qsize() == 0


# -- post-roll timing -----------------------------------------------------------


def test_processing_waits_for_the_post_roll_window(tmp_path):
    database = FakeDatabase()
    ring_buffers = {"cam_01": filled_ring_buffer()}
    writer = make_writer(tmp_path, database, ring_buffers, post_roll_seconds=0.3)
    writer.start()
    try:
        writer.on_alert(alert())
        time.sleep(0.1)
        assert database.calls == [], "should not have processed before the post-roll window"
        assert wait_for(lambda: database.calls)
    finally:
        writer.stop()


def test_stop_does_not_hang_waiting_out_a_long_post_roll(tmp_path):
    writer = make_writer(
        tmp_path, FakeDatabase(), {"cam_01": filled_ring_buffer()}, post_roll_seconds=30.0
    )
    writer.start()
    writer.on_alert(alert())

    started = time.monotonic()
    writer.stop(timeout=2.0)
    assert time.monotonic() - started < 2.5


# -- the real mux_clip, on its cheap path only -----------------------------------
#
# `mux_clip` opening a real `cv2.VideoWriter` is deliberately NOT exercised by a test
# in this file: `test_camera_api.py`'s unreachable-RTSP-host probes are known to leave
# OpenCV's FFmpeg backend stalling the next cv2 video open in the same test process for
# well over a minute (confirmed while writing this file - worse than that file's own
# documented ~30s figure), and no existing test in this codebase exercises
# `record.py:open_writer` with a real codec either, for the same reason. `mux_clip`
# reuses that same, already-production-proven helper; correctness of the actual mux is
# checked manually (Phase C's verification runs a pipeline against a real video file and
# confirms a real clip appears - see docs/phases/PHASE_C.md), not by a unit test that
# would inherit this suite's cross-file cv2 timing hazard.


def test_mux_clip_returns_none_for_fewer_than_two_frames(tmp_path):
    """The one branch of the real `mux_clip` that returns before ever touching cv2's
    VideoWriter, so it is safe to exercise directly."""
    assert mux_clip([], "evt", tmp_path) is None
    assert mux_clip([(1.0, jpeg())], "evt", tmp_path) is None
