"""EvidenceWriter (Expansion Plan Phase C).

Registered as a sink on `AlertBus` (`alerts/bus.py`'s existing `sinks` hook - no new hook
was needed). On every alert it waits `post_roll_seconds` - so the still-filling ring
buffer accumulates some "after" footage - then muxes whatever is currently buffered for
that camera into a clip and picks the frame closest to the alert's own timestamp as the
snapshot. Mirrors `AlertBus`'s own queue-plus-background-thread shape deliberately: a
new detection module never needs its own alerting code, and evidence capture never needs
its own concurrency model either.

Failure policy, same as the rest of this codebase: one bad job (camera gone, disk full,
no usable codec) must not kill the thread or block the next alert's evidence.

The actual `cv2.VideoWriter` mux (`mux_clip` below) is passed into `EvidenceWriter` as an
injectable `mux` callable rather than called directly, so tests can substitute a fast
fake instead of opening a real codec - this matters because `test_camera_api.py`'s
unreachable-RTSP-host probes are known to leave OpenCV's FFmpeg backend in a state that
stalls the *next* cv2 video open in the same process for up to ~30s (see that file's
module docstring); that is a real, separate finding worth a follow-up (a stuck camera
connection could stall evidence writing for every other camera in the process), not
something to paper over here, but a unit test asserting EvidenceWriter's own
orchestration logic should not have to pay for it.
"""

from __future__ import annotations

import logging
import queue
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perimeter.capture.ring_buffer import JpegRingBuffer
from perimeter.evidence.store import LocalEvidenceStore
from perimeter.record import open_writer

log = logging.getLogger("perimeter.evidence.writer")

DEFAULT_FPS_FALLBACK = 10.0

MuxFn = Callable[[list[tuple[float, bytes]], str, Path], Path | None]


def _decode(jpeg: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def mux_clip(frames: list[tuple[float, bytes]], event_id: str, tmp_dir: Path) -> Path | None:
    """Decode buffered JPEGs and write them out as one video file under `tmp_dir`.

    fps is derived from the actual frame timestamps (`len(frames) / span`), not a
    configured rate - that reflects the real capture cadence, drops included, more
    accurately than trusting a nominal decode_fps would.
    """
    if len(frames) < 2:
        return None  # too little buffered to produce a meaningful clip

    first_ts, first_jpeg = frames[0]
    last_ts, _ = frames[-1]
    span = last_ts - first_ts
    fps = (len(frames) - 1) / span if span > 0 else DEFAULT_FPS_FALLBACK

    first_image = _decode(first_jpeg)
    if first_image is None:
        return None
    height, width = first_image.shape[:2]

    writer, tmp_path = open_writer(tmp_dir / event_id, fps, (width, height))
    try:
        for _, jpeg in frames:
            image = _decode(jpeg)
            if image is not None:
                writer.write(image)
    finally:
        writer.release()
    return tmp_path


class EvidenceWriter:
    def __init__(
        self,
        database: Any,
        store: LocalEvidenceStore,
        ring_buffer_lookup: Callable[[str], JpegRingBuffer | None],
        post_roll_seconds: float = 5.0,
        mux: MuxFn = mux_clip,
    ) -> None:
        self.database = database
        self.store = store
        self.ring_buffer_lookup = ring_buffer_lookup
        self.post_roll_seconds = post_roll_seconds
        self._mux = mux

        self._queue: queue.Queue[tuple[dict[str, Any], float]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="evidence-writer", daemon=True)
        self._thread.start()
        log.info("evidence writer started (post_roll=%.1fs)", self.post_roll_seconds)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    # -- producer side (an AlertBus sink) -------------------------------------

    def on_alert(self, payload: dict[str, Any]) -> None:
        """`AlertBus` sink signature - called after persistence, so `payload["id"]` is
        already set. Never blocks the alert thread."""
        if "id" not in payload:
            return  # persistence failed - nothing to attach evidence to yet
        self._queue.put((payload, time.time() + self.post_roll_seconds))

    # -- consumer side ---------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload, deadline = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            remaining = deadline - time.time()
            if remaining > 0:
                # Interruptible: stop() should not have to wait out a long post-roll.
                self._stop.wait(remaining)
            try:
                self._process(payload)
            except Exception:  # noqa: BLE001 - one bad job must not kill this thread
                log.exception("evidence capture failed for event %s", payload.get("id"))
        log.info("evidence writer stopped")

    def _process(self, payload: dict[str, Any]) -> None:
        camera_id = payload.get("camera_id")
        event_id = payload.get("id")
        ring_buffer = self.ring_buffer_lookup(camera_id) if camera_id else None
        if ring_buffer is None:
            log.debug("no ring buffer for camera %r (event %s) - skipping", camera_id, event_id)
            return

        frames = ring_buffer.snapshot()
        if not frames:
            log.debug("ring buffer empty for camera %r (event %s) - skipping", camera_id, event_id)
            return

        # Independent try/excepts: a clip failure must not cost the (cheaper, more
        # likely to succeed) snapshot, and vice versa.
        snapshot_key = None
        try:
            snapshot_key = self._write_snapshot(camera_id, event_id, ring_buffer, payload.get("ts"))
        except Exception:  # noqa: BLE001
            log.exception("snapshot capture failed for event %s", event_id)

        clip_key = None
        try:
            clip_key = self._write_clip(camera_id, event_id, frames)
        except Exception:  # noqa: BLE001
            log.exception("clip capture failed for event %s", event_id)

        if snapshot_key or clip_key:
            self.database.update_evidence_paths(
                event_id, snapshot_path=snapshot_key, clip_path=clip_key
            )

    def _write_snapshot(
        self, camera_id: str, event_id: str, ring_buffer: JpegRingBuffer, ts: float | None
    ) -> str | None:
        jpeg = ring_buffer.closest_to(ts) if ts is not None else None
        if jpeg is None:
            return None
        return self.store.save_snapshot(camera_id, event_id, jpeg)

    def _write_clip(
        self, camera_id: str, event_id: str, frames: list[tuple[float, bytes]]
    ) -> str | None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="perimeter-evidence-"))
        try:
            tmp_path = self._mux(frames, event_id, tmp_dir)
            if tmp_path is None:
                return None
            return self.store.save_clip(camera_id, event_id, tmp_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
