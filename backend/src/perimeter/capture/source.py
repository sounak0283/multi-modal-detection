"""Capture thread (PLAN.md section 4).

Decodes continuously and publishes into a LatestSlot so the inference thread always
picks up the freshest frame. Reconnects with exponential backoff, and reports feed
health rather than failing silently - a camera that stops delivering must raise an
alert, not just log (PLAN.md section 4, "Camera health is an alert kind").
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from perimeter.capture.latest_slot import LatestSlot
from perimeter.settings import redact

log = logging.getLogger("perimeter.capture")

RECONNECT_BACKOFF_START_S = 1.0
RECONNECT_BACKOFF_MAX_S = 30.0

# OpenCV's FFmpeg backend opens streams under a process-wide lock, and its default
# open/read interrupt is 30 s. Measured: while one unreachable RTSP host is being
# opened, every other FFmpeg open in the process (other cameras reconnecting, a file
# source, the Test-connection probe) waits the full 30 s behind it. Bounding the
# timeout per capture bounds how long one dead camera can stall all the others.
OPEN_TIMEOUT_MS = 8_000


def open_capture(
    source: str | int, open_timeout_ms: int = OPEN_TIMEOUT_MS, read_timeout_ms: int | None = None
) -> cv2.VideoCapture:
    """A device index uses the platform's native backend; anything else goes through
    FFmpeg with bounded open (and optionally read) timeouts."""
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_ms)]
    if read_timeout_ms is not None:
        params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout_ms)]
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)


class FeedState(Enum):
    STARTING = "starting"
    LIVE = "live"
    LOST = "lost"


@dataclass(frozen=True)
class Frame:
    """A decoded frame plus the metadata downstream stages need.

    `seq` drives the inference cadence (person on n%2, fire on n%8+3), so it must come
    from the capture thread rather than being recounted downstream - dropped frames
    have to advance it, otherwise the two models would silently re-align onto the same
    frames whenever the pipeline fell behind.
    """

    seq: int
    ts: float  # unix epoch
    image: np.ndarray


class CaptureThread:
    def __init__(
        self,
        source: str | int,
        slot: LatestSlot[Frame],
        feed_lost_after_s: float = 10.0,
        on_state_change: Callable[[FeedState], None] | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        decode_fps: float | None = None,
    ) -> None:
        self.source = source
        self.slot = slot
        self.feed_lost_after_s = feed_lost_after_s
        self.on_state_change = on_state_change
        self.width = width
        self.height = height
        self.fps = fps
        # The tracker's timing (occlusion tolerance, hysteresis) is derived from
        # decode_fps, so frames must actually be published at that rate - not at whatever
        # rate the camera (30 fps) or a video file (as fast as it decodes) delivers.
        self.decode_fps = decode_fps

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state = FeedState.STARTING
        self._seq = 0
        self._last_frame_ts = 0.0
        self._actual_size: tuple[int, int] | None = None

    @property
    def actual_size(self) -> tuple[int, int] | None:
        """What the camera is really streaming, which may not be what we asked for."""
        return self._actual_size

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("capture thread already started")
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.slot.close()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    @property
    def state(self) -> FeedState:
        return self._state

    @property
    def frames_decoded(self) -> int:
        return self._seq

    def _set_state(self, state: FeedState) -> None:
        if state is self._state:
            return
        log.info("feed state %s -> %s", self._state.value, state.value)
        self._state = state
        if self.on_state_change is not None:
            self.on_state_change(state)

    # -- capture loop ------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        # Read timeout matches the feed-lost threshold: a stream that stalls mid-read must
        # be reported LOST after feed_lost_after_s, not after FFmpeg's 30 s default.
        cap = open_capture(self.source, read_timeout_ms=int(self.feed_lost_after_s * 1000))
        if not cap.isOpened():
            return cap

        # One-deep driver buffer. We drop-to-latest downstream anyway, and a deep
        # buffer only hands us stale frames after a hiccup.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.width and self.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self._actual_size = actual

        # A camera that quietly ignores the requested resolution is not cosmetic: box
        # sizes in pixels drive min_person_box_px for the identity layer, and the fire
        # model's small-object recall. Surfacing the mismatch beats debugging it later.
        if self.width and self.height and actual != (self.width, self.height):
            log.warning(
                "camera is streaming %dx%d, not the requested %dx%d - "
                "pixel-based thresholds are tuned for the configured resolution",
                actual[0], actual[1], self.width, self.height,
            )
        else:
            log.info("camera open at %dx%d", actual[0], actual[1])
        return cap

    def _run(self) -> None:
        backoff = RECONNECT_BACKOFF_START_S

        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self._check_feed_lost()
                # redact(): an RTSP URL embeds the camera password in plain text, and a
                # reconnect loop would otherwise write it to the log on every retry.
                log.warning("cannot open %s, retrying in %.0fs", redact(self.source), backoff)
                self._sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)
                continue

            log.info("capture connected to %s", redact(self.source))
            backoff = RECONNECT_BACKOFF_START_S
            self._last_frame_ts = time.time()

            # A recorded file has no clock of its own; pace reads to its native rate so it
            # plays back in real time. A live source paces itself and must be read as
            # fast as it delivers, or the driver buffer serves stale frames.
            file_fps = self._file_fps(cap)
            opened_at = time.perf_counter()
            read_count = 0
            publish_interval = 1.0 / self.decode_fps if self.decode_fps else 0.0
            next_publish = 0.0

            while not self._stop.is_set():
                if file_fps:
                    due = opened_at + read_count / file_fps
                    delay = due - time.perf_counter()
                    if delay > 0:
                        self._sleep(delay)
                ok, image = cap.read()
                read_count += 1
                if not ok or image is None:
                    self._check_feed_lost()
                    break

                now = time.perf_counter()
                self._last_frame_ts = time.time()
                self._set_state(FeedState.LIVE)
                if now < next_publish:
                    continue
                # Anchor to the schedule, not to `now`, so jitter does not erode the rate;
                # re-anchor after a long stall rather than bursting to catch up.
                next_publish = max(next_publish + publish_interval, now - publish_interval)
                self._seq += 1
                self.slot.publish(Frame(seq=self._seq, ts=self._last_frame_ts, image=image))

            cap.release()
            if not self._stop.is_set():
                self._sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)

        log.info("capture thread stopped")

    def _file_fps(self, cap: cv2.VideoCapture) -> float:
        if isinstance(self.source, int) or "://" in str(self.source):
            return 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if 0.0 < fps <= 240.0 else 0.0

    def _check_feed_lost(self) -> None:
        if time.time() - self._last_frame_ts >= self.feed_lost_after_s:
            self._set_state(FeedState.LOST)

    def _sleep(self, seconds: float) -> None:
        self._stop.wait(seconds)
