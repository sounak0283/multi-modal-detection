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

from fsbd.capture.latest_slot import LatestSlot
from fsbd.settings import redact

log = logging.getLogger("fsbd.capture")

RECONNECT_BACKOFF_START_S = 1.0
RECONNECT_BACKOFF_MAX_S = 30.0


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
    ) -> None:
        self.source = source
        self.slot = slot
        self.feed_lost_after_s = feed_lost_after_s
        self.on_state_change = on_state_change

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state = FeedState.STARTING
        self._seq = 0
        self._last_frame_ts = 0.0

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
        cap = (
            cv2.VideoCapture(self.source)
            if isinstance(self.source, int)
            else cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        )
        if cap.isOpened():
            # One-deep driver buffer. We drop-to-latest downstream anyway, and a deep
            # buffer only hands us stale frames after a hiccup.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok or image is None:
                    self._check_feed_lost()
                    break

                self._seq += 1
                self._last_frame_ts = time.time()
                self._set_state(FeedState.LIVE)
                self.slot.publish(Frame(seq=self._seq, ts=self._last_frame_ts, image=image))

            cap.release()
            if not self._stop.is_set():
                self._sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)

        log.info("capture thread stopped")

    def _check_feed_lost(self) -> None:
        if time.time() - self._last_frame_ts >= self.feed_lost_after_s:
            self._set_state(FeedState.LOST)

    def _sleep(self, seconds: float) -> None:
        self._stop.wait(seconds)
