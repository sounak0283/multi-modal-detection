"""Runtime pipeline: capture -> detect -> track -> boundary (PLAN.md section 4).

Threading model
---------------
Two threads plus the API's own. The capture thread decodes and publishes into a
single-slot mailbox; the inference thread takes only the freshest frame and drops the
rest. Alert delivery moves off this path in Phase 4.

Cadence
-------
Person detection runs on `seq % person_every_n == 0`. Fire/smoke will run on
`seq % 8 == 3` - deliberately offset so the two models never land on the same frame and
stack their latency onto every eighth one.

The tracker and boundary engine advance ONLY on detection frames. Their hysteresis is
counted in calls, so an extra call would shorten it, and calling the tracker with an
empty detection set on a skip frame would age tracks toward deletion rather than
interpolating.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from fsbd.boundary.engine import BoundaryEngine, BoundaryEvent
from fsbd.boundary.zones import ZoneStore, ZoneType
from fsbd.capture.latest_slot import LatestSlot
from fsbd.capture.source import CaptureThread, FeedState, Frame
from fsbd.detect.person import PersonDetector
from fsbd.detect.yolox_onnx import configure_opencv_threads
from fsbd.settings import Settings, redact
from fsbd.track.tracker import PersonTracker

log = logging.getLogger("fsbd.pipeline")

MAX_RECENT_EVENTS = 200
JPEG_QUALITY = 75

ZONE_COLOURS = {
    ZoneType.POLYGON: (80, 220, 80),
    ZoneType.TRIPWIRE: (0, 200, 255),
    ZoneType.EXCLUSION: (80, 80, 255),
    ZoneType.FIRE_ROI: (255, 140, 0),
}
TRACK_PALETTE = [
    (255, 128, 0), (0, 200, 255), (60, 220, 60), (200, 100, 255),
    (255, 80, 80), (255, 220, 0), (0, 160, 255), (180, 180, 180),
]


@dataclass
class PipelineStats:
    frames_decoded: int = 0
    detections_run: int = 0
    people_tracked: int = 0
    events_fired: int = 0
    feed_state: str = FeedState.STARTING.value
    inference_ms: float = 0.0
    dropped_frames: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def uptime_s(self) -> float:
        return time.time() - self.started_at


class Pipeline:
    def __init__(self, settings: Settings, store: ZoneStore, model_path: str) -> None:
        self.settings = settings
        self.store = store
        self.stats = PipelineStats()

        self.detector = PersonDetector(
            model_path,
            conf_threshold=settings.boundary.person_conf,
            intra_op_threads=settings.inference.intra_op_threads,
        )
        detection_fps = settings.inference.person_fps(settings.camera.decode_fps)
        self.tracker = PersonTracker(
            detection_fps=detection_fps,
            lost_track_seconds=settings.boundary.lost_track_seconds,
        )
        self.engine = BoundaryEngine(
            store,
            camera_id=settings.camera.id,
            default_min_frames=settings.boundary.default_min_frames,
        )

        self._slot: LatestSlot[Frame] = LatestSlot()
        self._capture = CaptureThread(
            settings.camera.inference_source,
            self._slot,
            on_state_change=self._on_feed_state,
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_raw_jpeg: bytes | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._events: deque[BoundaryEvent] = deque(maxlen=MAX_RECENT_EVENTS)
        self._tracked = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        configure_opencv_threads()
        log.info("pipeline starting on %s", redact(self.settings.camera.inference_source))
        self._capture.start()
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._capture.stop(timeout=timeout)
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _on_feed_state(self, state: FeedState) -> None:
        self.stats.feed_state = state.value

    # -- accessors used by the API ----------------------------------------

    @property
    def frame_size(self) -> tuple[int, int]:
        with self._lock:
            return self._frame_size

    def latest_jpeg(self, annotated: bool = True) -> bytes | None:
        with self._lock:
            return self._latest_jpeg if annotated else self._latest_raw_jpeg

    def recent_events(self, limit: int = 50) -> list[BoundaryEvent]:
        with self._lock:
            return list(self._events)[-limit:][::-1]

    # -- main loop ---------------------------------------------------------

    def _run(self) -> None:
        every_n = max(1, self.settings.inference.person_every_n)

        while not self._stop.is_set():
            frame = self._slot.get(timeout=0.5)
            if frame is None:
                continue

            self.stats.frames_decoded = self._capture.frames_decoded
            self.stats.dropped_frames = self._slot.stats().dropped

            is_detection_frame = frame.seq % every_n == 0
            if is_detection_frame:
                self._process_detection(frame)

            self._render(frame)

        log.info("inference thread stopped")

    def _process_detection(self, frame: Frame) -> None:
        started = time.perf_counter()
        detections = self.detector.detect(frame.image)
        tracked = self.tracker.update(detections)

        height, width = frame.image.shape[:2]
        events = self.engine.update(tracked, (width, height), now=frame.ts)

        self.stats.inference_ms = (time.perf_counter() - started) * 1000
        self.stats.detections_run += 1
        self.stats.people_tracked = len(tracked)
        self.stats.events_fired += len(events)

        with self._lock:
            self._tracked = tracked
            self._frame_size = (width, height)
            self._events.extend(events)

    # -- rendering ---------------------------------------------------------

    def _render(self, frame: Frame) -> None:
        raw_ok, raw_buf = cv2.imencode(
            ".jpg", frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        canvas = self._annotate(frame.image)
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        with self._lock:
            if ok:
                self._latest_jpeg = buf.tobytes()
            if raw_ok:
                self._latest_raw_jpeg = raw_buf.tobytes()

    def _annotate(self, image: np.ndarray) -> np.ndarray:
        canvas = image.copy()
        height, width = canvas.shape[:2]

        overlay = canvas.copy()
        for zone in self.store.zones(self.settings.camera.id):
            colour = ZONE_COLOURS.get(zone.type, (200, 200, 200))
            points = zone.pixel_points(width, height).astype(np.int32)

            if zone.is_area:
                cv2.fillPoly(overlay, [points], colour)
                cv2.polylines(canvas, [points], True, colour, 2)
            else:
                cv2.polylines(canvas, [points], False, colour, 3)
                self._draw_direction_arrow(canvas, points, colour)

            anchor = tuple(points[0])
            cv2.putText(
                canvas, zone.display_name, (anchor[0] + 4, anchor[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
            )
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

        with self._lock:
            tracked = self._tracked

        if tracked is not None:
            for box, track_id in zip(tracked.xyxy.astype(int), tracked.track_ids, strict=True):
                colour = TRACK_PALETTE[int(track_id) % len(TRACK_PALETTE)]
                x1, y1, x2, y2 = box
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    canvas, f"#{track_id}", (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
                )
                # The point the boundary engine actually tests.
                cv2.circle(canvas, ((x1 + x2) // 2, y2), 4, (255, 255, 255), -1)
                cv2.circle(canvas, ((x1 + x2) // 2, y2), 4, colour, 1)

        return canvas

    @staticmethod
    def _draw_direction_arrow(canvas, points: np.ndarray, colour) -> None:
        """Perpendicular arrow showing which crossing counts as ENTRY."""
        if len(points) < 2:
            return
        a, b = points[0], points[1]
        midpoint = ((a + b) / 2).astype(int)
        direction = b - a
        norm = float(np.hypot(*direction))
        if norm < 1:
            return
        normal = np.array([-direction[1], direction[0]]) / norm
        tip = (midpoint + normal * 28).astype(int)
        cv2.arrowedLine(canvas, tuple(midpoint), tuple(tip), colour, 2, tipLength=0.35)
