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
from fsbd.detect.person import MIN_IDENTITY_BOX_HEIGHT_PX, PersonDetector
from fsbd.detect.yolox_onnx import configure_opencv_threads
from fsbd.settings import Settings, describe_source
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
    render_fps: float = 0.0
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
        self._capture = self._build_capture(settings)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_raw_jpeg: bytes | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._events: deque[BoundaryEvent] = deque(maxlen=MAX_RECENT_EVENTS)
        self._tracked = None
        self._masked: np.ndarray = np.zeros(0, dtype=bool)
        self._last_render_ts: float = 0.0

    # -- lifecycle ---------------------------------------------------------

    def _build_capture(self, settings: Settings) -> CaptureThread:
        camera = settings.camera
        return CaptureThread(
            camera.inference_source,
            self._slot,
            on_state_change=self._on_feed_state,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
        )

    def start(self) -> None:
        configure_opencv_threads()
        log.info("pipeline starting on %s", describe_source(self.settings.camera))
        self._capture.start()
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._capture.stop(timeout=timeout)
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def reconfigure(self, settings: Settings) -> None:
        """Swap to a different camera without restarting the process.

        Only the capture thread is replaced. The detector, tracker and boundary engine
        survive - but tracker and engine state is reset, because track IDs and per-zone
        IN/OUT state from the old view are meaningless against a new one and would
        otherwise fire events on the first frame from the new camera.
        """
        log.info("reconfiguring camera -> %s", describe_source(settings.camera))
        self._capture.stop(timeout=3.0)

        self.settings = settings
        self._slot = LatestSlot()
        self._capture = self._build_capture(settings)

        self.tracker = PersonTracker(
            detection_fps=settings.inference.person_fps(settings.camera.decode_fps),
            lost_track_seconds=settings.boundary.lost_track_seconds,
        )
        self.engine = BoundaryEngine(
            self.store,
            camera_id=settings.camera.id,
            default_min_frames=settings.boundary.default_min_frames,
        )
        with self._lock:
            self._tracked = None
            self._latest_jpeg = None
            self._latest_raw_jpeg = None

        self.stats.feed_state = FeedState.STARTING.value
        self._capture.start()

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
        while not self._stop.is_set():
            every_n = max(1, self.settings.inference.person_every_n)
            frame = self._slot.get(timeout=0.5)

            if frame is None:
                # No frame arrived. If the feed is down, publish a rendered placeholder
                # rather than leaving the last good frame on screen: a frozen image is
                # indistinguishable from a still scene, so an operator watching the
                # dashboard would have no way to tell the camera had gone.
                if self._capture.state is FeedState.LOST:
                    self._render_disconnected()
                continue

            self.stats.frames_decoded = self._capture.frames_decoded
            self.stats.dropped_frames = self._slot.stats().dropped
            self._track_render_fps()

            if frame.seq % every_n == 0:
                self._process_detection(frame)

            self._render(frame)

        log.info("inference thread stopped")

    def _track_render_fps(self) -> None:
        now = time.perf_counter()
        if self._last_render_ts:
            dt = now - self._last_render_ts
            if dt > 0:
                # Exponential moving average: an instantaneous 1/dt reading jitters too
                # much to read off a live overlay.
                self.stats.render_fps = 0.9 * self.stats.render_fps + 0.1 * (1.0 / dt)
        self._last_render_ts = now

    def _process_detection(self, frame: Frame) -> None:
        started = time.perf_counter()
        detections = self.detector.detect(frame.image)
        tracked = self.tracker.update(detections)

        height, width = frame.image.shape[:2]
        events = self.engine.update(tracked, (width, height), now=frame.ts)

        # Why each track is or is not actionable, computed once here so the overlay does
        # not repeat the geometry work on every rendered frame.
        if len(tracked):
            masked = self.engine.suppressed_mask(tracked.foot_points(), width, height, "person")
        else:
            masked = np.zeros(0, dtype=bool)

        self.stats.inference_ms = (time.perf_counter() - started) * 1000
        self.stats.detections_run += 1
        self.stats.people_tracked = len(tracked)
        self.stats.events_fired += len(events)

        with self._lock:
            self._tracked = tracked
            self._masked = masked
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
            masked = self._masked

        if tracked is not None:
            for index in range(len(tracked)):
                self._draw_track(canvas, tracked, index, masked)

        self._draw_status_bar(canvas, len(tracked) if tracked is not None else 0)
        return canvas

    def _draw_track(self, canvas, tracked, index: int, masked: np.ndarray) -> None:
        """One track, labelled with *why* it is in the state it is in.

        A bare box and ID answers "is it detecting?" but not "why did nothing fire?",
        which is the question an installer is actually asking. The label distinguishes a
        suppressed detection from an undersized one from a normal tracked person.
        """
        x1, y1, x2, y2 = tracked.xyxy[index].astype(int)
        track_id = int(tracked.track_ids[index])
        score = float(tracked.scores[index])
        box_w, box_h = x2 - x1, y2 - y1

        is_masked = bool(masked[index]) if index < len(masked) else False
        too_small = box_h < MIN_IDENTITY_BOX_HEIGHT_PX

        if is_masked:
            colour = (90, 90, 255)
            label = f"#{track_id} MASKED (exclusion zone)"
        elif too_small:
            colour = (0, 165, 255)
            label = (
                f"#{track_id} {score:.2f} SMALL {box_w}x{box_h}px "
                f"(id needs {MIN_IDENTITY_BOX_HEIGHT_PX})"
            )
        else:
            colour = TRACK_PALETTE[track_id % len(TRACK_PALETTE)]
            label = f"#{track_id} {score:.2f} {box_w}x{box_h}px"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - text_h - 7), (x1 + text_w + 5, y1), colour, -1)
        cv2.putText(
            canvas, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1
        )

        # The point the boundary engine actually tests.
        foot = ((x1 + x2) // 2, y2)
        cv2.circle(canvas, foot, 4, (255, 255, 255), -1)
        cv2.circle(canvas, foot, 4, colour, 1)

    def _draw_status_bar(self, canvas, tracked_count: int) -> None:
        text = (
            f"{self.stats.render_fps:4.1f} fps | {self.stats.inference_ms:5.1f} ms | "
            f"{tracked_count} tracked | {self.stats.events_fired} events | "
            f"{self.settings.camera.id}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(
            canvas, text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1
        )

    def _render_disconnected(self) -> None:
        """Publish a rendered 'camera down' frame instead of a stale one.

        Without this the MJPEG stream simply stops updating, which on a quiet scene is
        visually identical to a working camera pointed at nothing happening.
        """
        width, height = self._frame_size if self._frame_size != (0, 0) else (960, 540)
        placeholder = np.zeros((height, width, 3), np.uint8)
        placeholder[:] = (26, 21, 18)

        cv2.putText(
            placeholder, "CAMERA DISCONNECTED", (40, height // 2 - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2,
        )
        cv2.putText(
            placeholder, f"reconnecting to {describe_source(self.settings.camera)}",
            (40, height // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1,
        )
        cv2.putText(
            placeholder, time.strftime("%Y-%m-%d %H:%M:%S"),
            (40, height // 2 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1,
        )

        ok, buf = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

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
