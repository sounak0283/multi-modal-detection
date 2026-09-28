"""Runtime pipeline: capture -> detect -> track -> boundary (PLAN.md section 4).

Threading model
---------------
Two threads plus the API's own. The capture thread decodes and publishes into a
single-slot mailbox; the inference thread takes only the freshest frame and drops the
rest. Alert delivery moves off this path in Phase 4.

The one exception: `capture_realtime=False` (the Video test page only) makes that
mailbox block instead of drop, so a finite recorded file gets every frame it decided to
keep processed, not whatever inference happened to still be free for.

Cadence
-------
Person detection runs on `seq % person_every_n == 0`. Fire/smoke (Expansion Plan Phase E)
runs on `seq % firesmoke_every_n == firesmoke_offset` - deliberately offset so the two
models never land on the same frame and stack their latency onto every eighth one.

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

from perimeter.alerts.bus import AlertBus
from perimeter.alerts.messages import (
    boundary_message,
    crowd_dispersed_message,
    crowd_message,
    firesmoke_message,
    ppe_message,
    unauthorized_access_message,
    welding_message,
)
from perimeter.boundary.crowd import CrowdEvent, CrowdMonitor
from perimeter.boundary.engine import BoundaryEngine, BoundaryEvent
from perimeter.boundary.zones import EventKind, Severity, Zone, ZoneStore, ZoneType
from perimeter.cameras.models import (
    MODULE_BOUNDARY,
    MODULE_CROWD,
    MODULE_FIRE_SMOKE,
    MODULE_IDENTITY,
    MODULE_PPE,
    MODULE_WELDING,
    Camera,
    describe_source,
)
from perimeter.capture.latest_slot import LatestSlot
from perimeter.capture.ring_buffer import JpegRingBuffer
from perimeter.capture.source import CaptureThread, FeedState, Frame
from perimeter.detect.firesmoke import FireSmokeDetector
from perimeter.detect.firesmoke import class_name as firesmoke_class_name
from perimeter.detect.person import MIN_IDENTITY_BOX_HEIGHT_PX, PersonDetector
from perimeter.detect.ppe import PPEClassifier, PPEMonitor, classify_state, person_crop
from perimeter.detect.ppe_helmet import PPEHelmetDetector
from perimeter.detect.temporal_gate import ConfirmedEvent, TemporalGate
from perimeter.detect.welding import SparkDetector, SparkEvent
from perimeter.detect.yolox_onnx import configure_opencv_threads
from perimeter.identity.resolver import IdentityResolver, IdentityResult
from perimeter.settings import Settings
from perimeter.track.tracker import PersonTracker

log = logging.getLogger("perimeter.pipeline")

MAX_RECENT_EVENTS = 200
# A PPE overlay mark lives for this many PPE checks after it was measured (checks run
# every ppe_every_n detection frames), so it does not blink between checks. Aged in
# detection frames, not wall-clock seconds: the Video test page runs faster than real
# time, where a seconds-based TTL kept marks alive across several scenes.
PPE_MARK_MAX_CHECKS = 2
ERROR_LOG_INTERVAL_S = 30.0
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
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def uptime_s(self) -> float:
        return time.time() - self.started_at


class Pipeline:
    """Capture -> detect -> track -> boundary, for one camera.

    Runs as one instance per camera under a `PipelineManager` (Expansion Plan Phase A.2)
    rather than the single global instance `main.py` used to construct directly. `camera`
    carries everything specific to this camera (source, resolution, enabled modules);
    `settings` carries the engineering constants shared across every camera
    (`inference.person_every_n`, `boundary.person_conf`, ...). Only the modules listed in
    `camera.enabled_modules` are built, and every module is opt-in: `"boundary"` builds
    the `PersonDetector` that person tracking and zone events run on (the zone
    `BoundaryEngine` itself always exists, since fire/smoke reads its zones); `fire_smoke`
    (Expansion Plan Phase E) additionally builds a `FireSmokeDetector` + `TemporalGate`
    if a model path was given and actually loads; `identity` (Expansion Plan Phase F)
    resolves a confirmed ENTRY's face via a shared, process-wide `IdentityResolver`
    passed in from outside rather than built per camera; `crowd` (Phase G) builds a
    `CrowdMonitor` unconditionally (no model); `ppe` (Phase H) builds a
    `PPEClassifier` + `PPEMonitor` the same optional, soft-fail way `fire_smoke` does -
    phone remains unimplemented (Phase I).
    """

    def __init__(
        self,
        settings: Settings,
        camera: Camera,
        store: ZoneStore,
        model_path: str,
        alerts: AlertBus | None = None,
        firesmoke_model_path: str | None = None,
        identity_resolver: IdentityResolver | None = None,
        ppe_model_path: str | None = None,
        capture_realtime: bool = True,
    ) -> None:
        self.settings = settings
        self.camera = camera
        self.store = store
        self.alerts = alerts
        self.firesmoke_model_path = firesmoke_model_path
        self.ppe_model_path = ppe_model_path
        # False only for the Video test page: process an uploaded file as fast as the
        # hardware allows instead of pacing it to its own real length, so results are
        # complete and reproducible regardless of how many modules are enabled (a
        # real-time-paced run drops more frames under heavier CPU load, which can hide a
        # short, genuine event - measured directly: crowd detection throughput fell from
        # ~2900 to ~1200 processed frames by the same point in one video once fire/smoke,
        # welding and identity all ran alongside it).
        self._capture_realtime = capture_realtime
        # Shared across every camera (Expansion Plan Phase F) - the models/gallery are
        # stateless per call and expensive to load, so one instance serves the whole
        # process rather than one per camera the way the fire/smoke detector does.
        self.identity_resolver = identity_resolver
        self.stats = PipelineStats()

        self.model_path = model_path
        # Person detection is the "boundary" module and, like every module, opt-in: a
        # camera with no modules selected loads no person model at all.
        self.detector: PersonDetector | None = self._build_detector(camera)
        self.tracker = self._build_tracker(camera)
        # Always built (no model, no cost): fire/smoke consults exclusion and fire-ROI
        # zones through the engine even when person detection is off.
        self.engine = BoundaryEngine(
            store,
            camera_id=camera.id,
            default_min_frames=settings.boundary.default_min_frames,
        )
        self.firesmoke_detector, self.firesmoke_gate = self._build_firesmoke(camera)
        self.crowd_monitor = self._build_crowd_monitor(camera)
        self.ppe_classifier, self.ppe_monitor = self._build_ppe(camera)
        self.spark_detector = SparkDetector() if MODULE_WELDING in camera.enabled_modules else None
        self._spark_box: tuple[tuple[float, ...], float] | None = None
        self._fire_boxes: tuple[list[tuple[str, float, tuple[int, ...], bool]], float] | None = None
        # track_id -> (head box relative to the person box, label, violation confirmed,
        # detections_run when measured) for the overlay.
        self._ppe_marks: dict[int, tuple[tuple[float, ...], str, bool, int]] = {}
        self._ppe_unsupported_warned: set[tuple[str, str]] = set()

        self._slot: LatestSlot[Frame] = LatestSlot(blocking=not capture_realtime)
        self._capture = self._build_capture(camera)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Evidence pre-roll (Expansion Plan Phase C). Holds RAW frames, never the
        # annotated overlay - evidence should show what actually happened, not the
        # dashboard's zone/track drawing on top of it.
        self.ring_buffer = JpegRingBuffer(settings.storage.ring_buffer_seconds)

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_raw_jpeg: bytes | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._events: deque[BoundaryEvent] = deque(maxlen=MAX_RECENT_EVENTS)
        self._tracked = None
        self._masked: np.ndarray = np.zeros(0, dtype=bool)
        self._last_render_ts: float = 0.0
        self._last_error_log: float = float("-inf")
        # Identity resolved on a track's ENTRY, remembered for its later EXIT (see the
        # events loop in _process_detection) - a person has usually turned away by the
        # time they exit, so re-running face detection there would mostly just fail;
        # reusing the ENTRY result is both cheaper and more reliable than trying again.
        # Value is (identity, frame.ts last touched) - pruning is on that timestamp, not
        # on whether the track appears in the *current* frame's tracked set: a track
        # PersonTracker briefly does not return (a missed detection, mid-occlusion) is
        # not the same as one ByteTrack has actually deleted, and evicting on the first
        # kind would routinely lose the identity well before the real EXIT arrives.
        self._identity_by_track: dict[int, tuple[IdentityResult, float]] = {}

    # -- lifecycle ---------------------------------------------------------

    def _build_detector(self, camera: Camera) -> PersonDetector | None:
        if MODULE_BOUNDARY not in camera.enabled_modules:
            return None
        if getattr(self, "detector", None) is not None:
            return self.detector  # no per-camera state; reuse across reconfigures
        return PersonDetector(
            self.model_path,
            conf_threshold=self.settings.boundary.person_conf,
            intra_op_threads=self.settings.inference.intra_op_threads,
        )

    def _build_tracker(self, camera: Camera) -> PersonTracker:
        return PersonTracker(
            detection_fps=self.settings.inference.person_fps(camera.decode_fps),
            lost_track_seconds=self.settings.boundary.lost_track_seconds,
            activation_threshold=self.settings.boundary.track_activation_threshold,
        )

    def _build_firesmoke(
        self, camera: Camera
    ) -> tuple[FireSmokeDetector | None, TemporalGate | None]:
        """Best-effort: fire/smoke is an optional module (Expansion Plan Phase E), never
        a reason to refuse to run a camera. `FileNotFoundError` is the expected case
        until real trained weights exist at the configured path - anything else is
        logged the same way, since a camera that can't do fire/smoke should still do
        boundary detection."""
        if MODULE_FIRE_SMOKE not in camera.enabled_modules or not self.firesmoke_model_path:
            return None, None
        try:
            detector = FireSmokeDetector(
                self.firesmoke_model_path,
                conf_fire=self.settings.firesmoke.conf_fire,
                conf_smoke=self.settings.firesmoke.conf_smoke,
                intra_op_threads=self.settings.inference.intra_op_threads,
            )
        except Exception as exc:  # noqa: BLE001 - any load failure degrades, never crashes
            log.warning(
                "fire/smoke enabled for %s but no usable model at %s (%s) - module inactive",
                camera.id, self.firesmoke_model_path, exc,
            )
            return None, None
        gate = TemporalGate(
            k=self.settings.firesmoke.gate_k,
            n=self.settings.firesmoke.gate_n,
            iou_threshold=self.settings.firesmoke.gate_iou,
        )
        log.info("fire/smoke detection active for %s", camera.id)
        return detector, gate

    def _build_crowd_monitor(self, camera: Camera) -> CrowdMonitor | None:
        """No model, no file to load - unlike fire/smoke/identity, `crowd` needs
        nothing more than the module flag to activate (Expansion Plan Phase G)."""
        if MODULE_CROWD not in camera.enabled_modules:
            return None
        return CrowdMonitor(
            eps_px=self.settings.crowd.eps_px,
            min_samples=self.settings.crowd.min_samples,
            eps_frac=self.settings.crowd.eps_frac,
        )

    def _build_ppe(
        self, camera: Camera
    ) -> tuple[PPEClassifier | PPEHelmetDetector | None, PPEMonitor | None]:
        """Best-effort, same posture as `_build_firesmoke` (Expansion Plan Phase H):
        weights are not in git, so a missing/unloadable model degrades to a clean, logged
        no-op rather than a startup failure. `ppe.backend` picks the model family."""
        if MODULE_PPE not in camera.enabled_modules or not self.ppe_model_path:
            return None, None
        ppe = self.settings.ppe
        threads = self.settings.inference.intra_op_threads
        try:
            if ppe.backend == "helmet_detector":
                classifier = PPEHelmetDetector(
                    self.ppe_model_path,
                    head_threshold=ppe.head_threshold,
                    helmet_threshold=ppe.helmet_threshold,
                    min_person_height_px=ppe.min_person_height_px,
                    intra_op_threads=threads,
                )
            else:
                classifier = PPEClassifier(self.ppe_model_path, intra_op_threads=threads)
        except Exception as exc:  # noqa: BLE001 - any load failure degrades, never crashes
            log.warning(
                "PPE enabled for %s but the %s model at %s could not be loaded (%s) - "
                "module inactive",
                camera.id, ppe.backend, self.ppe_model_path, exc,
            )
            return None, None
        log.info(
            "PPE compliance checking active for %s (%s, items: %s)",
            camera.id, ppe.backend, ", ".join(sorted(classifier.supported_items)),
        )
        return classifier, PPEMonitor()

    def _build_capture(self, camera: Camera) -> CaptureThread:
        return CaptureThread(
            camera.inference_source,
            self._slot,
            on_state_change=self._on_feed_state,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
            decode_fps=camera.decode_fps,
            realtime=self._capture_realtime,
        )

    def start(self) -> None:
        configure_opencv_threads()
        log.info("pipeline starting on %s", describe_source(self.camera))
        self._capture.start()
        self._thread = threading.Thread(
            target=self._run, name=f"inference-{self.camera.id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._capture.stop(timeout=timeout)
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def reconfigure(self, camera: Camera) -> None:
        """Swap to different camera settings (e.g. a new RTSP URL) without restarting.

        Only the capture thread is replaced. The detector survives if the boundary module
        stays enabled (it has no per-camera state); the tracker and boundary engine are
        rebuilt because track IDs and
        per-zone IN/OUT state from the old view are meaningless against a new one and
        would otherwise fire events on the first frame from the new camera.
        """
        if camera.id != self.camera.id:
            raise ValueError("reconfigure() cannot change a pipeline's camera id")
        log.info("reconfiguring %s -> %s", camera.id, describe_source(camera))
        self._capture.stop(timeout=3.0)

        self.camera = camera
        self._slot = LatestSlot(blocking=not self._capture_realtime)
        self._capture = self._build_capture(camera)

        self.detector = self._build_detector(camera)
        self.tracker = self._build_tracker(camera)
        self.engine = BoundaryEngine(
            self.store,
            camera_id=camera.id,
            default_min_frames=self.settings.boundary.default_min_frames,
        )
        self.firesmoke_detector, self.firesmoke_gate = self._build_firesmoke(camera)
        self.crowd_monitor = self._build_crowd_monitor(camera)
        self.ppe_classifier, self.ppe_monitor = self._build_ppe(camera)
        self.spark_detector = SparkDetector() if MODULE_WELDING in camera.enabled_modules else None
        self._spark_box = None
        self._fire_boxes = None
        with self._lock:
            self._tracked = None
            self._latest_jpeg = None
            self._latest_raw_jpeg = None
            self._identity_by_track = {}
        # Old buffered frames belong to the feed just replaced, not the new one.
        self.ring_buffer = JpegRingBuffer(self.settings.storage.ring_buffer_seconds)

        self.stats.feed_state = FeedState.STARTING.value
        self._capture.start()

    def play_once(self) -> None:
        """For a file source: stop at the end instead of looping (call before start())."""
        self._capture.loop = False

    @property
    def capture_finished(self) -> bool:
        return self._capture.finished

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
            try:
                self._step()
            except Exception:  # noqa: BLE001 - see below
                # One bad frame (a transient database error, an unexpected model
                # output) must never end detection for this camera. An uncaught exception
                # here used to kill the thread while the capture thread kept the feed
                # reporting "live" - a silently unwatched camera until restart.
                self.stats.errors += 1
                now = time.monotonic()
                if now - self._last_error_log >= ERROR_LOG_INTERVAL_S:
                    self._last_error_log = now
                    log.exception(
                        "inference step failed for %s (%d error(s) so far) - continuing",
                        self.camera.id, self.stats.errors,
                    )
                self._stop.wait(0.2)

        log.info("inference thread stopped")

    def _step(self) -> None:
        every_n = max(1, self.settings.inference.person_every_n)
        frame = self._slot.get(timeout=0.5)

        if frame is None:
            # No frame arrived. If the feed is down, publish a rendered placeholder
            # rather than leaving the last good frame on screen: a frozen image is
            # indistinguishable from a still scene, so an operator watching the
            # dashboard would have no way to tell the camera had gone.
            if self._capture.state is FeedState.LOST:
                self._render_disconnected()
            return

        with self._lock:
            self._frame_size = (frame.image.shape[1], frame.image.shape[0])
        self.stats.frames_decoded = self._capture.frames_decoded
        self.stats.dropped_frames = self._slot.stats().dropped
        self._track_render_fps()

        if self.detector is not None and frame.seq % every_n == 0:
            self._process_detection(frame)

        if self.firesmoke_detector is not None:
            fs_every_n = max(1, self.settings.inference.firesmoke_every_n)
            if frame.seq % fs_every_n == self.settings.inference.firesmoke_offset % fs_every_n:
                self._process_firesmoke(frame)

        if self.spark_detector is not None:
            self._process_sparks(frame)

        self._render(frame)

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
            masked = self.engine.suppressed_mask(
                tracked.foot_points(height), width, height, "person"
            )
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

        # Identity (Expansion Plan Phase F): resolved only for confirmed ENTRYs, and
        # only when the module is on for this camera - never every frame, never every
        # detection. `track_boxes` reuses the same detection-frame's raw pixel boxes,
        # not a second lookup.
        identity_active = (
            self.identity_resolver is not None and MODULE_IDENTITY in self.camera.enabled_modules
        )
        track_boxes = (
            {int(tid): tracked.xyxy[i] for i, tid in enumerate(tracked.track_ids)}
            if identity_active
            else {}
        )

        if identity_active:
            self._touch_live_identities(track_boxes.keys(), frame.ts)
            self._prune_stale_identities(frame.ts)

        for event in events:
            identity = (
                self._resolve_identity_for_event(event, frame.image, frame.ts, track_boxes)
                if identity_active
                else None
            )
            self._publish(event, identity, frame.video_pos_s)

        if self.crowd_monitor is not None:
            self._process_crowd(tracked, masked, width, height, frame.ts, frame.video_pos_s)

        if self.ppe_classifier is not None:
            ppe_every_n = max(1, self.settings.inference.ppe_every_n)
            if self.stats.detections_run % ppe_every_n == 0:
                self._process_ppe(frame, tracked, masked, width, height)

    def _touch_live_identities(self, live_track_ids, now_ts: float) -> None:
        """Refresh the "last seen" timestamp for every remembered identity whose track
        is present in this detection frame.

        Without this, the stored timestamp is stamped once at ENTRY and never updated,
        so `_prune_stale_identities` below measures "how long has this person been
        anywhere near this zone" instead of "how long has their track actually been
        gone" - a visit lasting longer than the occlusion tolerance (a very normal
        thing) would then evict the identity before the real EXIT ever arrives, which is
        exactly the bug this method exists to not have.
        """
        for tid in live_track_ids:
            entry = self._identity_by_track.get(tid)
            if entry is not None:
                self._identity_by_track[tid] = (entry[0], now_ts)

    def _prune_stale_identities(self, now_ts: float) -> None:
        """Garbage-collect remembered identities whose track has been genuinely absent
        (not refreshed by `_touch_live_identities`) for longer than the tracker's own
        occlusion tolerance - by the time ByteTrack would actually drop the track,
        remembering its identity has stopped being useful anyway.
        """
        ttl = self.settings.boundary.lost_track_seconds
        self._identity_by_track = {
            tid: (identity, ts)
            for tid, (identity, ts) in self._identity_by_track.items()
            if now_ts - ts <= ttl
        }

    def _resolve_identity_for_event(
        self,
        event: BoundaryEvent,
        frame_image: np.ndarray,
        frame_ts: float,
        track_boxes: dict[int, np.ndarray],
    ) -> IdentityResult | None:
        """Identity for one boundary event (Expansion Plan Phase F).

        ENTRY resolves fresh, from this same detection frame's raw pixel box, and
        remembers the result under this track's id. EXIT reuses whatever was resolved at
        that same track's ENTRY instead of re-running face detection - a person has
        usually turned away by the time they leave, so a second attempt would mostly
        just fail, and the identity does not change mid-visit anyway.
        """
        if event.kind is EventKind.ENTRY:
            box = track_boxes.get(event.track_id)
            if box is None:
                return None
            identity = self.identity_resolver.resolve(frame_image, tuple(float(v) for v in box))
            self._identity_by_track[event.track_id] = (identity, frame_ts)
            return identity
        if event.kind is EventKind.EXIT:
            entry = self._identity_by_track.get(event.track_id)
            return entry[0] if entry is not None else None
        return None

    def _process_ppe(
        self, frame: Frame, tracked, masked: np.ndarray, width: int, height: int
    ) -> None:
        """PPE compliance (Expansion Plan Phase H). Runs every `ppe_every_n` DETECTION
        frames (not every one) for a camera with the `ppe` module enabled - see
        `InferenceSettings.ppe_every_n`'s docstring for why the cadence is keyed off the
        detection counter rather than the raw camera frame the way fire/smoke's is."""
        assert self.ppe_classifier is not None and self.ppe_monitor is not None
        if not len(tracked):
            return
        feet = tracked.foot_points(height)
        keep = ~masked if len(masked) == len(feet) else np.ones(len(feet), dtype=bool)
        visible_feet = feet[keep]
        if not len(visible_feet):
            return
        visible_boxes = tracked.xyxy[keep]
        visible_ids = tracked.track_ids[keep]

        classifier = self.ppe_classifier
        supported = getattr(classifier, "supported_items", None)
        membership = self.engine.ppe_membership(visible_feet, width, height)
        marks: dict[int, tuple[tuple[float, ...], str, bool, int]] = {}
        for zone_id, mask in membership.items():
            zone = self.engine.zone_by_id(zone_id)
            if zone is None or not zone.ppe_required:
                continue
            checked_items = [
                item for item in zone.ppe_required
                if supported is None or item in supported
            ]
            for item in zone.ppe_required.difference(checked_items):
                if (zone_id, item) not in self._ppe_unsupported_warned:
                    self._ppe_unsupported_warned.add((zone_id, item))
                    log.warning(
                        "zone %s on %s requires %r, which the active PPE model (%s) cannot "
                        "detect - that item is never checked", zone.display_name,
                        self.camera.id, item, self.settings.ppe.backend,
                    )
            if not checked_items:
                continue
            for index in np.nonzero(mask)[0]:
                track_id = int(visible_ids[index])
                box = visible_boxes[index]
                x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                if hasattr(classifier, "classify_person"):
                    result = classifier.classify_person(frame.image, (x1, y1, x2, y2))
                    probabilities, head_box, label = (
                        result.probabilities, result.head_box, result.label
                    )
                else:  # duck-typed classifier exposing only classify(crop)
                    crop, _origin = person_crop(frame.image, (x1, y1, x2, y2))
                    probabilities, head_box, label = classifier.classify(crop), None, None
                for item in checked_items:
                    state = classify_state(probabilities.get(item, 0.0))
                    if self.ppe_monitor.update(zone_id, track_id, item, state):
                        self._publish_ppe(zone, item, track_id, frame.ts, frame.video_pos_s)
                if head_box is not None and label:
                    violating = self.ppe_monitor.is_confirmed(zone_id, track_id, "helmet")
                    pw, ph = max(1, x2 - x1), max(1, y2 - y1)
                    relative = (
                        (head_box[0] - x1) / pw, (head_box[1] - y1) / ph,
                        (head_box[2] - x1) / pw, (head_box[3] - y1) / ph,
                    )
                    marks[track_id] = (relative, label, violating, self.stats.detections_run)
        self.ppe_monitor.evict_stale()
        live = {int(t) for t in tracked.track_ids}
        oldest = self.stats.detections_run - self._ppe_mark_max_age()
        with self._lock:
            self._ppe_marks = {
                tid: mark for tid, mark in self._ppe_marks.items()
                if tid in live and mark[3] >= oldest
            }
            self._ppe_marks.update(marks)

    def _ppe_mark_max_age(self) -> int:
        return PPE_MARK_MAX_CHECKS * max(1, self.settings.inference.ppe_every_n)

    def _publish_ppe(
        self, zone: Zone, item: str, track_id: int, ts: float,
        video_pos_s: float | None = None,
    ) -> None:
        if self.alerts is None:
            return
        self.alerts.publish(
            {
                "ts": ts,
                "camera_id": self.camera.id,
                "kind": "ppe",
                "subtype": item,
                "zone_id": zone.id,
                "zone_name": zone.display_name,
                "track_id": track_id,
                "message": ppe_message(zone.display_name, item),
                "severity": zone.severity.value,
                "bbox": None,
                "foot_point": None,
                "identity_status": None,
                "identity_name": None,
                "video_pos_s": video_pos_s,
            }
        )

    def _process_crowd(
        self, tracked, masked: np.ndarray, width: int, height: int, ts: float,
        video_pos_s: float | None = None,
    ) -> None:
        """Crowd formation (Expansion Plan Phase G). Runs on every detection frame for
        a camera with the `crowd` module enabled - no model, so no cadence gating is
        needed the way fire/smoke's second, offset cadence is.

        Deliberately runs with zero tracked people too (an empty `tracked`, or everyone
        masked out): that is how a crowd disperses down to nothing, and each zone's
        hysteresis state machine must see that frame to ever confirm a dispersal."""
        assert self.crowd_monitor is not None
        feet = tracked.foot_points(height) if len(tracked) else np.zeros((0, 2), np.float32)
        # Exclusion zones suppress detections for every module, not just boundary -
        # someone standing in a masked-out area (a queue line, a waiting area marked as
        # an exclusion zone) should not count toward a crowd.
        visible = feet[~masked] if len(masked) == len(feet) else feet

        membership = self.engine.crowd_membership(visible, width, height)
        for zone_id, mask in membership.items():
            zone = self.engine.zone_by_id(zone_id)
            if zone is None or zone.crowd_threshold is None:
                continue
            min_frames = zone.crowd_min_frames or self.settings.boundary.default_min_frames
            event = self.crowd_monitor.update(
                zone_id, visible[mask], zone.crowd_threshold, min_frames, frame_height=height
            )
            if event is not None:
                self._publish_crowd(event, zone, ts, video_pos_s)

    def _publish_crowd(
        self, event: CrowdEvent, zone: Zone, ts: float, video_pos_s: float | None = None,
    ) -> None:
        if self.alerts is None:
            return
        message = (
            crowd_message(zone.display_name, event.cluster_size)
            if event.formed
            else crowd_dispersed_message(zone.display_name)
        )
        self.alerts.publish(
            {
                "ts": ts,
                "camera_id": self.camera.id,
                "kind": "crowd",
                "subtype": "formed" if event.formed else "dispersed",
                "zone_id": zone.id,
                "zone_name": zone.display_name,
                "track_id": None,
                "message": message,
                # A dispersal is informational, not a threat - it never needs the zone's
                # own (possibly high/critical) severity, which exists to size the alarm
                # for people gathering, not for them leaving.
                "severity": zone.severity.value if event.formed else "low",
                "bbox": None,
                "foot_point": None,
                "identity_status": None,
                "identity_name": None,
                "video_pos_s": video_pos_s,
            }
        )

    def _publish(
        self,
        event: BoundaryEvent,
        identity: IdentityResult | None = None,
        video_pos_s: float | None = None,
    ) -> None:
        """Hand an event to the alert thread for persistence and notification.

        Rendering the sentence here rather than at read time means the stored history
        always shows what was actually sent, even after the wording changes.
        """
        if self.alerts is None:
            return
        identity_status = identity.status if identity is not None else None
        identity_name = identity.name if identity is not None else None
        identity_confidence = identity.confidence if identity is not None else None
        self.alerts.publish(
            {
                "ts": event.ts,
                "camera_id": event.camera_id,
                "kind": "boundary",
                "subtype": event.kind.value,
                "zone_id": event.zone_id,
                "zone_name": event.zone_name,
                "track_id": event.track_id,
                "message": boundary_message(
                    event.kind, event.zone_name, identity_status, identity_name
                ),
                "severity": event.severity.value,
                "bbox": list(event.bbox),
                "foot_point": list(event.foot_point),
                "identity_status": identity_status,
                "identity_name": identity_name,
                "identity_confidence": identity_confidence,
                "video_pos_s": video_pos_s,
            }
        )
        self._check_access_control(event, identity, video_pos_s)

    def _check_access_control(
        self,
        event: BoundaryEvent,
        identity: IdentityResult | None,
        video_pos_s: float | None = None,
    ) -> None:
        """Restricted-zone allow-list (Expansion Plan Phase F.1).

        Only meaningful for an ENTRY into a zone that actually declares
        `authorized_person_ids`; every other event is unaffected. Anyone not positively
        matched to that list - known-but-unlisted, unknown_face, or no_face alike -
        raises a second, critical alert alongside (never instead of) the normal ENTRY
        log, with `cooldown_seconds: 0` on the matching rule bypassing cooldown
        unconditionally (CooldownGate's existing bypass, no changes needed there).
        """
        if self.alerts is None or event.kind is not EventKind.ENTRY or identity is None:
            return
        zone = self.engine.zone_by_id(event.zone_id)
        if zone is None or not zone.authorized_person_ids:
            return
        if identity.status == "known" and identity.person_id in zone.authorized_person_ids:
            return

        self.alerts.publish(
            {
                "ts": event.ts,
                "camera_id": event.camera_id,
                "kind": "access",
                "subtype": "unauthorized",
                "zone_id": event.zone_id,
                "zone_name": event.zone_name,
                "track_id": event.track_id,
                "message": unauthorized_access_message(
                    event.zone_name, identity.status, identity.name
                ),
                "severity": Severity.CRITICAL.value,
                "bbox": list(event.bbox),
                "foot_point": list(event.foot_point),
                "identity_status": identity.status,
                "identity_name": identity.name,
                "video_pos_s": video_pos_s,
            }
        )

    def _process_firesmoke(self, frame: Frame) -> None:
        """Fire/smoke detection tick (Expansion Plan Phase E; PLAN.md §5.5).

        `FireSmokeDetector` returns raw, class-filtered detections only - the zone-aware
        policy (per-class threshold, a fire_roi's sensitivity delta, an exclusion mask)
        lives here, the same way it already does for "person" in `_process_detection`.
        """
        assert self.firesmoke_detector is not None and self.firesmoke_gate is not None
        detections = self.firesmoke_detector.detect(frame.image)
        height, width = frame.image.shape[:2]

        accepted: list[tuple[str, tuple[float, float, float, float]]] = []
        seen: list[tuple[str, float, tuple[int, ...], bool]] = []
        for index in range(len(detections)):
            klass = firesmoke_class_name(int(detections.class_ids[index]))
            box = tuple(float(v) for v in detections.xyxy[index])
            score = float(detections.scores[index])
            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

            if self.engine.suppressed_mask(np.array([center]), width, height, klass)[0]:
                seen.append((klass, score, tuple(int(v) for v in box), False))
                continue  # an exclusion zone masks this class here (e.g. the welding bay)

            base_conf = (
                self.settings.firesmoke.conf_fire
                if klass == "fire"
                else self.settings.firesmoke.conf_smoke
            )
            delta = self.engine.confidence_delta(center, width, height)  # <= 0
            if score < base_conf + delta:
                seen.append((klass, score, tuple(int(v) for v in box), False))
                continue

            accepted.append((klass, box))
            seen.append((klass, score, tuple(int(v) for v in box), True))
        self._fire_boxes = (seen, time.time() + 1.5)

        for confirmed in self.firesmoke_gate.update(accepted):
            self._publish_firesmoke(confirmed, frame, width, height)

    def _publish_firesmoke(
        self, event: ConfirmedEvent, frame: Frame, width: int, height: int
    ) -> None:
        if self.alerts is None:
            return
        center = (
            (event.bbox[0] + event.bbox[2]) / 2.0,
            (event.bbox[1] + event.bbox[3]) / 2.0,
        )
        zone = self.engine.fire_roi_zone_at(center, width, height)
        zone_name = zone.display_name if zone is not None else None

        self.alerts.publish(
            {
                "ts": frame.ts,
                "camera_id": self.camera.id,
                "kind": event.klass,
                "subtype": None,
                "zone_id": zone.id if zone is not None else None,
                "zone_name": zone_name,
                "track_id": None,
                "message": firesmoke_message(event.klass, zone_name),
                "severity": "critical" if event.escalate else "high",
                "bbox": list(event.bbox),
                "foot_point": None,
                "identity_status": None,
                "identity_name": None,
                "video_pos_s": frame.video_pos_s,
            }
        )

    def _process_sparks(self, frame: Frame) -> None:
        assert self.spark_detector is not None
        event = self.spark_detector.update(frame.image, frame.ts)
        if event is not None:
            self._publish_sparks(event, frame)

    def _publish_sparks(self, event: SparkEvent, frame: Frame) -> None:
        self._spark_box = (event.bbox, time.time() + 5.0)
        if self.alerts is None:
            return
        self.alerts.publish(
            {
                "ts": frame.ts,
                "camera_id": self.camera.id,
                "kind": "welding",
                "subtype": "welding_spark",
                "zone_id": None,
                "zone_name": None,
                "track_id": None,
                "message": welding_message(),
                "severity": "low",
                "bbox": list(event.bbox),
                "foot_point": None,
                "identity_status": None,
                "identity_name": None,
                "video_pos_s": frame.video_pos_s,
            }
        )

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

        if raw_ok:
            self.ring_buffer.append(frame.ts, raw_buf.tobytes())

    def _annotate(self, image: np.ndarray) -> np.ndarray:
        canvas = image.copy()
        height, width = canvas.shape[:2]

        overlay = canvas.copy()
        for zone in self.store.zones(self.camera.id):
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

        fire = self._fire_boxes
        if fire is not None and time.time() < fire[1]:
            for klass, score, (x1, y1, x2, y2), counted in fire[0]:
                colour = (40, 40, 255) if counted else (150, 150, 150)
                label = f"{klass.upper()} {score:.2f}" + ("" if counted else " below threshold")
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(
                    canvas, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
                )
        with self._lock:
            ppe_marks = dict(self._ppe_marks)
        oldest = self.stats.detections_run - self._ppe_mark_max_age()
        for index in range(len(tracked) if tracked is not None and ppe_marks else 0):
            mark = ppe_marks.get(int(tracked.track_ids[index]))
            if mark is None or mark[3] < oldest:
                continue
            relative, label, violating, _measured = mark
            px1, py1, px2, py2 = (float(v) for v in tracked.xyxy[index])
            pw, ph = px2 - px1, py2 - py1
            head_box = (
                px1 + relative[0] * pw, py1 + relative[1] * ph,
                px1 + relative[2] * pw, py1 + relative[3] * ph,
            )
            if label == "HELMET":
                colour = (0, 200, 0)
            elif violating:
                colour = (0, 0, 255)
            else:  # bare head seen, not yet confirmed by the hysteresis
                colour, label = (0, 165, 255), "NO HELMET?"
            x1, y1, x2, y2 = (int(v) for v in head_box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                canvas, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
            )
        spark = self._spark_box
        if spark is not None and time.time() < spark[1]:
            x1, y1, x2, y2 = (int(v) for v in spark[0])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(
                canvas, "WELDING SPARKS", (x1, max(12, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
            )
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
            f"{self.camera.id}"
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
            placeholder, f"reconnecting to {describe_source(self.camera)}",
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
