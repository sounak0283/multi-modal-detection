"""Test-video sessions: play an uploaded file through the real detection pipeline.

A session is a temporary in-memory camera pointed at the file, run by the same `Pipeline`
class the live cameras use - same models, same cadences, same tracker and zone engine - so
what it detects is what a live camera would detect. It is deliberately NOT registered in the
camera registry (so the pipeline manager never sees or reconciles it and it never shows up on
the Cameras page), and its alerts go to the test page, not to the alert history, e-mail or S3,
unless the user ticks "send as real alerts".

Only one session runs at a time - each is a full detection pipeline on a CPU-only machine.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from perimeter.boundary.zones import zone_from_dict
from perimeter.cameras.models import (
    ALL_MODULES,
    MODULE_BOUNDARY,
    MODULE_CROWD,
    MODULE_FIRE_SMOKE,
    MODULE_IDENTITY,
    MODULE_PHONE,
    MODULE_PPE,
    MODULE_WELDING,
    REQUIRES_BOUNDARY,
    camera_from_dict,
)
from perimeter.testvideo.store import TestVideoStore, UploadedVideo

log = logging.getLogger("perimeter.testvideo.sessions")

MAX_ALERTS_KEPT = 500
TEST_PREFIX = "[TEST VIDEO] "

MODULE_INFO = {
    MODULE_BOUNDARY: "People & boundary zones (person detection and tracking)",
    MODULE_FIRE_SMOKE: "Fire & smoke",
    MODULE_CROWD: "Crowd formation",
    MODULE_PPE: "PPE compliance",
    MODULE_IDENTITY: "Identity / face recognition",
    MODULE_PHONE: "Phone near ear",
    MODULE_WELDING: "Welding sparks (soft alert)",
}


class SessionError(ValueError):
    """A request that cannot be started; the message is shown to the user as-is."""


class ZoneView:
    """Read-only stand-in for `ZoneStore`: shows one camera's zones to the session's pipeline
    (or none), so a test video can be checked against real boundaries without copying them
    and without ever being able to edit them. Duck-types only what `BoundaryEngine` uses."""

    def __init__(
        self, store: Any, source_camera_id: str | None, custom: list | None = None
    ) -> None:
        self._store = store
        self._source = source_camera_id
        self._custom = custom
        self.last_error = None

    def zones(self, camera_id: str | None = None) -> list:  # noqa: ARG002
        if self._custom is not None:
            return list(self._custom)
        return list(self._store.zones(self._source)) if self._source else []

    @property
    def version(self) -> int:
        if self._custom is not None:
            return 1
        return self._store.version if self._source else 0

    def refresh_in_background(self) -> None:
        return None

    def reload(self, force: bool = False) -> bool:  # noqa: ARG002
        return False


@dataclass
class TestAlert:
    id: str
    ts: float
    video_time_s: float
    kind: str
    subtype: str | None
    severity: str
    message: str
    zone_name: str | None = None
    identity_status: str | None = None
    identity_name: str | None = None
    identity_confidence: float | None = None
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class _SessionAlerts:
    """What the pipeline calls instead of the real `AlertBus`: `publish(payload)`."""

    def __init__(self, session: TestSession, real_bus: Any | None) -> None:
        self._session = session
        self._real_bus = real_bus

    def publish(self, payload: dict[str, Any]) -> None:
        session = self._session
        # `video_pos_s` is the frame's actual position in the source file's own
        # timeline - required once a session can run faster than real time (a whole
        # video finishes in well under its own length), where wall-clock elapsed since
        # session start no longer has anything to do with where in the video this
        # happened. Falls back to wall-clock elapsed for a live-camera source, where
        # "position in the file" has no meaning and realtime pacing keeps the two equal
        # anyway.
        video_pos_s = payload.get("video_pos_s")
        if video_pos_s is None:
            video_pos_s = time.time() - session.started_at
        alert = TestAlert(
            id=uuid.uuid4().hex[:12],
            ts=payload.get("ts") or time.time(),
            video_time_s=round(video_pos_s, 1),
            kind=str(payload.get("kind")),
            subtype=payload.get("subtype"),
            severity=str(payload.get("severity", "medium")),
            message=str(payload.get("message", "")),
            zone_name=payload.get("zone_name"),
            identity_status=payload.get("identity_status"),
            identity_name=payload.get("identity_name"),
            identity_confidence=payload.get("identity_confidence"),
            track_id=payload.get("track_id"),
        )
        with session.lock:
            session.alerts.append(alert)
        if self._real_bus is not None:
            # Opt-in: goes through the normal path (history, e-mail, clips), clearly labelled.
            self._real_bus.publish({**payload, "message": TEST_PREFIX + alert.message})


@dataclass
class TestSession:
    __test__ = False

    id: str
    video: UploadedVideo
    modules: list[str]
    zones_from: str | None
    loop: bool
    send_alerts: bool
    started_at: float = field(default_factory=time.time)
    pipeline: Any = None
    camera_id: str = ""
    stopped: bool = False
    alerts: deque = field(default_factory=lambda: deque(maxlen=MAX_ALERTS_KEPT))
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def status(self) -> str:
        if self.stopped:
            return "stopped"
        if self.pipeline is not None and getattr(self.pipeline, "capture_finished", False):
            return "finished"
        return "running"

    def to_dict(self, since: int = 0) -> dict[str, Any]:
        stats = getattr(self.pipeline, "stats", None)
        with self.lock:
            alerts = list(self.alerts)
        return {
            "id": self.id,
            "status": self.status,
            "video": self.video.to_dict(),
            "modules": self.modules,
            "zones_from": self.zones_from,
            "loop": self.loop,
            "send_alerts": self.send_alerts,
            "elapsed_s": round(time.time() - self.started_at, 1),
            "stats": {
                "frames": getattr(stats, "frames_decoded", 0),
                "detections": getattr(stats, "detections_run", 0),
                "people": getattr(stats, "people_tracked", 0),
                "inference_ms": round(getattr(stats, "inference_ms", 0.0), 1),
            },
            "alert_count": len(alerts),
            "alerts": [a.to_dict() for a in alerts[since:]],
        }


PipelineFactory = Callable[[Any, Any, Any], Any]  # (camera, zone_view, alerts_sink) -> pipeline


class TestSessionManager:
    __test__ = False

    def __init__(
        self,
        store: TestVideoStore,
        zone_store: Any,
        model_info: Callable[[], dict[str, Any]],
        pipeline_factory: PipelineFactory,
        real_bus: Any | None = None,
        camera_ids: Callable[[], list[str]] = lambda: [],
    ) -> None:
        self.store = store
        self._zone_store = zone_store
        self._model_info = model_info
        self._factory = pipeline_factory
        self._real_bus = real_bus
        self._camera_ids = camera_ids
        self._lock = threading.Lock()
        self._current: TestSession | None = None

    # -- what can be chosen ----------------------------------------------------

    def modules(self) -> list[dict[str, Any]]:
        """Every module with whether it can run here, and why not when it can't."""
        info = self._model_info()
        out = []
        for module in sorted(ALL_MODULES, key=lambda m: list(MODULE_INFO).index(m)):
            available, reason = True, ""
            if module == MODULE_FIRE_SMOKE and not info.get("firesmoke"):
                available, reason = False, "no fire/smoke model file is installed"
            elif module == MODULE_PPE and not info.get("ppe"):
                available, reason = False, "no PPE model file is installed"
            elif module == MODULE_IDENTITY and not info.get("identity"):
                available, reason = False, "face recognition is not enabled on this server"
            elif module == MODULE_PHONE:
                available, reason = False, "not implemented yet"
            out.append({
                "id": module,
                "label": MODULE_INFO.get(module, module),
                "available": available,
                "reason": reason,
                "requires_boundary": module in REQUIRES_BOUNDARY,
            })
        return out

    # -- lifecycle -------------------------------------------------------------

    def start(
        self,
        video_id: str,
        modules: list[str],
        zones_from: str | None = None,
        loop: bool = False,
        send_alerts: bool = False,
        zones: list[dict[str, Any]] | None = None,
    ) -> TestSession:
        video = self.store.get(video_id)
        if video is None:
            raise SessionError("That video no longer exists - upload it again.")
        chosen = [str(m).strip().lower() for m in modules]
        if not chosen:
            raise SessionError("Choose at least one model to run.")
        available = {m["id"]: m for m in self.modules()}
        for module in chosen:
            if module not in available:
                raise SessionError(f"Unknown model {module!r}.")
            if not available[module]["available"]:
                why = available[module]["reason"]
                raise SessionError(f"{MODULE_INFO[module]} is unavailable: {why}.")
        if zones_from and zones_from not in self._camera_ids():
            raise SessionError(f"Camera {zones_from!r} does not exist, so its zones cannot be used")

        custom_zones = None
        if zones:
            try:
                custom_zones = [zone_from_dict(z) for z in zones]
            except ValueError as exc:
                raise SessionError(f"Invalid boundary: {exc}") from exc

        path = self.store.path(video_id)
        session_id = secrets.token_hex(4)
        camera = camera_from_dict({
            "id": f"test_{session_id}",
            "name": f"Test: {video.name}",
            "source_type": "webcam",  # a file path in `source` is opened as a video file
            "source": str(path),
            "width": min(7680, max(64, video.width)),
            "height": min(7680, max(64, video.height)),
            "fps": int(min(240, max(1, round(video.fps or 30)))),
            "decode_fps": 12.0,
            "enabled": True,
            "enabled_modules": chosen,
        })
        session = TestSession(
            id=session_id, video=video, modules=sorted(camera.enabled_modules),
            zones_from=zones_from, loop=loop, send_alerts=send_alerts, camera_id=camera.id,
        )
        sink = _SessionAlerts(session, self._real_bus if send_alerts else None)

        with self._lock:
            self._stop_locked()
            view = ZoneView(self._zone_store, zones_from, custom_zones)
            session.pipeline = self._factory(camera, view, sink)
            if not loop and hasattr(session.pipeline, "play_once"):
                session.pipeline.play_once()
            session.pipeline.start()
            self._current = session
        log.info("test session %s started: %s with %s", session.id, video.name, session.modules)
        return session

    def current(self) -> TestSession | None:
        with self._lock:
            return self._current

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        session = self._current
        if session is None:
            return
        session.stopped = True
        try:
            session.pipeline.stop()
        except Exception:  # noqa: BLE001 - stopping must never raise into the request
            log.exception("could not stop test session %s cleanly", session.id)
        log.info("test session %s stopped", session.id)
        self._current = None

    def pipeline_for(self, camera_id: str) -> Any | None:
        """The running pipeline for a `test_*` camera id (streams, and evidence lookup)."""
        session = self.current()
        if session is not None and session.camera_id == camera_id and not session.stopped:
            return session.pipeline
        return None


def pipeline_factory_from(manager: Any) -> PipelineFactory:
    """Build session pipelines with the same models and settings as the live cameras'."""
    from perimeter.pipeline import Pipeline

    def build(camera: Any, zone_view: Any, sink: Any) -> Any:
        return Pipeline(
            manager.settings, camera, zone_view, manager.model_path,
            alerts=sink,
            firesmoke_model_path=manager.firesmoke_model_path,
            identity_resolver=manager.identity_resolver,
            ppe_model_path=manager.ppe_model_path,
            capture_realtime=False,
        )

    return build


def model_info_from(manager: Any) -> Callable[[], dict[str, Any]]:
    return lambda: {
        "firesmoke": bool(manager.firesmoke_model_path),
        "ppe": bool(manager.ppe_model_path),
        "identity": manager.identity_resolver is not None,
    }
