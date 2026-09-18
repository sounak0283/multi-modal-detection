"""PipelineManager - one Pipeline per enabled camera (Expansion Plan Phase A.2).

Replaces `main.py` constructing a single global `Pipeline`. A background thread polls
`CameraRegistry` (throttled the same way `ZoneStore.reload` throttles its own MongoDB
polling - see boundary.zones) and reconciles the set of running `Pipeline` instances
against it: a camera added or set `enabled: true` gets a new `Pipeline` started; one
removed or disabled gets stopped; one whose connection-relevant fields changed
(source/resolution/fps) is reconfigured in place via `Pipeline.reconfigure`.

`enabled_modules` is read by `Pipeline` itself (Expansion Plan §5) - this manager's job
ends at "does a Pipeline exist and is it pointed at the right camera fields", not at
deciding which detectors that Pipeline builds internally.
"""

from __future__ import annotations

import logging
import threading

from perimeter.alerts.bus import AlertBus
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.models import Camera
from perimeter.cameras.registry import CameraRegistry
from perimeter.identity.resolver import IdentityResolver
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings

log = logging.getLogger("perimeter.cameras.manager")

POLL_INTERVAL_S = 5.0

# Fields that require calling Pipeline.reconfigure() to actually take effect on an
# already-running camera - `_reconcile()` otherwise only updates its own bookkeeping
# (`self._cameras`), never Pipeline.camera itself, so a change to a field outside this
# list is silently invisible to a running pipeline until it is next restarted.
# Connection-related fields need the underlying CaptureThread rebound; `enabled_modules`
# doesn't strictly need that, but reconfigure() is the only existing mechanism that
# rebuilds a running Pipeline from a new Camera at all (Expansion Plan Phase E - toggling
# fire_smoke on the Cameras page for an already-running camera must not be a silent
# no-op until the whole process restarts).
_RECONNECT_FIELDS = (
    "source_type", "source", "rtsp_url", "substream", "width", "height", "fps", "decode_fps",
    "enabled_modules",
)


class PipelineManager:
    def __init__(
        self,
        settings: Settings,
        registry: CameraRegistry,
        store: ZoneStore,
        model_path: str,
        alerts: AlertBus | None = None,
        poll_interval: float = POLL_INTERVAL_S,
        firesmoke_model_path: str | None = None,
        identity_resolver: IdentityResolver | None = None,
        ppe_model_path: str | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.store = store
        self.model_path = model_path
        self.alerts = alerts
        self.firesmoke_model_path = firesmoke_model_path
        self.identity_resolver = identity_resolver
        self.ppe_model_path = ppe_model_path
        self.poll_interval = poll_interval

        self._pipelines: dict[str, Pipeline] = {}
        self._cameras: dict[str, Camera] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Guards notify_camera_changed() and the poll loop: without it, creating a
        # camera through the API would try to open real hardware and load the ONNX
        # model immediately, even under --no-pipeline or in a unit test that never
        # called start() - both legitimate "API only, no cameras running" states.
        self._started = False

    def start(self) -> None:
        self._started = True
        self._reconcile()
        self._thread = threading.Thread(target=self._run, name="pipeline-manager", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        with self._lock:
            for pipeline in self._pipelines.values():
                pipeline.stop(timeout=timeout)
            self._pipelines.clear()
            self._cameras.clear()

    def get(self, camera_id: str) -> Pipeline | None:
        with self._lock:
            return self._pipelines.get(camera_id)

    def all(self) -> dict[str, Pipeline]:
        with self._lock:
            return dict(self._pipelines)

    def notify_camera_changed(self, camera_id: str) -> None:
        """Called by the API right after a create/update/delete so a change is picked up
        immediately rather than waiting for the next poll tick. A no-op if this manager
        was never started (--no-pipeline, or a test that only exercises the API)."""
        if self._started:
            self._reconcile()

    # -- internals -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                self._reconcile()
            except Exception:  # noqa: BLE001 - one bad poll must not kill the manager
                log.exception("pipeline reconciliation failed")

    def _reconcile(self) -> None:
        try:
            cameras = {c.id: c for c in self.registry.list() if c.enabled}
        except Exception:  # noqa: BLE001 - a transient Mongo hiccup keeps existing pipelines running
            log.exception("could not load camera registry; leaving existing pipelines as-is")
            return

        with self._lock:
            current_ids = set(self._pipelines)
            desired_ids = set(cameras)

            for camera_id in current_ids - desired_ids:
                log.info("camera %s removed/disabled - stopping its pipeline", camera_id)
                self._pipelines.pop(camera_id).stop()
                self._cameras.pop(camera_id, None)

            for camera_id in desired_ids - current_ids:
                self._start_pipeline(cameras[camera_id])

            for camera_id in desired_ids & current_ids:
                new_camera = cameras[camera_id]
                old_camera = self._cameras[camera_id]
                if new_camera == old_camera:
                    continue
                if any(
                    getattr(new_camera, f) != getattr(old_camera, f) for f in _RECONNECT_FIELDS
                ):
                    log.info("camera %s connection settings changed - reconfiguring", camera_id)
                    self._pipelines[camera_id].reconfigure(new_camera)
                self._cameras[camera_id] = new_camera

    def _start_pipeline(self, camera: Camera) -> None:
        log.info("starting pipeline for camera %s", camera)
        pipeline = Pipeline(
            self.settings, camera, self.store, self.model_path,
            alerts=self.alerts, firesmoke_model_path=self.firesmoke_model_path,
            identity_resolver=self.identity_resolver, ppe_model_path=self.ppe_model_path,
        )
        pipeline.start()
        self._pipelines[camera.id] = pipeline
        self._cameras[camera.id] = camera
