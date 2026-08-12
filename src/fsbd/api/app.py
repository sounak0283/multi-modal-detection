"""Dashboard API (PLAN.md section 6.6).

Serves the zone editor and the live view. Endpoints are deliberately small: the browser
draws, this validates and persists, and the engine hot-reloads from the same file.

Note the trust boundary. This binds to 127.0.0.1 by default because it has no
authentication - it is a commissioning tool, not an internet-facing service. Exposing it
on 0.0.0.0 without a reverse proxy and auth in front would let anyone on the network
redraw the zones that arm the site.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from fsbd.alerts.messages import boundary_message
from fsbd.boundary.geometry import polygon_warnings, validate_polygon, validate_tripwire
from fsbd.boundary.zones import AREA_TYPES, ZoneStore, ZoneType, zone_from_dict
from fsbd.capture.probe import probe_source
from fsbd.pipeline import Pipeline
from fsbd.settings import (
    CameraSettings,
    Settings,
    describe_source,
    load_settings,
    redact,
    save_camera_settings,
)

log = logging.getLogger("fsbd.api")

WEB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web"
STREAM_BOUNDARY = "frame"

MAX_DIMENSION = 7680  # 8K; beyond this is a typo, not a camera
MIN_DIMENSION = 64


def _camera_from_payload(payload: dict[str, Any], current: CameraSettings) -> CameraSettings:
    """Build CameraSettings from a partial dashboard payload.

    Fields absent from the payload keep their current value. The RTSP URL specifically:
    an absent or empty key means "leave what is stored alone", because the browser is
    never sent the real URL and therefore cannot echo it back. Only a non-empty string
    replaces it, and the sentinel "" with `clear_rtsp_url` true erases it.
    """

    def as_int(key: str, fallback: int, low: int, high: int) -> int:
        if key not in payload or payload[key] in (None, ""):
            return fallback
        try:
            value = int(payload[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a whole number") from exc
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        return value

    def as_float(key: str, fallback: float, low: float, high: float) -> float:
        if key not in payload or payload[key] in (None, ""):
            return fallback
        try:
            value = float(payload[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        return value

    use_cctv = bool(payload.get("use_cctv", current.use_cctv))

    rtsp_url = current.cctv_rtsp_url
    if payload.get("clear_rtsp_url"):
        rtsp_url = ""
    elif payload.get("rtsp_url"):
        rtsp_url = str(payload["rtsp_url"]).strip()

    if use_cctv and not rtsp_url:
        raise ValueError("CCTV is selected but no RTSP URL is set.")

    source = str(payload.get("source", current.source)).strip() or "0"
    substream = payload.get("substream")
    if substream is None:
        resolved_substream = current.substream
    else:
        resolved_substream = str(substream).strip() or None

    return CameraSettings(
        id=str(payload.get("id", current.id)).strip() or "cam_01",
        use_cctv=use_cctv,
        source=source,
        cctv_rtsp_url=rtsp_url,
        substream=resolved_substream,
        width=as_int("width", current.width, MIN_DIMENSION, MAX_DIMENSION),
        height=as_int("height", current.height, MIN_DIMENSION, MAX_DIMENSION),
        fps=as_int("fps", current.fps, 1, 240),
        decode_fps=as_float("decode_fps", current.decode_fps, 1.0, 120.0),
        autostart=bool(payload.get("autostart", current.autostart)),
    )


def _database_health(database: Any | None) -> dict[str, Any]:
    """Storage status, reported separately from pipeline status.

    They fail independently and the operator response differs: a dead camera means the
    site is unwatched, a dead database means alerts are firing but not being recorded.
    Collapsing them into one indicator would hide that distinction.
    """
    if database is None:
        return {"configured": False, "connected": False}
    return {"configured": True, "connected": database.healthy(), "dsn": database.dsn.safe}


def create_app(
    settings: Settings,
    store: ZoneStore,
    pipeline: Pipeline | None,
    database: Any | None = None,
) -> FastAPI:
    app = FastAPI(title="Fire, Smoke & Boundary Detection", docs_url="/api/docs")
    router = APIRouter(prefix="/api")

    # Settings are replaced wholesale when the Camera page saves, so they live in a
    # mutable box rather than being captured by value in each closure.
    state: dict[str, Any] = {"settings": settings, "database": database}

    def require_pipeline() -> Pipeline:
        if pipeline is None:
            raise HTTPException(503, "pipeline is not running")
        return pipeline

    # -- pages -------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/{filename}.js", include_in_schema=False)
    def script(filename: str) -> FileResponse:
        path = (WEB_DIR / f"{filename}.js").resolve()
        # Path traversal guard: a crafted filename must not escape web/.
        if not path.is_file() or WEB_DIR.resolve() not in path.parents:
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type="application/javascript")

    @app.get("/{filename}.css", include_in_schema=False)
    def stylesheet(filename: str) -> FileResponse:
        path = (WEB_DIR / f"{filename}.css").resolve()
        if not path.is_file() or WEB_DIR.resolve() not in path.parents:
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type="text/css")

    # -- zones -------------------------------------------------------------

    @router.get("/zones")
    def get_zones() -> dict[str, Any]:
        store.reload()
        return {
            "camera_id": state["settings"].camera.id,
            "zones": [z.to_dict() for z in store.zones()],
            "error": store.last_error,
            "version": store.version,
        }

    @router.put("/zones")
    def put_zones(payload: dict[str, Any]) -> dict[str, Any]:
        raw = payload.get("zones")
        if not isinstance(raw, list):
            raise HTTPException(400, "body must be {\"zones\": [...]}")

        try:
            zones = [zone_from_dict(z) for z in raw]
        except ValueError as exc:
            # 400 with the parser's own message: the editor shows it verbatim, so it has
            # to name the zone and the problem.
            raise HTTPException(400, str(exc)) from exc

        ids = [z.id for z in zones]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise HTTPException(400, f"duplicate zone id(s): {sorted(duplicates)}")

        store.save(zones)
        log.info("zones updated via dashboard: %d zone(s)", len(zones))
        return {"saved": len(zones), "version": store.version}

    @router.post("/zones/validate")
    def validate_zone(payload: dict[str, Any]) -> dict[str, Any]:
        """Check a single zone without saving - drives the editor's live feedback."""
        points = payload.get("points") or []
        try:
            zone_type = ZoneType(str(payload.get("type", "polygon")))
        except ValueError:
            raise HTTPException(400, f"unknown zone type {payload.get('type')!r}") from None

        errors = (
            validate_polygon(points) if zone_type in AREA_TYPES else validate_tripwire(points)
        )
        return {
            "valid": not errors,
            "errors": [{"field": e.field, "message": e.message} for e in errors],
            "warnings": polygon_warnings(points) if zone_type in AREA_TYPES else [],
        }

    # -- camera settings ---------------------------------------------------

    @router.get("/camera")
    def get_camera() -> dict[str, Any]:
        """Current camera settings.

        The RTSP URL is returned REDACTED and is write-only from the browser's point of
        view: you can replace it, never read it back. The dashboard has no authentication,
        so a readable password field would hand the camera to anyone who reached the page.
        `rtsp_url_set` tells the UI whether to show "configured" or an empty field.
        """
        camera = state["settings"].camera
        return {
            "id": camera.id,
            "use_cctv": camera.use_cctv,
            "source": str(camera.source),
            "rtsp_url_redacted": redact(camera.cctv_rtsp_url) if camera.cctv_rtsp_url else "",
            "rtsp_url_set": bool(camera.cctv_rtsp_url),
            "substream_set": bool(camera.substream),
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "decode_fps": camera.decode_fps,
            "autostart": camera.autostart,
            "resolved": describe_source(camera),
            "actual_size": list(pipeline.frame_size) if pipeline else None,
            "person_fps": round(
                state["settings"].inference.person_fps(camera.decode_fps), 2
            ),
        }

    @router.put("/camera")
    def put_camera(payload: dict[str, Any]) -> dict[str, Any]:
        """Persist camera settings to .env and hot-reconnect.

        Written to .env rather than app.yaml because every field is a per-deployment
        fact and one of them is a password: the git-ignored file is a stronger guarantee
        than remembering to redact a tracked one.
        """
        current = state["settings"].camera
        try:
            camera = _camera_from_payload(payload, current)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        env_file = state["settings"].env_file
        save_camera_settings(camera, env_file=env_file)
        state["settings"] = load_settings(env_file=env_file)

        if pipeline is not None:
            pipeline.reconfigure(state["settings"])

        return {"saved": True, "resolved": describe_source(state["settings"].camera)}

    @router.post("/camera/test")
    def test_camera(payload: dict[str, Any]) -> dict[str, Any]:
        """Open the source, grab one frame, report what actually came back.

        Reports the ACTUAL resolution, not the requested one. A camera silently ignoring
        a 1280x720 request changes what every pixel-based threshold means, and finding
        that out during commissioning is far cheaper than inferring it from poor recall.
        """
        current = state["settings"].camera
        try:
            candidate = _camera_from_payload(payload, current)
            source = candidate.resolved_source
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        return probe_source(source, candidate.width, candidate.height, candidate.fps)

    # -- video -------------------------------------------------------------

    @router.get("/snapshot")
    def snapshot(annotated: bool = False) -> Response:
        """A single JPEG. The editor freezes one of these as its drawing backdrop."""
        jpeg = require_pipeline().latest_jpeg(annotated=annotated)
        if jpeg is None:
            raise HTTPException(503, "no frame available yet")
        return Response(jpeg, media_type="image/jpeg")

    @router.get("/stream")
    def stream(annotated: bool = True) -> StreamingResponse:
        """MJPEG live view - the 'live test mode' from PLAN.md section 6.6 step 7."""
        active = require_pipeline()

        def frames():
            last: bytes | None = None
            while True:
                jpeg = active.latest_jpeg(annotated=annotated)
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield (
                        b"--" + STREAM_BOUNDARY.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                time.sleep(0.05)

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}",
        )

    # -- status ------------------------------------------------------------

    @router.get("/events")
    def events(
        limit: int = 50, kind: str | None = None, zone_id: str | None = None
    ) -> dict[str, Any]:
        """Alert history.

        Served from PostgreSQL when it is available, falling back to the pipeline's
        in-memory ring otherwise. The fallback matters: a database outage must not make
        the dashboard look like nothing is happening, which would be indistinguishable
        from a dead camera.
        """
        database = state["database"]
        if database is not None:
            try:
                return {
                    "events": database.recent_events(limit=limit, kind=kind, zone_id=zone_id),
                    "source": "postgres",
                }
            except Exception as exc:  # noqa: BLE001 - fall back rather than 500
                log.warning("event query failed, serving from memory: %s", exc)

        active = require_pipeline()
        return {
            "source": "memory",
            "events": [
                {
                    "id": None,
                    "ts": datetime.fromtimestamp(e.ts, tz=UTC).isoformat(),
                    "camera_id": e.camera_id,
                    "kind": "boundary",
                    "subtype": e.kind.value,
                    "zone_id": e.zone_id,
                    "zone_name": e.zone_name,
                    "track_id": e.track_id,
                    "severity": e.severity.value,
                    "message": boundary_message(e.kind, e.zone_name),
                    "bbox": list(e.bbox),
                    "delivered": False,
                }
                for e in active.recent_events(limit)
            ],
        }

    @router.get("/events/summary")
    def events_summary() -> dict[str, Any]:
        database = state["database"]
        if database is None:
            return {"available": False, "counts": {}, "today": {}}
        try:
            midnight = datetime.now(UTC).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return {
                "available": True,
                "counts": database.event_counts(),
                "today": database.event_counts(since=midnight),
            }
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": str(exc), "counts": {}, "today": {}}

    @router.get("/health")
    def health() -> dict[str, Any]:
        active_settings = state["settings"]
        payload: dict[str, Any] = {
            "camera_id": active_settings.camera.id,
            # describe_source(): the camera URL embeds a password and this endpoint ends
            # up in screenshots and support bundles. It also never raises on a
            # half-configured camera, which is a legitimate state on a fresh install.
            "source": describe_source(active_settings.camera),
            "zones": len(store.zones()),
            "zones_error": store.last_error,
            "pipeline": pipeline is not None,
            "database": _database_health(state["database"]),
        }
        if pipeline is not None:
            stats = pipeline.stats
            payload |= {
                "feed_state": stats.feed_state,
                "frames_decoded": stats.frames_decoded,
                "dropped_frames": stats.dropped_frames,
                "detections_run": stats.detections_run,
                "people_tracked": stats.people_tracked,
                "events_fired": stats.events_fired,
                "inference_ms": round(stats.inference_ms, 1),
                "render_fps": round(stats.render_fps, 1),
                "uptime_s": round(stats.uptime_s, 1),
                "frame_size": list(pipeline.frame_size),
            }
        return payload

    @router.get("/track-states")
    def track_states() -> JSONResponse:
        active = require_pipeline()
        return JSONResponse(active.engine.track_states())

    app.include_router(router)
    return app
