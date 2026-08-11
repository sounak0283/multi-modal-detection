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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from fsbd.boundary.geometry import polygon_warnings, validate_polygon, validate_tripwire
from fsbd.boundary.zones import AREA_TYPES, ZoneStore, ZoneType, zone_from_dict
from fsbd.pipeline import Pipeline
from fsbd.settings import Settings, redact

log = logging.getLogger("fsbd.api")

WEB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web"
STREAM_BOUNDARY = "frame"


def create_app(settings: Settings, store: ZoneStore, pipeline: Pipeline | None) -> FastAPI:
    app = FastAPI(title="Fire, Smoke & Boundary Detection", docs_url="/api/docs")
    router = APIRouter(prefix="/api")

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
            "camera_id": settings.camera.id,
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
    def events(limit: int = 50) -> dict[str, Any]:
        active = require_pipeline()
        return {
            "events": [
                {
                    "ts": e.ts,
                    "zone_id": e.zone_id,
                    "zone_name": e.zone_name,
                    "kind": e.kind.value,
                    "track_id": e.track_id,
                    "severity": e.severity.value,
                    "bbox": list(e.bbox),
                    "description": e.describe(),
                }
                for e in active.recent_events(limit)
            ]
        }

    @router.get("/health")
    def health() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "camera_id": settings.camera.id,
            # redact(): the camera URL embeds a password and this endpoint ends up in
            # screenshots and support bundles.
            "source": redact(settings.camera.inference_source),
            "zones": len(store.zones()),
            "zones_error": store.last_error,
            "pipeline": pipeline is not None,
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
