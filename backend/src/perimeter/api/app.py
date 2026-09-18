"""Dashboard API (Expansion Plan Phase A; PLAN.md section 6.6).

Serves the camera registry, the zone editor and the live view for any number of cameras.
Endpoints are deliberately small: the browser draws/configures, this validates and
persists, and `PipelineManager`/`ZoneStore` pick the change up without a restart.

Every route below requires a logged-in session (Expansion Plan Phase B): a signed,
httpOnly session cookie, not a JWT - see `auth/sessions.py`. Two roles gate what a
session can do - `admin` (full read/write) and `operator` (read-only) - both funnelled
through the single `require_role()` dependency defined in `create_app` below. Static
frontend assets (`_mount_frontend`) are the one thing served without a session, since
they carry no secrets and the SPA itself is what calls `/api/auth/me` to decide whether
to show the login page.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from perimeter.alerts.cooldown import alert_config_from_dict
from perimeter.auth.models import Role, User, normalise_email, user_from_dict, validate_password
from perimeter.auth.passwords import hash_password, verify_password
from perimeter.auth.registry import UserRegistry
from perimeter.auth.sessions import Session, SessionManager
from perimeter.boundary.geometry import polygon_warnings, validate_polygon, validate_tripwire
from perimeter.boundary.zones import AREA_TYPES, ZoneStore, ZoneType, zone_from_dict
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.models import Camera, camera_from_dict, describe_source
from perimeter.cameras.registry import CameraRegistry
from perimeter.capture.probe import probe_source
from perimeter.identity.gallery import PersonRecord
from perimeter.identity.resolver import IdentityResolver
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings

log = logging.getLogger("perimeter.api")

# Enrollment requires one photo per pose (Expansion Plan Phase F). A single canonical
# photo matches poorly against a live boundary-crossing frame, which rarely presents a
# straight-on face - five angles give the matcher something close to every real crossing.
REQUIRED_POSES = {"front", "left", "right", "up", "down"}

# Checked on create only: a camera id is used verbatim in /api/cameras/{id}/..., so "/" or
# whitespace would make the camera unreachable (and undeletable) through its own routes.
CAMERA_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")

# Storage-outage tolerance for authentication (see get_current_user).
USER_CACHE_MAX_AGE_S = 15 * 60
STORAGE_RETRY_AFTER_S = 15.0
STORAGE_DOWN_DETAIL = "storage is temporarily unavailable - live detection continues"

SESSION_COOKIE = "perimeter_session"
ALL_ROLES = (Role.ADMIN.value, Role.OPERATOR.value)
ADMIN_ONLY = (Role.ADMIN.value,)

WEB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web"
# Vite build output. Not in git - built from frontend/ during packaging.
DIST_DIR = WEB_DIR / "dist"
STREAM_BOUNDARY = "frame"


def _database_health(database: Any | None) -> dict[str, Any]:
    """Storage status, reported separately from pipeline status.

    They fail independently and the operator response differs: a dead camera means a
    site is unwatched, a dead database means alerts (and camera/zone configuration
    changes) are not being recorded. Collapsing them into one indicator would hide that
    distinction.
    """
    if database is None:
        return {"configured": False, "connected": False}
    # Prefer the instant driver-state check: a blocking ping here stalled the health
    # endpoint for the full driver timeout during exactly the outage it should report.
    available = getattr(database, "available", None)
    connected = available() if available is not None else database.healthy()
    return {"configured": True, "connected": connected, "db_name": database.db_name}


def _camera_public(camera: Camera, pipeline: Pipeline | None, settings: Settings) -> dict[str, Any]:
    payload = camera.to_public_dict()
    payload["resolved"] = describe_source(camera)
    payload["running"] = pipeline is not None
    payload["person_fps"] = round(settings.inference.person_fps(camera.decode_fps), 2)
    if pipeline is not None:
        stats = pipeline.stats
        payload["stats"] = {
            "feed_state": stats.feed_state,
            "frames_decoded": stats.frames_decoded,
            "dropped_frames": stats.dropped_frames,
            "detections_run": stats.detections_run,
            "people_tracked": stats.people_tracked,
            "zone_occupancy": pipeline.engine.zone_occupancy(),
            "events_fired": stats.events_fired,
            "inference_ms": round(stats.inference_ms, 1),
            "render_fps": round(stats.render_fps, 1),
            "uptime_s": round(stats.uptime_s, 1),
            "frame_size": list(pipeline.frame_size),
        }
    return payload


def create_app(
    settings: Settings,
    store: ZoneStore,
    registry: CameraRegistry,
    manager: PipelineManager,
    users: UserRegistry,
    database: Any | None = None,
    alerts: Any | None = None,
    evidence_store: Any | None = None,
    alert_config_store: Any | None = None,
    identity_resolver: IdentityResolver | None = None,
) -> FastAPI:
    app = FastAPI(title="Fire, Smoke & Boundary Detection", docs_url="/api/docs")

    @app.exception_handler(PyMongoError)
    async def _storage_unavailable(_request: Request, exc: PyMongoError) -> JSONResponse:
        # A database outage is a 503 the dashboard can explain, not a 500 stack trace.
        log.warning("request failed, storage unavailable: %s", exc.__class__.__name__)
        return JSONResponse(
            status_code=503,
            content={"detail": STORAGE_DOWN_DETAIL},
        )
    router = APIRouter(prefix="/api")
    sessions = SessionManager(settings.auth.session_secret, settings.auth.session_max_age_s)

    # -- auth ------------------------------------------------------------------

    def get_current_user(request: Request) -> User:
        """Resolve the session cookie to a live, active user - on every request.

        Deliberately re-reads from `UserRegistry` rather than trusting anything cached
        in the cookie: a role change or deactivation must take effect on the very next
        request, not the next login (see `auth/sessions.py`'s module docstring).
        """
        session = _session_or_401(request)
        user = _lookup_user(session.user_id)
        if user is None or not user.active:
            raise HTTPException(401, "account no longer active")
        if user.session_version != session.session_version:
            raise HTTPException(401, "session has been signed out")
        return user

    def _session_or_401(request: Request) -> Session:
        session = sessions.resolve(request.cookies.get(SESSION_COOKIE))
        if session is None or _is_revoked(session.session_id):
            raise HTTPException(401, "not authenticated")
        return session

    # -- storage-outage tolerance for auth --------------------------------------
    # Every route authenticates against MongoDB. Without a fallback, a database outage
    # made the whole dashboard - live view and in-memory live events included - return
    # errors after a 10 s driver timeout per request, while detection itself carried on.
    # When storage is reachable a user is still re-read on every request; only while it
    # is not do recently verified accounts keep working, and a short circuit breaker
    # stops each request from waiting out the driver timeout again.
    user_cache: dict[str, tuple[User, float]] = {}
    revoked_locally: dict[str, float] = {}
    storage_down_until = [0.0]

    def _storage_failed() -> None:
        storage_down_until[0] = time.monotonic() + STORAGE_RETRY_AFTER_S

    def _cached_user_or_503(user_id: str) -> User:
        cached = user_cache.get(user_id)
        if cached is not None and time.monotonic() - cached[1] <= USER_CACHE_MAX_AGE_S:
            return cached[0]
        raise HTTPException(503, "account storage is temporarily unavailable")

    def _lookup_user(user_id: str) -> User | None:
        if time.monotonic() < storage_down_until[0] or not _storage_up():
            return _cached_user_or_503(user_id)
        try:
            user = users.get(user_id)
        except PyMongoError:
            _storage_failed()
            return _cached_user_or_503(user_id)
        if user is None:
            user_cache.pop(user_id, None)
        else:
            user_cache[user_id] = (user, time.monotonic())
        return user

    def _is_revoked(session_id: str) -> bool:
        if session_id in revoked_locally:
            return True
        if time.monotonic() < storage_down_until[0] or not _storage_up():
            return False
        try:
            return users.is_session_revoked(session_id)
        except PyMongoError:
            _storage_failed()
            return False

    def require_role(*roles: str):
        """The single dependency every route in this file runs through (Expansion Plan
        Phase B §8): resolves the session, then checks the role against `roles`.
        """

        def _dependency(user: User = Depends(get_current_user)) -> User:
            if user.role.value not in roles:
                raise HTTPException(403, f"requires role: {' or '.join(roles)}")
            return user

        return _dependency

    # Built once rather than calling require_role(...) inline at each route: every route
    # below references one of these two, not a fresh call - clearer, and avoids
    # reconstructing the same closure on every request.
    require_admin = require_role(*ADMIN_ONLY)
    require_any_role = require_role(*ALL_ROLES)

    def get_user_or_404(user_id: str) -> User:
        user = users.get(user_id)
        if user is None:
            raise HTTPException(404, f"user {user_id!r} not found")
        return user

    def _set_session_cookie(response: Response, user: User) -> None:
        response.set_cookie(
            SESSION_COOKIE,
            sessions.issue(user.id, user.session_version),
            max_age=settings.auth.session_max_age_s,
            httponly=True,
            secure=settings.auth.cookie_secure,
            samesite="lax",
            path="/",
        )

    @router.post("/auth/login")
    def login(payload: dict[str, Any], response: Response) -> dict[str, Any]:
        email = normalise_email(payload.get("email", ""))
        password = str(payload.get("password", ""))
        user = users.get_by_email(email) if email else None
        # Same generic message whether the email is unknown or the password is wrong -
        # confirming an account exists is its own information leak.
        if user is None or not user.active or not verify_password(password, user.password_hash):
            raise HTTPException(401, "invalid email or password")
        user_cache[user.id] = (user, time.monotonic())
        _set_session_cookie(response, user)
        return user.to_public_dict()

    @router.post("/auth/logout")
    def logout(
        request: Request, response: Response, _user: User = Depends(get_current_user)
    ) -> dict[str, Any]:
        # Clearing the cookie alone leaves a copied token valid until it expires. Only
        # this session is revoked - the same account signed in on another screen stays in.
        session_id = _session_or_401(request).session_id
        now = time.monotonic()
        for sid, expires in list(revoked_locally.items()):
            if expires <= now:
                revoked_locally.pop(sid, None)
        revoked_locally[session_id] = now + sessions.max_age_s
        try:
            users.revoke_session(session_id, sessions.max_age_s)
        except PyMongoError as exc:
            log.warning("logout revocation not persisted (storage unavailable): %s", exc)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    @router.get("/auth/me")
    def me(user: User = Depends(require_any_role)) -> dict[str, Any]:
        return user.to_public_dict()

    # -- users (admin only) ------------------------------------------------------

    @router.get("/users")
    def list_users(_admin: User = Depends(require_admin)) -> dict[str, Any]:
        return {"users": [u.to_public_dict() for u in users.list()]}

    @router.post("/users")
    def create_user(
        payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            password = validate_password(str(payload.get("password", "")))
            user = user_from_dict(payload | {"password_hash": hash_password(password)})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            users.create(user)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return user.to_public_dict()

    @router.put("/users/{user_id}")
    def update_user(
        user_id: str, payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        current = get_user_or_404(user_id)
        merged = current.to_dict()
        merged["email"] = current.email  # email (= id) is immutable via this route
        merged["role"] = payload.get("role", current.role.value)
        merged["active"] = payload.get("active", current.active)
        new_password = payload.get("password")

        if _would_remove_last_admin(current, merged):
            raise HTTPException(409, "cannot remove the last active admin account")

        try:
            if new_password:
                merged["password_hash"] = hash_password(validate_password(str(new_password)))
                merged["session_version"] = current.session_version + 1
            user = user_from_dict(merged)
            users.update(user_id, user)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return user.to_public_dict()

    @router.delete("/users/{user_id}")
    def delete_user(
        user_id: str, _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        current = get_user_or_404(user_id)
        if current.role is Role.ADMIN and current.active and users.count_active_admins() <= 1:
            raise HTTPException(409, "cannot remove the last active admin account")
        try:
            users.delete(user_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"deleted": user_id}

    def _would_remove_last_admin(current: User, merged: dict[str, Any]) -> bool:
        was_active_admin = current.role is Role.ADMIN and current.active
        stays_active_admin = (
            str(merged.get("role")) == Role.ADMIN.value and bool(merged.get("active"))
        )
        return was_active_admin and not stays_active_admin and users.count_active_admins() <= 1

    def _storage_up() -> bool:
        available = getattr(database, "available", None)
        return available is None or available()

    def _require_storage() -> None:
        if not _storage_up():
            raise HTTPException(503, STORAGE_DOWN_DETAIL)

    def get_camera_or_404(camera_id: str) -> Camera:
        try:
            if not _storage_up():
                raise ServerSelectionTimeoutError("no reachable server")
            camera = registry.get(camera_id)
        except PyMongoError:
            # The live view polls this every couple of seconds; during an outage answer
            # from the running pipeline so the operator keeps seeing the camera.
            pipeline = manager.get(camera_id)
            if pipeline is None:
                raise HTTPException(503, STORAGE_DOWN_DETAIL) from None
            return pipeline.camera
        if camera is None:
            raise HTTPException(404, f"camera {camera_id!r} not found")
        return camera

    # -- cameras -------------------------------------------------------------

    @router.get("/cameras")
    def list_cameras(_user: User = Depends(require_any_role)) -> dict[str, Any]:
        pipelines = manager.all()
        try:
            if not _storage_up():
                raise ServerSelectionTimeoutError("no reachable server")
            cameras = registry.list()
        except PyMongoError:
            cameras = [p.camera for p in pipelines.values()]
        return {"cameras": [_camera_public(c, pipelines.get(c.id), settings) for c in cameras]}

    @router.post("/cameras")
    def create_camera(
        payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            camera = camera_from_dict(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not CAMERA_ID_PATTERN.fullmatch(camera.id):
            raise HTTPException(
                400,
                f"camera id {camera.id!r} must be 1-64 letters, digits, '_' or '-', starting "
                "with a letter or digit - it is used verbatim in URLs",
            )
        try:
            registry.create(camera)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        manager.notify_camera_changed(camera.id)
        return _camera_public(camera, manager.get(camera.id), settings)

    @router.get("/cameras/{camera_id}")
    def get_camera(
        camera_id: str, _user: User = Depends(require_any_role)
    ) -> dict[str, Any]:
        camera = get_camera_or_404(camera_id)
        return _camera_public(camera, manager.get(camera_id), settings)

    @router.put("/cameras/{camera_id}")
    def update_camera(
        camera_id: str, payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        current = get_camera_or_404(camera_id)
        merged = current.to_dict()
        merged.update({k: v for k, v in payload.items() if k != "rtsp_url" or v})
        merged["id"] = camera_id  # id is immutable via this route
        try:
            camera = camera_from_dict(merged)
            registry.update(camera_id, camera)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        manager.notify_camera_changed(camera_id)
        return _camera_public(camera, manager.get(camera_id), settings)

    @router.delete("/cameras/{camera_id}")
    def delete_camera(
        camera_id: str, _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            registry.delete(camera_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        manager.notify_camera_changed(camera_id)
        return {"deleted": camera_id}

    @router.post("/cameras/test")
    def test_camera(
        payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        """Open a candidate source, grab one frame, report what actually came back.

        Reports the ACTUAL resolution, not the requested one - a camera silently
        ignoring a 1280x720 request changes what every pixel-based threshold means, and
        finding that out during commissioning is far cheaper than inferring it from poor
        recall. Works for both a not-yet-created camera (the "add camera" form) and an
        existing one being re-tested, since the payload always carries the full shape.
        """
        try:
            candidate = camera_from_dict(payload | {"id": payload.get("id") or "_test"})
            source = candidate.resolved_source
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return probe_source(source, candidate.width, candidate.height, candidate.fps)

    # -- zones (per camera) --------------------------------------------------

    @router.get("/cameras/{camera_id}/zones")
    def get_zones(
        camera_id: str, _user: User = Depends(require_any_role)
    ) -> dict[str, Any]:
        get_camera_or_404(camera_id)
        store.reload()
        return {
            "camera_id": camera_id,
            "zones": [z.to_dict() for z in store.zones(camera_id)],
            "error": store.last_error,
            "version": store.version,
        }

    @router.put("/cameras/{camera_id}/zones")
    def put_zones(
        camera_id: str, payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        get_camera_or_404(camera_id)
        raw = payload.get("zones")
        if not isinstance(raw, list):
            raise HTTPException(400, "body must be {\"zones\": [...]}")

        try:
            zones = [zone_from_dict(z) for z in raw]
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        try:
            store.save(zones, camera_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        log.info("zones updated via dashboard: %d zone(s) for %s", len(zones), camera_id)
        return {"saved": len(zones), "version": store.version}

    @router.post("/zones/validate")
    def validate_zone(
        payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
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
            "warnings": polygon_warnings(points) if zone_type in AREA_TYPES and not errors else [],
        }

    # -- video (per camera) ---------------------------------------------------

    def require_pipeline(camera_id: str) -> Pipeline:
        get_camera_or_404(camera_id)
        pipeline = manager.get(camera_id)
        if pipeline is None:
            raise HTTPException(503, f"camera {camera_id!r} has no running pipeline")
        return pipeline

    @router.get("/cameras/{camera_id}/snapshot")
    def snapshot(
        camera_id: str, annotated: bool = False, _user: User = Depends(require_any_role)
    ) -> Response:
        """A single JPEG. The editor freezes one of these as its drawing backdrop."""
        jpeg = require_pipeline(camera_id).latest_jpeg(annotated=annotated)
        if jpeg is None:
            raise HTTPException(503, "no frame available yet")
        return Response(jpeg, media_type="image/jpeg")

    @router.get("/cameras/{camera_id}/stream")
    def stream(
        camera_id: str, annotated: bool = True, _user: User = Depends(require_any_role)
    ) -> StreamingResponse:
        """MJPEG live view - the 'live test mode' from PLAN.md section 6.6 step 7."""
        active = require_pipeline(camera_id)

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

    @router.get("/cameras/{camera_id}/track-states")
    def track_states(
        camera_id: str, _user: User = Depends(require_any_role)
    ) -> JSONResponse:
        active = require_pipeline(camera_id)
        return JSONResponse(active.engine.track_states())

    # -- events (across cameras, optionally filtered) -------------------------

    @router.get("/events")
    def events(
        limit: int = 50,
        kind: str | None = None,
        zone_id: str | None = None,
        camera_id: str | None = None,
        identity_status: str | None = None,
        _user: User = Depends(require_any_role),
    ) -> dict[str, Any]:
        """Alert history from MongoDB.

        Unlike the single-camera version there is no in-memory fallback here: with many
        cameras and no per-camera in-process ring guaranteed to still exist (a camera can
        be removed from the registry while its history should remain queryable), MongoDB
        is the only source of truth for history. `/api/events/live` remains the
        in-memory "what is happening right now" view for when storage is degraded.

        `identity_status=known` is what the People > Recognition log view filters on -
        every other consumer (AlertHistory) leaves it unset and sees everything.
        """
        if database is None:
            raise HTTPException(503, "event storage is not configured")
        _require_storage()
        try:
            return {
                "events": database.recent_events(
                    limit=limit,
                    kind=kind,
                    zone_id=zone_id,
                    camera_id=camera_id,
                    identity_status=identity_status,
                ),
                "source": "mongodb",
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, f"event query failed: {exc}") from exc

    @router.get("/events/live")
    def events_live(
        limit: int = 20,
        camera_id: str | None = None,
        _user: User = Depends(require_any_role),
    ) -> dict[str, Any]:
        """Alerts held in memory, never from the database.

        Deliberately a separate endpoint from /events. The Live view answers "what is
        happening", the History view answers "what has been recorded" - and those differ
        precisely when something is wrong.
        """
        if alerts is None:
            return {"events": []}
        recent = alerts.recent(limit)
        if camera_id:
            recent = [e for e in recent if e.get("camera_id") == camera_id]
        return {"events": recent}

    @router.get("/events/summary")
    def events_summary(
        camera_id: str | None = None, _user: User = Depends(require_any_role)
    ) -> dict[str, Any]:
        if database is None or not _storage_up():
            return {"available": False, "counts": {}, "today": {}, "entries_by_zone_today": {}}
        try:
            midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            return {
                "available": True,
                "counts": database.event_counts(camera_id=camera_id),
                "today": database.event_counts(since=midnight, camera_id=camera_id),
                "entries_by_zone_today": database.zone_entry_counts(
                    since=midnight, camera_id=camera_id
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "available": False,
                "error": str(exc),
                "counts": {},
                "today": {},
                "entries_by_zone_today": {},
            }

    # open_writer() (record.py) falls through mp4v -> XVID if the platform's OpenCV
    # build lacks an MP4 encoder, so the file actually on disk may be .avi - the
    # content-type served must match what is really there, not always assume .mp4.
    CLIP_MEDIA_TYPES = {".mp4": "video/mp4", ".avi": "video/x-msvideo"}

    def _evidence_file(
        event_id: str, field: str, default_media_type: str, media_types_by_suffix: dict[str, str]
    ) -> FileResponse:
        """Shared lookup for /snapshot and /clip: event exists, has that evidence key,
        and the file it points to is actually still on disk (Expansion Plan Phase C)."""
        if database is None:
            raise HTTPException(503, "event storage is not configured")
        if evidence_store is None:
            raise HTTPException(503, "evidence storage is not configured")
        event = database.get_event(event_id)
        if event is None:
            raise HTTPException(404, f"event {event_id!r} not found")
        key = event.get(field)
        if not key:
            raise HTTPException(404, f"no {field.removesuffix('_path')} captured for this event")
        path = evidence_store.resolve(key)
        if path is None:
            raise HTTPException(404, "evidence file is missing")
        media_type = media_types_by_suffix.get(path.suffix.lower(), default_media_type)
        return FileResponse(path, media_type=media_type)

    @router.get("/events/{event_id}/snapshot")
    def event_snapshot(
        event_id: str, _user: User = Depends(require_any_role)
    ) -> FileResponse:
        return _evidence_file(event_id, "snapshot_path", "image/jpeg", {})

    @router.get("/events/{event_id}/clip")
    def event_clip(event_id: str, _user: User = Depends(require_any_role)) -> FileResponse:
        return _evidence_file(event_id, "clip_path", "video/mp4", CLIP_MEDIA_TYPES)

    # -- alert config (Expansion Plan Phase D) --------------------------------

    @router.get("/alert-config")
    def get_alert_config(_user: User = Depends(require_any_role)) -> dict[str, Any]:
        """Recipients/severity/cooldown aren't secrets - readable by both roles, same
        tier as zones."""
        if alert_config_store is None:
            raise HTTPException(503, "alert config storage is not configured")
        alert_config_store.reload()
        return alert_config_store.get().to_dict()

    @router.put("/alert-config")
    def put_alert_config(
        payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        if alert_config_store is None:
            raise HTTPException(503, "alert config storage is not configured")
        try:
            config = alert_config_from_dict(payload)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        alert_config_store.save(config)
        return config.to_dict()

    # -- persons / identity gallery (Expansion Plan Phase F) -------------------

    def _refresh_gallery() -> None:
        """Rebuild the shared in-memory gallery after any enrollment/removal - mirrors
        `ZoneStore.reload`'s hot-reload posture rather than a live DB read per match."""
        if identity_resolver is None or database is None:
            return
        persons = [
            PersonRecord(
                id=p["id"], name=p["name"], external_id=p.get("external_id", ""),
                active=p.get("active", True), embeddings=p.get("embeddings", []),
            )
            for p in database.list_persons()
        ]
        identity_resolver.gallery.rebuild(persons)

    def _person_public(person: dict[str, Any]) -> dict[str, Any]:
        """Never echoes the biometric embeddings back to the browser."""
        person = dict(person)
        person.pop("embeddings", None)
        return person

    @router.get("/persons")
    def list_persons(_user: User = Depends(require_any_role)) -> dict[str, Any]:
        if database is None:
            raise HTTPException(503, "person storage is not configured")
        return {"persons": [_person_public(p) for p in database.list_persons()]}

    def _decode_image(image_b64: str, pose: str):
        try:
            image_bytes = base64.b64decode(str(image_b64).split(",")[-1])
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001 - any decode failure is a 400, not a 500
            raise HTTPException(400, f"could not decode the '{pose}' photo: {exc}") from exc
        if image is None:
            raise HTTPException(400, f"could not decode the '{pose}' photo")
        return image

    @router.post("/persons")
    def create_person(
        payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        """Enrol one person from `REQUIRED_POSES` photos, one per named pose. Each photo
        is decoded and embedded server-side (YuNet + SFace) - the browser never computes
        or sees a face embedding. Requiring several angles (rather than one canonical
        photo) measurably improves live match quality, since a boundary crossing rarely
        presents a straight-on face the way an enrollment photo does."""
        if database is None:
            raise HTTPException(503, "person storage is not configured")
        if identity_resolver is None:
            raise HTTPException(
                503, "identity is not enabled on this deployment (see docs/phases/PHASE_F.md)"
            )
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(400, "name is required")
        poses = payload.get("poses")
        if not isinstance(poses, dict) or set(poses) != REQUIRED_POSES:
            raise HTTPException(
                400, f"poses must be provided for exactly: {', '.join(sorted(REQUIRED_POSES))}"
            )

        embeddings: list[list[float]] = []
        for pose in sorted(REQUIRED_POSES):
            image_b64 = poses.get(pose)
            if not image_b64:
                raise HTTPException(400, f"the '{pose}' photo is required")
            image = _decode_image(image_b64, pose)
            face = identity_resolver.detector.largest(image)
            if face is None:
                raise HTTPException(400, f"no face found in the '{pose}' photo")
            embedding = identity_resolver.embedder.embed(image, face)
            embeddings.append([float(v) for v in embedding])

        person_id = str(payload.get("id") or uuid4())
        database.insert_person(
            {
                "id": person_id,
                "name": name,
                "external_id": str(payload.get("external_id", "")),
                "active": True,
                "embeddings": embeddings,
            }
        )
        _refresh_gallery()
        log.info(
            "person enrolled via dashboard: %s (%s), %d pose(s)",
            person_id, name, len(embeddings),
        )
        return _person_public(database.get_person(person_id))

    @router.patch("/persons/{person_id}")
    def update_person(
        person_id: str, payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        if database is None:
            raise HTTPException(503, "person storage is not configured")
        if database.get_person(person_id) is None:
            raise HTTPException(404, f"person {person_id!r} not found")
        updates: dict[str, Any] = {}
        if "active" in payload:
            updates["active"] = bool(payload["active"])
        if "name" in payload:
            new_name = str(payload["name"]).strip()
            if not new_name:
                raise HTTPException(400, "name cannot be blank")
            updates["name"] = new_name
        database.update_person(person_id, updates)
        _refresh_gallery()
        return _person_public(database.get_person(person_id))

    @router.delete("/persons/{person_id}")
    def delete_person(
        person_id: str, _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        """Purges the stored embedding along with the record - no soft delete of
        biometric data (Expansion Plan §8)."""
        if database is None:
            raise HTTPException(503, "person storage is not configured")
        if database.get_person(person_id) is None:
            raise HTTPException(404, f"person {person_id!r} not found")
        database.delete_person(person_id)
        _refresh_gallery()
        log.info("person removed via dashboard: %s", person_id)
        return {"deleted": person_id}

    @router.post("/dev/simulate-alert")
    def simulate_alert(
        payload: dict[str, Any], _admin: User = Depends(require_admin)
    ) -> dict[str, Any]:
        """Inject a fake fire/smoke event so the dashboard popup can be exercised before
        the real fire/smoke detector exists (PLAN.md section 5). Never persisted, never
        sent to a notification sink - visible only through /api/events/live.

        404s when PERIMETER_DEV_TOOLS is not set - admin-only session auth (Phase B) is a
        second, independent gate on top of that flag, not a replacement for it.
        """
        if not settings.api.dev_tools:
            raise HTTPException(404, "not found")
        if alerts is None:
            raise HTTPException(503, "alert bus is not running")

        kind = payload.get("kind")
        if kind not in ("fire", "smoke"):
            raise HTTPException(400, "kind must be \"fire\" or \"smoke\"")

        camera_id = payload.get("camera_id")
        if not camera_id:
            cameras = registry.list()
            if not cameras:
                raise HTTPException(400, "no cameras exist to attribute the test alert to")
            camera_id = cameras[0].id

        alerts.publish_test(
            {
                "ts": time.time(),
                "camera_id": camera_id,
                "kind": kind,
                "subtype": "test",
                "zone_id": None,
                "zone_name": None,
                "track_id": None,
                "message": f"[TEST] Simulated {kind} detected on {camera_id}",
                "severity": "high",
                "bbox": None,
                "foot_point": None,
                "identity_status": None,
                "identity_name": None,
            }
        )
        return {"ok": True}

    @router.get("/health")
    def health(_user: User = Depends(require_any_role)) -> dict[str, Any]:
        pipelines = manager.all()
        try:
            if not _storage_up():
                raise ServerSelectionTimeoutError("no reachable server")
            cameras = registry.list()
        except PyMongoError:
            # Still answer during a storage outage: report what is actually running.
            cameras = [p.camera for p in pipelines.values()]
        return {
            "database": _database_health(database),
            "dev_tools": settings.api.dev_tools,
            "identity_enabled": identity_resolver is not None,
            "cameras": [
                {
                    "id": c.id,
                    "name": c.name,
                    "enabled": c.enabled,
                    "running": c.id in pipelines,
                    "feed_state": pipelines[c.id].stats.feed_state if c.id in pipelines else None,
                }
                for c in cameras
            ],
        }

    app.include_router(router)
    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built React dashboard.

    Mounted AFTER the API router so /api always wins. One process on one port keeps
    deployment to a single service on a customer's box - no separate web server to
    install, configure and keep patched.
    """
    index = DIST_DIR / "index.html"
    if not index.is_file():
        @app.get("/", include_in_schema=False)
        def missing_build() -> HTMLResponse:
            # A blank page would look like a crashed backend; say what is actually wrong.
            return HTMLResponse(
                "<h1>Dashboard not built</h1>"
                "<p>Run <code>npm install &amp;&amp; npm run build</code> in "
                "<code>frontend/</code>, then reload.</p>",
                status_code=503,
            )
        log.warning("no built frontend at %s - run the frontend build", DIST_DIR)
        return

    assets = DIST_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def spa_root() -> FileResponse:
        return FileResponse(index)

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str) -> FileResponse:
        """Serve real files, otherwise hand back index.html.

        The catch-all must not shadow the API: a mistyped /api/... would otherwise
        return the HTML page with a 200, and the client would fail on JSON parsing
        instead of showing a clean 404.
        """
        if path.startswith("api/"):
            raise HTTPException(404, "not found")

        candidate = (DIST_DIR / path).resolve()
        if candidate.is_file() and DIST_DIR.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)
