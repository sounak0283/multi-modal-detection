"""Routes for the dashboard's Video test page and its on/off setting.

Kept out of `app.py` so that file does not keep growing. Everything under /test-video is
admin-only (an upload writes to disk and a session uses a full detection pipeline's worth of
CPU) and answers 404 while an admin has switched the section off.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse

from perimeter.appsettings import AppSettings, AppSettingsStore
from perimeter.testvideo.sessions import SessionError, TestSessionManager
from perimeter.testvideo.store import TestVideoStore, UploadError

log = logging.getLogger("perimeter.api.testvideo")

STREAM_BOUNDARY = "frame"
UPLOAD_CHUNK_LIMIT_NOTE = "The video is larger than the upload limit ({mb} MB)."


def add_test_video_routes(
    router: APIRouter,
    *,
    require_admin: Callable[..., Any],
    require_any_role: Callable[..., Any],
    app_settings: AppSettingsStore,
    store: TestVideoStore | None,
    sessions: TestSessionManager | None,
    list_cameras: Callable[[], list[dict[str, Any]]],
) -> None:
    def guard() -> tuple[TestVideoStore, TestSessionManager]:
        if store is None or sessions is None or not app_settings.get().video_test_enabled:
            raise HTTPException(404, "The video test section is switched off.")
        return store, sessions

    # -- the on/off setting ------------------------------------------------------

    @router.get("/app-settings")
    def get_app_settings(_user: Any = Depends(require_any_role)) -> dict[str, Any]:
        return {**app_settings.get().to_dict(), "video_test_available": store is not None}

    @router.put("/app-settings")
    def put_app_settings(payload: dict[str, Any], _admin: Any = Depends(require_admin)) -> dict:
        value = payload.get("video_test_enabled")
        if not isinstance(value, bool):
            raise HTTPException(400, "video_test_enabled must be true or false")
        app_settings.save(AppSettings(video_test_enabled=value))
        if not value and sessions is not None:
            sessions.stop()  # switching the section off must also end a running test
        return app_settings.get().to_dict()

    # -- what the page needs to render -------------------------------------------

    @router.get("/test-video/status")
    def status(_admin: Any = Depends(require_admin)) -> dict[str, Any]:
        video_store, manager = guard()
        current = manager.current()
        return {
            "modules": manager.modules(),
            "uploads": [v.to_dict() for v in video_store.list()],
            "cameras": list_cameras(),
            "max_upload_mb": video_store.max_bytes // (1024 * 1024),
            "retention_days": video_store.retention_days,
            "session": current.to_dict() if current else None,
        }

    # -- uploads -----------------------------------------------------------------

    @router.post("/test-video/upload")
    async def upload(request: Request, _admin: Any = Depends(require_admin)) -> dict[str, Any]:
        """The video is the raw request body (no multipart parsing, so no extra dependency
        and no whole-file buffering); its name travels in the X-Filename header."""
        video_store, _ = guard()
        max_mb = video_store.max_bytes // (1024 * 1024)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > video_store.max_bytes:
            raise HTTPException(413, UPLOAD_CHUNK_LIMIT_NOTE.format(mb=max_mb))
        filename = unquote(request.headers.get("x-filename", ""))
        try:
            video_id, path = video_store.new_upload(filename)
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc

        size = 0
        try:
            with open(path, "wb") as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > video_store.max_bytes:
                        raise HTTPException(413, UPLOAD_CHUNK_LIMIT_NOTE.format(mb=max_mb))
                    out.write(chunk)
        except BaseException:
            video_store.discard(path)
            raise
        if size == 0:
            video_store.discard(path)
            raise HTTPException(400, "The upload was empty.")
        try:
            video = await run_in_threadpool(video_store.finalize, video_id, path, filename)
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc
        return video.to_dict()

    @router.get("/test-video/uploads/{video_id}/frame")
    async def upload_frame(video_id: str, _admin: Any = Depends(require_admin)) -> Response:
        video_store, _ = guard()
        jpeg = await run_in_threadpool(video_store.frame_jpeg, video_id)
        if jpeg is None:
            raise HTTPException(404, "video not found")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.delete("/test-video/uploads/{video_id}")
    def delete_upload(video_id: str, _admin: Any = Depends(require_admin)) -> dict[str, Any]:
        video_store, manager = guard()
        current = manager.current()
        if current is not None and current.video.id == video_id:
            manager.stop()
        if not video_store.delete(video_id):
            raise HTTPException(404, "video not found")
        return {"deleted": video_id}

    # -- a running test -----------------------------------------------------------

    @router.post("/test-video/session")
    def start_session(payload: dict[str, Any], _admin: Any = Depends(require_admin)) -> dict:
        _, manager = guard()
        modules = payload.get("modules")
        if not isinstance(modules, list):
            raise HTTPException(400, "modules must be a list of model names")
        try:
            session = manager.start(
                video_id=str(payload.get("video_id", "")),
                modules=modules,
                zones_from=payload.get("zones_from") or None,
                loop=bool(payload.get("loop", False)),
                send_alerts=bool(payload.get("send_alerts", False)),
                zones=payload.get("zones") if isinstance(payload.get("zones"), list) else None,
            )
        except SessionError as exc:
            raise HTTPException(400, str(exc)) from exc
        return session.to_dict()

    @router.get("/test-video/session")
    def get_session(since: int = 0, _admin: Any = Depends(require_admin)) -> dict[str, Any]:
        """`since` = how many alerts the page already has, so a poll returns only new ones."""
        _, manager = guard()
        current = manager.current()
        return {"session": current.to_dict(since=max(0, since)) if current else None}

    @router.delete("/test-video/session")
    def stop_session(_admin: Any = Depends(require_admin)) -> dict[str, Any]:
        _, manager = guard()
        manager.stop()
        return {"stopped": True}

    @router.get("/test-video/session/stream")
    def session_stream(
        annotated: bool = True, _admin: Any = Depends(require_admin)
    ) -> StreamingResponse:
        """Live annotated view of the running test - the same MJPEG a camera's live view uses."""
        _, manager = guard()
        current = manager.current()
        if current is None:
            raise HTTPException(404, "no test is running")
        pipeline = current.pipeline

        def frames():
            last: bytes | None = None
            while not current.stopped:
                jpeg = pipeline.latest_jpeg(annotated=annotated)
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield (
                        b"--" + STREAM_BOUNDARY.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                time.sleep(0.05)

        return StreamingResponse(
            frames(), media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}"
        )
