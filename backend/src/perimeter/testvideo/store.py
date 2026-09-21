"""Uploaded test videos on local disk (the dashboard's "Video test" page).

Files are stored as `<32-hex-id><ext>` with a `<id>.json` sidecar (original name, size,
duration...). The id is generated here and validated by pattern on every lookup, and only a
short list of video extensions is accepted, so a client can never choose a path. Nothing here
serves the raw file back out: the page only ever plays it through the detection pipeline.
Videos older than `retention_days` are deleted automatically - they are test material, not
evidence, and can contain people.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2

log = logging.getLogger("perimeter.testvideo.store")

VIDEO_ID = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})


class UploadError(ValueError):
    """The upload is not usable (bad type, too big, unreadable). The message is shown to the
    user as-is."""


@dataclass(frozen=True)
class UploadedVideo:
    id: str
    name: str
    size_bytes: int
    uploaded_at: str
    duration_s: float
    fps: float
    width: int
    height: int
    frames: int

    def to_dict(self) -> dict:
        return asdict(self)


def probe_video(path: Path) -> dict:
    """Open the file and read one frame, or raise UploadError. A file that merely has a video
    extension (a renamed document, a truncated download) must fail here, at upload time, not
    later as a camera that never connects."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise UploadError("This file could not be opened as a video.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise UploadError("The video opened but contains no readable frames.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        height, width = frame.shape[:2]
    finally:
        cap.release()
    fps = fps if 0.0 < fps <= 240.0 else 0.0
    duration = frames / fps if fps and frames > 0 else 0.0
    return {"fps": fps, "frames": max(frames, 0), "width": width, "height": height,
            "duration_s": duration}


class TestVideoStore:
    __test__ = False  # not a pytest test class, despite the name

    def __init__(
        self,
        directory: Path,
        max_bytes: int,
        retention_days: int = 7,
        probe: Callable[[Path], dict] = probe_video,
    ) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        self.retention_days = retention_days
        self._probe = probe
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- writing -------------------------------------------------------------

    def new_upload(self, filename: str) -> tuple[str, Path]:
        """Reserve an id and a path for an incoming file (raises UploadError for a bad type)."""
        ext = Path(filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise UploadError(f"Unsupported file type {ext or '(none)'}. Use one of: {allowed}")
        video_id = uuid.uuid4().hex
        return video_id, self.directory / f"{video_id}{ext}"

    def finalize(self, video_id: str, path: Path, original_name: str) -> UploadedVideo:
        """Verify the written file is a readable video and record its metadata; on any
        failure the file is removed so a rejected upload leaves nothing behind."""
        try:
            info = self._probe(path)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        video = UploadedVideo(
            id=video_id,
            name=Path(original_name).name[:200] or path.name,
            size_bytes=path.stat().st_size,
            uploaded_at=datetime.now(UTC).isoformat(),
            duration_s=round(info["duration_s"], 2),
            fps=round(info["fps"], 2),
            width=info["width"],
            height=info["height"],
            frames=info["frames"],
        )
        (self.directory / f"{video_id}.json").write_text(json.dumps(video.to_dict()))
        log.info("test video uploaded: %s (%s, %.1fs)", video.name, video_id, video.duration_s)
        return video

    def discard(self, path: Path) -> None:
        path.unlink(missing_ok=True)

    # -- reading -------------------------------------------------------------

    def path(self, video_id: str) -> Path | None:
        if not VIDEO_ID.fullmatch(video_id or ""):
            return None
        for ext in ALLOWED_EXTENSIONS:
            candidate = self.directory / f"{video_id}{ext}"
            if candidate.is_file():
                return candidate
        return None

    def frame_jpeg(self, video_id: str, at: float = 0.25) -> bytes | None:
        """One still from the video (default a quarter of the way in, to skip black openers),
        used as the backdrop for drawing boundaries. Same pixel size as the pipeline sees."""
        path = self.path(video_id)
        video = self.get(video_id)
        if path is None or video is None:
            return None
        cap = cv2.VideoCapture(str(path))
        try:
            if video.frames > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(video.frames * min(max(at, 0.0), 0.9)))
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else None

    def get(self, video_id: str) -> UploadedVideo | None:
        if self.path(video_id) is None:
            return None
        meta = self.directory / f"{video_id}.json"
        try:
            return UploadedVideo(**json.loads(meta.read_text()))
        except (OSError, ValueError, TypeError):
            return None

    def list(self) -> list[UploadedVideo]:
        self.prune_expired()
        videos = []
        for meta in self.directory.glob("*.json"):
            video = self.get(meta.stem)
            if video is not None:
                videos.append(video)
        return sorted(videos, key=lambda v: v.uploaded_at, reverse=True)

    def delete(self, video_id: str) -> bool:
        path = self.path(video_id)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        (self.directory / f"{video_id}.json").unlink(missing_ok=True)
        return True

    def prune_expired(self) -> int:
        cutoff = time.time() - self.retention_days * 86400
        removed = 0
        for path in list(self.directory.iterdir()):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        return removed
