"""Test-connection probe for the dashboard's Camera page.

Opens a candidate source, grabs one frame, and reports what actually came back. Kept
separate from CaptureThread because it must be short-lived and must never disturb the
running capture - it opens its own handle and releases it immediately.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2

from perimeter.capture.source import open_capture
from perimeter.settings import redact

log = logging.getLogger("perimeter.capture.probe")

# FFmpeg gives up opening after this; below the wall-clock bound so a dead host is
# reported by FFmpeg itself and the probe slot (and FFmpeg's global open lock, see
# capture/source.py) is released promptly rather than 30 s later.
PROBE_OPEN_TIMEOUT_MS = 6_000
# Hard wall-clock bound on the whole probe.
PROBE_TIMEOUT_S = 8.0

# Giving up on a probe does not stop the FFmpeg call underneath it - the worker thread
# keeps blocking until FFmpeg's own 30 s default expires. An installer facing an
# unreachable camera will click Test repeatedly, and without a cap each click would leave
# another thread stuck inside FFmpeg holding a socket. Two concurrent probes is enough to
# never make a legitimate user wait, and small enough that repeated clicking is harmless.
MAX_CONCURRENT_PROBES = 2
_probe_slots = threading.Semaphore(MAX_CONCURRENT_PROBES)


def probe_source(
    source: str | int,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> dict[str, Any]:
    """Try to open `source` and read one frame, with a hard wall-clock bound.

    Returns the ACTUAL resolution rather than the requested one. A camera that quietly
    ignores a 1280x720 request changes what every pixel-based threshold means -
    min_person_box_px for the identity layer, small-object recall for fire - so the
    mismatch is surfaced here rather than inferred from poor results weeks later.

    FFmpeg's open is bounded by `PROBE_OPEN_TIMEOUT_MS` (OpenCV's per-capture
    CAP_PROP_OPEN_TIMEOUT_MSEC - the OPENCV_FFMPEG_CAPTURE_OPTIONS env var was measured to
    have no effect on this build). The wall-clock bound here is belt-and-braces for
    anything that still blocks: the worker thread is a daemon, so if it is stuck when we
    give up it finishes and releases its own handle later.
    """
    if not _probe_slots.acquire(blocking=False):
        return {
            "ok": False,
            "error": "A connection test is already running. Wait for it to finish "
            "- an unreachable camera can take up to 30s to give up underneath.",
            "elapsed_ms": 0,
        }

    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result.update(_probe_blocking(source, width, height, fps))
        finally:
            # Released by the worker, not by the waiter: the slot is occupied for as
            # long as the FFmpeg call actually blocks, which is the thing being capped.
            _probe_slots.release()

    thread = threading.Thread(target=worker, name="camera-probe", daemon=True)
    started = time.perf_counter()
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        return {
            "ok": False,
            "error": (
                f"No response within {timeout_s:.0f}s from {redact(source)}. "
                f"The host is probably unreachable, or a firewall is dropping the "
                f"connection rather than refusing it."
            ),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
    return result


def _probe_blocking(
    source: str | int,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cap: cv2.VideoCapture | None = None

    try:
        cap = open_capture(source, open_timeout_ms=PROBE_OPEN_TIMEOUT_MS)

        if not cap.isOpened():
            return {
                "ok": False,
                "error": f"Could not open {redact(source)}. "
                f"Check the URL, credentials, and that the camera is reachable.",
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)

        ok, frame = cap.read()
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        if not ok or frame is None:
            return {
                "ok": False,
                "error": "Opened the source but could not read a frame. "
                "For RTSP this usually means the wrong stream path or a codec the "
                "bundled FFmpeg cannot decode.",
                "elapsed_ms": elapsed_ms,
            }

        actual_h, actual_w = frame.shape[:2]
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        result: dict[str, Any] = {
            "ok": True,
            "source": redact(source),
            "actual_width": actual_w,
            "actual_height": actual_h,
            "reported_fps": round(reported_fps, 1) if reported_fps > 0 else None,
            "elapsed_ms": elapsed_ms,
            "warnings": [],
        }

        if width and height and (actual_w, actual_h) != (width, height):
            result["warnings"].append(
                f"Camera is streaming {actual_w}x{actual_h}, not the requested "
                f"{width}x{height}. Pixel-based thresholds are tuned for the configured "
                f"resolution and may need adjusting."
            )
        if elapsed_ms > 4000:
            result["warnings"].append(
                f"Took {elapsed_ms / 1000:.1f}s to produce a frame - a slow or congested "
                f"link will show up as latency on every alert."
            )
        return result

    except cv2.error as exc:
        return {
            "ok": False,
            "error": f"OpenCV rejected the source: {exc}",
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
    finally:
        if cap is not None:
            cap.release()
