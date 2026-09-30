"""EvidenceWriter (Expansion Plan Phase C).

Registered as a sink on `AlertBus` (`alerts/bus.py`'s existing `sinks` hook - no new hook
was needed). On every alert it waits `post_roll_seconds` - so the still-filling ring
buffer accumulates some "after" footage - then muxes whatever is currently buffered for
that camera into a clip and picks the frame closest to the alert's own timestamp as the
snapshot. Mirrors `AlertBus`'s own queue-plus-background-thread shape deliberately: a
new detection module never needs its own alerting code, and evidence capture never needs
its own concurrency model either.

Failure policy, same as the rest of this codebase: one bad job (camera gone, disk full,
no usable codec) must not kill the thread or block the next alert's evidence.

The actual `cv2.VideoWriter` mux (`mux_clip` below) is passed into `EvidenceWriter` as an
injectable `mux` callable rather than called directly, so tests can substitute a fast
fake instead of opening a real codec - this matters because `test_camera_api.py`'s
unreachable-RTSP-host probes are known to leave OpenCV's FFmpeg backend in a state that
stalls the *next* cv2 video open in the same process for up to ~30s (see that file's
module docstring); that is a real, separate finding worth a follow-up (a stuck camera
connection could stall evidence writing for every other camera in the process), not
something to paper over here, but a unit test asserting EvidenceWriter's own
orchestration logic should not have to pay for it.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import queue
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perimeter.capture.ring_buffer import JpegRingBuffer
from perimeter.evidence.policy import ClipPolicy
from perimeter.evidence.store import LocalEvidenceStore, S3EvidenceStore
from perimeter.record import open_writer
from perimeter.timefmt import local_time

log = logging.getLogger("perimeter.evidence.writer")

DEFAULT_FPS_FALLBACK = 10.0

MuxFn = Callable[[list[tuple[float, bytes]], str, Path], Path | None]
# notify(payload, followup=, only_rules=) -> ids of the alert rules it was sent to.
NotifyFn = Callable[..., set[str]]

# Waits between clip save attempts (seconds). The first attempt is immediate; a slow or
# briefly-down S3 is exactly what these ride out.
DEFAULT_RETRY_BACKOFF_S = (5.0, 30.0, 120.0)
# Each pending retry holds ~15s of JPEG frames (10-15 MB) in memory, so cap how many can
# wait at once; past that a failed clip is given up on immediately rather than growing RAM.
MAX_PENDING_RETRIES = 20
# Cooldown scope for alert emails that carry no video (see CooldownGate.allow).
NO_VIDEO_SCOPE = "novideo"
# Longest the writer will hold an alert back, beyond the post-roll, waiting for the buffer to
# reach `min_clip_seconds`. Past it the clip is saved as-is rather than delaying the email
# indefinitely (a camera that just dropped may never fill the buffer).
MAX_EXTRA_WAIT_S = 15.0
# A ring buffer bounded to N seconds never spans a full N (its oldest frame is dropped the
# moment it is N seconds old), so "full" means within this of its capacity - otherwise, with
# ring_buffer_seconds == min_clip_seconds (the shipped config), every clip waited the whole
# MAX_EXTRA_WAIT_S for a length it could never reach, delaying every alert email by it.
# 1 s is above the gap between frames down to 1 fps.
BUFFER_FULL_SLACK_S = 1.0
# Normal (non-urgent) clips are muxed and uploaded in parallel, so a burst of alerts - several
# people entering at once - does not queue each email behind every clip before it.
NORMAL_WORKERS = 3


@dataclass
class _Job:
    payload: dict[str, Any]
    deadline: float
    skip: bool = False  # policy says no clip: just send the email
    urgent: bool = False  # email now, video follows


@dataclass
class _RetryJob:
    due: float
    payload: dict[str, Any]
    camera_id: str
    event_id: str
    frames: list[tuple[float, bytes]]
    tmp_dir: Path
    tmp_path: Path | None
    attempts: int
    sent_rules: set[str]


def _decode(jpeg: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


# Browsers (Chrome, Edge, Firefox) only play H.264 (or VP8/VP9) inside an MP4. The recorder's
# default "mp4v" is MPEG-4 Part 2: valid, decodable by OpenCV, and completely unplayable in
# the dashboard's <video> tag. VP8/VP9 are browser-safe but take 24s / 145s to encode a 20s
# clip here (measured) - far too slow beside live detection - so H.264 comes first, and mp4v
# stays only as a last resort so evidence is still saved when no H.264 encoder exists.
BROWSER_CODEC_CANDIDATES: tuple[tuple[str, str], ...] = (("avc1", ".mp4"), ("mp4v", ".mp4"))
H264_TAGS = {"h264", "avc1", "x264"}


def _video_tag(path: Path) -> str:
    """The codec tag actually written into `path`, e.g. 'h264' or 'FMP4' - read back from
    the file, because OpenCV reports a VideoWriter as opened even when the requested
    encoder is missing and something else was substituted."""
    cap = cv2.VideoCapture(str(path))
    try:
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip(chr(0) + " ")


def _even(image: np.ndarray) -> np.ndarray:
    """H.264 needs even width and height; trim a stray odd row/column (never scales)."""
    height, width = image.shape[:2]
    return image[: height - height % 2, : width - width % 2]


def mux_clip(frames: list[tuple[float, bytes]], event_id: str, tmp_dir: Path) -> Path | None:
    """Decode buffered JPEGs and write them out as one browser-playable video under `tmp_dir`.

    fps is derived from the actual frame timestamps (`len(frames) / span`), not a
    configured rate - that reflects the real capture cadence, drops included, more
    accurately than trusting a nominal decode_fps would.
    """
    if len(frames) < 2:
        return None  # too little buffered to produce a meaningful clip

    first_ts, first_jpeg = frames[0]
    last_ts, _ = frames[-1]
    span = last_ts - first_ts
    fps = (len(frames) - 1) / span if span > 0 else DEFAULT_FPS_FALLBACK

    first_image = _decode(first_jpeg)
    if first_image is None:
        return None
    height, width = _even(first_image).shape[:2]

    last_error = "no encoder"
    for index, candidate in enumerate(BROWSER_CODEC_CANDIDATES):
        is_last = index == len(BROWSER_CODEC_CANDIDATES) - 1
        try:
            writer, tmp_path = open_writer(tmp_dir / event_id, fps, (width, height), (candidate,))
        except RuntimeError as exc:  # this encoder cannot even be opened - try the next one
            last_error = str(exc)
            if is_last:
                raise
            continue
        try:
            for _, jpeg in frames:
                image = _decode(jpeg)
                if image is not None:
                    writer.write(_even(image))
        finally:
            writer.release()

        tag = _video_tag(tmp_path)
        if candidate[0] == "avc1" and tag.lower() not in H264_TAGS:
            last_error = f"asked for H.264 but the file holds {tag!r}"
            tmp_path.unlink(missing_ok=True)
            if not is_last:
                continue
            raise RuntimeError(last_error)
        if tag.lower() not in H264_TAGS:
            log.warning(
                "clip for %s is %r, not H.264 - it is saved but will NOT play in a browser "
                "(no H.264 encoder in this OpenCV build; see backend README, video codecs)",
                event_id, tag,
            )
        return tmp_path
    raise RuntimeError(last_error)


class EvidenceWriter:
    def __init__(
        self,
        database: Any,
        store: LocalEvidenceStore,
        ring_buffer_lookup: Callable[[str], JpegRingBuffer | None],
        post_roll_seconds: float = 5.0,
        mux: MuxFn = mux_clip,
        notify: NotifyFn | None = None,
        link_expiry_hours: float = 72.0,
        upload_attempts: int = 4,
        retry_backoff_s: tuple[float, ...] = DEFAULT_RETRY_BACKOFF_S,
        min_clip_seconds: float = 0.0,
        max_extra_wait_s: float = MAX_EXTRA_WAIT_S,
        policy: ClipPolicy | None = None,
        urgent_post_roll_seconds: float = 1.0,
    ) -> None:
        self.database = database
        self.store = store
        self.ring_buffer_lookup = ring_buffer_lookup
        self.post_roll_seconds = post_roll_seconds
        self._mux = mux
        # The alert email waits for this writer (so it can carry the video link) - see
        # `_notify`. `None` (tests, or evidence disabled) means nothing is notified here.
        self._notify_fn = notify
        self.link_expiry_s = link_expiry_hours * 3600.0
        self.upload_attempts = max(1, upload_attempts)
        self.retry_backoff_s = retry_backoff_s
        self.min_clip_seconds = min_clip_seconds
        # None records every alert and treats none as urgent (the original behaviour).
        self.policy = policy
        self.urgent_post_roll_seconds = urgent_post_roll_seconds
        self.max_extra_wait_s = max_extra_wait_s

        self._retries: list[tuple[float, int, _RetryJob]] = []
        self._retry_seq = itertools.count()
        self._retry_lock = threading.Lock()
        self._retry_thread: threading.Thread | None = None

        # Two workers so an urgent alert (fire, smoke, crowd) is never queued behind a normal
        # clip that is still muxing or uploading. The urgent worker also handles the cheap
        # "no clip needed" jobs, which just send the email.
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._urgent_queue: queue.Queue[_Job] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._urgent_thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        for index in range(NORMAL_WORKERS):
            thread = threading.Thread(
                target=self._run, args=(self._queue,), name=f"evidence-writer-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        self._urgent_thread = threading.Thread(
            target=self._run, args=(self._urgent_queue,), name="evidence-urgent", daemon=True
        )
        self._urgent_thread.start()
        self._retry_thread = threading.Thread(
            target=self._retry_loop, name="evidence-retry", daemon=True
        )
        self._retry_thread.start()
        log.info("evidence writer started (post_roll=%.1fs)", self.post_roll_seconds)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout)
        self._threads = []
        if self._urgent_thread is not None:
            self._urgent_thread.join(timeout)
            self._urgent_thread = None
        if self._retry_thread is not None:
            self._retry_thread.join(timeout)
            self._retry_thread = None
        with self._retry_lock:
            for _due, _seq, job in self._retries:
                shutil.rmtree(job.tmp_dir, ignore_errors=True)
            self._retries.clear()

    # -- producer side (an AlertBus sink) -------------------------------------

    def on_alert(self, payload: dict[str, Any]) -> None:
        """`AlertBus` sink signature - called after persistence, so `payload["id"]` is
        already set. Never blocks the alert thread: it only decides and enqueues."""
        if "id" not in payload:
            return  # persistence failed - nothing to attach evidence to yet
        now = time.time()
        decision = self.policy.decide(payload) if self.policy else None
        if decision is not None and not decision.record:
            log.debug("no clip for event %s: %s", payload["id"], decision.reason)
            self._urgent_queue.put(_Job(payload, now, skip=True))
        elif decision is not None and decision.urgent:
            self._urgent_queue.put(
                _Job(payload, now + self.urgent_post_roll_seconds, urgent=True)
            )
        else:
            self._queue.put(_Job(payload, now + self.post_roll_seconds))

    # -- consumer side ---------------------------------------------------------

    def _run(self, jobs: queue.Queue[_Job]) -> None:
        while not self._stop.is_set():
            try:
                job = jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._handle(job)
            except Exception:  # noqa: BLE001 - one bad job must not kill this thread
                log.exception("evidence capture failed for event %s", job.payload.get("id"))
        log.info("evidence worker stopped")

    def _handle(self, job: _Job) -> None:
        payload = job.payload
        if job.skip:
            self.database.update_evidence_paths(payload["id"], clip_status="skipped")
            # Own cooldown scope: this email has no video and must not use up the cooldown
            # of a video-carrying alert for the same camera and zone.
            self._notify(payload, scope=NO_VIDEO_SCOPE)
            return
        pre_sent: set[str] | None = None
        if job.urgent:
            # Email first, at once: the video is a follow-up, not something fire waits for.
            pre_sent = self._notify(payload)
        remaining = job.deadline - time.time()
        if remaining > 0:
            # Interruptible: stop() should not have to wait out a post-roll.
            self._stop.wait(remaining)
        self._process(payload, pre_sent=pre_sent, urgent=job.urgent)

    def _process(
        self, payload: dict[str, Any], pre_sent: set[str] | None = None, urgent: bool = False
    ) -> None:
        camera_id = payload.get("camera_id")
        event_id = payload.get("id")
        ring_buffer = self.ring_buffer_lookup(camera_id) if camera_id else None
        if ring_buffer is None:
            # The camera (or, for a Video test session, the whole temporary pipeline) is
            # already gone by the time this job was dequeued - most often a backlog of
            # queued clip jobs still outstanding when a short-lived session/camera stops.
            # Without this, clip_status was left at its initial unset value forever,
            # indistinguishable from "still pending" (found live: several Video test
            # clips stuck at None permanently after the video reached its end).
            log.warning(
                "no ring buffer for camera %r (event %s) - camera is no longer running, "
                "evidence cannot be collected", camera_id, event_id,
            )
            if event_id:
                self.database.update_evidence_paths(event_id, clip_status="failed")
            self._first_email(payload, pre_sent)
            return

        frames = self._collect_frames(ring_buffer, urgent=urgent)
        if not frames:
            # Same "must not leave clip_status unset forever" reasoning as the missing
            # ring buffer case just above - this is also a terminal path (no retry is
            # scheduled for it), so clip_status has to say so.
            log.warning(
                "ring buffer empty for camera %r (event %s) - evidence cannot be collected",
                camera_id, event_id,
            )
            if event_id:
                self.database.update_evidence_paths(event_id, clip_status="failed")
            self._first_email(payload, pre_sent)
            return

        # Independent try/excepts: a clip failure must not cost the (cheaper, more
        # likely to succeed) snapshot, and vice versa.
        snapshot_key = None
        try:
            snapshot_key = self._write_snapshot(camera_id, event_id, ring_buffer, payload.get("ts"))
        except Exception:  # noqa: BLE001
            log.exception("snapshot capture failed for event %s", event_id)

        tmp_dir = Path(tempfile.mkdtemp(prefix="perimeter-evidence-"))
        tmp_path: Path | None = None
        clip_key = None
        errored = False
        try:
            tmp_path = self._mux(frames, event_id, tmp_dir)
            if tmp_path is not None:
                clip_key = self.store.save_clip(camera_id, event_id, tmp_path)
        except Exception:  # noqa: BLE001
            errored = True
            log.exception("clip capture failed for event %s", event_id)

        if clip_key is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._record(event_id, snapshot_key, clip_key, "saved", 1)
            linked = self._with_link(payload, clip_key)
            if pre_sent is not None:
                # Urgent: the alert email already went out; this is the follow-up with video.
                self._notify(linked, followup=True, only_rules=pre_sent)
            else:
                self._notify(linked)
            return

        # No clip yet. The alert must not wait on a slow or down store: send the email
        # now without a video, and keep trying in the background.
        # An ERROR while muxing/uploading is worth retrying (a codec hiccup, a network blip);
        # a mux that simply returned nothing (too few frames buffered) is not.
        retryable = errored
        self._record(event_id, snapshot_key, None, "pending" if retryable else "failed", 1)
        sent = self._first_email(payload, pre_sent)
        if not retryable:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return
        self._schedule_retry(
            _RetryJob(
                due=0.0, payload=payload, camera_id=camera_id, event_id=event_id,
                frames=frames, tmp_dir=tmp_dir, tmp_path=tmp_path, attempts=1, sent_rules=sent,
            )
        )

    # -- helpers ---------------------------------------------------------------

    def _first_email(self, payload: dict[str, Any], pre_sent: set[str] | None) -> set[str]:
        """The alert email, unless an urgent alert already sent it (then just its rules)."""
        return pre_sent if pre_sent is not None else self._notify(payload)

    def _collect_frames(
        self, ring_buffer: JpegRingBuffer, urgent: bool = False
    ) -> list[tuple[float, bytes]]:
        """The buffered footage, held back (up to `max_extra_wait_s`) until it spans at least
        `min_clip_seconds`. The wait is what keeps a clip from coming out short when an alert
        lands right after a start or a camera reconnect, before the buffer has filled."""
        frames = ring_buffer.snapshot()
        waited = 0.0
        target = self.min_clip_seconds
        capacity = getattr(ring_buffer, "seconds", None)
        if capacity is not None:
            target = min(target, capacity - BUFFER_FULL_SLACK_S)
        while (
            self.min_clip_seconds > 0
            and len(frames) >= 2
            and frames[-1][0] - frames[0][0] < target
            and not urgent  # urgent alerts never wait for a fuller buffer
            and waited < self.max_extra_wait_s
            and not self._stop.is_set()
        ):
            step = min(0.5, self.max_extra_wait_s - waited)
            self._stop.wait(step)
            waited += step
            frames = ring_buffer.snapshot()
        return frames

    def _backend_name(self) -> str:
        return "s3" if isinstance(self.store, S3EvidenceStore) else "local"

    def _record(
        self,
        event_id: str,
        snapshot_key: str | None,
        clip_key: str | None,
        status: str,
        attempts: int,
    ) -> None:
        self.database.update_evidence_paths(
            event_id,
            snapshot_path=snapshot_key,
            clip_path=clip_key,
            snapshot_uri=self.store.uri(snapshot_key) if snapshot_key else None,
            clip_uri=self.store.uri(clip_key) if clip_key else None,
            clip_status=status,
            clip_attempts=attempts,
            evidence_backend=self._backend_name(),
        )

    def _with_link(self, payload: dict[str, Any], clip_key: str) -> dict[str, Any]:
        """A copy of `payload` carrying a presigned video link (remote stores only), minted
        once here for the email. The link and its expiry are also recorded on the event as an
        audit trail of what was sent; the permanent reference stays `clip_uri`, and the
        dashboard mints fresh short-lived links on demand rather than reusing this one."""
        try:
            url = self.store.presigned_url(clip_key, self.link_expiry_s)
        except Exception:  # noqa: BLE001 - a missing link must never cost the email
            log.exception("could not create a video link for event %s", payload.get("id"))
            return payload
        if not url:
            return payload
        expires = datetime.now(UTC) + timedelta(seconds=self.link_expiry_s)
        # Audit record of exactly what the email carried. Bookkeeping only: failing to
        # write it must never cost the email.
        try:
            self.database.update_evidence_paths(
                payload["id"], clip_link_url=url, clip_link_expires_at=expires
            )
        except Exception:  # noqa: BLE001
            log.exception("could not record the video link for event %s", payload.get("id"))
        return {
            **payload,
            "clip_url": url,
            "clip_link_expires_at": local_time(expires, seconds=False),
        }

    def _notify(
        self,
        payload: dict[str, Any],
        followup: bool = False,
        only_rules: set[str] | None = None,
        scope: str = "",
    ) -> set[str]:
        if self._notify_fn is None:
            return set()
        kwargs: dict[str, Any] = {"followup": followup, "only_rules": only_rules}
        if scope:
            kwargs["scope"] = scope
        try:
            return self._notify_fn(payload, **kwargs) or set()
        except Exception:  # noqa: BLE001 - notification trouble must not kill evidence writing
            log.exception("alert notification failed for event %s", payload.get("id"))
            return set()

    # -- retry -----------------------------------------------------------------

    def _schedule_retry(self, job: _RetryJob) -> None:
        with self._retry_lock:
            if len(self._retries) >= MAX_PENDING_RETRIES:
                log.error("too many clip uploads pending - giving up on event %s", job.event_id)
                self.database.update_evidence_paths(job.event_id, clip_status="failed")
                shutil.rmtree(job.tmp_dir, ignore_errors=True)
                return
            wait = self.retry_backoff_s[min(job.attempts - 1, len(self.retry_backoff_s) - 1)]
            job.due = time.monotonic() + wait
            heapq.heappush(self._retries, (job.due, next(self._retry_seq), job))

    def _retry_loop(self) -> None:
        while not self._stop.is_set():
            job = None
            with self._retry_lock:
                if self._retries and self._retries[0][0] <= time.monotonic():
                    job = heapq.heappop(self._retries)[2]
            if job is None:
                self._stop.wait(0.2)
                continue
            try:
                self._retry_once(job)
            except Exception:  # noqa: BLE001 - one bad job must not kill the retry thread
                log.exception("clip retry crashed for event %s", job.event_id)
                shutil.rmtree(job.tmp_dir, ignore_errors=True)

    def _retry_once(self, job: _RetryJob) -> None:
        job.attempts += 1
        clip_key = None
        try:
            if job.tmp_path is None:
                job.tmp_path = self._mux(job.frames, job.event_id, job.tmp_dir)
            if job.tmp_path is not None:
                clip_key = self.store.save_clip(job.camera_id, job.event_id, job.tmp_path)
        except Exception:  # noqa: BLE001
            log.exception(
                "clip save attempt %d/%d failed for event %s",
                job.attempts, self.upload_attempts, job.event_id,
            )

        if clip_key is not None:
            shutil.rmtree(job.tmp_dir, ignore_errors=True)
            self._record(job.event_id, None, clip_key, "saved", job.attempts)
            log.info("clip for event %s saved on attempt %d", job.event_id, job.attempts)
            self._notify(
                self._with_link(job.payload, clip_key), followup=True, only_rules=job.sent_rules
            )
            return

        if job.attempts >= self.upload_attempts:
            log.error(
                "giving up on the clip for event %s after %d attempts", job.event_id, job.attempts
            )
            self.database.update_evidence_paths(
                job.event_id, clip_status="failed", clip_attempts=job.attempts
            )
            shutil.rmtree(job.tmp_dir, ignore_errors=True)
            return

        self.database.update_evidence_paths(
            job.event_id, clip_status="pending", clip_attempts=job.attempts
        )
        self._schedule_retry(job)

    def _write_snapshot(
        self, camera_id: str, event_id: str, ring_buffer: JpegRingBuffer, ts: float | None
    ) -> str | None:
        jpeg = ring_buffer.closest_to(ts) if ts is not None else None
        if jpeg is None:
            return None
        return self.store.save_snapshot(camera_id, event_id, jpeg)
