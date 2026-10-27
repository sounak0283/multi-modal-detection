"""EmailSink (Expansion Plan Phase D).

Registered (via `alerts/cooldown.py:route()`) as an `AlertBus` sink. Its own
queue-plus-background-thread, mirroring `evidence/writer.py:EvidenceWriter` -
deliberately not a bare function called straight from `AlertBus._notify()`, because a
slow/hung SMTP connection would otherwise stall that one notify loop for every other
sink and every subsequent alert. Same failure posture as the rest of this codebase's
sinks: a bad send is logged and dropped, never fatal - email is a best-effort notification
channel, not the record of truth (the `events` collection is). The one exception is the
[VIDEO READY] email, which is retried a few times (VIDEO_EMAIL_RETRY_BACKOFF_S).
"""

from __future__ import annotations

import logging
import queue
import smtplib
import threading
from collections.abc import Callable
from email.message import EmailMessage
from typing import Any

from perimeter.timefmt import local_time

log = logging.getLogger("perimeter.alerts.sinks.smtp")

SendFn = Callable[[EmailMessage], None]

# Concurrent email sends. Small on purpose: Gmail rate-limits, and alerts are rare.
DEFAULT_WORKERS = 3
# A failed [VIDEO READY] email is tried again after each of these waits, then given up.
# Only the video email: its alert email already reached people, and the video is the part
# that is worth a retry when a send hits a transient Gmail/network error.
VIDEO_EMAIL_RETRY_BACKOFF_S = (10.0, 30.0, 60.0)


def _default_send(
    host: str, port: int, user: str, password: str, use_tls: bool
) -> SendFn:
    """Builds the real sender: connect, STARTTLS, auth, send, disconnect - once per
    message, since alerts are rare enough that connection reuse isn't worth the
    complexity of keeping a long-lived SMTP session healthy across retries/reconnects."""

    def _send(message: EmailMessage) -> None:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            if use_tls:
                smtp.starttls()
            if user:
                smtp.login(user, password)
            smtp.send_message(message)

    return _send


class EmailSink:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        from_addr: str,
        use_tls: bool = True,
        send: SendFn | None = None,
        on_result: Callable[[str | None, str | None], None] | None = None,
        workers: int = DEFAULT_WORKERS,
        video_retry_backoff_s: tuple[float, ...] = VIDEO_EMAIL_RETRY_BACKOFF_S,
    ) -> None:
        # on_result(event_id, error) - error is None on success. Lets the caller record
        # delivery on the stored event; without it `delivered` stays False forever and
        # nobody can tell a working sink from a broken one.
        self._on_result = on_result
        self.from_addr = from_addr or user
        self._send = send or _default_send(host, port, user, password, use_tls)

        # (message, event id, is the [VIDEO READY] email, send attempt number)
        self._queue: queue.Queue[tuple[EmailMessage, str | None, bool, int]] = queue.Queue()
        self.video_retry_backoff_s = tuple(video_retry_backoff_s)
        # Retries wait on timers, never inside a worker, so a waiting retry never holds up
        # anyone else's alert email.
        self._retry_timers: list[threading.Timer] = []
        self._timer_lock = threading.Lock()
        # Each send opens its own SMTP session (a few seconds to 20s on Gmail). One worker
        # meant a video email queued behind other alerts' emails, a full minute late, so a
        # few sends may run at once.
        self._workers = max(1, workers)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        for index in range(self._workers):
            thread = threading.Thread(target=self._run, name=f"email-sink-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)
        log.info("email sink started (from=%s, %d sender(s))", self.from_addr, self._workers)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._timer_lock:
            for timer in self._retry_timers:
                timer.cancel()
            self._retry_timers = []
        for thread in self._threads:
            thread.join(timeout)
        self._threads = []

    # -- producer side (an AlertBus sink, via cooldown.route()) ---------------

    def on_alert(
        self, payload: dict[str, Any], to: tuple[str, ...], followup: bool = False
    ) -> None:
        message = _build_message(payload, self.from_addr, to, followup=followup)
        self._queue.put((message, payload.get("id"), followup, 1))

    def _report(self, event_id: str | None, error: str | None) -> None:
        if self._on_result is None or not event_id:
            return
        try:
            self._on_result(event_id, error)
        except Exception:  # noqa: BLE001 - bookkeeping must never kill the send thread
            log.exception("could not record delivery result for event %s", event_id)

    # -- consumer side ---------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                message, event_id, followup, attempt = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._send(message)
            except Exception as exc:  # noqa: BLE001 - a failed send must not kill this thread
                log.exception("email send failed for event %s", event_id)
                self._report(event_id, f"{type(exc).__name__}: {exc}")
                if followup:
                    self._retry_video_email(message, event_id, attempt)
            else:
                log.info("email sent for event %s to %s", event_id, message["To"])
                self._report(event_id, None)
        log.info("email sink stopped")

    def _retry_video_email(self, message: EmailMessage, event_id: str | None, attempt: int) -> None:
        if attempt > len(self.video_retry_backoff_s):
            log.error(
                "giving up on the video email for event %s after %d attempts", event_id, attempt
            )
            return
        wait = self.video_retry_backoff_s[attempt - 1]
        log.warning(
            "video email for event %s failed (attempt %d) - retrying in %.0fs",
            event_id, attempt, wait,
        )
        timer = threading.Timer(
            wait, lambda: self._stop.is_set()
            or self._queue.put((message, event_id, True, attempt + 1)),
        )
        timer.daemon = True
        with self._timer_lock:
            if self._stop.is_set():
                return
            self._retry_timers = [t for t in self._retry_timers if t.is_alive()] + [timer]
        timer.start()


def _build_message(
    payload: dict[str, Any], from_addr: str, to: tuple[str, ...], followup: bool = False
) -> EmailMessage:
    severity = str(payload.get("severity", "medium")).upper()
    kind = payload.get("kind", "alert")
    camera_id = payload.get("camera_id", "unknown camera")

    message = EmailMessage()
    prefix = "[VIDEO READY] " if followup else ""
    message["Subject"] = f"{prefix}[{severity}] {kind} — {camera_id}"
    message["From"] = from_addr
    message["To"] = ", ".join(to)

    ts = payload.get("ts")
    when = local_time(ts) if isinstance(ts, int | float) else "unknown time"
    lines = [
        payload.get("message", "(no message)"),
        "",
        f"Camera:   {camera_id}",
        f"Kind:     {kind}" + (f" / {payload['subtype']}" if payload.get("subtype") else ""),
        f"Zone:     {payload.get('zone_name') or payload.get('zone_id') or '-'}",
        f"Severity: {severity}",
        f"Time:     {when}",
    ]
    clip_url = payload.get("clip_url")
    if clip_url:
        lines += ["", f"Video:    {clip_url}"]
        if payload.get("clip_link_expires_at"):
            lines.append(f"          (link expires {payload['clip_link_expires_at']})")
    elif followup:
        lines += ["", "Video:    (not available)"]
    message.set_content("\n".join(lines))
    return message
