"""Alert delivery thread (PLAN.md section 4).

Persistence and notification live off the inference hot path. A database round-trip or
an HTTP POST that blocks for three seconds while holding the inference loop would stall
the entire pipeline, so events are handed to this thread and it deals with the slow
parts.

Failure policy: **detection outranks persistence.** If PostgreSQL is unreachable the
pipeline keeps detecting, the dashboard keeps serving from its in-memory buffer, and
this thread keeps retrying. A database outage must never take the cameras offline -
that would turn a storage problem into a security incident.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("fsbd.alerts")

QUEUE_MAXSIZE = 1000
RETRY_BACKOFF_START_S = 2.0
RETRY_BACKOFF_MAX_S = 60.0


@dataclass
class AlertStats:
    queued: int = 0
    persisted: int = 0
    failed: int = 0
    dropped: int = 0
    pending_retry: int = 0
    last_error: str | None = None


@dataclass
class _Pending:
    payload: dict[str, Any]
    attempts: int = 0
    next_attempt: float = field(default_factory=time.monotonic)


class AlertBus:
    """Queue in, PostgreSQL (and later notification sinks) out."""

    def __init__(
        self,
        database: Any | None = None,
        sinks: list[Callable[[dict[str, Any]], None]] | None = None,
    ) -> None:
        self.database = database
        self.sinks = sinks or []
        self.stats = AlertStats()

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._retry: list[_Pending] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="alerts", daemon=True)
        self._thread.start()
        log.info("alert bus started (persistence=%s)", "on" if self.database else "off")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    # -- producer side -----------------------------------------------------

    def publish(self, payload: dict[str, Any]) -> None:
        """Called from the inference thread. Never blocks.

        A full queue drops the OLDEST pending alert rather than the new one: if the
        backlog is being shed, the most recent state of the site is the more useful
        thing to keep.
        """
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.stats.dropped += 1
                self._queue.put_nowait(payload)
                log.warning("alert queue full - dropped the oldest pending alert")
            except queue.Empty:  # pragma: no cover - racing with the consumer
                pass
        self.stats.queued += 1

    # -- consumer side -----------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain_retries()
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._handle(payload, attempts=0)

        # Best-effort flush so a clean shutdown does not lose queued alerts.
        while True:
            try:
                self._handle(self._queue.get_nowait(), attempts=0)
            except queue.Empty:
                break
        log.info("alert bus stopped")

    def _handle(self, payload: dict[str, Any], attempts: int) -> None:
        if self.database is None:
            self._notify(payload)
            return

        try:
            event_id = self.database.insert_event(payload)
            payload["id"] = event_id
            self.stats.persisted += 1
            self.stats.last_error = None
        except Exception as exc:  # noqa: BLE001 - any driver error must be retried
            self.stats.failed += 1
            self.stats.last_error = str(exc)
            self._schedule_retry(payload, attempts + 1)
            log.warning("could not persist alert (attempt %d): %s", attempts + 1, exc)
            # Still notify: a delivered warning with no database row beats silence.
            self._notify(payload)
            return

        self._notify(payload)

    def _notify(self, payload: dict[str, Any]) -> None:
        for sink in self.sinks:
            try:
                sink(payload)
            except Exception:  # noqa: BLE001 - one broken sink must not stop the others
                log.exception("alert sink failed")

    def _schedule_retry(self, payload: dict[str, Any], attempts: int) -> None:
        backoff = min(RETRY_BACKOFF_START_S * (2 ** (attempts - 1)), RETRY_BACKOFF_MAX_S)
        with self._lock:
            self._retry.append(
                _Pending(
                    payload=payload,
                    attempts=attempts,
                    next_attempt=time.monotonic() + backoff,
                )
            )
            self.stats.pending_retry = len(self._retry)

    def _drain_retries(self) -> None:
        if not self._retry:
            return
        now = time.monotonic()
        with self._lock:
            due = [p for p in self._retry if p.next_attempt <= now]
            self._retry = [p for p in self._retry if p.next_attempt > now]
            self.stats.pending_retry = len(self._retry)

        for pending in due:
            if self.database is None:
                continue
            try:
                pending.payload["id"] = self.database.insert_event(pending.payload)
                self.stats.persisted += 1
                self.stats.last_error = None
                log.info("alert persisted on retry %d", pending.attempts)
            except Exception as exc:  # noqa: BLE001
                self.stats.last_error = str(exc)
                self._schedule_retry(pending.payload, pending.attempts + 1)
