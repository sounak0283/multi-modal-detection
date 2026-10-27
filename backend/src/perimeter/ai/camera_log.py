"""Camera online/offline history (Phase J), so the chat can answer "which camera goes
missing, and at what time".

Nothing recorded this before: `/api/health` only knows the state right now. This samples
the same state every `ai.camera_log_seconds` and stores one row per *change* in
`camera_status` - `{camera_id, state, start, end, last_seen}`, `end` None while current.
It only reads pipeline state; the pipeline itself is not touched.

States: live | lost (feed stopped responding) | starting | stopped (enabled but not
running) | disabled. A server restart closes the open rows at their `last_seen`, so time
the whole system was off shows up as a gap rather than as the camera being "live".
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pymongo import ASCENDING

log = logging.getLogger("perimeter.ai.camera_log")

CAMERA_STATUS = "camera_status"

StatesFn = Callable[[], dict[str, str]]  # camera id -> state


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class CameraStatusLog:
    def __init__(self, db: Any, states: StatesFn, interval_s: float = 30.0,
                 retention_days: int = 30) -> None:
        self.rows = db[CAMERA_STATUS]
        self._states = states
        self.interval_s = interval_s
        self.retention_days = retention_days
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.rows.create_index([("camera_id", ASCENDING), ("start", ASCENDING)])
        self.rows.create_index([("end", ASCENDING)])
        self.close_open_rows()
        self._thread = threading.Thread(target=self._run, name="camera-status-log", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:  # noqa: BLE001 - a bad sample must not end the log
                log.exception("camera status sample failed")
            self._stop.wait(self.interval_s)

    def close_open_rows(self) -> None:
        """At startup: rows still open from the last run end where they were last seen."""
        for row in self.rows.find({"end": None}):
            self.rows.update_one({"_id": row["_id"]}, {"$set": {"end": row["last_seen"]}})

    def sample(self, now: datetime | None = None) -> None:
        now = now or _now()
        states = self._states()
        open_rows = {row["camera_id"]: row for row in self.rows.find({"end": None})}
        for camera_id, state in states.items():
            row = open_rows.pop(camera_id, None)
            if row is not None and row["state"] == state:
                self.rows.update_one({"_id": row["_id"]}, {"$set": {"last_seen": now}})
                continue
            if row is not None:
                self.rows.update_one({"_id": row["_id"]}, {"$set": {"end": now, "last_seen": now}})
            self.rows.insert_one({"camera_id": camera_id, "state": state, "start": now,
                                  "end": None, "last_seen": now})
        for row in open_rows.values():  # camera deleted
            self.rows.update_one({"_id": row["_id"]}, {"$set": {"end": now, "last_seen": now}})
        self.rows.delete_many({"end": {"$lt": now - timedelta(days=self.retention_days)}})

    # -- reading ----------------------------------------------------------------------

    def history(self, since: datetime, until: datetime,
                camera_id: str | None = None) -> list[dict[str, Any]]:
        """Every non-live period (and every gap with no record, i.e. the server was off)
        overlapping [since, until], oldest first."""
        query: dict[str, Any] = {
            "start": {"$lt": until},
            "$or": [{"end": None}, {"end": {"$gt": since}}],
        }
        if camera_id:
            query["camera_id"] = camera_id
        rows = list(self.rows.find(query).sort("start", ASCENDING))
        periods: list[dict[str, Any]] = []
        last_end: dict[str, datetime] = {}
        for camera in {row["camera_id"] for row in rows}:
            # The row before the window, so a gap that began before `since` is still seen.
            first = min(row["start"] for row in rows if row["camera_id"] == camera)
            before = self.rows.find_one({"camera_id": camera, "start": {"$lt": first}},
                                        sort=[("start", -1)])
            if before is not None:
                last_end[camera] = _aware(before["end"] or before["last_seen"])
        for row in rows:
            start, end = _aware(row["start"]), _aware(row["end"] or row["last_seen"])
            previous = last_end.get(row["camera_id"])
            if previous is not None and (start - previous).total_seconds() > 2 * self.interval_s:
                periods.append(_period(row["camera_id"], "no_record", max(previous, since),
                                       min(start, until)))
            last_end[row["camera_id"]] = end
            if row["state"] != "live":
                periods.append(_period(row["camera_id"], row["state"], max(start, since),
                                       min(end, until)))
        return periods


def _period(camera_id: str, state: str, start: datetime, end: datetime) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "state": state,
        "start": start,
        "end": end,
        "minutes": round(max(0.0, (end - start).total_seconds()) / 60.0, 1),
    }


def states_from(manager: Any, registry: Any) -> StatesFn:
    """Read-only view of each configured camera's state, as `/api/health` reports it."""

    def states() -> dict[str, str]:
        running = manager.all()
        out: dict[str, str] = {}
        for camera in registry.list():
            if not camera.enabled:
                out[camera.id] = "disabled"
            elif camera.id not in running:
                out[camera.id] = "stopped"
            else:
                out[camera.id] = running[camera.id].stats.feed_state
        return out

    return states
