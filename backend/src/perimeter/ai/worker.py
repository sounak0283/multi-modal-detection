"""Background analysis of alert clips (Phase J steps 2 + 2b).

The worker watches the events collection for newly saved clips instead of being called by
the evidence writer, so nothing in detection, alerting or email changes or waits on it: a
slow or failed Bedrock call can only delay an analysis, never an alert.

For each clip it runs Pegasus (`video.py`, stored in `video_analyses`) and then frame RAG
(`frames.py`, the search index), one job at a time. Auto jobs are alerts of
`ai.auto_analyze_kinds`, at most `ai.daily_cap` per rolling 24 h. A manual request from the
dashboard goes first, works for any alert with a clip, and is not capped.

The verdict (likely_real / likely_false / unsure) is stored but kept out of `public()`:
the dashboard does not show it yet.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from perimeter.ai.bedrock import AIUnavailableError
from perimeter.ai.frames import FrameRAG
from perimeter.ai.video import PegasusAnalyzer, S3UnreadableError
from perimeter.settings import AISettings
from perimeter.testvideo.sessions import TEST_PREFIX

log = logging.getLogger("perimeter.ai.worker")

ANALYSES = "video_analyses"
# Clips saved shortly before a restart are still picked up; older ones are only analysed
# on request, so turning this on never sets off a backlog of past alerts.
LOOKBACK_S = 15 * 60
SCAN_LIMIT = 50
MAX_ATTEMPTS = 3
RETRY_AFTER_S = 5 * 60
RETENTION_SWEEP_S = 3600
HIDDEN_FIELDS = ("verdict",)
# Errors no retry can fix: no clip, AI switched off, a clip that cannot be decoded.
PERMANENT_ERRORS = (ValueError, FileNotFoundError, AIUnavailableError, S3UnreadableError)


class AnalysisError(Exception):
    """A request the worker cannot take (unknown alert, no clip)."""


def _now() -> datetime:
    return datetime.now(UTC)


def _event_dict(doc: dict[str, Any]) -> dict[str, Any]:
    return {**doc, "id": str(doc["_id"])}


class AnalysisWorker:
    def __init__(
        self,
        ai: AISettings,
        db: Any,  # a pymongo Database
        analyzer: PegasusAnalyzer,
        rag: FrameRAG | None = None,
        retention_days: int = 30,
    ) -> None:
        self.ai = ai
        self.events = db["events"]
        self.analyses = db[ANALYSES]
        self.analyzer = analyzer
        self.rag = rag
        self.retention_days = retention_days
        self.started_at = _now()
        self._manual: queue.Queue[str] = queue.Queue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sweep = float("-inf")

    # -- lifecycle -----------------------------------------------------------

    def ensure_indexes(self) -> None:
        self.analyses.create_index([("status", ASCENDING), ("updated_at", ASCENDING)])
        self.analyses.create_index([("trigger", ASCENDING), ("created_at", DESCENDING)])
        self.analyses.create_index([("event_ts", ASCENDING)])

    def start(self) -> None:
        if self._thread is not None:
            return
        self.ensure_indexes()
        self.recover_interrupted()
        self._thread = threading.Thread(target=self._run, name="ai-analysis", daemon=True)
        self._thread.start()
        log.info(
            "AI analysis worker started (auto: %s, cap %d/24h, frame RAG %s)",
            ", ".join(self.ai.auto_analyze_kinds) or "off", self.ai.daily_cap,
            "on" if self.rag else "off",
        )

    def recover_interrupted(self) -> None:
        """A job cut off by a restart is retried on the next pass, not stuck at "running"."""
        self.analyses.update_many(
            {"status": {"$in": ["queued", "running"]}},
            {"$set": {"status": "failed", "error": "interrupted by a restart",
                      "updated_at": _now() - timedelta(seconds=RETRY_AFTER_S)}},
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 - the worker must outlive any one bad pass
                log.exception("AI analysis pass failed")
            self._wake.wait(self.ai.poll_seconds)
            self._wake.clear()

    # -- requests from the dashboard -------------------------------------------

    def request(self, event_id: str) -> dict[str, Any]:
        """Queue a manual analysis (or a re-run) of one alert's clip."""
        event = self._event(event_id)
        if event is None:
            raise AnalysisError(f"alert {event_id!r} not found")
        if not event.get("clip_path"):
            raise AnalysisError("this alert has no saved clip to analyse")
        current = self.analyses.find_one({"_id": event["id"]})
        if current and current.get("status") in ("queued", "running"):
            return self.public(current)
        doc = self._new_doc(event, "manual")
        self.analyses.replace_one({"_id": event["id"]}, doc, upsert=True)
        self._manual.put(event["id"])
        self._wake.set()
        return self.public(doc)

    def status(self, event_id: str) -> dict[str, Any] | None:
        doc = self.analyses.find_one({"_id": str(event_id)})
        return self.public(doc) if doc else None

    @staticmethod
    def public(doc: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in doc.items() if k not in ("_id", "raw")}
        out["event_id"] = str(doc["_id"])
        analysis = dict(out.get("analysis") or {})
        for name in HIDDEN_FIELDS:
            analysis.pop(name, None)
        if "analysis" in out:
            out["analysis"] = analysis
        for key, value in list(out.items()):
            if isinstance(value, datetime):
                out[key] = value.isoformat()
        return out

    # -- one pass ---------------------------------------------------------------

    def run_once(self) -> int:
        """Manual requests first, then retries, then new auto clips. Returns jobs run."""
        done = self._drain_manual()
        for event_id in self._retry_ids():
            if self._stop.is_set():
                return done
            event = self._event(event_id)
            if event is None:  # the alert itself is gone (retention) - nothing to retry
                self.analyses.update_one(
                    {"_id": event_id},
                    {"$set": {"attempts": MAX_ATTEMPTS, "error": "the alert no longer exists"}},
                )
                continue
            self._process(event)
            done += 1 + self._drain_manual()
        for event in self._new_auto_events():
            if self._stop.is_set() or not self._claim(event):
                continue
            self._process(event)
            done += 1 + self._drain_manual()
        self._sweep_retention()
        return done

    def _drain_manual(self) -> int:
        done = 0
        while not self._stop.is_set():
            try:
                event_id = self._manual.get_nowait()
            except queue.Empty:
                return done
            event = self._event(event_id)
            if event is not None:
                self._process(event)
                done += 1
        return done

    def _new_auto_events(self) -> list[dict[str, Any]]:
        if not self.ai.auto_analyze_kinds:
            return []
        room = self.ai.daily_cap - self.analyses.count_documents(
            {"trigger": "auto", "created_at": {"$gte": _now() - timedelta(hours=24)}}
        )
        if room <= 0:
            return []
        query: dict[str, Any] = {
            "clip_status": "saved",
            "kind": {"$in": list(self.ai.auto_analyze_kinds)},
            "ts": {"$gte": self.started_at - timedelta(seconds=LOOKBACK_S)},
        }
        if self.ai.skip_test_video:
            query["message"] = {"$not": re.compile("^" + re.escape(TEST_PREFIX))}
        docs = list(self.events.find(query).sort("ts", ASCENDING).limit(SCAN_LIMIT))
        ids = [str(d["_id"]) for d in docs]
        known = {d["_id"] for d in self.analyses.find({"_id": {"$in": ids}}, {"_id": 1})}
        return [_event_dict(d) for d in docs if str(d["_id"]) not in known][:room]

    def _retry_ids(self) -> list[str]:
        cursor = self.analyses.find(
            {"status": "failed", "attempts": {"$lt": MAX_ATTEMPTS},
             "updated_at": {"$lte": _now() - timedelta(seconds=RETRY_AFTER_S)}},
            {"_id": 1},
        ).limit(SCAN_LIMIT)
        return [d["_id"] for d in cursor]

    def _claim(self, event: dict[str, Any]) -> bool:
        try:
            self.analyses.insert_one(self._new_doc(event, "auto"))
            return True
        except DuplicateKeyError:  # already analysed, queued, or requested meanwhile
            return False

    def _new_doc(self, event: dict[str, Any], trigger: str) -> dict[str, Any]:
        now = _now()
        return {
            "_id": event["id"],
            "trigger": trigger,
            "status": "queued",
            "attempts": 0,
            "kind": event.get("kind"),
            "camera_id": event.get("camera_id"),
            "event_ts": event.get("ts"),
            "created_at": now,
            "updated_at": now,
        }

    # -- one job ------------------------------------------------------------------

    def _process(self, event: dict[str, Any]) -> None:
        event_id = event["id"]
        self.analyses.update_one(
            {"_id": event_id},
            {"$set": {"status": "running", "updated_at": _now()}, "$inc": {"attempts": 1}},
        )
        started = time.perf_counter()
        updates: dict[str, Any] = {}
        summary = ""
        people_count = None
        try:
            result = self.analyzer.analyze(event)
            data = result.to_dict()
            updates.update({
                "status": "done", "error": None, "raw": data.pop("raw", None),
                "analysis": data,
            })
            summary = " ".join(p for p in (result.summary, result.findings) if p)
            if isinstance(result.people_count, int):
                people_count = result.people_count
        except Exception as exc:  # noqa: BLE001 - recorded on the job; the worker carries on
            permanent = isinstance(exc, PERMANENT_ERRORS)
            log.warning("video analysis of %s failed%s: %s", event_id,
                        " (not retried)" if permanent else "", exc)
            updates.update({"status": "failed", "error": str(exc)[:500]})
            if permanent:
                updates["attempts"] = MAX_ATTEMPTS

        if self.rag is not None:
            try:
                updates["frames_count"] = self.rag.index_event(
                    event, summary=summary, people_count=people_count
                )
                updates["frames_status"] = "done"
                updates["frames_error"] = None
            except Exception as exc:  # noqa: BLE001
                log.warning("frame indexing of %s failed: %s", event_id, exc)
                updates.update({"frames_status": "failed", "frames_error": str(exc)[:500]})

        updates["seconds"] = round(time.perf_counter() - started, 2)
        updates["updated_at"] = _now()
        self.analyses.update_one({"_id": event_id}, {"$set": updates})
        log.info("analysed %s alert %s in %.1fs (%s)", event.get("kind"), event_id,
                 updates["seconds"], updates["status"])

    # -- housekeeping ----------------------------------------------------------------

    def _event(self, event_id: str) -> dict[str, Any] | None:
        try:
            doc = self.events.find_one({"_id": ObjectId(event_id)})
        except (InvalidId, TypeError):
            return None
        return _event_dict(doc) if doc else None

    def _sweep_retention(self) -> None:
        """Same retention as the alerts themselves (`storage.retention_days`)."""
        if time.monotonic() - self._last_sweep < RETENTION_SWEEP_S:
            return
        self._last_sweep = time.monotonic()
        cutoff = _now() - timedelta(days=self.retention_days)
        self.analyses.delete_many({"event_ts": {"$lt": cutoff}})
        if self.rag is not None:
            self.rag.index.purge_before(cutoff.timestamp())
