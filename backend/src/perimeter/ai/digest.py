"""Daily digest (Phase J, UX item 5): yesterday's alerts in a few lines, emailed each
morning at `ai.digest_time` (server local time).

The numbers are computed here, not by the model: counts by type and camera, the busiest
hour, a comparison with the 7 days before, alerts that recur at the same hour on most
days, camera outages, and the video-analysis summaries of serious alerts. Claude only
writes them up. Video test alerts are counted separately and left out of everything else.

Stored in `ai_digests` (one per day, id = the date), so the dashboard can show past ones
and a restart never sends the same day twice.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from email.message import EmailMessage
from typing import Any

from perimeter.ai.bedrock import LazyClient, converse_text
from perimeter.ai.queries import event_query, is_test_video, local_day_bounds, now_local, to_local
from perimeter.settings import AISettings
from perimeter.timefmt import local_time

log = logging.getLogger("perimeter.ai.digest")

DIGESTS = "ai_digests"
SERIOUS_KINDS = ("fire", "smoke", "access", "crowd", "ppe")
RECURRING_MIN_DAYS = 4  # of the last 7, at the same hour
CHECK_EVERY_S = 60.0

PROMPT = """Write the daily safety digest for {day} from these figures (JSON). Plain text
for an email - no markdown (no ** or #) - under 200 words: a one-line headline, then
short bullet points starting with "- ". Say what
changed compared with the previous 7 days, any alert recurring at the same hour, cameras
that were offline, and the notable alerts with what their video showed. If nothing
happened, say so in one line. Do not invent anything not in the figures.

{stats}"""


class DigestService:
    def __init__(
        self,
        ai: AISettings,
        db: Any,
        client: LazyClient,
        send: Callable[[EmailMessage], None] | None = None,
        from_addr: str = "",
        recipients: Callable[[], list[str]] | None = None,
        camera_log: Any = None,
    ) -> None:
        self.ai = ai
        self.events = db["events"]
        self.analyses = db["video_analyses"]
        self.digests = db[DIGESTS]
        self._client = client
        self._send = send
        self.from_addr = from_addr
        self._recipients = recipients or (lambda: [])
        self.camera_log = camera_log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- schedule ------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None or not self.ai.digest_enabled:
            return
        self._thread = threading.Thread(target=self._run, name="ai-digest", daemon=True)
        self._thread.start()
        log.info("daily digest scheduled for %s", self.ai.digest_time)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_due()
            except Exception:  # noqa: BLE001 - try again at the next check
                log.exception("daily digest failed")
            self._stop.wait(CHECK_EVERY_S)

    def run_due(self, now: datetime | None = None) -> dict[str, Any] | None:
        """Yesterday's digest, once it is past `digest_time` and not yet made."""
        now = now or now_local()
        hour, minute = (int(p) for p in self.ai.digest_time.split(":"))
        if (now.hour, now.minute) < (hour, minute):
            return None
        yesterday = now.date() - timedelta(days=1)
        if self.digests.find_one({"_id": yesterday.isoformat()}, {"_id": 1}):
            return None
        return self.generate(yesterday)

    # -- making one ----------------------------------------------------------------------------

    def generate(self, day: date, email: bool = True) -> dict[str, Any]:
        with self._lock:
            stats = self.stats(day)
            text = converse_text(
                self._client.get(), self.ai,
                [{"text": PROMPT.format(day=day.strftime("%A %d %B %Y"), stats=stats)}],
                max_tokens=700,
            ).strip()
            doc = {"_id": day.isoformat(), "day": day.isoformat(), "text": text,
                   "stats": stats, "created_at": datetime.now(UTC), "emailed_to": [],
                   "email_error": None}
            if email:
                doc["emailed_to"], doc["email_error"] = self._email(day, text)
            self.digests.replace_one({"_id": doc["_id"]}, doc, upsert=True)
            log.info("daily digest for %s made (emailed to %d)", day, len(doc["emailed_to"]))
            return public(doc)

    def stats(self, day: date) -> dict[str, Any]:
        start, end = local_day_bounds(day)
        week_start = start - timedelta(days=7)
        docs = list(self.events.find(event_query(since=week_start, until=end)))
        today = [d for d in docs if start <= to_local(d["ts"]) < end]
        real = [d for d in today if not is_test_video(d)]
        before = [d for d in docs if to_local(d["ts"]) < start and not is_test_video(d)]

        by_kind = Counter(d.get("kind") for d in real)
        avg_kind = {k: round(v / 7, 1) for k, v in Counter(d.get("kind") for d in before).items()}
        hours = Counter(to_local(d["ts"]).hour for d in real)

        recurring = []
        week = [d for d in docs if not is_test_video(d)]
        days_at = defaultdict(set)
        for d in week:
            moment = to_local(d["ts"])
            days_at[(d.get("kind"), d.get("camera_id"), moment.hour)].add(moment.date())
        for (kind, camera, hour), days in days_at.items():
            if len(days) >= RECURRING_MIN_DAYS:
                recurring.append({"kind": kind, "camera": camera, "hour": f"{hour:02d}:00",
                                  "days_of_last_8": len(days)})

        outages = []
        if self.camera_log is not None:
            for p in self.camera_log.history(start, end):
                if p["state"] != "disabled" and p["minutes"] >= 1:
                    outages.append({"camera": p["camera_id"], "state": p["state"],
                                    "from": local_time(to_local(p["start"])),
                                    "to": local_time(to_local(p["end"])),
                                    "minutes": p["minutes"]})

        serious_ids = [str(d["_id"]) for d in real if d.get("kind") in SERIOUS_KINDS]
        notable = []
        for a in self.analyses.find({"_id": {"$in": serious_ids}, "status": "done"}).limit(8):
            notable.append({"kind": a.get("kind"), "camera": a.get("camera_id"),
                            "time": local_time(to_local(a.get("event_ts"))),
                            "video_summary": (a.get("analysis") or {}).get("summary")})

        return {
            "day": day.isoformat(),
            "total": len(real),
            "by_type": dict(by_kind.most_common()),
            "previous_7_day_average_by_type": avg_kind,
            "by_camera": dict(Counter(d.get("camera_id") for d in real).most_common()),
            "busiest_hour": f"{hours.most_common(1)[0][0]:02d}:00" if hours else None,
            "recurring_same_hour": recurring[:10],
            "camera_outages": outages[:20],
            "notable_alerts": notable,
            "video_test_alerts": len(today) - len(real),
        }

    def _email(self, day: date, text: str) -> tuple[list[str], str | None]:
        to = [r for r in (self.ai.digest_recipients or tuple(self._recipients())) if r]
        if self._send is None or not to:
            return [], "email is not configured" if self._send is None else "no recipients"
        message = EmailMessage()
        message["Subject"] = f"Daily safety digest - {day.strftime('%d %b %Y')}"
        message["From"] = self.from_addr
        message["To"] = ", ".join(to)
        message.set_content(text + "\n\n-- Perimeter (AI summary; check the dashboard for "
                            "the full alert history)")
        try:
            self._send(message)
        except Exception as exc:  # noqa: BLE001 - the digest is still stored and viewable
            log.exception("digest email failed")
            return [], f"{type(exc).__name__}: {exc}"
        return to, None

    # -- reading ---------------------------------------------------------------------------------

    def latest(self, limit: int = 14) -> list[dict[str, Any]]:
        return [public(d) for d in self.digests.find().sort("_id", -1).limit(limit)]


def public(doc: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k != "_id"}
    if isinstance(out.get("created_at"), datetime):
        out["created_at"] = out["created_at"].isoformat()
    return out
