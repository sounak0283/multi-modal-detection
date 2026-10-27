"""Suggested questions for the chat (Phase J, UX item 2).

Rule-based on purpose: they appear instantly and cost nothing. They follow the page the
user is on, what has happened in the last day, and - in an alert's own chat - the alert's
type, with a generic set for any type not listed (a new module's alerts still get good
suggestions).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

MAX_SUGGESTIONS = 4

GENERAL = [
    "What happened in the last 24 hours?",
    "Is any alert repeating at the same time every day?",
    "Did any camera go offline this week?",
    "Which camera had the most alerts this week?",
]

BY_VIEW = {
    "live": ["How do I see all cameras at once?"],
    "boundaries": ["How do I draw a boundary?", "Which boundary has the most alerts?"],
    "cameras": ["How do I add an RTSP camera?", "Which cameras are offline right now?"],
    "history": ["Summarise this week's alerts by type", "What time of day are alerts busiest?"],
    "evidence": ["Find clips of people without helmets", "Show me this week's fire alerts"],
    "recognition": ["Who was recognised today?"],
    "alerts": ["How do I set up email alerts?"],
    "persons": ["How do I enrol a person?"],
    "users": ["What can an operator do compared to an admin?"],
    "videotest": ["How does Video test work?"],
}

BY_KIND = {
    "fire": ["Is the fire growing?", "Is anyone near the fire?"],
    "smoke": ["Where is the smoke coming from?", "Is the smoke spreading?"],
    "crowd": ["What is the crowd doing?", "How many people are in the group?"],
    "ppe": ["Who is missing a helmet?", "What is the worker doing?"],
    "access": ["Who entered the restricted area?", "Did they leave again?"],
    "boundary": ["Who crossed and where did they go?"],
    "welding": ["Is this welding or a fire?"],
}
ALERT_GENERIC = [
    "What happens in this clip?",
    "How many people are visible?",
    "Has this happened before at this camera?",
]


def suggestions(db: Any, view: str | None = None, event_id: str | None = None) -> list[str]:
    if event_id:
        try:
            event = db["events"].find_one({"_id": ObjectId(event_id)}, {"kind": 1, "clip_path": 1})
        except (InvalidId, TypeError):
            event = None
        if event is not None:
            asks = [*BY_KIND.get(str(event.get("kind")), []), *ALERT_GENERIC]
            if not event.get("clip_path"):
                asks = ["What does the snapshot show?", "Has this happened before at this camera?"]
            return asks[:MAX_SUGGESTIONS]

    asks: list[str] = []
    day_ago = datetime.now(UTC) - timedelta(days=1)
    for kind in ("fire", "smoke"):
        if db["events"].find_one({"kind": kind, "ts": {"$gte": day_ago}}, {"_id": 1}):
            asks.append(f"Tell me about today's {kind} alerts")
    asks += BY_VIEW.get(view or "", [])
    asks += [q for q in GENERAL if q not in asks]
    return asks[:MAX_SUGGESTIONS]
