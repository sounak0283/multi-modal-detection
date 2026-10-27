"""Plain-English alert search (Phase J, UX item 6).

"fire alerts on cam_01 last night" -> Claude fills in a fixed set of filters (forced tool
call, so the answer is always structured) -> the events collection is queried with them.
The model only chooses filters; it never sees or invents the alerts themselves. The
filters come back with the results, so the page shows exactly what was searched.
"""

from __future__ import annotations

from typing import Any

from perimeter.ai.bedrock import LazyClient
from perimeter.ai.catalog import known_kinds
from perimeter.ai.queries import SOURCES, event_query, in_hours, now_local, to_local
from perimeter.settings import AISettings
from perimeter.timefmt import local_time

MAX_RESULTS = 500
HOUR_SCAN_LIMIT = 5000

FILTER_TOOL = {"toolSpec": {
    "name": "set_filters",
    "description": "The alert-history filters that match the request.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "one alert type, or empty"},
            "camera_id": {"type": "string", "description": "one camera id, or empty"},
            "zone": {"type": "string", "description": "boundary/zone name words, or empty"},
            "since": {"type": "string", "description": "ISO local start, or empty"},
            "until": {"type": "string", "description": "ISO local end (exclusive), or empty"},
            "hour_from": {"type": "integer", "description": "local hour 0-23 the alerts "
                          "must start at or after, every day (e.g. 22 for 'at night')"},
            "hour_to": {"type": "integer", "description": "local hour 0-23 they must be "
                        "before, every day (e.g. 6); wraps past midnight"},
            "text": {"type": "string", "description": "words that must be in the alert "
                     "message, only if no other filter covers them"},
            "source": {"type": "string", "enum": list(SOURCES)},
            "explanation": {"type": "string", "description": "one short sentence saying "
                            "what is being shown, e.g. 'Fire alerts on cam_01, 22:00-06:00 "
                            "last night'"},
        },
        "required": ["explanation"],
    }},
}}

PROMPT = """Turn this alert-history search into filters. Now (site local time): {now}.
Alert types: {kinds}. Cameras: {cameras}.
"last night" = yesterday 18:00 to today 06:00 unless said otherwise; "today" starts at
local midnight; "this week" starts Monday. Leave a filter empty if not asked for. Use
source "real" for "real cameras", "test" for "test video".
Search: {query}"""


class AlertSearch:
    def __init__(self, ai: AISettings, db: Any, client: LazyClient, cameras) -> None:
        self.ai = ai
        self.events = db["events"]
        self._client = client
        self._cameras = cameras

    def parse(self, query: str) -> dict[str, Any]:
        cameras = ", ".join(f"{c['id']} ({c['name']})" for c in self._cameras()) or "none"
        response = self._client.get().converse(
            modelId=self.ai.chat_model,
            messages=[{"role": "user", "content": [{"text": PROMPT.format(
                now=local_time(now_local()), kinds=", ".join(known_kinds()),
                cameras=cameras, query=query[:500],
            )}]}],
            toolConfig={"tools": [FILTER_TOOL], "toolChoice": {"tool": {"name": "set_filters"}}},
            inferenceConfig={"maxTokens": 400, "temperature": 0},
        )
        for part in response["output"]["message"]["content"]:
            if "toolUse" in part:
                return {k: v for k, v in part["toolUse"]["input"].items() if v not in (None, "")}
        raise ValueError("the model did not return filters")

    def search(self, query: str, limit: int = 200) -> dict[str, Any]:
        filters = self.parse(query)
        return {"filters": filters, "explanation": filters.get("explanation", ""),
                "events": self.run(filters, limit)}

    def run(self, filters: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
        from perimeter.store.db import _doc_to_dict

        limit = max(1, min(int(limit), MAX_RESULTS))
        query = event_query(
            kind=filters.get("kind"), camera_id=filters.get("camera_id"),
            zone=filters.get("zone"), since=filters.get("since"), until=filters.get("until"),
            source=filters.get("source") or "all", text=filters.get("text"),
        )
        hour_from, hour_to = filters.get("hour_from"), filters.get("hour_to")
        hourly = hour_from is not None or hour_to is not None
        cursor = self.events.find(query).sort("ts", -1).limit(HOUR_SCAN_LIMIT if hourly else limit)
        out = []
        for doc in cursor:
            if hourly and not in_hours(to_local(doc.get("ts")), hour_from, hour_to):
                continue
            out.append(_doc_to_dict(doc))
            if len(out) >= limit:
                break
        return out
