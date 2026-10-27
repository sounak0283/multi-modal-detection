"""The assistant's tools (Phase J steps 3-5).

Everything here reads: alerts, statistics, camera history, video analyses, footage, the
help guide. The one write is `remember` (the assistant's own memory). Nothing a tool does
can change a camera, a boundary, an alert rule or an account.

Each tool is generic over alert types and modules - the type list comes from the alert
catalogue - so a new detection module's alerts can be asked about without changes here.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from perimeter.ai import help as product_help
from perimeter.ai.catalog import known_kinds
from perimeter.ai.frames import extract_keyframes
from perimeter.ai.queries import (
    GROUP_FIELDS,
    SEVERITIES,
    SOURCES,
    alert_stats,
    event_query,
    is_test_video,
    now_local,
    parse_when,
    to_local,
)
from perimeter.timefmt import local_time

log = logging.getLogger("perimeter.ai.tools")

MAX_LIST = 50
# Frames the assistant looks at itself. At 360 px a hard hat is a few pixels and the model
# guessed (measured on a welding clip); 720 px shows it.
LOOK_FRAME_HEIGHT = 720
# Frames for the people inspection keep more detail: each person is cropped out of them.
INSPECT_FRAME_HEIGHT = 1440
MAX_TOOL_TEXT = 12_000
HIDDEN_ANALYSIS_FIELDS = ("verdict",)


@dataclass
class ToolContext:
    user_id: str
    is_admin: bool = False
    event_id: str | None = None  # set in an alert's own chat


@dataclass
class ToolResult:
    data: Any
    images: list[bytes] = field(default_factory=list)

    def content(self) -> list[dict[str, Any]]:
        text = json.dumps(self.data, default=str, ensure_ascii=False)
        if len(text) > MAX_TOOL_TEXT:
            text = text[:MAX_TOOL_TEXT] + " ...(truncated - narrow the query)"
        blocks: list[dict[str, Any]] = [{"text": text}]
        blocks += [{"image": {"format": "jpeg", "source": {"bytes": jpeg}}} for jpeg in self.images]
        return blocks


@dataclass
class Tool:
    name: str
    label: str  # shown in the chat while it runs
    description: str
    properties: dict[str, Any]
    run: Callable[[ToolContext, dict[str, Any]], ToolResult]
    required: tuple[str, ...] = ()

    def spec(self) -> dict[str, Any]:
        return {"toolSpec": {
            "name": self.name,
            "description": self.description,
            "inputSchema": {"json": {
                "type": "object", "properties": self.properties, "required": list(self.required),
            }},
        }}


TIME_HELP = "ISO local date/time, e.g. 2026-10-27T22:00 (server local time if no offset)"
FILTERS = {
    "kind": {"type": "string", "description": "alert type, e.g. fire, ppe, boundary"},
    "camera_id": {"type": "string", "description": "camera id, e.g. cam_01"},
    "zone": {"type": "string", "description": "boundary/zone name or id (partial match)"},
    "since": {"type": "string", "description": f"from (inclusive). {TIME_HELP}"},
    "until": {"type": "string", "description": f"to (exclusive). {TIME_HELP}"},
    "source": {"type": "string", "enum": list(SOURCES),
               "description": "real cameras only, Video test only, or all (default)"},
    "severity": {"type": "array", "items": {"type": "string", "enum": list(SEVERITIES)},
                 "description": "only these severity levels, e.g. ['high', 'critical'] for "
                                "'high severity or worse'"},
}


def _event_row(doc: dict[str, Any], analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "id": str(doc["_id"]),
        "time": local_time(to_local(doc.get("ts"))) if doc.get("ts") else None,
        "kind": doc.get("kind"),
        "subtype": doc.get("subtype"),
        "camera_id": doc.get("camera_id"),
        "zone": doc.get("zone_name") or doc.get("zone_id"),
        "message": doc.get("message"),
        "severity": doc.get("severity"),
        "has_video": bool(doc.get("clip_path")),
        "has_snapshot": bool(doc.get("snapshot_path")),
        "test_video": is_test_video(doc),
    }
    if doc.get("identity_status"):
        row["person"] = doc.get("identity_name") or doc.get("identity_status")
    if analysis:
        row["video_summary"] = analysis.get("summary")
    return row


def _visible_analysis(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc or doc.get("status") != "done":
        return None
    analysis = dict(doc.get("analysis") or {})
    for name in HIDDEN_ANALYSIS_FIELDS:
        analysis.pop(name, None)
    return {k: analysis.get(k) for k in ("summary", "findings", "people_count", "activities")}


class ToolBox:
    def __init__(
        self,
        db: Any,
        store: Any,
        cameras: Callable[[], list[dict[str, Any]]],
        analyzer: Any = None,
        rag: Any = None,
        memory: Any = None,
        camera_log: Any = None,
        frame_height: int = 360,
        inspector: Any = None,
    ) -> None:
        self.inspector = inspector
        self.events = db["events"]
        self.analyses = db["video_analyses"]
        self.store = store
        self.cameras = cameras
        self.analyzer = analyzer
        self.rag = rag
        self.memory = memory
        self.camera_log = camera_log
        self.frame_height = frame_height
        self.tools = {tool.name: tool for tool in self._build()}

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.tools.values()]

    def label(self, name: str) -> str:
        tool = self.tools.get(name)
        return tool.label if tool else name

    def run(self, name: str, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"no tool named {name!r}")
        return tool.run(ctx, args or {})

    # -- the tools -----------------------------------------------------------------

    def _build(self) -> list[Tool]:
        kinds = ", ".join(known_kinds())
        tools = [
            Tool(
                "list_alerts", "Checking alerts",
                f"List recorded alerts, newest first, with filters. Alert types: {kinds}. "
                "Each row says whether it has video and its video summary if analysed.",
                {**FILTERS,
                 "text": {"type": "string", "description": "words in the alert message"},
                 "limit": {"type": "integer", "description": f"max rows (<= {MAX_LIST})"}},
                self._list_alerts,
            ),
            Tool(
                "alert_stats", "Counting alerts",
                "Count alerts grouped by one or two of: " + ", ".join(GROUP_FIELDS)
                + " (day/hour/weekday in local time; grouping by day lists days with zero "
                "alerts too). Use this for counts and patterns: how many (of a severity), "
                "busiest hour, alerts every day or every other day, which camera or zone "
                "has the most.",
                {**FILTERS, "group_by": {"type": "array", "items": {"type": "string",
                                         "enum": list(GROUP_FIELDS)}, "maxItems": 2}},
                self._alert_stats, required=("group_by",),
            ),
            Tool(
                "alert_details", "Opening the alert",
                "Everything about one alert: details, evidence, and the AI video analysis "
                "(summary, people, activities) if it has run.",
                {"event_id": {"type": "string"}},
                self._alert_details,
            ),
            Tool(
                "camera_outages", "Checking camera history",
                "When cameras were not live: feed lost, stopped, starting, disabled, or no "
                "record (server off). With per-camera totals and the local start hours, to "
                "spot outages at a fixed time each day.",
                {"camera_id": FILTERS["camera_id"], "since": FILTERS["since"],
                 "until": FILTERS["until"]},
                self._camera_outages,
            ),
            Tool(
                "list_cameras", "Checking cameras",
                "Configured cameras: id, name, enabled, current state, modules, boundaries.",
                {}, self._list_cameras,
            ),
            Tool(
                "product_help", "Reading the guide",
                "How to use this dashboard: every page, setting and AI feature.",
                {}, lambda _ctx, _args: ToolResult(product_help.guide()),
            ),
        ]
        if self.rag is not None:
            tools.append(Tool(
                "search_footage", "Searching footage",
                "Find moments in analysed alert clips by describing them (e.g. 'person "
                "without a helmet near the gate'). Returns matching frames/clip summaries "
                "with alert id, camera, time and caption. Embedding search: check the "
                "captions - it can rank 'with helmet' near 'without helmet'.",
                {"query": {"type": "string"}, "kind": FILTERS["kind"],
                 "camera_id": FILTERS["camera_id"], "since": FILTERS["since"],
                 "limit": {"type": "integer"}},
                self._search_footage, required=("query",),
            ))
        if self.inspector is not None and self.inspector.available:
            how = ("A specialist vision check boxes every person and reports, person by "
                   "person, clothing and whether a helmet/vest is ON them (yes / no / cannot "
                   "tell), marking the person the alert flagged. Use its report as the facts "
                   "about who wears what - do not contradict it.")
        else:
            how = "You see the image yourself."
        question = {"type": "string", "description": "what you want to know about the people"}
        tools += [
            Tool(
                "look_at_snapshot", "Looking at the snapshot",
                f"Look at the alert's snapshot (the moment of the alert). {how}",
                {"event_id": {"type": "string"}, "question": question},
                self._look_at_snapshot,
            ),
            Tool(
                "look_at_frames", "Looking at the video",
                "Look at a few frames spread across the alert's video clip (still images). "
                f"{how} Prefer ask_video for motion or what happens over time.",
                {"event_id": {"type": "string"}, "question": question,
                 "count": {"type": "integer", "description": "1-4 frames (default 3)"}},
                self._look_at_frames,
            ),
        ]
        if self.analyzer is not None:
            tools.append(Tool(
                "ask_video", "Watching the video",
                "Ask the video-understanding model one question about an alert's clip; it "
                "watches the whole clip, motion included. Good for what happens over time; "
                "unreliable for who wears what (it mixes people up) - check that yourself "
                "with look_at_snapshot / look_at_frames. Slower (~10-20 s).",
                {"event_id": {"type": "string"}, "question": {"type": "string"}},
                self._ask_video, required=("question",),
            ))
        if self.memory is not None:
            tools += [
                Tool(
                    "recall_memory", "Remembering",
                    "Search long-term memory (site facts, this user's notes, alert notes).",
                    {"query": {"type": "string"}}, self._recall, required=("query",),
                ),
                Tool(
                    "remember", "Saving to memory",
                    "Save a durable fact. scope: system (about the site, shared with "
                    "everyone), user (about this user), alert (about one alert - give "
                    "event_id). Only save what the user said or confirmed.",
                    {"text": {"type": "string"},
                     "scope": {"type": "string", "enum": ["system", "user", "alert"]},
                     "event_id": {"type": "string"}},
                    self._remember, required=("text", "scope"),
                ),
            ]
        return tools

    def _event(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        event_id = str(args.get("event_id") or ctx.event_id or "").strip()
        try:
            doc = self.events.find_one({"_id": ObjectId(event_id)})
        except (InvalidId, TypeError):
            doc = None
        if doc is None:
            raise ValueError(f"alert {event_id!r} not found")
        return doc

    def _filters(self, args: dict[str, Any]) -> dict[str, Any]:
        return {key: args.get(key) for key in FILTERS if args.get(key) not in (None, "")}

    def _list_alerts(self, _ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        limit = max(1, min(int(args.get("limit") or 20), MAX_LIST))
        query = event_query(text=args.get("text"), **self._filters(args))
        total = self.events.count_documents(query)
        docs = list(self.events.find(query).sort("ts", -1).limit(limit))
        ids = [str(d["_id"]) for d in docs]
        analyses = {
            a["_id"]: _visible_analysis(a) for a in self.analyses.find({"_id": {"$in": ids}})
        }
        return ToolResult({
            "total_matching": total,
            "shown": len(docs),
            "alerts": [_event_row(d, analyses.get(str(d["_id"]))) for d in docs],
        })

    def _alert_stats(self, _ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        group_by = args.get("group_by") or ["kind"]
        if isinstance(group_by, str):
            group_by = [group_by]
        return ToolResult(alert_stats(self.events, list(group_by), **self._filters(args)))

    def _alert_details(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        doc = self._event(ctx, args)
        analysis_doc = self.analyses.find_one({"_id": str(doc["_id"])})
        row = _event_row(doc)
        row.update({
            "subtype": doc.get("subtype"),
            "track_id": doc.get("track_id"),
            "identity_status": doc.get("identity_status"),
            "identity_confidence": doc.get("identity_confidence"),
            "email_sent": doc.get("delivered"),
            "clip_status": doc.get("clip_status"),
        })
        row["video_analysis"] = _visible_analysis(analysis_doc) or (
            {"status": analysis_doc.get("status"), "error": analysis_doc.get("error")}
            if analysis_doc else "not analysed"
        )
        return ToolResult(row)

    def _camera_outages(self, _ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if self.camera_log is None:
            return ToolResult({"error": "camera history is not being recorded"})
        until = parse_when(args.get("until")) or now_local()
        since = parse_when(args.get("since")) or until - timedelta(days=7)
        periods = self.camera_log.history(since, until, args.get("camera_id") or None)
        first = self.camera_log.rows.find_one({}, sort=[("start", 1)])
        totals: dict[str, dict[str, Any]] = {}
        for p in periods:
            entry = totals.setdefault(p["camera_id"], {"periods": 0, "minutes": 0.0,
                                                       "start_hours": Counter()})
            entry["periods"] += 1
            entry["minutes"] = round(entry["minutes"] + p["minutes"], 1)
            entry["start_hours"][f"{to_local(p['start']).hour:02d}:00"] += 1
        return ToolResult({
            "recorded_since": local_time(to_local(first["start"])) if first else None,
            "per_camera": {
                cam: {**t, "start_hours": dict(t["start_hours"].most_common())}
                for cam, t in totals.items()
            },
            "periods": [
                {**p, "start": local_time(to_local(p["start"])),
                 "end": local_time(to_local(p["end"]))}
                for p in periods[-100:]
            ],
        })

    def _list_cameras(self, _ctx: ToolContext, _args: dict[str, Any]) -> ToolResult:
        return ToolResult({"cameras": self.cameras()})

    def _search_footage(self, _ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        since = parse_when(args.get("since"))
        hits = self.rag.search(
            str(args.get("query") or ""),
            limit=max(1, min(int(args.get("limit") or 8), 20)),
            kind=args.get("kind") or None,
            camera_id=args.get("camera_id") or None,
            since=since.timestamp() if since else None,
        )
        return ToolResult({"results": [h.to_dict() for h in hits]})

    def _inspect(self, doc: dict[str, Any], images: list[tuple[str, bytes]],
                 question: str) -> ToolResult:
        """The specialist per-person report on `images` (see inspect_people.py)."""
        result = self.inspector.inspect(
            images, question, flagged_bbox=doc.get("bbox") if doc.get("kind") == "ppe" else None,
            about=f"Alert: {doc.get('message')} (type {doc.get('kind')}).",
        )
        return ToolResult({"event_id": str(doc["_id"]), "checked": result["images"],
                           "people_report": result["report"]})

    def _look_at_snapshot(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        doc = self._event(ctx, args)
        if not doc.get("snapshot_path"):
            return ToolResult({"error": "this alert has no snapshot"})
        jpeg = self.store.read(doc["snapshot_path"])
        if self.inspector is not None and self.inspector.available:
            return self._inspect(doc, [("Snapshot", jpeg)], str(args.get("question") or ""))
        return ToolResult({"event_id": str(doc["_id"]), "image": "snapshot attached"},
                          images=[jpeg])

    def _look_at_frames(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        doc = self._event(ctx, args)
        if not doc.get("clip_path"):
            return ToolResult({"error": "this alert has no saved video"})
        count = max(1, min(int(args.get("count") or 3), 4))
        inspecting = self.inspector is not None and self.inspector.available
        frames = extract_keyframes(self.store.read(doc["clip_path"]), count,
                                   INSPECT_FRAME_HEIGHT if inspecting else LOOK_FRAME_HEIGHT)
        if inspecting:
            # The flagged box belongs to the alert's moment, not to these other frames.
            return self._inspect({**doc, "bbox": None},
                                 [(f"Frame at {offset:.1f} s", jpeg) for offset, jpeg in frames],
                                 str(args.get("question") or ""))
        return ToolResult(
            {"event_id": str(doc["_id"]),
             "frames_at_seconds": [offset for offset, _jpeg in frames]},
            images=[jpeg for _offset, jpeg in frames],
        )

    def _ask_video(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        doc = self._event(ctx, args)
        if not doc.get("clip_path"):
            return ToolResult({"error": "this alert has no saved video"})
        event = {**doc, "id": str(doc["_id"])}
        return ToolResult({"event_id": event["id"],
                           "answer": self.analyzer.ask(event, str(args["question"]))})

    def _recall(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        found = self.memory.recall(str(args["query"]), ctx.user_id, ctx.event_id, limit=8)
        return ToolResult({"memories": [{"scope": m["scope"], "text": m["text"],
                                         "event_id": m.get("event_id")} for m in found]})

    def _remember(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        scope = str(args.get("scope") or "")
        event_id = args.get("event_id") or (ctx.event_id if scope == "alert" else None)
        saved = self.memory.add(str(args.get("text") or ""), scope, user_id=ctx.user_id,
                                event_id=event_id, source="chat")
        return ToolResult({"saved": saved["text"], "scope": saved["scope"]})


def cameras_from(registry: Any, zones: Any, manager: Any) -> Callable[[], list[dict[str, Any]]]:
    """The camera list the assistant sees: config plus live state, read-only."""

    def cameras() -> list[dict[str, Any]]:
        running = manager.all()
        out = []
        for camera in registry.list():
            pipeline = running.get(camera.id)
            out.append({
                "id": camera.id,
                "name": camera.name or camera.id,
                "enabled": camera.enabled,
                "state": pipeline.stats.feed_state if pipeline else "not running",
                "modules": sorted(camera.enabled_modules),
                "boundaries": [z.name or z.id for z in zones.zones(camera.id)],
            })
        return out

    return cameras
