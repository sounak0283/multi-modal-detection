"""Alert queries shared by the chat tools, plain-English search and the daily digest.

Times people type or the model writes ("2026-10-27T22:00") with no offset are this
server's local time - the same clock alert emails are written in (timefmt.py). Stored
times stay UTC. Grouping by day/hour is done here in Python, in local time, so "every day
at 2 am" means 2 am on site, not in UTC.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Any

from perimeter.testvideo.sessions import TEST_PREFIX

GROUP_FIELDS = ("kind", "subtype", "severity", "camera", "zone", "day", "hour", "weekday")
SOURCES = ("all", "real", "test")
SEVERITIES = ("low", "medium", "high", "critical")  # every alert carries one of these
STATS_SCAN_LIMIT = 20_000
MAX_FILLED_DAYS = 400
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def local_tz():
    return datetime.now().astimezone().tzinfo


def now_local() -> datetime:
    return datetime.now(local_tz())


def parse_when(value: Any) -> datetime | None:
    """An ISO date/time as an aware datetime; no offset means server local time."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"not an ISO date/time: {value!r}") from exc
    return moment if moment.tzinfo else moment.replace(tzinfo=local_tz())


def aware(value: Any) -> datetime | None:
    """A stored (UTC, possibly naive) datetime as aware UTC."""
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def to_local(value: Any) -> datetime | None:
    moment = aware(value)
    return moment.astimezone(local_tz()) if moment else None


def local_day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=local_tz())
    return start, start + timedelta(days=1)


def is_test_video(event: dict[str, Any]) -> bool:
    return str(event.get("message") or "").startswith(TEST_PREFIX)


def event_query(
    kind: str | None = None,
    camera_id: str | None = None,
    zone: str | None = None,
    since: Any = None,
    until: Any = None,
    source: str = "all",
    text: str | None = None,
    severity: Any = None,
) -> dict[str, Any]:
    """A MongoDB filter on the events collection. `severity` is one level or a list of
    levels (e.g. ["high", "critical"])."""
    query: dict[str, Any] = {}
    conditions: list[dict[str, Any]] = []
    if kind:
        query["kind"] = str(kind).strip().lower()
    if severity:
        levels = [severity] if isinstance(severity, str) else list(severity)
        levels = [str(level).strip().lower() for level in levels if str(level).strip()]
        unknown = [level for level in levels if level not in SEVERITIES]
        if unknown:
            raise ValueError(f"severity must be among {SEVERITIES}, got {unknown}")
        if levels:
            query["severity"] = {"$in": levels}
    if camera_id:
        query["camera_id"] = str(camera_id).strip()
    if zone:
        pattern = re.compile(re.escape(str(zone).strip()), re.IGNORECASE)
        conditions.append({"$or": [{"zone_name": pattern}, {"zone_id": pattern}]})
    start, end = parse_when(since), parse_when(until)
    if start or end:
        query["ts"] = {}
        if start:
            query["ts"]["$gte"] = start.astimezone(UTC)
        if end:
            query["ts"]["$lt"] = end.astimezone(UTC)
    source = (source or "all").lower()
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    test = re.compile("^" + re.escape(TEST_PREFIX))
    if source == "real":
        conditions.append({"message": {"$not": test}})
    elif source == "test":
        conditions.append({"message": test})
    if text:
        conditions.append({"message": re.compile(re.escape(str(text).strip()), re.IGNORECASE)})
    if conditions:
        query["$and"] = conditions
    return query


def in_hours(moment: datetime | None, hour_from: int | None, hour_to: int | None) -> bool:
    """Whether a local time falls in [hour_from, hour_to), wrapping midnight (22 -> 6)."""
    if hour_from is None and hour_to is None:
        return True
    if moment is None:
        return False
    hour = moment.hour
    start = 0 if hour_from is None else int(hour_from) % 24
    end = 24 if hour_to is None else int(hour_to) % 24 or 24
    return start <= hour < end if start < end else hour >= start or hour < end


def group_key(event: dict[str, Any], field: str) -> str:
    moment = to_local(event.get("ts"))
    if field == "kind":
        return str(event.get("kind") or "?")
    if field == "subtype":
        return str(event.get("subtype") or "-")
    if field == "severity":
        return str(event.get("severity") or "-")
    if field == "camera":
        return str(event.get("camera_id") or "?")
    if field == "zone":
        return str(event.get("zone_name") or event.get("zone_id") or "-")
    if moment is None:
        return "?"
    if field == "day":
        return moment.strftime("%Y-%m-%d")
    if field == "hour":
        return f"{moment.hour:02d}:00"
    if field == "weekday":
        return WEEKDAYS[moment.weekday()]
    raise ValueError(f"cannot group by {field!r}; use one of {GROUP_FIELDS}")


def alert_stats(
    events_collection: Any,
    group_by: list[str],
    since: Any = None,
    until: Any = None,
    **filters: Any,
) -> dict[str, Any]:
    """Alert counts grouped by up to two fields. Grouping by day fills in days with no
    alerts, so "every other day" patterns show as zeros between counts."""
    fields = [f for f in group_by if f][:2] or ["kind"]
    for field in fields:
        if field not in GROUP_FIELDS:
            raise ValueError(f"cannot group by {field!r}; use one of {GROUP_FIELDS}")
    query = event_query(since=since, until=until, **filters)
    cursor = events_collection.find(
        query, {"ts": 1, "kind": 1, "subtype": 1, "severity": 1, "camera_id": 1,
                "zone_name": 1, "zone_id": 1, "message": 1},
    ).sort("ts", 1).limit(STATS_SCAN_LIMIT)
    events = list(cursor)
    counts = Counter(tuple(group_key(e, f) for f in fields) for e in events)
    if fields == ["day"] and events:
        first = (parse_when(since) or to_local(events[0]["ts"])).astimezone(local_tz()).date()
        end = parse_when(until)
        if end:
            last = (end - timedelta(microseconds=1)).astimezone(local_tz()).date()
        else:
            last = now_local().date()
        day = first
        while day <= last and len(counts) <= MAX_FILLED_DAYS:
            counts.setdefault((day.isoformat(),), 0)
            day += timedelta(days=1)
    rows = [
        {**dict(zip(fields, key, strict=True)), "count": count}
        for key, count in sorted(counts.items())
    ]
    return {
        "group_by": fields,
        "total": len(events),
        "truncated": len(events) >= STATS_SCAN_LIMIT,
        "rows": rows,
    }
