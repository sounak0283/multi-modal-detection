"""MongoDB event store (Expansion Plan, Phase A.1).

Replaces the PostgreSQL/`pg8000` implementation. `pymongo` is synchronous, matching this
codebase's existing all-synchronous threading model - the `AlertBus` background thread
writes, FastAPI's sync `def` routes read, and neither assumes an event loop. `MongoClient`
pools connections internally, so the hand-rolled `LifoQueue` pool the Postgres version
needed does not carry over.

The public method contract is unchanged from the Postgres version on purpose
(`insert_event`, `recent_events`, `event_counts`, `zone_entry_counts`, `healthy`,
`purge_older_than`, `mark_delivered`) so `AlertBus` and the API routes needed almost no
changes for this migration. The one visible difference: an event id is now a MongoDB
ObjectId string, not an integer.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

log = logging.getLogger("perimeter.store")

EVENTS = "events"
PERSONS = "persons"

# How long a driver call waits for a reachable server. Long enough to ride out an Atlas
# primary election; short enough that a real outage does not pin API worker threads.
CONNECT_TIMEOUT_MS = 5_000


class Database:
    """MongoDB Atlas (or any MongoDB-compatible server) access for events and persons."""

    def __init__(self, url: str, db_name: str = "perimeter") -> None:
        self.url = url
        self.db_name = db_name
        self.client: MongoClient = MongoClient(url, serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS)
        self.db = self.client[db_name]
        log.info("connected to MongoDB database %r", db_name)

    def close(self) -> None:
        self.client.close()

    # -- schema (indexes only - Mongo has no schema to migrate) -------------

    def ensure_indexes(self) -> None:
        """Idempotent index creation. Safe to call on every boot."""
        events = self.db[EVENTS]
        events.create_index([("ts", DESCENDING)])
        events.create_index([("camera_id", ASCENDING), ("ts", DESCENDING)])
        events.create_index([("kind", ASCENDING), ("ts", DESCENDING)])
        events.create_index([("zone_id", ASCENDING), ("ts", DESCENDING)])
        events.create_index(
            [("delivered", ASCENDING)], partialFilterExpression={"delivered": False}
        )
        self.db[PERSONS].create_index("id", unique=True)
        log.info("indexes ensured")

    def available(self) -> bool:
        """Instant, no network I/O: whether the driver's background monitor currently
        knows a reachable server. Lets request paths fail fast during an outage instead
        of each waiting out the server-selection timeout."""
        return self.client.topology_description.has_readable_server()

    def healthy(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    # -- writes --------------------------------------------------------------

    def insert_event(self, event: dict[str, Any]) -> str:
        """Insert one alert and return its id (a MongoDB ObjectId, as a string).

        `ts` is stored as a timezone-aware UTC datetime, mirroring the Postgres
        implementation's TIMESTAMPTZ behaviour: an alert log whose timestamps shift
        meaning is useless as evidence.
        """
        ts = event.get("ts")
        if isinstance(ts, int | float):
            ts = datetime.fromtimestamp(ts, tz=UTC)
        elif isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        elif ts is None:
            ts = datetime.now(UTC)

        document = {
            "ts": ts,
            "camera_id": event["camera_id"],
            "kind": event["kind"],
            "subtype": event.get("subtype"),
            "zone_id": event.get("zone_id"),
            "zone_name": event.get("zone_name"),
            "track_id": event.get("track_id"),
            "message": event["message"],
            "severity": event.get("severity", "medium"),
            "confidence": event.get("confidence"),
            "bbox": _coerce_coords(event.get("bbox")),
            "foot_point": _coerce_coords(event.get("foot_point")),
            "person_id": event.get("person_id"),
            "identity_status": event.get("identity_status"),
            "identity_name": event.get("identity_name"),
            "identity_confidence": event.get("identity_confidence"),
            "snapshot_path": event.get("snapshot_path"),
            "clip_path": event.get("clip_path"),
            "delivered": False,
            "delivery_attempts": 0,
            "last_attempt_ts": None,
            "delivery_error": None,
            "created_at": datetime.now(UTC),
        }
        result = self.db[EVENTS].insert_one(document)
        return str(result.inserted_id)

    def mark_delivered(self, event_id: str, error: str | None = None) -> None:
        self.db[EVENTS].update_one(
            {"_id": ObjectId(event_id)},
            {
                "$set": {
                    "delivered": error is None,
                    "last_attempt_ts": datetime.now(UTC),
                    "delivery_error": error,
                },
                "$inc": {"delivery_attempts": 1},
            },
        )

    def update_evidence_paths(
        self, event_id: str, snapshot_path: str | None = None, clip_path: str | None = None
    ) -> None:
        """Called by `EvidenceWriter` once a clip/snapshot has actually been written -
        the columns exist since Phase A but nothing ever set them before Phase C."""
        updates: dict[str, Any] = {}
        if snapshot_path is not None:
            updates["snapshot_path"] = snapshot_path
        if clip_path is not None:
            updates["clip_path"] = clip_path
        if not updates:
            return
        self.db[EVENTS].update_one({"_id": ObjectId(event_id)}, {"$set": updates})

    # -- reads -----------------------------------------------------------------

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """A single event by id. Returns `None` (never raises) for a malformed id -
        this is reached from a user-supplied URL path parameter, so a clean 404 beats
        a 500."""
        try:
            object_id = ObjectId(event_id)
        except (InvalidId, TypeError):
            return None
        doc = self.db[EVENTS].find_one({"_id": object_id})
        return _doc_to_dict(doc) if doc else None

    def recent_events(
        self,
        limit: int = 50,
        kind: str | None = None,
        zone_id: str | None = None,
        camera_id: str | None = None,
        since: datetime | None = None,
        identity_status: str | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if kind:
            query["kind"] = kind
        if zone_id:
            query["zone_id"] = zone_id
        if camera_id:
            query["camera_id"] = camera_id
        if since:
            query["ts"] = {"$gte": since}
        if identity_status:
            query["identity_status"] = identity_status

        cursor = (
            self.db[EVENTS]
            .find(query)
            .sort("ts", DESCENDING)
            .limit(max(1, min(limit, 500)))
        )
        return [_doc_to_dict(doc) for doc in cursor]

    def event_counts(
        self, since: datetime | None = None, camera_id: str | None = None
    ) -> dict[str, int]:
        match: dict[str, Any] = {}
        if since:
            match["ts"] = {"$gte": since}
        if camera_id:
            match["camera_id"] = camera_id
        pipeline: list[dict[str, Any]] = []
        if match:
            pipeline.append({"$match": match})
        pipeline.append({"$group": {"_id": "$kind", "count": {"$sum": 1}}})
        return {doc["_id"]: doc["count"] for doc in self.db[EVENTS].aggregate(pipeline)}

    def zone_entry_counts(
        self, since: datetime | None = None, camera_id: str | None = None
    ) -> dict[str, int]:
        """People counted entering each zone - a cumulative complement to the boundary
        engine's live occupancy, which only knows who is inside *right now*.
        """
        match: dict[str, Any] = {"kind": "boundary", "subtype": "entry", "zone_id": {"$ne": None}}
        if since:
            match["ts"] = {"$gte": since}
        if camera_id:
            match["camera_id"] = camera_id
        pipeline = [
            {"$match": match},
            {"$group": {"_id": "$zone_id", "count": {"$sum": 1}}},
        ]
        return {doc["_id"]: doc["count"] for doc in self.db[EVENTS].aggregate(pipeline)}

    def purge_older_than(self, cutoff: datetime) -> int:
        """Retention. Expansion Plan §8 / PLAN.md §10.3 require a stated, enforced period."""
        result = self.db[EVENTS].delete_many({"ts": {"$lt": cutoff}})
        if result.deleted_count:
            log.info("purged %d event(s) older than %s", result.deleted_count, cutoff.isoformat())
        return int(result.deleted_count)

    # -- persons (Expansion Plan Phase F) ------------------------------------

    def insert_person(self, person: dict[str, Any]) -> str:
        """Enrol one person. `embeddings` is a list of up to 5 128-d SFace vectors, one
        per captured pose (front/left/right/up/down) - a delete purges all of them
        outright (see `delete_person`), never a soft delete of the biometric data
        itself (Expansion Plan §8)."""
        document = {
            "id": person["id"],
            "name": person["name"],
            "external_id": person.get("external_id", ""),
            "active": bool(person.get("active", True)),
            "embeddings": [[float(v) for v in e] for e in person.get("embeddings", [])],
            "created_at": datetime.now(UTC),
        }
        self.db[PERSONS].insert_one(document)
        return person["id"]

    def list_persons(self, active_only: bool = False) -> list[dict[str, Any]]:
        query = {"active": True} if active_only else {}
        return [_person_doc_to_dict(doc) for doc in self.db[PERSONS].find(query)]

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        doc = self.db[PERSONS].find_one({"id": person_id})
        return _person_doc_to_dict(doc) if doc else None

    def update_person(self, person_id: str, updates: dict[str, Any]) -> None:
        if not updates:
            return
        self.db[PERSONS].update_one({"id": person_id}, {"$set": updates})

    def delete_person(self, person_id: str) -> None:
        """Purges the stored embedding along with the record - there is no soft delete
        of biometric data (Expansion Plan §8)."""
        self.db[PERSONS].delete_one({"id": person_id})


def _coerce_coords(value: Any) -> list[float] | None:
    """Tolerate numpy scalars in a coordinate list, same as the Postgres version's `_json`."""
    if value is None:
        return None
    return json.loads(
        json.dumps(
            list(value) if isinstance(value, tuple) else value,
            default=lambda o: float(o) if hasattr(o, "__float__") else str(o),
        )
    )


def _person_doc_to_dict(doc: dict[str, Any]) -> dict[str, Any]:
    doc = dict(doc)
    doc.pop("_id", None)
    created = doc.get("created_at")
    if isinstance(created, datetime):
        doc["created_at"] = created.isoformat()
    return doc


def _doc_to_dict(doc: dict[str, Any]) -> dict[str, Any]:
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    ts = doc.get("ts")
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        doc["ts"] = ts.isoformat()
    return doc
