"""Zone definitions and MongoDB persistence (Expansion Plan Phase A.3; PLAN.md sections 6.2-6.5).

Four zone types:

| type        | shape          | purpose                                          |
|-------------|----------------|--------------------------------------------------|
| `polygon`   | closed polygon | person inside/outside -> ENTRY / EXIT            |
| `tripwire`  | line/polyline  | directional crossing -> A->B / B->A              |
| `exclusion` | closed polygon | SUPPRESS detections inside this region           |
| `fire_roi`  | closed polygon | bias fire/smoke confidence in this region        |

The exclusion mask is the highest value-per-hour feature in the system. The worst
false-alarm sources are fixed in place - the welding bay, the beacon on the forklift
charger, the skylight that throws sunlight at 16:00 - and letting the installer draw a
polygon around each does more for the false-alarm rate than another twenty thousand
training images.

Storage: MongoDB, not a YAML file
----------------------------------
This used to be a single `zones.yaml` with a top-level `cameras: {cam_id: {zones: [...]}}`
mapping, hot-reloaded on file mtime change. Multi-camera (Expansion Plan Phase A) means
zones are now created and edited from the dashboard at runtime by any number of cameras,
which fits a database far better than a git-reviewed file - so zones live in the `zones`
MongoDB collection, one document per zone, tagged with `camera_id`. `zones.example.yaml`
stays in the repo as documentation of the zone shape, but the running app no longer reads
it.

`Zone` itself is unchanged in shape from the file-based version - it has no `camera_id`
field, because a single zone's definition has never depended on which camera it belongs
to. `ZoneStore` is what attaches/strips `camera_id` when talking to MongoDB, exactly as it
previously attached/stripped it when talking to the YAML file's `cameras` mapping.

Hot reload
----------
Installers iterate ten to twenty times on site. A process restart per edit turns a
twenty-minute commissioning into an afternoon, so the store re-reads periodically (a
throttled poll of the `zones` collection, replacing the old mtime check - MongoDB has no
mtime) and swaps atomically in memory. A document set that fails validation is REJECTED
and the previous good configuration is kept - a bad edit must never silently disarm a
live site.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from perimeter.boundary.geometry import (
    ValidationError,
    to_pixels,
    validate_polygon,
    validate_tripwire,
)
from perimeter.boundary.schedule import Schedule

log = logging.getLogger("perimeter.boundary.zones")

# How often reload() is willing to actually hit MongoDB when not forced. A Mongo round
# trip on every detection frame (6-8 Hz, times N cameras) would be wasteful; a few
# seconds of staleness on a dashboard edit is an acceptable trade the file-based mtime
# check made too (it just happened to be nearly free to check).
REFRESH_INTERVAL_S = 2.0


class ZoneType(Enum):
    POLYGON = "polygon"
    TRIPWIRE = "tripwire"
    EXCLUSION = "exclusion"
    FIRE_ROI = "fire_roi"


class Direction(Enum):
    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"
    BOTH = "both"


class EventKind(Enum):
    ENTRY = "entry"
    EXIT = "exit"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


DETECT_CLASSES = frozenset({"person", "fire", "smoke"})
# Mirrors detect/ppe.py:PPE_ITEMS without importing from detect/ - the same
# "zone-domain module declares its own copy, kept in sync by convention" precedent
# DETECT_CLASSES above already sets for person/fire/smoke (Expansion Plan Phase H).
PPE_ITEMS = frozenset({"helmet", "vest", "gloves", "shoes", "glasses"})
AREA_TYPES = {ZoneType.POLYGON, ZoneType.EXCLUSION, ZoneType.FIRE_ROI}


@dataclass(frozen=True)
class Zone:
    id: str
    type: ZoneType
    points: list[list[float]]  # normalised 0-1
    name: str = ""
    detect: frozenset[str] = frozenset({"person"})
    events: frozenset[EventKind] = frozenset({EventKind.ENTRY, EventKind.EXIT})
    direction: Direction = Direction.BOTH
    min_frames: int | None = None  # None -> use the global default
    schedule: Schedule = field(default_factory=Schedule)
    severity: Severity = Severity.MEDIUM
    applies_to: frozenset[str] = frozenset()  # exclusion / fire_roi target classes
    conf_delta: float = 0.0  # fire_roi only; negative = more sensitive

    # -- fields reserved for later phases (Expansion Plan §6) -----------------
    # Inert until their owning module ships: a restricted-area allow-list (Phase F.1),
    # crowd-formation thresholds (Phase G), required PPE items (Phase H) and the
    # phone-near-ear opt-in (Phase I). Declared now so the `zones` collection schema does
    # not need a second migration when those phases land.
    authorized_person_ids: frozenset[str] = frozenset()
    crowd_threshold: int | None = None
    crowd_min_frames: int | None = None
    ppe_required: frozenset[str] = frozenset()
    phone_restricted: bool = False

    @property
    def display_name(self) -> str:
        return self.name or self.id

    @property
    def is_area(self) -> bool:
        return self.type in AREA_TYPES

    @property
    def emits_events(self) -> bool:
        """Exclusion and fire ROI zones modify other detections; they never fire alerts."""
        return self.type in {ZoneType.POLYGON, ZoneType.TRIPWIRE}

    def pixel_points(self, width: int, height: int) -> np.ndarray:
        return to_pixels(self.points, width, height)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "points": [[round(x, 5), round(y, 5)] for x, y in self.points],
        }
        if self.name:
            payload["name"] = self.name
        if self.emits_events:
            payload["detect"] = sorted(self.detect)
            payload["events"] = sorted(e.value for e in self.events)
            payload["severity"] = self.severity.value
            if self.min_frames is not None:
                payload["min_frames"] = self.min_frames
        if self.type is ZoneType.TRIPWIRE:
            payload["direction"] = self.direction.value
        if self.applies_to:
            payload["applies_to"] = sorted(self.applies_to)
        if self.type is ZoneType.FIRE_ROI and self.conf_delta:
            payload["conf_delta"] = self.conf_delta
        if self.authorized_person_ids:
            payload["authorized_person_ids"] = sorted(self.authorized_person_ids)
        if self.crowd_threshold is not None:
            payload["crowd_threshold"] = self.crowd_threshold
        if self.crowd_min_frames is not None:
            payload["crowd_min_frames"] = self.crowd_min_frames
        if self.ppe_required:
            payload["ppe_required"] = sorted(self.ppe_required)
        if self.phone_restricted:
            payload["phone_restricted"] = self.phone_restricted
        schedule = self.schedule.to_dict()
        if schedule is not None:
            payload["schedule"] = schedule
        return payload


def _parse_enum(enum_cls, value: Any, field_name: str, zone_id: str):
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in enum_cls)
        raise ValueError(f"zone {zone_id!r}: invalid {field_name} {value!r} (expected: {valid})")\
            from exc


def zone_from_dict(data: dict[str, Any]) -> Zone:
    """Parse and validate one zone. Raises ValueError with an actionable message."""
    if not isinstance(data, dict):
        raise ValueError("each zone must be a mapping")

    zone_id = str(data.get("id", "")).strip()
    if not zone_id:
        raise ValueError("zone is missing 'id'")

    zone_type = _parse_enum(ZoneType, data.get("type", "polygon"), "type", zone_id)
    points = data.get("points")

    errors: list[ValidationError] = (
        validate_polygon(points) if zone_type in AREA_TYPES else validate_tripwire(points)
    )
    if errors:
        detail = "; ".join(f"{e.field}: {e.message}" for e in errors)
        raise ValueError(f"zone {zone_id!r}: {detail}")

    detect = frozenset(str(c).lower() for c in data.get("detect", ["person"]))
    unknown = detect - DETECT_CLASSES
    if unknown:
        raise ValueError(f"zone {zone_id!r}: unknown detect class(es) {sorted(unknown)}")

    events = frozenset(
        _parse_enum(EventKind, e, "event", zone_id) for e in data.get("events", ["entry", "exit"])
    )

    min_frames = data.get("min_frames")
    if min_frames is not None:
        min_frames = int(min_frames)
        if min_frames < 1:
            raise ValueError(f"zone {zone_id!r}: min_frames must be >= 1")

    applies_to = frozenset(str(c).lower() for c in data.get("applies_to", []))
    unknown_applies = applies_to - DETECT_CLASSES
    if unknown_applies:
        raise ValueError(
            f"zone {zone_id!r}: unknown applies_to class(es) {sorted(unknown_applies)}"
        )

    if zone_type in {ZoneType.EXCLUSION, ZoneType.FIRE_ROI} and not applies_to:
        raise ValueError(
            f"zone {zone_id!r}: {zone_type.value} zones need 'applies_to' "
            f"(which classes they suppress or bias)"
        )

    try:
        schedule = Schedule.from_dict(data.get("schedule"))
    except ValueError as exc:
        raise ValueError(f"zone {zone_id!r}: {exc}") from exc

    crowd_threshold = data.get("crowd_threshold")
    if crowd_threshold is not None:
        crowd_threshold = int(crowd_threshold)
        if crowd_threshold < 1:
            raise ValueError(f"zone {zone_id!r}: crowd_threshold must be >= 1")

    crowd_min_frames = data.get("crowd_min_frames")
    if crowd_min_frames is not None:
        crowd_min_frames = int(crowd_min_frames)
        if crowd_min_frames < 1:
            raise ValueError(f"zone {zone_id!r}: crowd_min_frames must be >= 1")

    ppe_required = frozenset(str(p).lower() for p in data.get("ppe_required", []))
    unknown_ppe = ppe_required - PPE_ITEMS
    if unknown_ppe:
        raise ValueError(f"zone {zone_id!r}: unknown ppe_required item(s) {sorted(unknown_ppe)}")

    return Zone(
        id=zone_id,
        type=zone_type,
        points=[[float(x), float(y)] for x, y in points],
        name=str(data.get("name", "")),
        detect=detect,
        events=events,
        direction=_parse_enum(Direction, data.get("direction", "both"), "direction", zone_id),
        min_frames=min_frames,
        schedule=schedule,
        severity=_parse_enum(Severity, data.get("severity", "medium"), "severity", zone_id),
        applies_to=applies_to,
        conf_delta=float(data.get("conf_delta", 0.0)),
        authorized_person_ids=frozenset(
            str(p) for p in data.get("authorized_person_ids", [])
        ),
        crowd_threshold=crowd_threshold,
        crowd_min_frames=crowd_min_frames,
        ppe_required=ppe_required,
        phone_restricted=bool(data.get("phone_restricted", False)),
    )


class ZoneStore:
    """Thread-safe zone configuration, MongoDB-backed with a throttled poll for hot reload."""

    def __init__(
        self,
        collection: Collection,
        camera_id: str = "cam_01",
        refresh_interval: float = REFRESH_INTERVAL_S,
    ) -> None:
        self._collection = collection
        self._collection.create_index("camera_id")
        self.camera_id = camera_id
        self._refresh_interval = refresh_interval
        self._lock = threading.Lock()
        self._zones_by_camera: dict[str, list[Zone]] = {}
        self._last_error: str | None = None
        # Bumped on every successful load or save. The engine caches compiled shapely
        # geometry against this, so a reload invalidates the cache without the engine
        # having to diff zone contents on every frame.
        self._version = 0
        self._last_poll = 0.0
        self._refresher: threading.Thread | None = None
        self.reload(force=True)

    # -- reading -----------------------------------------------------------

    def zones(self, camera_id: str | None = None) -> list[Zone]:
        with self._lock:
            return list(self._zones_by_camera.get(camera_id or self.camera_id, []))

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def reload(self, force: bool = False) -> bool:
        """Re-read from MongoDB, throttled to at most once per REFRESH_INTERVAL_S.

        A document set that fails validation is rejected and the previous good
        configuration is kept: a bad edit mid-save must never silently disarm a live site.
        """
        now = time.monotonic()
        if not force and now - self._last_poll < self._refresh_interval:
            return False
        self._last_poll = now

        try:
            parsed: dict[str, list[Zone]] = {}
            for doc in self._collection.find({}):
                camera_id = str(doc.pop("camera_id"))
                doc.pop("_id", None)
                parsed.setdefault(camera_id, []).append(zone_from_dict(doc))
        except (ValueError, KeyError) as exc:
            with self._lock:
                self._last_error = str(exc)
            log.error("zones collection rejected, keeping previous config: %s", exc)
            return False
        except PyMongoError as exc:
            # Unreachable database: keep enforcing the last good zones. A storage outage
            # must never disarm a live site.
            with self._lock:
                self._last_error = f"zone storage unavailable: {exc.__class__.__name__}"
            log.warning("could not reload zones, keeping previous config: %s", exc)
            return False

        for camera_id, zones in parsed.items():
            ids = [z.id for z in zones]
            duplicates = {i for i in ids if ids.count(i) > 1}
            if duplicates:
                with self._lock:
                    self._last_error = (
                        f"camera {camera_id!r}: duplicate zone id(s) {sorted(duplicates)}"
                    )
                log.error("zones collection rejected: %s", self._last_error)
                return False

        with self._lock:
            self._zones_by_camera = parsed
            self._last_error = None
            self._version += 1

        total = sum(len(z) for z in parsed.values())
        log.info("zones loaded: %d zone(s) across %d camera(s)", total, len(parsed))
        return True

    def refresh_in_background(self) -> None:
        """Hot-reload without blocking the caller - for the per-camera inference thread,
        which must never wait on the database (a MongoDB outage blocks each query for the
        full server-selection timeout). With `refresh_interval <= 0` it reloads
        synchronously, so a test can edit the collection and see it on the next call."""
        if self._refresh_interval <= 0:
            self.reload()
            return
        with self._lock:
            due = time.monotonic() - self._last_poll >= self._refresh_interval
            busy = self._refresher is not None and self._refresher.is_alive()
            if not due or busy:
                return
            self._refresher = threading.Thread(
                target=self.reload, name="zone-refresh", daemon=True
            )
            self._refresher.start()

    # -- writing -----------------------------------------------------------

    def save(self, zones: list[Zone], camera_id: str | None = None) -> None:
        """Replace one camera's zones in MongoDB.

        Not wrapped in a multi-document transaction: Atlas supports them, but the
        replace-many-documents-for-one-camera operation is scoped to a single camera and
        a torn write here (delete succeeds, insert partially fails) is recoverable by
        re-saving from the dashboard, which already holds the full desired zone list in
        memory - unlike a torn file write, it cannot leave the collection unparseable.
        """
        target = camera_id or self.camera_id
        ids = [z.id for z in zones]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate zone id(s): {sorted(duplicates)}")

        # Never write something we cannot read back.
        docs = [z.to_dict() | {"camera_id": target} for z in zones]
        for doc in docs:
            check = dict(doc)
            check.pop("camera_id")
            zone_from_dict(check)

        self._collection.delete_many({"camera_id": target})
        if docs:
            self._collection.insert_many(docs)

        with self._lock:
            merged = dict(self._zones_by_camera)
            merged[target] = list(zones)
            self._zones_by_camera = merged
            self._last_error = None
            self._version += 1
        log.info("zones saved: %d zone(s) for %s", len(zones), target)
