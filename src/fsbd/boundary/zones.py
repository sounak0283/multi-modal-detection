"""Zone definitions, YAML persistence and hot-reload (PLAN.md sections 6.2-6.5).

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

Hot reload
----------
Installers iterate ten to twenty times on site. A process restart per edit turns a
twenty-minute commissioning into an afternoon, so the store re-reads on mtime change and
swaps atomically. A file that fails validation is REJECTED and the previous good
configuration is kept - a syntax error must never disarm a live site.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fsbd.boundary.geometry import (
    ValidationError,
    to_pixels,
    validate_polygon,
    validate_tripwire,
)
from fsbd.boundary.schedule import Schedule

log = logging.getLogger("fsbd.boundary.zones")


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


DETECT_CLASSES = frozenset({"person", "fire", "smoke"})
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
    )


def parse_zones(data: dict[str, Any] | None) -> dict[str, list[Zone]]:
    """Parse the whole document. Returns {camera_id: [Zone, ...]}."""
    if not data:
        return {}

    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("zones file must have a top-level 'cameras' mapping")

    result: dict[str, list[Zone]] = {}
    for camera_id, camera_data in cameras.items():
        raw_zones = (camera_data or {}).get("zones") or []
        zones = [zone_from_dict(z) for z in raw_zones]

        seen: set[str] = set()
        for zone in zones:
            if zone.id in seen:
                raise ValueError(f"camera {camera_id!r}: duplicate zone id {zone.id!r}")
            seen.add(zone.id)
        result[str(camera_id)] = zones
    return result


def dump_zones(zones_by_camera: dict[str, list[Zone]]) -> str:
    payload = {
        "cameras": {
            camera_id: {"zones": [z.to_dict() for z in zones]}
            for camera_id, zones in zones_by_camera.items()
        }
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


class ZoneStore:
    """Thread-safe zone configuration with mtime-based hot reload."""

    def __init__(self, path: Path | str, camera_id: str = "cam_01") -> None:
        self.path = Path(path)
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._zones_by_camera: dict[str, list[Zone]] = {}
        self._mtime: float | None = None
        self._last_error: str | None = None
        # Bumped on every successful load or save. The engine caches compiled shapely
        # geometry against this, so a hot reload invalidates the cache without the
        # engine having to diff zone contents on every frame.
        self._version = 0
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
        """Re-read if the file changed. Returns True if the configuration was replaced.

        A file that fails validation is rejected and the previous good configuration is
        kept: a syntax error mid-edit must never silently disarm a live site.
        """
        if not self.path.is_file():
            with self._lock:
                if force:
                    self._zones_by_camera = {}
                    self._mtime = None
            return False

        mtime = self.path.stat().st_mtime
        if not force and mtime == self._mtime:
            return False

        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
            parsed = parse_zones(raw)
        except (yaml.YAMLError, ValueError) as exc:
            with self._lock:
                self._last_error = str(exc)
                # Advance mtime anyway so a broken file is not re-parsed every tick.
                self._mtime = mtime
            log.error("zones file rejected, keeping previous config: %s", exc)
            return False

        with self._lock:
            self._zones_by_camera = parsed
            self._mtime = mtime
            self._last_error = None
            self._version += 1

        total = sum(len(z) for z in parsed.values())
        log.info("zones loaded: %d zone(s) across %d camera(s)", total, len(parsed))
        return True

    # -- writing -----------------------------------------------------------

    def save(self, zones: list[Zone], camera_id: str | None = None) -> None:
        """Replace one camera's zones and write the file atomically.

        Atomic because the engine may reload at any moment; a half-written file would
        be read as a validation failure and, worse, could briefly present as an empty
        configuration.
        """
        target = camera_id or self.camera_id
        with self._lock:
            merged = dict(self._zones_by_camera)
            merged[target] = list(zones)

        text = dump_zones(merged)
        parse_zones(yaml.safe_load(text))  # never write something we cannot read back

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(self.path)

        with self._lock:
            self._zones_by_camera = merged
            self._mtime = self.path.stat().st_mtime
            self._last_error = None
            self._version += 1
        log.info("zones saved: %d zone(s) for %s", len(zones), target)
