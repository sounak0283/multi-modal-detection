"""Boundary state machine (PLAN.md section 6.3).

Rev 1 of the plan contained two functional bugs here. Both are fixed, and both are
documented in place because the naive implementation reintroduces them.

**Bug 1 - a time-based cooldown swallows EXIT events.** Rev 1 guarded emission with
`now - last_event_ts > cooldown_s` at 30 s. A person entering at t=0 and leaving at t=10
had their EXIT suppressed, and because the guard also used `state_age == min_frames`
(true for exactly one update), the event never re-fired - it was lost permanently.
Entering and leaving inside 30 s is the *normal* case, so entry/exit pairing was broken
by default. The `emitted` flag replaces the timer: state transitions are inherently
self-limiting and the hysteresis already does the rate-limiting.

**Bug 2 - uninitialised tracks fire phantom ENTRYs.** Rev 1 never defined the state of a
new track. ByteTrack reassigns IDs after occlusion, which in a warehouse with racking
happens constantly. If a new track defaults to OUT, every person who appears already
inside a zone fires a false ENTRY - including the same person who just acquired a new ID
while standing still. Tracks are therefore born already settled at their observed
position, and only an observed *transition* emits.

The accepted trade-off: someone who crosses a boundary while fully occluded and
re-emerges with a new track ID produces no event. That is the right trade against
ID-churn alert floods, but it is a decision, not an oversight, and belongs in the
customer-facing limitations.

Cadence note: this runs at the person-detection rate (~6-8 Hz), not the camera frame
rate, so `min_frames: 4` is ~0.5-0.7 s of hysteresis (PLAN.md section 4).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from shapely.geometry import Polygon

from perimeter.boundary.geometry import FRAME_EDGE_TOLERANCE, Side, side_of_polyline
from perimeter.boundary.zones import Direction, EventKind, Severity, Zone, ZoneStore, ZoneType
from perimeter.track.tracker import TrackedDetections

log = logging.getLogger("perimeter.boundary.engine")

# State tokens. Polygon zones use IN/OUT; tripwires use the two sides. Both drive the
# same transition machine.
STATE_IN, STATE_OUT = "in", "out"

# Track states are dropped after this many updates without being seen, so a long-running
# process does not accumulate state for every person who ever walked past.
STALE_TRACK_UPDATES = 300
# Detection frames a track may go unseen and still count toward live occupancy - absorbs a
# single missed detection without letting departed tracks linger in the headcount.
OCCUPANCY_GRACE_UPDATES = 3
# Tracks first seen during this many updates after start-up are never treated as having
# just walked into view: they were already there when the camera came up (~1 s at the
# usual 6 Hz detection rate).
STARTUP_GRACE_UPDATES = 6
# A track inside an area zone that was last seen touching the frame edge and then goes
# unseen for this many updates (~1 s) has walked out of view: it gets an EXIT. The mirror
# of the edge-born ENTRY in `_advance` - with a zone covering most of the frame, nobody
# is ever observed "outside" it on camera, so without this no EXIT could fire.
LEFT_VIEW_AFTER_UPDATES = 6


@dataclass(frozen=True)
class BoundaryEvent:
    ts: float
    camera_id: str
    zone_id: str
    zone_name: str
    kind: EventKind
    track_id: int
    severity: Severity
    bbox: tuple[float, float, float, float]  # normalised 0-1
    foot_point: tuple[float, float]  # normalised 0-1

    def describe(self) -> str:
        verb = "entered" if self.kind is EventKind.ENTRY else "exited"
        return f"track #{self.track_id} {verb} {self.zone_name}"


@dataclass
class _ZoneTrackState:
    """Per (zone, track) state.

    Two states, not one. `confirmed` is the last state we actually settled on and
    reported; `candidate` is what the current observations say. An event fires only when
    a candidate survives the hysteresis AND differs from what was confirmed.

    A single state plus an `emitted` flag - which is what PLAN.md rev 2 section 6.3
    specified - is not enough. Sequence with min_frames=3: person outside (confirmed
    OUT), flickers inside for one frame (candidate IN, age 1, never confirmed), returns
    outside. With one state, the return to OUT looks like a fresh transition that has not
    been emitted, and fires a **spurious EXIT with no matching ENTRY**. Comparing against
    `confirmed` makes the round trip a no-op, which is what it actually was.
    """

    confirmed: str
    candidate: str
    age: int
    last_seen: int  # update counter, for stale-state cleanup
    # Last observed geometry (pixels), for the left-the-view EXIT - see
    # LEFT_VIEW_AFTER_UPDATES.
    last_box: tuple[float, float, float, float] | None = None
    last_point: tuple[float, float] | None = None
    at_edge: bool = False


def _touches_frame_edge(box: np.ndarray, width: int, height: int) -> bool:
    """True when an xyxy pixel box reaches within `FRAME_EDGE_TOLERANCE` of any side of
    the frame - i.e. part of the person is out of view."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    margin_x = width * FRAME_EDGE_TOLERANCE
    margin_y = height * FRAME_EDGE_TOLERANCE
    return (
        x1 <= margin_x or y1 <= margin_y or x2 >= width - margin_x or y2 >= height - margin_y
    )


@dataclass
class _CompiledZone:
    """Zone geometry projected into pixel space, cached across frames."""

    zone: Zone
    polygon: Polygon | None = None
    vertices: np.ndarray | None = None
    _cache: dict = field(default_factory=dict)


class BoundaryEngine:
    """Evaluates tracked people against zones and emits entry/exit events."""

    def __init__(
        self,
        store: ZoneStore,
        camera_id: str = "cam_01",
        default_min_frames: int = 4,
    ) -> None:
        self.store = store
        self.camera_id = camera_id
        self.default_min_frames = default_min_frames

        self._states: dict[tuple[str, int], _ZoneTrackState] = {}
        self._updates = 0
        self._compiled: list[_CompiledZone] = []
        self._compiled_key: tuple[int, int, int] | None = None

    # -- geometry cache ----------------------------------------------------

    def _compile(self, width: int, height: int) -> list[_CompiledZone]:
        key = (self.store.version, width, height)
        if key == self._compiled_key:
            return self._compiled

        compiled: list[_CompiledZone] = []
        for zone in self.store.zones(self.camera_id):
            pixels = zone.pixel_points(width, height)
            if zone.is_area:
                compiled.append(_CompiledZone(zone=zone, polygon=Polygon(pixels)))
            else:
                compiled.append(_CompiledZone(zone=zone, vertices=pixels))

        self._compiled, self._compiled_key = compiled, key
        # Zones may have been deleted or reshaped; stale per-zone state would otherwise
        # make a redrawn zone emit against a position observed under the old geometry.
        live_ids = {c.zone.id for c in compiled}
        self._states = {k: v for k, v in self._states.items() if k[0] in live_ids}
        return compiled

    # -- state evaluation --------------------------------------------------

    @staticmethod
    def _state_for(
        compiled: _CompiledZone, point: tuple[float, float], previous: str | None
    ) -> str:
        zone = compiled.zone
        if zone.is_area:
            assert compiled.polygon is not None
            from shapely.geometry import Point

            return STATE_IN if compiled.polygon.covers(Point(point)) else STATE_OUT

        assert compiled.vertices is not None
        side = side_of_polyline(point, compiled.vertices)
        if side is Side.ON:
            # Exactly collinear. Holding the previous state avoids a spurious transition
            # for someone standing precisely on the line.
            return previous if previous is not None else Side.A.value
        return side.value

    @staticmethod
    def _event_kind(zone: Zone, new_state: str) -> EventKind:
        """Which event a transition into `new_state` represents.

        Polygon: IN is an entry. Tripwire: crossing onto side B is an entry, matching the
        a_to_b direction convention.
        """
        if zone.is_area:
            return EventKind.ENTRY if new_state == STATE_IN else EventKind.EXIT
        return EventKind.ENTRY if new_state == Side.B.value else EventKind.EXIT

    @staticmethod
    def _direction_allows(zone: Zone, kind: EventKind) -> bool:
        if zone.type is not ZoneType.TRIPWIRE or zone.direction is Direction.BOTH:
            return True
        if zone.direction is Direction.A_TO_B:
            return kind is EventKind.ENTRY
        return kind is EventKind.EXIT

    # -- exclusion masks ---------------------------------------------------

    def suppressed_mask(
        self, foot_points_px: np.ndarray, width: int, height: int, klass: str = "person"
    ) -> np.ndarray:
        """True for detections whose foot point falls inside an active exclusion zone.

        The single highest-value feature per hour of work: the worst false-alarm sources
        are fixed in place, and letting the installer mask them out beats any amount of
        additional training data.
        """
        mask = np.zeros(len(foot_points_px), dtype=bool)
        if not len(foot_points_px):
            return mask

        from shapely.geometry import Point

        now = datetime.now()
        for compiled in self._compile(width, height):
            zone = compiled.zone
            if zone.type is not ZoneType.EXCLUSION or klass not in zone.applies_to:
                continue
            if not zone.schedule.is_active(now):
                continue
            assert compiled.polygon is not None
            for index, point in enumerate(foot_points_px):
                if not mask[index] and compiled.polygon.covers(Point(tuple(point))):
                    mask[index] = True
        return mask

    def confidence_delta(self, point_px: tuple[float, float], width: int, height: int) -> float:
        """Confidence adjustment for fire/smoke at a point, from any fire_roi zone.

        Negative means more sensitive: a missed fire in the server room costs more than a
        false alarm there does.
        """
        from shapely.geometry import Point

        now = datetime.now()
        delta = 0.0
        for compiled in self._compile(width, height):
            zone = compiled.zone
            if zone.type is not ZoneType.FIRE_ROI or not zone.schedule.is_active(now):
                continue
            assert compiled.polygon is not None
            if compiled.polygon.covers(Point(point_px)):
                delta = min(delta, zone.conf_delta)
        return delta

    def fire_roi_zone_at(
        self, point_px: tuple[float, float], width: int, height: int
    ) -> Zone | None:
        """The fire_roi zone (if any) containing `point_px`, so an alert can name a real
        place ("Fire detected in Server Room") rather than always falling back to the
        bare message. Same `_compile()`-iteration shape as `confidence_delta` above."""
        from shapely.geometry import Point

        now = datetime.now()
        for compiled in self._compile(width, height):
            zone = compiled.zone
            if zone.type is not ZoneType.FIRE_ROI or not zone.schedule.is_active(now):
                continue
            assert compiled.polygon is not None
            if compiled.polygon.covers(Point(point_px)):
                return zone
        return None

    def crowd_membership(
        self, foot_points_px: np.ndarray, width: int, height: int
    ) -> dict[str, np.ndarray]:
        """Zone id -> boolean mask of which foot points fall inside it, for every
        POLYGON zone that declares `crowd_threshold` (Expansion Plan Phase G). Scoped to
        POLYGON only, matching `zone_occupancy()`'s existing scoping - crowd formation is
        an "area people occupy" concept, not one exclusion/fire_roi/tripwire zones have.
        """
        result: dict[str, np.ndarray] = {}
        if not len(foot_points_px):
            return result

        from shapely.geometry import Point

        now = datetime.now()
        for compiled in self._compile(width, height):
            zone = compiled.zone
            if zone.type is not ZoneType.POLYGON or zone.crowd_threshold is None:
                continue
            if not zone.schedule.is_active(now):
                continue
            assert compiled.polygon is not None
            result[zone.id] = np.array(
                [compiled.polygon.covers(Point(tuple(p))) for p in foot_points_px],
                dtype=bool,
            )
        return result

    def ppe_membership(
        self, foot_points_px: np.ndarray, width: int, height: int
    ) -> dict[str, np.ndarray]:
        """Zone id -> boolean mask of which foot points fall inside it, for every
        POLYGON zone that declares `ppe_required` (Expansion Plan Phase H). Same shape
        as `crowd_membership` above - POLYGON-only, same "area people occupy" scoping.
        """
        result: dict[str, np.ndarray] = {}
        if not len(foot_points_px):
            return result

        from shapely.geometry import Point

        now = datetime.now()
        for compiled in self._compile(width, height):
            zone = compiled.zone
            if zone.type is not ZoneType.POLYGON or not zone.ppe_required:
                continue
            if not zone.schedule.is_active(now):
                continue
            assert compiled.polygon is not None
            result[zone.id] = np.array(
                [compiled.polygon.covers(Point(tuple(p))) for p in foot_points_px],
                dtype=bool,
            )
        return result

    def zone_by_id(self, zone_id: str) -> Zone | None:
        """Look up one of this camera's zones by id, for policy checks keyed off a
        `BoundaryEvent.zone_id` (e.g. the restricted-zone allow-list, Expansion Plan
        Phase F.1) that need the full `Zone`, not just its name."""
        for zone in self.store.zones(self.camera_id):
            if zone.id == zone_id:
                return zone
        return None

    # -- main entry point --------------------------------------------------

    def update(
        self,
        tracked: TrackedDetections,
        frame_size: tuple[int, int],
        now: float | None = None,
    ) -> list[BoundaryEvent]:
        """Advance by one DETECTION frame and return any events fired.

        Call once per person-detection frame, never on skip frames - the hysteresis is
        counted in calls, so an extra call shortens it.
        """
        width, height = frame_size
        self.store.refresh_in_background()  # never blocks inference on the database
        compiled_zones = self._compile(width, height)
        self._updates += 1

        timestamp = now if now is not None else time.time()
        wall_clock = datetime.fromtimestamp(timestamp)
        events: list[BoundaryEvent] = []

        if len(tracked):
            feet = tracked.foot_points(height)
            suppressed = self.suppressed_mask(feet, width, height, "person")
        else:
            feet = np.zeros((0, 2), np.float32)
            suppressed = np.zeros(0, dtype=bool)

        for compiled in compiled_zones:
            zone = compiled.zone
            if not zone.emits_events or "person" not in zone.detect:
                continue
            if not zone.schedule.is_active(wall_clock):
                continue

            min_frames = zone.min_frames or self.default_min_frames

            for index in range(len(tracked)):
                if suppressed[index]:
                    continue
                track_id = int(tracked.track_ids[index])
                event = self._advance(
                    compiled, track_id, tuple(feet[index]), min_frames, timestamp,
                    tracked.xyxy[index], width, height,
                )
                if event is not None:
                    events.append(event)

        events.extend(self._left_view_exits(compiled_zones, wall_clock, timestamp, width, height))
        self._evict_stale()
        for event in events:
            log.info("%s (severity=%s)", event.describe(), event.severity.value)
        return events

    def _advance(
        self,
        compiled: _CompiledZone,
        track_id: int,
        point: tuple[float, float],
        min_frames: int,
        timestamp: float,
        box: np.ndarray,
        width: int,
        height: int,
    ) -> BoundaryEvent | None:
        zone = compiled.zone
        key = (zone.id, track_id)
        existing = self._states.get(key)
        observed = self._state_for(compiled, point, existing.candidate if existing else None)

        if existing is None:
            if (
                zone.is_area
                and observed == STATE_IN
                and self._updates > STARTUP_GRACE_UPDATES
                and _touches_frame_edge(box, width, height)
            ):
                # Born inside an area zone at the edge of the frame: the person just
                # walked (or leaned) into view, and the zone reaches the edge they came
                # from, so the "outside" they crossed from is off-camera. Seed as OUT
                # with an IN candidate - the normal hysteresis then confirms the ENTRY.
                # Without this, a zone covering most of the frame can never fire one.
                self._states[key] = state = _ZoneTrackState(
                    confirmed=STATE_OUT, candidate=STATE_IN, age=1, last_seen=self._updates
                )
                self._remember_geometry(state, box, point, width, height)
                return None
            # Born already settled at the observed position: confirmed == candidate, so
            # the initial observation is NOT an event. This is what stops ID churn from
            # firing a phantom ENTRY for everyone already standing inside a zone - churn
            # happens mid-frame, where a person is occluded, not at the frame edge.
            self._states[key] = state = _ZoneTrackState(
                confirmed=observed, candidate=observed, age=min_frames, last_seen=self._updates
            )
            self._remember_geometry(state, box, point, width, height)
            return None

        existing.last_seen = self._updates
        self._remember_geometry(existing, box, point, width, height)
        if observed != existing.candidate:
            existing.candidate = observed
            existing.age = 1
        else:
            existing.age += 1

        # No time-based cooldown: that is what swallowed EXIT events in rev 1.
        # The `candidate != confirmed` test is what makes a brief unconfirmed excursion
        # and return a no-op rather than a spurious event.
        if existing.age < min_frames or existing.candidate == existing.confirmed:
            return None

        kind = self._event_kind(zone, existing.candidate)
        existing.confirmed = existing.candidate

        if kind not in zone.events or not self._direction_allows(zone, kind):
            return None

        return self._make_event(zone, kind, track_id, timestamp, box, point, width, height)

    @staticmethod
    def _remember_geometry(
        state: _ZoneTrackState, box: np.ndarray, point: tuple[float, float],
        width: int, height: int,
    ) -> None:
        state.last_box = tuple(float(v) for v in box[:4])
        state.last_point = (float(point[0]), float(point[1]))
        state.at_edge = _touches_frame_edge(box, width, height)

    def _left_view_exits(
        self, compiled_zones: list[_CompiledZone], wall_clock: datetime, timestamp: float,
        width: int, height: int,
    ) -> list[BoundaryEvent]:
        """EXITs for tracks that walked out of view while inside an area zone.

        Only tracks last seen at the frame edge qualify: a person who vanishes mid-frame
        was occluded or missed by the detector, and has not left.
        """
        events: list[BoundaryEvent] = []
        cutoff = self._updates - LEFT_VIEW_AFTER_UPDATES
        for compiled in compiled_zones:
            zone = compiled.zone
            if not zone.is_area or not zone.emits_events or "person" not in zone.detect:
                continue
            if not zone.schedule.is_active(wall_clock):
                continue
            for (zone_id, track_id), state in self._states.items():
                if zone_id != zone.id or state.confirmed != STATE_IN:
                    continue
                if not state.at_edge or state.last_seen > cutoff or state.last_box is None:
                    continue
                state.confirmed = state.candidate = STATE_OUT
                if EventKind.EXIT not in zone.events:
                    continue
                events.append(self._make_event(
                    zone, EventKind.EXIT, track_id, timestamp,
                    np.asarray(state.last_box), state.last_point, width, height,
                ))
        return events

    def _make_event(
        self, zone: Zone, kind: EventKind, track_id: int, timestamp: float,
        box: np.ndarray, point: tuple[float, float], width: int, height: int,
    ) -> BoundaryEvent:
        return BoundaryEvent(
            ts=timestamp,
            camera_id=self.camera_id,
            zone_id=zone.id,
            zone_name=zone.display_name,
            kind=kind,
            track_id=track_id,
            severity=zone.severity,
            bbox=(
                float(box[0]) / width,
                float(box[1]) / height,
                float(box[2]) / width,
                float(box[3]) / height,
            ),
            # float(), not just division: `point` comes from a numpy float32 array, and
            # np.float32 / int stays np.float32. json.dumps refuses numpy scalars, so
            # leaving them here made every database insert fail with "Object of type
            # float32 is not JSON serializable" - while detection carried on looking
            # perfectly healthy.
            foot_point=(float(point[0]) / width, float(point[1]) / height),
        )

    def _evict_stale(self) -> None:
        cutoff = self._updates - STALE_TRACK_UPDATES
        if cutoff <= 0:
            return
        stale = [k for k, v in self._states.items() if v.last_seen < cutoff]
        for key in stale:
            del self._states[key]

    # -- introspection, used by the dashboard ------------------------------

    def track_states(self) -> dict[str, dict[int, str]]:
        """Current IN/OUT (or side) per zone per track, for the live test view."""
        result: dict[str, dict[int, str]] = {}
        for (zone_id, track_id), state in self._states.items():
            result.setdefault(zone_id, {})[track_id] = state.confirmed
        return result

    def zone_occupancy(self) -> dict[str, int]:
        """Current headcount per polygon zone, from confirmed IN states.

        Scoped to `polygon` zones only: exclusion and fire ROI zones are not "areas
        people occupy" for this purpose (they modify other detections rather than
        representing a space), and a tripwire has no inside/outside to count. Every
        polygon zone is included even at zero, so the dashboard shows "0" for an empty
        zone rather than omitting it.
        """
        counts: dict[str, int] = {
            z.id: 0 for z in self.store.zones(self.camera_id) if z.type is ZoneType.POLYGON
        }
        # States outlive their track by STALE_TRACK_UPDATES so a briefly lost person keeps
        # their IN/OUT history - but a headcount "now" must only include people actually
        # still being tracked, or a departed track (or one re-identified under a new id)
        # is counted alongside the person for the better part of a minute.
        recent = self._updates - OCCUPANCY_GRACE_UPDATES
        for (zone_id, _track_id), state in self._states.items():
            if zone_id in counts and state.confirmed == STATE_IN and state.last_seen >= recent:
                counts[zone_id] += 1
        return counts
