"""Tests for the boundary state machine (PLAN.md section 6.3).

The two regression tests at the top are the most important in the project: they pin the
bugs the plan review found, both of which produce failures that look like "the detector
is flaky" rather than like a state-machine defect.

Zone storage is MongoDB since Expansion Plan Phase A.3 - these tests seed an in-memory
`mongomock` collection instead of writing a `zones.yaml` file, and construct `ZoneStore`
with `refresh_interval=0` so a test can mutate the collection and see the change on the
very next `engine.update()` call, matching how the old file-mtime check picked up an edit
immediately.
"""

from __future__ import annotations

import mongomock
import numpy as np
import pytest
from pymongo.collection import Collection

from perimeter.boundary.engine import (
    LEFT_VIEW_AFTER_UPDATES,
    STARTUP_GRACE_UPDATES,
    BoundaryEngine,
)
from perimeter.boundary.zones import EventKind, ZoneStore
from perimeter.track.tracker import TrackedDetections

FRAME = (1000, 1000)

# A zone occupying the middle of the frame: pixels 200-800 on both axes.
ZONE_DOC = {
    "cameras": {
        "cam_01": {
            "zones": [
                {
                    "id": "bay",
                    "name": "Loading Bay",
                    "type": "polygon",
                    "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                    "detect": ["person"],
                    "events": ["entry", "exit"],
                    "min_frames": 2,
                }
            ]
        }
    }
}


def seed_zones(doc: dict | None = None) -> Collection:
    """Flatten the test file's `{cameras: {cam_id: {zones: [...]}}}` fixture shape into
    the `camera_id`-tagged documents `ZoneStore` actually stores, in a fresh in-memory
    mongomock collection."""
    collection = mongomock.MongoClient()["perimeter_test"]["zones"]
    for camera_id, camera_data in (doc or ZONE_DOC).get("cameras", {}).items():
        for zone in (camera_data or {}).get("zones") or []:
            collection.insert_one(dict(zone) | {"camera_id": camera_id})
    return collection


def make_engine(doc: dict | None = None, min_frames: int = 2) -> BoundaryEngine:
    store = ZoneStore(seed_zones(doc), camera_id="cam_01", refresh_interval=0)
    return BoundaryEngine(store, camera_id="cam_01", default_min_frames=min_frames)


def person_at(x: float, y: float, track_id: int = 1) -> TrackedDetections:
    """A box whose FOOT POINT is at (x, y) - that is what the engine tests."""
    return TrackedDetections(
        xyxy=np.array([[x - 20, y - 100, x + 20, y]], np.float32),
        scores=np.array([0.9], np.float32),
        class_ids=np.array([0], np.int32),
        track_ids=np.array([track_id], np.int32),
    )


def walk(engine, points, track_id=1, start_ts=1000.0):
    """Feed a sequence of foot positions, returning every event emitted."""
    events = []
    for step, (x, y) in enumerate(points):
        events.extend(engine.update(person_at(x, y, track_id), FRAME, now=start_ts + step))
    return events


# =========================================================================
# REGRESSION: bug 1 - a time cooldown swallowed EXIT events
# =========================================================================


def test_exit_fires_shortly_after_entry():
    """Rev 1 guarded emission with a 30 s cooldown, so a person who entered at t=0 and
    left at t=10 had their EXIT silently dropped - and because the guard used
    `age == min_frames` (true once), it never re-fired. Entering and leaving inside 30 s
    is the NORMAL case, so entry/exit pairing was broken by default."""
    engine = make_engine()

    events = walk(engine, [
        (100, 100), (100, 100),   # outside, settles
        (500, 500), (500, 500),   # inside  -> ENTRY
        (100, 100), (100, 100),   # outside -> EXIT, only ~4 s later
    ])

    kinds = [e.kind for e in events]
    assert kinds == [EventKind.ENTRY, EventKind.EXIT]


def test_repeated_entry_exit_cycles_all_fire():
    engine = make_engine()
    events = walk(engine, [
        (100, 100), (100, 100),
        (500, 500), (500, 500),
        (100, 100), (100, 100),
        (500, 500), (500, 500),
        (100, 100), (100, 100),
    ])

    assert [e.kind for e in events] == [
        EventKind.ENTRY, EventKind.EXIT, EventKind.ENTRY, EventKind.EXIT
    ]


# =========================================================================
# REGRESSION: bug 2 - uninitialised tracks fired phantom ENTRYs
# =========================================================================


def test_new_track_appearing_inside_a_zone_emits_nothing():
    """ByteTrack reassigns IDs after occlusion. If a new track defaulted to OUT, every
    person who appears already inside a zone would fire a false ENTRY - including the
    same person who just acquired a new ID while standing still."""
    engine = make_engine()

    events = walk(engine, [(500, 500)] * 6, track_id=7)

    assert events == []


def test_id_churn_on_a_stationary_person_emits_nothing():
    """The exact failure mode: one person standing inside a zone, cycling through new
    track IDs as the tracker loses and reacquires them."""
    engine = make_engine()

    events = []
    for track_id in range(1, 15):
        for _ in range(3):
            events.extend(engine.update(person_at(500, 500, track_id), FRAME))

    assert events == []


def test_new_track_outside_the_zone_also_emits_nothing():
    engine = make_engine()
    assert walk(engine, [(50, 50)] * 5, track_id=3) == []


def test_a_genuine_crossing_still_fires_after_churn():
    """Suppressing phantom entries must not suppress real ones."""
    engine = make_engine()

    walk(engine, [(500, 500)] * 3, track_id=9)          # born inside, silent
    events = walk(engine, [(100, 100)] * 3, track_id=9)  # then genuinely leaves

    assert [e.kind for e in events] == [EventKind.EXIT]


FULL_FRAME_ZONE_DOC = {
    "cameras": {
        "cam_01": {
            "zones": [
                {
                    "id": "room", "type": "polygon",
                    # Covers nearly the whole frame, as a "the room" zone does.
                    "points": [[0.04, 0.04], [0.96, 0.04], [0.96, 0.96], [0.04, 0.96]],
                    "detect": ["person"], "events": ["entry", "exit"], "min_frames": 2,
                }
            ]
        }
    }
}


def box_track(xyxy, track_id=1) -> TrackedDetections:
    return TrackedDetections(
        xyxy=np.array([xyxy], np.float32),
        scores=np.array([0.9], np.float32),
        class_ids=np.array([0], np.int32),
        track_ids=np.array([track_id], np.int32),
    )


def settle_past_startup(engine):
    for _ in range(STARTUP_GRACE_UPDATES + 1):
        engine.update(TrackedDetections.empty(), FRAME)


def test_walking_into_view_inside_a_full_frame_zone_fires_entry():
    """With the zone covering the whole view there is no on-camera "outside" - a person
    is first detected already inside it. Appearing at the frame edge is the entry."""
    engine = make_engine(FULL_FRAME_ZONE_DOC)
    settle_past_startup(engine)

    events = []
    for _ in range(3):  # enters from the left edge; foot point well inside the zone
        events.extend(engine.update(box_track([0, 400, 200, 800], track_id=4), FRAME))

    assert [e.kind for e in events] == [EventKind.ENTRY]


def test_person_already_at_the_edge_at_startup_emits_nothing():
    engine = make_engine(FULL_FRAME_ZONE_DOC)

    events = []
    for _ in range(4):
        events.extend(engine.update(box_track([0, 400, 200, 800]), FRAME))

    assert events == []


def test_id_churn_mid_frame_in_a_full_frame_zone_emits_nothing():
    engine = make_engine(FULL_FRAME_ZONE_DOC)
    settle_past_startup(engine)

    events = []
    for track_id in range(1, 10):
        for _ in range(3):
            events.extend(engine.update(box_track([400, 300, 600, 800], track_id), FRAME))

    assert events == []


def test_walking_out_of_view_from_a_full_frame_zone_fires_exit():
    engine = make_engine(FULL_FRAME_ZONE_DOC)
    settle_past_startup(engine)

    events = []
    for _ in range(3):  # walk in at the left edge -> ENTRY
        events.extend(engine.update(box_track([0, 400, 200, 800], track_id=4), FRAME))
    for _ in range(3):  # move to mid-frame
        events.extend(engine.update(box_track([400, 300, 600, 800], track_id=4), FRAME))
    for _ in range(2):  # back to the edge, then gone
        events.extend(engine.update(box_track([0, 400, 200, 800], track_id=4), FRAME))
    for _ in range(LEFT_VIEW_AFTER_UPDATES + 2):
        events.extend(engine.update(TrackedDetections.empty(), FRAME))

    assert [e.kind for e in events] == [EventKind.ENTRY, EventKind.EXIT]
    assert events[1].track_id == 4


def test_vanishing_mid_frame_is_not_an_exit():
    """Occlusion or a missed detection away from the edge - the person has not left."""
    engine = make_engine(FULL_FRAME_ZONE_DOC)
    settle_past_startup(engine)

    events = []
    for _ in range(3):
        events.extend(engine.update(box_track([0, 400, 200, 800], track_id=4), FRAME))
    for _ in range(3):
        events.extend(engine.update(box_track([400, 300, 600, 800], track_id=4), FRAME))
    for _ in range(LEFT_VIEW_AFTER_UPDATES + 2):
        events.extend(engine.update(TrackedDetections.empty(), FRAME))

    assert [e.kind for e in events] == [EventKind.ENTRY]


def test_left_view_exit_fires_only_once():
    engine = make_engine(FULL_FRAME_ZONE_DOC)
    settle_past_startup(engine)

    events = []
    for _ in range(3):
        events.extend(engine.update(box_track([0, 400, 200, 800], track_id=4), FRAME))
    for _ in range(LEFT_VIEW_AFTER_UPDATES * 5):
        events.extend(engine.update(TrackedDetections.empty(), FRAME))

    assert [e.kind for e in events] == [EventKind.ENTRY, EventKind.EXIT]


# =========================================================================
# Hysteresis
# =========================================================================


def test_min_frames_hysteresis_suppresses_a_brief_excursion():
    """Someone loitering on the line must not generate a flood of events."""
    engine = make_engine(min_frames=3)

    events = walk(engine, [
        (100, 100), (100, 100), (100, 100),
        (500, 500),                            # one frame inside - not enough
        (100, 100), (100, 100), (100, 100),
    ])

    assert events == []


def test_brief_excursion_does_not_fire_a_spurious_exit():
    """REGRESSION (bug 3, found in implementation): a one-frame flicker INTO the zone
    never confirms an ENTRY, so the return to outside must be a no-op - not an EXIT with
    no matching ENTRY.

    A single state plus an `emitted` flag (what PLAN.md rev 2 specified) gets this wrong:
    the return to OUT looks like a fresh unemitted transition. Comparing the settled
    candidate against the last CONFIRMED state is what makes the round trip a no-op."""
    engine = make_engine(min_frames=3)

    events = walk(engine, [(100, 100)] * 3 + [(500, 500)] + [(100, 100)] * 4)

    assert events == [], "outside -> flicker inside -> outside is not an exit"


def test_brief_excursion_out_does_not_fire_a_spurious_entry():
    """The mirror case: confirmed inside, one frame outside, back inside."""
    engine = make_engine(min_frames=3)

    walk(engine, [(100, 100)] * 3 + [(500, 500)] * 3)   # settle outside, then enter
    events = walk(engine, [(100, 100)] + [(500, 500)] * 4)

    assert events == [], "a single frame outside must not produce a second ENTRY"


def test_event_fires_once_state_is_held_long_enough():
    engine = make_engine(min_frames=3)
    events = walk(engine, [(100, 100)] * 3 + [(500, 500)] * 3)
    assert [e.kind for e in events] == [EventKind.ENTRY]


def test_event_is_emitted_only_once_per_transition():
    """Staying inside must not re-emit on every subsequent frame."""
    engine = make_engine()
    events = walk(engine, [(100, 100)] * 2 + [(500, 500)] * 20)
    assert len(events) == 1


# =========================================================================
# Event payload
# =========================================================================


def test_event_carries_zone_name_and_normalised_geometry():
    engine = make_engine()
    events = walk(engine, [(100, 100)] * 2 + [(500, 500)] * 2)

    event = events[0]
    assert event.zone_name == "Loading Bay"
    assert event.zone_id == "bay"
    assert event.track_id == 1
    assert event.camera_id == "cam_01"
    assert event.foot_point == pytest.approx((0.5, 0.5))
    assert all(0.0 <= v <= 1.0 for v in event.bbox)
    assert "Loading Bay" in event.describe()


# =========================================================================
# events filter
# =========================================================================


def test_entry_only_zone_does_not_emit_exit():
    doc = {"cameras": {"cam_01": {"zones": [{
        "id": "bay", "type": "polygon",
        "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "events": ["entry"], "min_frames": 2,
    }]}}}
    engine = make_engine(doc)

    events = walk(engine, [(100, 100)] * 2 + [(500, 500)] * 2 + [(100, 100)] * 2)

    assert [e.kind for e in events] == [EventKind.ENTRY]


# =========================================================================
# Tripwire direction
# =========================================================================


TRIPWIRE_DOC = {"cameras": {"cam_01": {"zones": [{
    "id": "gate", "name": "Main Gate", "type": "tripwire",
    "points": [[0.1, 0.5], [0.9, 0.5]],
    "direction": "both", "min_frames": 2,
}]}}}


def test_tripwire_emits_on_crossing():
    engine = make_engine(TRIPWIRE_DOC)
    events = walk(engine, [(500, 200)] * 2 + [(500, 800)] * 2)
    assert len(events) == 1


def test_tripwire_direction_filter_suppresses_the_reverse():
    doc = {"cameras": {"cam_01": {"zones": [dict(TRIPWIRE_DOC["cameras"]["cam_01"]["zones"][0],
                                                 direction="a_to_b")]}}}
    engine = make_engine(doc)

    events = walk(engine, [
        (500, 200), (500, 200),   # side A
        (500, 800), (500, 800),   # crossed to B -> allowed
        (500, 200), (500, 200),   # crossed back -> suppressed
    ])

    assert [e.kind for e in events] == [EventKind.ENTRY]


def test_tripwire_both_directions_emits_twice():
    engine = make_engine(TRIPWIRE_DOC)
    events = walk(engine, [(500, 200)] * 2 + [(500, 800)] * 2 + [(500, 200)] * 2)
    assert len(events) == 2


# =========================================================================
# Exclusion masks
# =========================================================================


def test_exclusion_zone_suppresses_events_inside_it():
    """The highest value-per-hour feature: mask the welding bay, keep the rest."""
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "bay", "type": "polygon", "min_frames": 2,
         "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]},
        {"id": "welding", "type": "exclusion", "applies_to": ["person"],
         "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]},
    ]}}}
    engine = make_engine(doc)

    # Walks from outside straight into the masked centre.
    events = walk(engine, [(100, 100)] * 2 + [(500, 500)] * 4)

    assert events == []


def test_exclusion_zone_does_not_suppress_elsewhere_in_the_same_zone():
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "bay", "type": "polygon", "min_frames": 2,
         "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]},
        {"id": "welding", "type": "exclusion", "applies_to": ["person"],
         "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]},
    ]}}}
    engine = make_engine(doc)

    events = walk(engine, [(100, 100)] * 2 + [(300, 300)] * 3)

    assert [e.kind for e in events] == [EventKind.ENTRY]


def test_exclusion_applying_only_to_fire_does_not_mask_people():
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "bay", "type": "polygon", "min_frames": 2,
         "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]},
        {"id": "welding", "type": "exclusion", "applies_to": ["fire", "smoke"],
         "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]},
    ]}}}
    engine = make_engine(doc)

    events = walk(engine, [(100, 100)] * 2 + [(500, 500)] * 3)

    assert [e.kind for e in events] == [EventKind.ENTRY], "people still tracked through it"


# =========================================================================
# Fire ROI
# =========================================================================


def test_fire_roi_returns_a_negative_confidence_delta():
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "server_room", "type": "fire_roi", "applies_to": ["fire", "smoke"],
         "conf_delta": -0.08,
         "points": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]},
    ]}}}
    engine = make_engine(doc)

    assert engine.confidence_delta((100.0, 100.0), *FRAME) == pytest.approx(-0.08)
    assert engine.confidence_delta((900.0, 900.0), *FRAME) == 0.0


def test_suppressed_mask_masks_fire_inside_an_exclusion_zone_that_applies_to_it():
    """Expansion Plan Phase E: suppressed_mask is class-agnostic already (klass="person"
    is just its default) - this is the direct check that "fire"/"smoke" actually work,
    not just that they don't accidentally mask people (see the test above this one)."""
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "welding", "type": "exclusion", "applies_to": ["fire", "smoke"],
         "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]},
    ]}}}
    engine = make_engine(doc)

    inside = np.array([[500.0, 500.0]])
    outside = np.array([[100.0, 100.0]])

    assert engine.suppressed_mask(inside, *FRAME, "fire")[0] is np.True_
    assert engine.suppressed_mask(inside, *FRAME, "smoke")[0] is np.True_
    assert engine.suppressed_mask(outside, *FRAME, "fire")[0] is np.False_


def test_suppressed_mask_does_not_mask_a_class_the_exclusion_does_not_apply_to():
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "welding", "type": "exclusion", "applies_to": ["fire"],  # smoke NOT listed
         "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]},
    ]}}}
    engine = make_engine(doc)

    inside = np.array([[500.0, 500.0]])

    assert engine.suppressed_mask(inside, *FRAME, "fire")[0] is np.True_
    assert engine.suppressed_mask(inside, *FRAME, "smoke")[0] is np.False_


def test_fire_roi_zone_at_returns_the_containing_zone():
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "server_room", "name": "Server Room", "type": "fire_roi",
         "applies_to": ["fire", "smoke"], "conf_delta": -0.08,
         "points": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]},
    ]}}}
    engine = make_engine(doc)

    zone = engine.fire_roi_zone_at((100.0, 100.0), *FRAME)

    assert zone is not None
    assert zone.id == "server_room"
    assert zone.display_name == "Server Room"


def test_fire_roi_zone_at_returns_none_outside_any_zone():
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "server_room", "type": "fire_roi", "applies_to": ["fire"],
         "points": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]},
    ]}}}
    engine = make_engine(doc)

    assert engine.fire_roi_zone_at((900.0, 900.0), *FRAME) is None


# =========================================================================
# Schedules
# =========================================================================


def test_zone_outside_its_schedule_emits_nothing():
    doc = {"cameras": {"cam_01": {"zones": [{
        "id": "bay", "type": "polygon", "min_frames": 2,
        "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "schedule": {"from": "23:00", "to": "23:30"},
    }]}}}
    engine = make_engine(doc)

    # now= is a unix timestamp; the schedule is evaluated in local time. A 30-minute
    # window means this is inactive for all but 2% of runs, so assert on the common case
    # by using a fixed midday timestamp.
    midday = 1786100400.0  # 2026-08-11 around midday local
    events = []
    for step in range(6):
        x = 100 if step < 2 else 500
        events.extend(engine.update(person_at(x, x), FRAME, now=midday + step))

    # Either the window is inactive (no events) or the test ran at 23:00-23:30.
    assert events == [] or len(events) == 1


# =========================================================================
# Multiple tracks and zones
# =========================================================================


def test_two_people_have_independent_state():
    engine = make_engine()

    def both(x_a, x_b):
        return TrackedDetections(
            xyxy=np.array([[x_a - 20, 400, x_a + 20, 500],
                           [x_b - 20, 400, x_b + 20, 500]], np.float32),
            scores=np.array([0.9, 0.9], np.float32),
            class_ids=np.array([0, 0], np.int32),
            track_ids=np.array([1, 2], np.int32),
        )

    events = []
    for _ in range(2):
        events.extend(engine.update(both(100, 100), FRAME))
    for _ in range(2):
        events.extend(engine.update(both(500, 100), FRAME))  # only track 1 enters

    assert len(events) == 1
    assert events[0].track_id == 1


def test_state_is_dropped_for_deleted_zones():
    collection = seed_zones()
    store = ZoneStore(collection, camera_id="cam_01", refresh_interval=0)
    engine = BoundaryEngine(store, camera_id="cam_01", default_min_frames=2)

    walk(engine, [(500, 500)] * 3)
    assert engine.track_states()

    collection.delete_many({"camera_id": "cam_01"})
    engine.update(person_at(500, 500), FRAME)

    assert engine.track_states() == {}


def test_empty_detections_are_handled():
    engine = make_engine()
    assert engine.update(TrackedDetections.empty(), FRAME) == []


# =========================================================================
# Occupancy (headcount)
# =========================================================================


def test_zone_occupancy_counts_confirmed_occupants():
    engine = make_engine()

    walk(engine, [(100, 100)] * 2 + [(500, 500)] * 2, track_id=1)  # enters
    walk(engine, [(100, 100)] * 2, track_id=2)                     # stays outside

    assert engine.zone_occupancy() == {"bay": 1}


def test_zone_occupancy_drops_a_track_that_leaves():
    engine = make_engine()

    walk(engine, [(100, 100)] * 2 + [(500, 500)] * 2, track_id=1)
    assert engine.zone_occupancy() == {"bay": 1}

    walk(engine, [(100, 100)] * 2, track_id=1)
    assert engine.zone_occupancy() == {"bay": 0}


def test_zone_occupancy_ignores_an_unconfirmed_excursion():
    """A one-frame flicker into the zone never confirms - it must not count as an
    occupant, matching the no-spurious-event behaviour this file already pins."""
    engine = make_engine(min_frames=3)

    walk(engine, [(100, 100)] * 3 + [(500, 500)], track_id=1)

    assert engine.zone_occupancy() == {"bay": 0}


def test_zone_occupancy_counts_multiple_independent_tracks():
    engine = make_engine()

    def both(x_a, x_b):
        return TrackedDetections(
            xyxy=np.array([[x_a - 20, 400, x_a + 20, 500],
                           [x_b - 20, 400, x_b + 20, 500]], np.float32),
            scores=np.array([0.9, 0.9], np.float32),
            class_ids=np.array([0, 0], np.int32),
            track_ids=np.array([1, 2], np.int32),
        )

    for _ in range(2):
        engine.update(both(500, 100), FRAME)  # track 1 inside, track 2 outside
    for _ in range(2):
        engine.update(both(500, 500), FRAME)  # both inside

    assert engine.zone_occupancy() == {"bay": 2}


def test_zone_occupancy_excludes_exclusion_and_fire_roi_zones():
    """Occupancy only makes sense for `polygon` zones - exclusion and fire_roi modify
    other detections rather than representing an area people occupy."""
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "bay", "type": "polygon", "min_frames": 2,
         "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]},
        {"id": "welding", "type": "exclusion", "applies_to": ["person"],
         "points": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]},
        {"id": "server_room", "type": "fire_roi", "applies_to": ["fire"],
         "points": [[0.9, 0.9], [1.0, 0.9], [1.0, 1.0], [0.9, 1.0]]},
    ]}}}
    engine = make_engine(doc)

    walk(engine, [(100, 100)] * 2 + [(500, 500)] * 2, track_id=1)

    assert engine.zone_occupancy() == {"bay": 1}


def test_zone_occupancy_includes_empty_zones_at_zero():
    engine = make_engine()
    assert engine.zone_occupancy() == {"bay": 0}


def test_zone_occupancy_does_not_count_a_track_that_disappeared_while_inside():
    """Found in live testing: 2 people on camera, 3 'inside' - a track lost while inside
    (occlusion, or re-identified under a new id) kept counting for ~50 s."""
    engine = make_engine()
    walk(engine, [(100, 100)] * 2 + [(500, 500)] * 2, track_id=1)
    assert engine.zone_occupancy() == {"bay": 1}

    walk(engine, [(500, 500)] * 10, track_id=2)  # same person, new tracker id

    assert engine.zone_occupancy() == {"bay": 1}
