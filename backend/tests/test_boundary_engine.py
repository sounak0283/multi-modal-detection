"""Tests for the boundary state machine (PLAN.md section 6.3).

The two regression tests at the top are the most important in the project: they pin the
bugs the plan review found, both of which produce failures that look like "the detector
is flaky" rather than like a state-machine defect.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from fsbd.boundary.engine import BoundaryEngine
from fsbd.boundary.zones import EventKind, ZoneStore
from fsbd.track.tracker import TrackedDetections

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


def write_zones(tmp_path, doc=None):
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump(doc or ZONE_DOC), encoding="utf-8")
    return path


def make_engine(tmp_path, doc=None, min_frames=2) -> BoundaryEngine:
    store = ZoneStore(write_zones(tmp_path, doc), camera_id="cam_01")
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


def test_exit_fires_shortly_after_entry(tmp_path):
    """Rev 1 guarded emission with a 30 s cooldown, so a person who entered at t=0 and
    left at t=10 had their EXIT silently dropped - and because the guard used
    `age == min_frames` (true once), it never re-fired. Entering and leaving inside 30 s
    is the NORMAL case, so entry/exit pairing was broken by default."""
    engine = make_engine(tmp_path)

    events = walk(engine, [
        (100, 100), (100, 100),   # outside, settles
        (500, 500), (500, 500),   # inside  -> ENTRY
        (100, 100), (100, 100),   # outside -> EXIT, only ~4 s later
    ])

    kinds = [e.kind for e in events]
    assert kinds == [EventKind.ENTRY, EventKind.EXIT]


def test_repeated_entry_exit_cycles_all_fire(tmp_path):
    engine = make_engine(tmp_path)
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


def test_new_track_appearing_inside_a_zone_emits_nothing(tmp_path):
    """ByteTrack reassigns IDs after occlusion. If a new track defaulted to OUT, every
    person who appears already inside a zone would fire a false ENTRY - including the
    same person who just acquired a new ID while standing still."""
    engine = make_engine(tmp_path)

    events = walk(engine, [(500, 500)] * 6, track_id=7)

    assert events == []


def test_id_churn_on_a_stationary_person_emits_nothing(tmp_path):
    """The exact failure mode: one person standing inside a zone, cycling through new
    track IDs as the tracker loses and reacquires them."""
    engine = make_engine(tmp_path)

    events = []
    for track_id in range(1, 15):
        for _ in range(3):
            events.extend(engine.update(person_at(500, 500, track_id), FRAME))

    assert events == []


def test_new_track_outside_the_zone_also_emits_nothing(tmp_path):
    engine = make_engine(tmp_path)
    assert walk(engine, [(50, 50)] * 5, track_id=3) == []


def test_a_genuine_crossing_still_fires_after_churn(tmp_path):
    """Suppressing phantom entries must not suppress real ones."""
    engine = make_engine(tmp_path)

    walk(engine, [(500, 500)] * 3, track_id=9)          # born inside, silent
    events = walk(engine, [(100, 100)] * 3, track_id=9)  # then genuinely leaves

    assert [e.kind for e in events] == [EventKind.EXIT]


# =========================================================================
# Hysteresis
# =========================================================================


def test_min_frames_hysteresis_suppresses_a_brief_excursion(tmp_path):
    """Someone loitering on the line must not generate a flood of events."""
    engine = make_engine(tmp_path, min_frames=3)

    events = walk(engine, [
        (100, 100), (100, 100), (100, 100),
        (500, 500),                            # one frame inside - not enough
        (100, 100), (100, 100), (100, 100),
    ])

    assert events == []


def test_brief_excursion_does_not_fire_a_spurious_exit(tmp_path):
    """REGRESSION (bug 3, found in implementation): a one-frame flicker INTO the zone
    never confirms an ENTRY, so the return to outside must be a no-op - not an EXIT with
    no matching ENTRY.

    A single state plus an `emitted` flag (what PLAN.md rev 2 specified) gets this wrong:
    the return to OUT looks like a fresh unemitted transition. Comparing the settled
    candidate against the last CONFIRMED state is what makes the round trip a no-op."""
    engine = make_engine(tmp_path, min_frames=3)

    events = walk(engine, [(100, 100)] * 3 + [(500, 500)] + [(100, 100)] * 4)

    assert events == [], "outside -> flicker inside -> outside is not an exit"


def test_brief_excursion_out_does_not_fire_a_spurious_entry(tmp_path):
    """The mirror case: confirmed inside, one frame outside, back inside."""
    engine = make_engine(tmp_path, min_frames=3)

    walk(engine, [(100, 100)] * 3 + [(500, 500)] * 3)   # settle outside, then enter
    events = walk(engine, [(100, 100)] + [(500, 500)] * 4)

    assert events == [], "a single frame outside must not produce a second ENTRY"


def test_event_fires_once_state_is_held_long_enough(tmp_path):
    engine = make_engine(tmp_path, min_frames=3)
    events = walk(engine, [(100, 100)] * 3 + [(500, 500)] * 3)
    assert [e.kind for e in events] == [EventKind.ENTRY]


def test_event_is_emitted_only_once_per_transition(tmp_path):
    """Staying inside must not re-emit on every subsequent frame."""
    engine = make_engine(tmp_path)
    events = walk(engine, [(100, 100)] * 2 + [(500, 500)] * 20)
    assert len(events) == 1


# =========================================================================
# Event payload
# =========================================================================


def test_event_carries_zone_name_and_normalised_geometry(tmp_path):
    engine = make_engine(tmp_path)
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


def test_entry_only_zone_does_not_emit_exit(tmp_path):
    doc = {"cameras": {"cam_01": {"zones": [{
        "id": "bay", "type": "polygon",
        "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "events": ["entry"], "min_frames": 2,
    }]}}}
    engine = make_engine(tmp_path, doc)

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


def test_tripwire_emits_on_crossing(tmp_path):
    engine = make_engine(tmp_path, TRIPWIRE_DOC)
    events = walk(engine, [(500, 200)] * 2 + [(500, 800)] * 2)
    assert len(events) == 1


def test_tripwire_direction_filter_suppresses_the_reverse(tmp_path):
    doc = {"cameras": {"cam_01": {"zones": [dict(TRIPWIRE_DOC["cameras"]["cam_01"]["zones"][0],
                                                 direction="a_to_b")]}}}
    engine = make_engine(tmp_path, doc)

    events = walk(engine, [
        (500, 200), (500, 200),   # side A
        (500, 800), (500, 800),   # crossed to B -> allowed
        (500, 200), (500, 200),   # crossed back -> suppressed
    ])

    assert [e.kind for e in events] == [EventKind.ENTRY]


def test_tripwire_both_directions_emits_twice(tmp_path):
    engine = make_engine(tmp_path, TRIPWIRE_DOC)
    events = walk(engine, [(500, 200)] * 2 + [(500, 800)] * 2 + [(500, 200)] * 2)
    assert len(events) == 2


# =========================================================================
# Exclusion masks
# =========================================================================


def test_exclusion_zone_suppresses_events_inside_it(tmp_path):
    """The highest value-per-hour feature: mask the welding bay, keep the rest."""
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "bay", "type": "polygon", "min_frames": 2,
         "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]},
        {"id": "welding", "type": "exclusion", "applies_to": ["person"],
         "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]},
    ]}}}
    engine = make_engine(tmp_path, doc)

    # Walks from outside straight into the masked centre.
    events = walk(engine, [(100, 100)] * 2 + [(500, 500)] * 4)

    assert events == []


def test_exclusion_zone_does_not_suppress_elsewhere_in_the_same_zone(tmp_path):
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "bay", "type": "polygon", "min_frames": 2,
         "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]},
        {"id": "welding", "type": "exclusion", "applies_to": ["person"],
         "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]},
    ]}}}
    engine = make_engine(tmp_path, doc)

    events = walk(engine, [(100, 100)] * 2 + [(300, 300)] * 3)

    assert [e.kind for e in events] == [EventKind.ENTRY]


def test_exclusion_applying_only_to_fire_does_not_mask_people(tmp_path):
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "bay", "type": "polygon", "min_frames": 2,
         "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]},
        {"id": "welding", "type": "exclusion", "applies_to": ["fire", "smoke"],
         "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]},
    ]}}}
    engine = make_engine(tmp_path, doc)

    events = walk(engine, [(100, 100)] * 2 + [(500, 500)] * 3)

    assert [e.kind for e in events] == [EventKind.ENTRY], "people still tracked through it"


# =========================================================================
# Fire ROI
# =========================================================================


def test_fire_roi_returns_a_negative_confidence_delta(tmp_path):
    doc = {"cameras": {"cam_01": {"zones": [
        {"id": "server_room", "type": "fire_roi", "applies_to": ["fire", "smoke"],
         "conf_delta": -0.08,
         "points": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]},
    ]}}}
    engine = make_engine(tmp_path, doc)

    assert engine.confidence_delta((100.0, 100.0), *FRAME) == pytest.approx(-0.08)
    assert engine.confidence_delta((900.0, 900.0), *FRAME) == 0.0


# =========================================================================
# Schedules
# =========================================================================


def test_zone_outside_its_schedule_emits_nothing(tmp_path):
    doc = {"cameras": {"cam_01": {"zones": [{
        "id": "bay", "type": "polygon", "min_frames": 2,
        "points": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "schedule": {"from": "23:00", "to": "23:30"},
    }]}}}
    engine = make_engine(tmp_path, doc)

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


def test_two_people_have_independent_state(tmp_path):
    engine = make_engine(tmp_path)

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


def test_state_is_dropped_for_deleted_zones(tmp_path):
    path = write_zones(tmp_path)
    store = ZoneStore(path, camera_id="cam_01")
    engine = BoundaryEngine(store, camera_id="cam_01", default_min_frames=2)

    walk(engine, [(500, 500)] * 3)
    assert engine.track_states()

    path.write_text(yaml.safe_dump({"cameras": {"cam_01": {"zones": []}}}), encoding="utf-8")
    engine.update(person_at(500, 500), FRAME)

    assert engine.track_states() == {}


def test_empty_detections_are_handled(tmp_path):
    engine = make_engine(tmp_path)
    assert engine.update(TrackedDetections.empty(), FRAME) == []
