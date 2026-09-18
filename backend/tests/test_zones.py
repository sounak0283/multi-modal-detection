"""Tests for zone parsing, persistence and hot reload (Expansion Plan Phase A.3;
PLAN.md sections 6.2-6.5).

The reload behaviour carries a safety requirement, not just a convenience one: a bad
write to the `zones` collection must never disarm a live site. Storage moved from a
YAML file to MongoDB in Phase A.3 - these tests seed and mutate an in-memory `mongomock`
collection directly (the Mongo equivalent of hand-editing the old `zones.yaml`) rather
than writing files.
"""

from __future__ import annotations

import mongomock
import pytest

from perimeter.boundary.zones import (
    Direction,
    EventKind,
    Severity,
    ZoneStore,
    ZoneType,
    zone_from_dict,
)

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


def polygon_zone(**overrides) -> dict:
    return {"id": "bay", "type": "polygon", "points": SQUARE, **overrides}


def collection():
    return mongomock.MongoClient()["perimeter_test"]["zones"]


def seed(coll, camera_id: str, *zones: dict) -> None:
    for zone in zones:
        coll.insert_one(dict(zone) | {"camera_id": camera_id})


# -- parsing --------------------------------------------------------------


def test_minimal_polygon_zone_gets_sensible_defaults():
    zone = zone_from_dict(polygon_zone())

    assert zone.type is ZoneType.POLYGON
    assert zone.detect == frozenset({"person"})
    assert zone.events == {EventKind.ENTRY, EventKind.EXIT}
    assert zone.severity is Severity.MEDIUM
    assert zone.schedule.always_on
    assert zone.min_frames is None, "None means fall back to the global default"
    assert zone.authorized_person_ids == frozenset()
    assert zone.crowd_threshold is None
    assert zone.ppe_required == frozenset()
    assert zone.phone_restricted is False


def test_display_name_falls_back_to_id():
    assert zone_from_dict(polygon_zone()).display_name == "bay"
    assert zone_from_dict(polygon_zone(name="Loading Bay")).display_name == "Loading Bay"


def test_all_four_zone_types_parse():
    types = {
        "polygon": polygon_zone(),
        "tripwire": {"id": "w", "type": "tripwire", "points": [[0.1, 0.5], [0.9, 0.5]]},
        "exclusion": {"id": "x", "type": "exclusion", "points": SQUARE,
                      "applies_to": ["fire", "smoke"]},
        "fire_roi": {"id": "r", "type": "fire_roi", "points": SQUARE,
                     "applies_to": ["fire"], "conf_delta": -0.08},
    }
    for name, data in types.items():
        assert zone_from_dict(data).type.value == name


def test_only_polygon_and_tripwire_emit_events():
    assert zone_from_dict(polygon_zone()).emits_events
    assert not zone_from_dict(
        {"id": "x", "type": "exclusion", "points": SQUARE, "applies_to": ["fire"]}
    ).emits_events


def test_tripwire_direction_parses():
    zone = zone_from_dict(
        {"id": "w", "type": "tripwire", "points": [[0.1, 0.5], [0.9, 0.5]],
         "direction": "a_to_b"}
    )
    assert zone.direction is Direction.A_TO_B


def test_reserved_fields_for_later_phases_parse():
    """crowd/PPE/phone/allow-list fields (Expansion Plan Phases F.1, G, H, I) - inert
    today, but must round-trip correctly since the `zones` collection carries them now."""
    zone = zone_from_dict(polygon_zone(
        authorized_person_ids=["alice", "bob"],
        crowd_threshold=5,
        crowd_min_frames=10,
        ppe_required=["helmet", "vest"],
        phone_restricted=True,
    ))

    assert zone.authorized_person_ids == frozenset({"alice", "bob"})
    assert zone.crowd_threshold == 5
    assert zone.crowd_min_frames == 10
    assert zone.ppe_required == frozenset({"helmet", "vest"})
    assert zone.phone_restricted is True


# -- validation errors ----------------------------------------------------


def test_missing_id_is_rejected():
    with pytest.raises(ValueError, match="missing 'id'"):
        zone_from_dict({"type": "polygon", "points": SQUARE})


def test_self_intersecting_polygon_is_rejected_with_the_zone_id():
    with pytest.raises(ValueError, match="bay"):
        zone_from_dict(polygon_zone(points=[[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]]))


def test_unknown_detect_class_is_rejected():
    with pytest.raises(ValueError, match="unknown detect"):
        zone_from_dict(polygon_zone(detect=["person", "cat"]))


def test_unknown_zone_type_lists_the_valid_ones():
    with pytest.raises(ValueError, match="expected"):
        zone_from_dict({"id": "z", "type": "rectangle", "points": SQUARE})


def test_exclusion_zone_without_applies_to_is_rejected():
    """An exclusion mask that suppresses nothing is silently useless, so it is an error."""
    with pytest.raises(ValueError, match="applies_to"):
        zone_from_dict({"id": "x", "type": "exclusion", "points": SQUARE})


def test_min_frames_must_be_positive():
    with pytest.raises(ValueError, match="min_frames"):
        zone_from_dict(polygon_zone(min_frames=0))


def test_crowd_threshold_must_be_positive():
    with pytest.raises(ValueError, match="crowd_threshold"):
        zone_from_dict(polygon_zone(crowd_threshold=0))


# -- round trip -----------------------------------------------------------


def test_round_trip_preserves_every_field():
    original = zone_from_dict(polygon_zone(
        name="Loading Bay",
        detect=["person"],
        events=["entry"],
        min_frames=6,
        severity="high",
        schedule={"days": ["mon", "fri"], "from": "18:00", "to": "06:00"},
        authorized_person_ids=["alice"],
        crowd_threshold=5,
        ppe_required=["helmet"],
        phone_restricted=True,
    ))

    restored = zone_from_dict(original.to_dict())

    assert restored == original


# -- store: load, save, hot reload ---------------------------------------


def test_store_loads_zones():
    coll = collection()
    seed(coll, "cam_01", polygon_zone())

    store = ZoneStore(coll, camera_id="cam_01")

    assert len(store.zones()) == 1
    assert store.last_error is None


def test_store_handles_an_empty_collection():
    store = ZoneStore(collection(), camera_id="cam_01")
    assert store.zones() == []


def test_store_reloads_after_a_direct_collection_change():
    coll = collection()
    seed(coll, "cam_01", polygon_zone())
    store = ZoneStore(coll, camera_id="cam_01", refresh_interval=0)
    assert len(store.zones()) == 1

    seed(coll, "cam_01", polygon_zone(id="second"))
    store.reload(force=True)

    assert len(store.zones()) == 2


def test_a_bad_duplicate_write_is_rejected_and_the_previous_config_survives():
    """A concurrent or malformed write must never disarm a live site - the Mongo
    analogue of the old "a syntax error saved mid-edit" file-based guarantee."""
    coll = collection()
    seed(coll, "cam_01", polygon_zone())
    store = ZoneStore(coll, camera_id="cam_01", refresh_interval=0)

    # Bypasses ZoneStore.save()'s own validation, simulating a bad direct write.
    seed(coll, "cam_01", polygon_zone())  # same id "bay" again -> duplicate
    store.reload(force=True)

    assert len(store.zones()) == 1, "previous good config retained"
    assert store.last_error is not None
    assert "duplicate" in store.last_error


def test_a_semantically_invalid_write_is_also_rejected():
    coll = collection()
    seed(coll, "cam_01", polygon_zone())
    store = ZoneStore(coll, camera_id="cam_01", refresh_interval=0)

    seed(coll, "cam_02", {"id": "bad", "type": "polygon", "points": [[0.1, 0.1]]})
    store.reload(force=True)

    assert len(store.zones("cam_01")) == 1
    assert "at least 3" in (store.last_error or "")


def test_save_writes_and_is_readable_back():
    coll = collection()
    store = ZoneStore(coll, camera_id="cam_01")

    store.save([zone_from_dict(polygon_zone(name="Bay"))])

    assert ZoneStore(coll, camera_id="cam_01").zones()[0].name == "Bay"


def test_save_preserves_other_cameras():
    coll = collection()
    seed(coll, "cam_01", polygon_zone())
    seed(coll, "cam_02", polygon_zone(id="other"))
    store = ZoneStore(coll, camera_id="cam_01")

    store.save([zone_from_dict(polygon_zone(id="replaced"))])

    reloaded = ZoneStore(coll, camera_id="cam_02")
    assert [z.id for z in reloaded.zones()] == ["other"]
    assert [z.id for z in reloaded.zones("cam_01")] == ["replaced"]


def test_save_rejects_duplicate_ids_in_the_same_batch():
    store = ZoneStore(collection(), camera_id="cam_01")
    with pytest.raises(ValueError, match="duplicate"):
        store.save([zone_from_dict(polygon_zone()), zone_from_dict(polygon_zone())])


def test_version_increments_on_change():
    """The engine caches compiled geometry against this."""
    coll = collection()
    seed(coll, "cam_01", polygon_zone())
    store = ZoneStore(coll, camera_id="cam_01")

    before = store.version
    store.save([zone_from_dict(polygon_zone(id="new"))])

    assert store.version > before
