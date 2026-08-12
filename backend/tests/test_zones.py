"""Tests for zone parsing, persistence and hot reload (PLAN.md sections 6.2-6.5).

The reload behaviour carries a safety requirement, not just a convenience one: a syntax
error saved mid-edit must never disarm a live site.
"""

from __future__ import annotations

import pytest
import yaml

from fsbd.boundary.zones import (
    Direction,
    EventKind,
    Severity,
    ZoneStore,
    ZoneType,
    dump_zones,
    parse_zones,
    zone_from_dict,
)

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


def doc(*zones) -> dict:
    return {"cameras": {"cam_01": {"zones": list(zones)}}}


def polygon_zone(**overrides) -> dict:
    return {"id": "bay", "type": "polygon", "points": SQUARE, **overrides}


# -- parsing --------------------------------------------------------------


def test_minimal_polygon_zone_gets_sensible_defaults():
    zone = zone_from_dict(polygon_zone())

    assert zone.type is ZoneType.POLYGON
    assert zone.detect == frozenset({"person"})
    assert zone.events == {EventKind.ENTRY, EventKind.EXIT}
    assert zone.severity is Severity.MEDIUM
    assert zone.schedule.always_on
    assert zone.min_frames is None, "None means fall back to the global default"


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


def test_duplicate_zone_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_zones(doc(polygon_zone(), polygon_zone()))


def test_document_without_cameras_key_is_rejected():
    with pytest.raises(ValueError, match="cameras"):
        parse_zones({"zones": []})


def test_empty_document_is_empty_not_an_error():
    assert parse_zones(None) == {}


# -- round trip -----------------------------------------------------------


def test_round_trip_preserves_every_field():
    original = zone_from_dict(polygon_zone(
        name="Loading Bay",
        detect=["person"],
        events=["entry"],
        min_frames=6,
        severity="high",
        schedule={"days": ["mon", "fri"], "from": "18:00", "to": "06:00"},
    ))

    restored = zone_from_dict(original.to_dict())

    assert restored == original


def test_dump_and_reparse_is_stable():
    zones = parse_zones(doc(
        polygon_zone(name="Bay"),
        {"id": "gate", "type": "tripwire", "points": [[0.1, 0.5], [0.9, 0.5]],
         "direction": "b_to_a"},
        {"id": "weld", "type": "exclusion", "points": SQUARE, "applies_to": ["fire"]},
    ))

    reparsed = parse_zones(yaml.safe_load(dump_zones(zones)))

    assert reparsed == zones


# -- store: load, save, hot reload ---------------------------------------


def test_store_loads_zones(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump(doc(polygon_zone())), encoding="utf-8")

    store = ZoneStore(path, camera_id="cam_01")

    assert len(store.zones()) == 1
    assert store.last_error is None


def test_store_handles_a_missing_file(tmp_path):
    store = ZoneStore(tmp_path / "absent.yaml", camera_id="cam_01")
    assert store.zones() == []


def test_store_reloads_on_mtime_change(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump(doc(polygon_zone())), encoding="utf-8")
    store = ZoneStore(path, camera_id="cam_01")
    assert len(store.zones()) == 1

    path.write_text(yaml.safe_dump(doc(polygon_zone(), polygon_zone(id="second"))), "utf-8")
    # mtime resolution can be coarse; force the check the way the engine would after a save.
    store.reload(force=True)

    assert len(store.zones()) == 2


def test_a_broken_file_is_rejected_and_the_previous_config_survives(tmp_path):
    """A syntax error saved mid-edit must never disarm a live site."""
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump(doc(polygon_zone())), encoding="utf-8")
    store = ZoneStore(path, camera_id="cam_01")

    path.write_text("cameras: [this is not: valid: yaml", encoding="utf-8")
    store.reload(force=True)

    assert len(store.zones()) == 1, "previous good config retained"
    assert store.last_error is not None


def test_a_semantically_invalid_file_is_also_rejected(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump(doc(polygon_zone())), encoding="utf-8")
    store = ZoneStore(path, camera_id="cam_01")

    path.write_text(yaml.safe_dump(doc({"id": "bad", "type": "polygon",
                                        "points": [[0.1, 0.1]]})), encoding="utf-8")
    store.reload(force=True)

    assert len(store.zones()) == 1
    assert "at least 3" in (store.last_error or "")


def test_save_writes_and_is_readable_back(tmp_path):
    path = tmp_path / "zones.yaml"
    store = ZoneStore(path, camera_id="cam_01")

    store.save([zone_from_dict(polygon_zone(name="Bay"))])

    assert path.is_file()
    assert ZoneStore(path, camera_id="cam_01").zones()[0].name == "Bay"


def test_save_leaves_no_temp_file(tmp_path):
    path = tmp_path / "zones.yaml"
    store = ZoneStore(path, camera_id="cam_01")
    store.save([zone_from_dict(polygon_zone())])

    assert list(tmp_path.glob("*.tmp")) == []


def test_save_preserves_other_cameras(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump({"cameras": {
        "cam_01": {"zones": [polygon_zone()]},
        "cam_02": {"zones": [polygon_zone(id="other")]},
    }}), encoding="utf-8")
    store = ZoneStore(path, camera_id="cam_01")

    store.save([zone_from_dict(polygon_zone(id="replaced"))])

    reloaded = ZoneStore(path, camera_id="cam_02")
    assert [z.id for z in reloaded.zones()] == ["other"]
    assert [z.id for z in reloaded.zones("cam_01")] == ["replaced"]


def test_version_increments_on_change(tmp_path):
    """The engine caches compiled geometry against this."""
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump(doc(polygon_zone())), encoding="utf-8")
    store = ZoneStore(path, camera_id="cam_01")

    before = store.version
    store.save([zone_from_dict(polygon_zone(id="new"))])

    assert store.version > before
