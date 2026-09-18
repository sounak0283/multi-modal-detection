"""Tests for the MongoDB event store (Expansion Plan Phase A.1).

Runs against `mongomock` rather than a live MongoDB server - fast, no external
dependency, and exercises the same PyMongo API surface `Database` uses.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import mongomock
import pytest

from perimeter.store.db import Database


@pytest.fixture
def database() -> Database:
    db = Database.__new__(Database)  # skip __init__'s real MongoClient(url) connection
    db.url = "mongodb://mock"
    db.db_name = "perimeter_test"
    db.client = mongomock.MongoClient()
    db.db = db.client[db.db_name]
    db.ensure_indexes()
    return db


def make_event(**overrides) -> dict:
    payload = {
        "camera_id": "cam_01",
        "kind": "boundary",
        "subtype": "entry",
        "zone_id": "bay",
        "zone_name": "Loading Bay",
        "message": "Someone entered Loading Bay",
        "severity": "medium",
        "bbox": [0.1, 0.1, 0.3, 0.4],
        "foot_point": [0.2, 0.4],
    }
    payload.update(overrides)
    return payload


def test_insert_returns_a_string_id(database):
    event_id = database.insert_event(make_event())
    assert isinstance(event_id, str)
    assert len(event_id) == 24  # a MongoDB ObjectId, hex-encoded


def test_recent_events_returns_newest_first(database):
    database.insert_event(make_event(ts=1000.0, message="first"))
    database.insert_event(make_event(ts=2000.0, message="second"))

    events = database.recent_events(limit=10)

    assert [e["message"] for e in events] == ["second", "first"]


def test_recent_events_filters_by_kind_zone_and_camera(database):
    database.insert_event(make_event(kind="boundary", zone_id="bay", camera_id="cam_01"))
    database.insert_event(make_event(kind="fire", zone_id=None, camera_id="cam_02"))

    assert len(database.recent_events(kind="fire")) == 1
    assert len(database.recent_events(zone_id="bay")) == 1
    assert len(database.recent_events(camera_id="cam_02")) == 1
    assert len(database.recent_events()) == 2


def test_recent_events_respects_since(database):
    now = datetime.now(UTC)
    database.insert_event(make_event(ts=now - timedelta(days=2)))
    database.insert_event(make_event(ts=now))

    assert len(database.recent_events(since=now - timedelta(hours=1))) == 1


def test_recent_events_limit_is_clamped(database):
    for i in range(5):
        database.insert_event(make_event(message=f"event {i}"))

    assert len(database.recent_events(limit=0)) == 1  # clamped up to the floor
    assert len(database.recent_events(limit=999)) == 5  # never more than exist


def test_event_counts_groups_by_kind(database):
    database.insert_event(make_event(kind="boundary"))
    database.insert_event(make_event(kind="boundary"))
    database.insert_event(make_event(kind="fire"))

    assert database.event_counts() == {"boundary": 2, "fire": 1}


def test_event_counts_can_be_scoped_to_a_camera(database):
    database.insert_event(make_event(kind="boundary", camera_id="cam_01"))
    database.insert_event(make_event(kind="boundary", camera_id="cam_02"))

    assert database.event_counts(camera_id="cam_01") == {"boundary": 1}


def test_zone_entry_counts_only_counts_boundary_entries(database):
    database.insert_event(make_event(kind="boundary", subtype="entry", zone_id="bay"))
    database.insert_event(make_event(kind="boundary", subtype="exit", zone_id="bay"))
    database.insert_event(make_event(kind="fire", subtype="entry", zone_id="bay"))

    assert database.zone_entry_counts() == {"bay": 1}


def test_mark_delivered_records_success(database):
    event_id = database.insert_event(make_event())
    database.mark_delivered(event_id)

    stored = database.recent_events(limit=1)[0]
    assert stored["delivered"] is True
    assert stored["delivery_attempts"] == 1
    assert stored["delivery_error"] is None


def test_mark_delivered_records_failure_and_increments_attempts(database):
    event_id = database.insert_event(make_event())
    database.mark_delivered(event_id, error="smtp timeout")
    database.mark_delivered(event_id, error="smtp timeout")

    stored = database.recent_events(limit=1)[0]
    assert stored["delivered"] is False
    assert stored["delivery_attempts"] == 2
    assert stored["delivery_error"] == "smtp timeout"


def test_purge_older_than_removes_only_old_events(database):
    now = datetime.now(UTC)
    database.insert_event(make_event(ts=now - timedelta(days=40)))
    database.insert_event(make_event(ts=now))

    removed = database.purge_older_than(now - timedelta(days=30))

    assert removed == 1
    assert len(database.recent_events()) == 1


def test_healthy_reports_true_for_a_reachable_database(database):
    assert database.healthy() is True


def test_coordinates_are_stored_as_plain_lists_not_json_strings(database):
    """Unlike the Postgres/JSONB version, bbox/foot_point are native Mongo arrays -
    readable directly by any Mongo tool without a second JSON-decode step."""
    event_id = database.insert_event(make_event(bbox=[0.0, 0.1, 0.2, 0.3]))
    stored = database.recent_events(limit=1)[0]

    assert stored["id"] == event_id
    assert stored["bbox"] == [0.0, 0.1, 0.2, 0.3]
