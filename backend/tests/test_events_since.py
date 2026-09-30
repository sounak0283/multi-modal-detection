"""GET /api/events?since=... - the Alert history period filter (Today / This week / This
month / Last 6 months). The dashboard computes the start in the viewer's timezone and
sends it with an offset; the endpoint must pass it to the database as an aware UTC
datetime. The database's own `since` filtering is covered in test_store_mongo.py."""

from __future__ import annotations

from datetime import UTC, datetime

import mongomock
from conftest import authed_client, seed_admin

from perimeter.api.app import create_app
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import Settings


class RecordingDatabase:
    def __init__(self):
        self.calls = []

    def recent_events(self, **kwargs):
        self.calls.append(kwargs)
        return []


def build(tmp_path):
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")
    database = RecordingDatabase()
    app = create_app(settings, store, registry, manager, users, database=database)
    return authed_client(app), database


def test_no_since_means_all_time(tmp_path):
    client, database = build(tmp_path)
    assert client.get("/api/events").status_code == 200
    assert database.calls[-1]["since"] is None


def test_a_local_midnight_with_an_offset_becomes_the_same_instant_in_utc(tmp_path):
    client, database = build(tmp_path)
    # Midnight in India (UTC+05:30) is 18:30 the previous day in UTC.
    response = client.get("/api/events", params={"since": "2026-09-29T00:00:00+05:30"})
    assert response.status_code == 200
    assert database.calls[-1]["since"] == datetime(2026, 9, 28, 18, 30, tzinfo=UTC)


def test_the_z_suffix_the_browser_sends_is_accepted(tmp_path):
    client, database = build(tmp_path)
    response = client.get("/api/events", params={"since": "2026-09-28T18:30:00.000Z"})
    assert response.status_code == 200
    assert database.calls[-1]["since"] == datetime(2026, 9, 28, 18, 30, tzinfo=UTC)


def test_a_malformed_since_is_a_clear_400(tmp_path):
    client, _database = build(tmp_path)
    response = client.get("/api/events", params={"since": "last tuesday"})
    assert response.status_code == 400
    assert "ISO-8601" in response.json()["detail"]


# -- Alert history cards must agree with the list ----------------------------------------

from datetime import timedelta  # noqa: E402

from perimeter.store.db import Database  # noqa: E402


def real_database():
    db = Database.__new__(Database)  # mongomock instead of a live MongoDB
    db.url, db.db_name = "mongodb://mock", "perimeter_test"
    db.client = mongomock.MongoClient()
    db.db = db.client[db.db_name]
    db.ensure_indexes()
    db.available = lambda: True  # mongomock has no server topology to ask
    return db


def build_with(tmp_path, database):
    db = mongomock.MongoClient()["perimeter_app"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")
    return authed_client(create_app(settings, store, registry, manager, users, database=database))


def seed(database, now):
    rows = [  # (days ago, kind, zone, camera)
        (0, "boundary", "bay", "cam_01"), (0, "boundary", "gate", "cam_01"),
        (0, "fire", None, "cam_02"), (0, "ppe", "bay", "cam_01"),
        (3, "boundary", "bay", "cam_01"), (3, "smoke", None, "cam_02"),
        (20, "boundary", "bay", "cam_02"), (200, "boundary", "bay", "cam_01"),
    ]
    for days, kind, zone, camera in rows:
        database.insert_event({"camera_id": camera, "kind": kind, "zone_id": zone,
                               "message": kind, "severity": "medium",
                               "ts": now - timedelta(days=days, minutes=1)})


def test_cards_and_list_agree_for_every_filter_combination(tmp_path):
    database = real_database()
    now = datetime.now(UTC)
    seed(database, now)
    client = build_with(tmp_path, database)

    periods = [None, now - timedelta(days=1), now - timedelta(days=7), now - timedelta(days=30)]
    for since in periods:
        for zone in (None, "bay", "gate"):
            for camera in (None, "cam_01", "cam_02"):
                params = {k: v for k, v in (("since", since and since.isoformat()),
                                           ("zone_id", zone), ("camera_id", camera)) if v}
                window = client.get("/api/events/summary", params=params).json()["window"]
                for kind in (None, "boundary", "fire", "smoke", "ppe"):
                    listed = client.get("/api/events", params={**params, "limit": 500,
                                                               **({"kind": kind} if kind else {})})
                    card = sum(window.values()) if kind is None else window.get(kind, 0)
                    assert card == len(listed.json()["events"]), (params, kind)


def test_summary_window_rejects_a_malformed_since(tmp_path):
    client = build_with(tmp_path, real_database())
    assert client.get("/api/events/summary", params={"since": "yesterday"}).status_code == 400
