"""A MongoDB outage must never stop detection or disarm a site (PLAN.md section 8,
"Detection outranks persistence").

Found by live testing: a stopped database killed the per-camera inference thread (the
zone hot-reload raised inside it) while the camera kept reporting "live", every API
route returned a 500 after a 10 s driver timeout, and alert-email routing failed.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mongomock
import pytest
from conftest import authed_client, seed_admin
from pymongo.errors import ServerSelectionTimeoutError

from perimeter.alerts.config_store import AlertConfigStore
from perimeter.api.app import create_app
from perimeter.boundary.zones import ZoneStore, zone_from_dict
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.models import camera_from_dict
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import Settings

PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")
ZONE = {"id": "bay", "type": "polygon", "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}


class Outage:
    """Wraps a collection; while `down` is set every call raises like an unreachable server."""

    def __init__(self, collection, delay_s: float = 0.0):
        self._collection = collection
        self.down = False
        self.delay_s = delay_s

    def __getattr__(self, name):
        attr = getattr(self._collection, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            if self.down:
                time.sleep(self.delay_s)
                raise ServerSelectionTimeoutError("simulated outage")
            return attr(*args, **kwargs)

        return call


# -- zones -------------------------------------------------------------------


def test_zone_reload_during_an_outage_keeps_enforcing_the_last_good_zones():
    collection = Outage(mongomock.MongoClient()["t"]["zones"])
    store = ZoneStore(collection, refresh_interval=0)
    store.save([zone_from_dict(ZONE)], "cam_01")

    collection.down = True

    assert store.reload(force=True) is False
    assert [z.id for z in store.zones("cam_01")] == ["bay"]
    assert "unavailable" in store.last_error


def test_background_refresh_never_blocks_the_caller_on_a_slow_database():
    collection = Outage(mongomock.MongoClient()["t"]["zones"], delay_s=2.0)
    store = ZoneStore(collection, refresh_interval=0.01)
    collection.down = True
    time.sleep(0.02)

    started = time.perf_counter()
    for _ in range(50):
        store.refresh_in_background()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5


# -- alert routing -----------------------------------------------------------


def test_alert_config_reload_during_an_outage_keeps_the_last_good_routing():
    collection = Outage(mongomock.MongoClient()["t"]["alert_config"])
    store = AlertConfigStore(collection, refresh_interval=0)
    store.save(
        __import__("perimeter.alerts.cooldown", fromlist=["x"]).alert_config_from_dict(
            {"sinks": [{"id": "r", "type": "smtp", "to": ["ops@example.com"]}]}
        )
    )

    collection.down = True

    assert store.reload(force=True) is False
    assert [r.id for r in store.get().sinks] == ["r"]


# -- inference thread ----------------------------------------------------------


@pytest.mark.skipif(not PERSON_MODEL.is_file(), reason="yolox_nano.onnx not present")
def test_an_exception_in_one_inference_step_does_not_end_detection():
    from perimeter.pipeline import Pipeline

    camera = camera_from_dict({"id": "cam_01", "source_type": "webcam", "source": 0})
    store = ZoneStore(mongomock.MongoClient()["t"]["zones"], refresh_interval=0)
    pipeline = Pipeline(Settings(), camera, store, str(PERSON_MODEL))
    calls = {"n": 0}

    def flaky_step():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ServerSelectionTimeoutError("simulated outage")
        if calls["n"] >= 3:
            pipeline._stop.set()

    pipeline._step = flaky_step
    thread = threading.Thread(target=pipeline._run, daemon=True)
    thread.start()
    thread.join(5)

    assert not thread.is_alive()
    assert calls["n"] >= 3
    assert pipeline.stats.errors == 1


# -- API ---------------------------------------------------------------------------


def build_outage_app(tmp_path):
    db = mongomock.MongoClient()["perimeter_test"]
    users_collection = Outage(db["users"])
    cameras_collection = Outage(db["cameras"])
    users = seed_admin(users_collection)
    registry = CameraRegistry(cameras_collection)
    store = ZoneStore(db["zones"], refresh_interval=0)
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")
    app = create_app(settings, store, registry, manager, users)
    return app, users_collection, cameras_collection


def test_a_signed_in_operator_keeps_the_dashboard_during_an_outage(tmp_path):
    app, users_collection, cameras_collection = build_outage_app(tmp_path)
    client = authed_client(app)
    assert client.get("/api/auth/me").status_code == 200

    users_collection.down = True
    cameras_collection.down = True

    assert client.get("/api/auth/me").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_storage_backed_routes_return_503_not_500_during_an_outage(tmp_path):
    app, users_collection, cameras_collection = build_outage_app(tmp_path)
    client = authed_client(app)
    client.get("/api/auth/me")

    cameras_collection.down = True

    assert client.get("/api/cameras/cam_01/zones").status_code == 503
    # the camera list degrades to "what is running" rather than failing outright
    assert client.get("/api/cameras").status_code == 200


def test_an_unknown_session_during_an_outage_is_503_not_a_crash(tmp_path):
    app, users_collection, _cameras = build_outage_app(tmp_path)
    users_collection.down = True

    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/api/auth/login", json={"email": "admin@test.local", "password": "admin-password"}
    )
    assert response.status_code == 503


def test_logout_during_an_outage_still_revokes_the_session(tmp_path):
    app, users_collection, _cameras = build_outage_app(tmp_path)
    client = authed_client(app)
    copied = dict(client.cookies)
    users_collection.down = True

    assert client.post("/api/auth/logout").status_code == 200

    from fastapi.testclient import TestClient

    replay = TestClient(app)
    replay.cookies.update(copied)
    assert replay.get("/api/auth/me").status_code == 401
