"""The dashboard's Video test feature: upload store, sessions, routes, and the on/off setting.

Sessions use a fake pipeline (no models, no threads of their own) except one test that builds
the REAL `Pipeline` to prove fire/smoke runs with no boundary module at all.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mongomock
import numpy as np
import pytest
from conftest import ADMIN_EMAIL, authed_client, seed_admin
from fastapi.testclient import TestClient

from perimeter.api.app import create_app
from perimeter.appsettings import AppSettings, AppSettingsStore
from perimeter.auth.models import Role, User
from perimeter.auth.passwords import hash_password
from perimeter.boundary.zones import ZoneStore, zone_from_dict
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import Settings
from perimeter.testvideo.sessions import (
    TEST_PREFIX,
    SessionError,
    TestSessionManager,
    ZoneView,
    _SessionAlerts,
)
from perimeter.testvideo.store import TestVideoStore, UploadError

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


def fake_probe(_path):
    return {"fps": 25.0, "frames": 250, "width": 640, "height": 360, "duration_s": 10.0}


def make_store(tmp_path, max_bytes=10_000, retention_days=7, probe=fake_probe):
    return TestVideoStore(tmp_path / "videos", max_bytes, retention_days, probe=probe)


def add_video(store, name="clip.mp4", data=b"x" * 100):
    video_id, path = store.new_upload(name)
    path.write_bytes(data)
    return store.finalize(video_id, path, name)


# -- the upload store ------------------------------------------------------------------------


def test_upload_is_stored_with_metadata(tmp_path):
    store = make_store(tmp_path)
    video = add_video(store, "Warehouse cam.mp4")

    assert video.name == "Warehouse cam.mp4" and video.duration_s == 10.0 and video.fps == 25.0
    assert store.get(video.id).size_bytes == 100
    assert [v.id for v in store.list()] == [video.id]


@pytest.mark.parametrize("name", ["notes.txt", "malware.exe", "video", "clip.mp4.zip", ""])
def test_only_video_file_types_are_accepted(tmp_path, name):
    with pytest.raises(UploadError, match="Unsupported file type"):
        make_store(tmp_path).new_upload(name)


def test_client_supplied_names_can_never_choose_a_path(tmp_path):
    store = make_store(tmp_path)
    video_id, path = store.new_upload("../../etc/passwd.mp4")
    assert path.parent == store.directory and path.name == f"{video_id}.mp4"
    assert store.path("../../etc/passwd") is None
    assert store.path("not-a-valid-id") is None


def test_an_unreadable_file_is_rejected_and_removed(tmp_path):
    def bad_probe(_path):
        raise UploadError("This file could not be opened as a video.")

    store = make_store(tmp_path, probe=bad_probe)
    video_id, path = store.new_upload("broken.mp4")
    path.write_bytes(b"not a video")

    with pytest.raises(UploadError):
        store.finalize(video_id, path, "broken.mp4")
    assert not path.exists()  # a rejected upload leaves nothing behind
    assert store.list() == []


def test_delete_removes_the_file_and_its_metadata(tmp_path):
    store = make_store(tmp_path)
    video = add_video(store)
    assert store.delete(video.id) is True
    assert store.get(video.id) is None and store.delete(video.id) is False
    assert list(store.directory.iterdir()) == []


def test_old_uploads_are_pruned(tmp_path):
    store = make_store(tmp_path, retention_days=7)
    old, fresh = add_video(store, "old.mp4"), add_video(store, "fresh.mp4")
    ancient = time.time() - 30 * 86400
    for p in store.directory.glob(f"{old.id}*"):
        os.utime(p, (ancient, ancient))

    assert [v.id for v in store.list()] == [fresh.id]


def test_the_real_probe_rejects_a_file_that_is_not_a_video(tmp_path):
    from perimeter.testvideo.store import probe_video

    junk = tmp_path / "fake.mp4"
    junk.write_bytes(b"this is just text pretending to be a video" * 20)
    with pytest.raises(UploadError):
        probe_video(junk)


# -- sessions ----------------------------------------------------------------------------------


class FakePipeline:
    def __init__(self, camera, zone_view, sink):
        self.camera, self.zone_view, self.sink = camera, zone_view, sink
        self.started = self.stopped = self.once = False
        self.capture_finished = False
        self.stats = SimpleNamespace(
            frames_decoded=5, detections_run=2, people_tracked=1, inference_ms=12.34
        )

    def play_once(self):
        self.once = True

    def start(self):
        self.started = True

    def stop(self, timeout=None):
        self.stopped = True

    def latest_jpeg(self, annotated=True):
        return b"\xff\xd8jpeg"


class RecordingBus:
    def __init__(self):
        self.published = []

    def publish(self, payload):
        self.published.append(payload)


def make_manager(tmp_path, *, info=None, real_bus=None, cameras=("cam_01",), zone_store=None):
    store = make_store(tmp_path)
    created = []

    def factory(camera, zone_view, sink):
        pipeline = FakePipeline(camera, zone_view, sink)
        created.append(pipeline)
        return pipeline

    manager = TestSessionManager(
        store,
        zone_store or ZoneStore(mongomock.MongoClient()["t"]["zones"], refresh_interval=0),
        model_info=lambda: info or {"firesmoke": True, "ppe": False, "identity": True},
        pipeline_factory=factory,
        real_bus=real_bus,
        camera_ids=lambda: list(cameras),
    )
    return manager, store, created


def test_a_session_runs_the_chosen_models_on_a_temporary_camera(tmp_path):
    manager, store, created = make_manager(tmp_path)
    video = add_video(store)

    session = manager.start(video.id, ["fire_smoke"], loop=False)

    pipeline = created[0]
    assert pipeline.started and pipeline.once  # plays once, not looped
    assert pipeline.camera.enabled_modules == frozenset({"fire_smoke"})
    assert pipeline.camera.id.startswith("test_") and session.status == "running"
    assert str(store.path(video.id)) == str(pipeline.camera.source)


def test_fire_smoke_alone_does_not_pull_in_the_boundary_module(tmp_path):
    manager, store, created = make_manager(tmp_path)
    manager.start(add_video(store).id, ["fire_smoke"])
    assert "boundary" not in created[0].camera.enabled_modules


def test_dependent_models_bring_the_boundary_module_with_them(tmp_path):
    manager, store, created = make_manager(tmp_path)
    manager.start(add_video(store).id, ["identity"])
    assert created[0].camera.enabled_modules == frozenset({"boundary", "identity"})


def test_several_models_can_run_together(tmp_path):
    manager, store, created = make_manager(tmp_path)
    manager.start(add_video(store).id, ["fire_smoke", "boundary", "crowd"])
    assert created[0].camera.enabled_modules == frozenset({"fire_smoke", "boundary", "crowd"})


def test_loop_keeps_the_video_repeating(tmp_path):
    manager, store, created = make_manager(tmp_path)
    manager.start(add_video(store).id, ["boundary"], loop=True)
    assert created[0].once is False


def test_starting_a_second_session_stops_the_first(tmp_path):
    manager, store, created = make_manager(tmp_path)
    video = add_video(store)
    manager.start(video.id, ["boundary"])
    manager.start(video.id, ["fire_smoke"])
    assert created[0].stopped and not created[1].stopped
    assert manager.current().modules == ["fire_smoke"]


def test_stop_ends_the_session(tmp_path):
    manager, store, created = make_manager(tmp_path)
    manager.start(add_video(store).id, ["boundary"])
    manager.stop()
    assert created[0].stopped and manager.current() is None


@pytest.mark.parametrize(
    ("modules", "fragment"),
    [
        ([], "at least one"),
        (["teleportation"], "Unknown model"),
        (["ppe"], "unavailable"),  # no PPE model in make_manager's info
        (["phone"], "unavailable"),
    ],
)
def test_bad_model_choices_are_rejected_with_a_reason(tmp_path, modules, fragment):
    manager, store, _ = make_manager(tmp_path)
    with pytest.raises(SessionError, match=fragment):
        manager.start(add_video(store).id, modules)


def test_a_model_without_its_file_is_reported_unavailable_with_the_reason(tmp_path):
    none_installed = {"firesmoke": False, "ppe": False, "identity": False}
    manager, _, _ = make_manager(tmp_path, info=none_installed)
    by_id = {m["id"]: m for m in manager.modules()}
    assert not by_id["fire_smoke"]["available"] and "model" in by_id["fire_smoke"]["reason"]
    assert by_id["boundary"]["available"]
    assert by_id["identity"]["requires_boundary"] is True


def test_unknown_video_or_camera_is_rejected(tmp_path):
    manager, store, _ = make_manager(tmp_path)
    with pytest.raises(SessionError, match="no longer exists"):
        manager.start("f" * 32, ["boundary"])
    with pytest.raises(SessionError, match="does not exist"):
        manager.start(add_video(store).id, ["boundary"], zones_from="nope")


def test_finished_status_when_the_video_ends(tmp_path):
    manager, store, created = make_manager(tmp_path)
    session = manager.start(add_video(store).id, ["boundary"])
    created[0].capture_finished = True
    assert session.status == "finished"


def test_session_report_includes_stats_and_only_new_alerts(tmp_path):
    manager, store, created = make_manager(tmp_path)
    session = manager.start(add_video(store).id, ["boundary"])
    for i in range(3):
        created[0].sink.publish({"kind": "boundary", "subtype": "entry", "severity": "high",
                                 "message": f"alert {i}", "track_id": i})

    full = session.to_dict()
    assert full["alert_count"] == 3 and full["stats"]["inference_ms"] == 12.3
    assert [a["message"] for a in session.to_dict(since=2)["alerts"]] == ["alert 2"]


def test_alerts_stay_on_the_test_page_by_default(tmp_path):
    bus = RecordingBus()
    manager, store, created = make_manager(tmp_path, real_bus=bus)
    manager.start(add_video(store).id, ["boundary"], send_alerts=False)
    created[0].sink.publish({"kind": "fire", "severity": "high", "message": "Fire detected"})
    assert bus.published == []  # no history, no e-mail, no S3


def test_send_alerts_forwards_them_to_the_real_bus_clearly_labelled(tmp_path):
    bus = RecordingBus()
    manager, store, created = make_manager(tmp_path, real_bus=bus)
    manager.start(add_video(store).id, ["boundary"], send_alerts=True)
    created[0].sink.publish({"kind": "fire", "severity": "high", "message": "Fire detected"})
    assert bus.published[0]["message"] == TEST_PREFIX + "Fire detected"


def test_pipeline_for_finds_only_the_running_test_camera(tmp_path):
    manager, store, created = make_manager(tmp_path)
    session = manager.start(add_video(store).id, ["boundary"])
    assert manager.pipeline_for(session.camera_id) is created[0]
    assert manager.pipeline_for("cam_01") is None
    manager.stop()
    assert manager.pipeline_for(session.camera_id) is None


def test_zone_view_shows_one_cameras_zones_read_only():
    store = ZoneStore(mongomock.MongoClient()["t"]["zones"], refresh_interval=0)
    store.save([zone_from_dict({"id": "bay", "type": "polygon", "points": SQUARE})], "cam_01")

    assert [z.id for z in ZoneView(store, "cam_01").zones("test_abc")] == ["bay"]
    assert ZoneView(store, None).zones("test_abc") == []
    assert not hasattr(ZoneView(store, "cam_01"), "save")  # cannot edit the real zones


def test_session_alerts_sink_records_the_video_time():
    session = SimpleNamespace(started_at=time.time() - 5, alerts=[], lock=threading.Lock())
    sink = _SessionAlerts(session, None)
    sink.publish({"kind": "smoke", "severity": "high", "message": "Smoke detected"})
    assert 4.5 <= session.alerts[0].video_time_s <= 6


# -- fire/smoke with no boundary, on the REAL pipeline ------------------------------------------

FIRESMOKE_MODEL = Path("models/firesmoke/model.onnx")
PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")


@pytest.mark.skipif(
    not (FIRESMOKE_MODEL.is_file() and PERSON_MODEL.is_file()), reason="model files not present"
)
def test_fire_smoke_runs_without_the_boundary_module():
    """The request: fire/smoke must not depend on boundary. With only fire_smoke enabled no
    person detector is built, and a frame still flows through the fire/smoke path."""
    from perimeter.cameras.models import camera_from_dict
    from perimeter.capture.source import Frame
    from perimeter.pipeline import Pipeline

    camera = camera_from_dict(
        {"id": "test_x", "source_type": "webcam", "source": 0, "enabled_modules": ["fire_smoke"]}
    )
    zones = ZoneView(ZoneStore(mongomock.MongoClient()["t"]["zones"], refresh_interval=0), None)
    pipeline = Pipeline(Settings(), camera, zones, str(PERSON_MODEL),
                        firesmoke_model_path=str(FIRESMOKE_MODEL))

    assert pipeline.detector is None  # no YOLOX person model loaded
    assert pipeline.firesmoke_detector is not None

    frame = Frame(seq=3, ts=time.time(), image=np.zeros((360, 640, 3), np.uint8))
    pipeline._slot.publish(frame)
    pipeline._step()  # the real per-frame path

    assert pipeline.stats.errors == 0
    assert pipeline.stats.detections_run == 0  # person detection never ran
    assert pipeline.latest_jpeg() is not None  # and the annotated frame was still rendered


# -- the routes ---------------------------------------------------------------------------------


def build_app(tmp_path, *, enabled=True, max_bytes=10_000, with_operator=False, info=None):
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    zone_store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    if with_operator:
        users.create(User(id="op@test.local", email="op@test.local", role=Role.OPERATOR,
                          password_hash=hash_password("op-password", rounds=4)))
    settings = Settings()
    manager = PipelineManager(settings, registry, zone_store, "unused.onnx")
    store = make_store(tmp_path, max_bytes=max_bytes)
    sessions_manager, _, created = make_manager(tmp_path)
    sessions_manager.store = store
    installed = info or {"firesmoke": True, "ppe": False, "identity": True}
    sessions_manager._model_info = lambda: installed
    app = create_app(
        settings, zone_store, registry, manager, users,
        app_settings_store=AppSettingsStore(None, AppSettings(video_test_enabled=enabled)),
        test_video_store=store, test_sessions=sessions_manager,
    )
    return app, store, sessions_manager, created


def upload(client, name="clip.mp4", body=b"video-bytes"):
    return client.post("/api/test-video/upload", content=body, headers={"X-Filename": name})


def test_admin_can_upload_and_the_video_is_listed(tmp_path):
    client = authed_client(build_app(tmp_path)[0])
    resp = upload(client, "my%20video.mp4")
    assert resp.status_code == 200 and resp.json()["name"] == "my video.mp4"

    status = client.get("/api/test-video/status").json()
    assert [u["name"] for u in status["uploads"]] == ["my video.mp4"]
    assert {m["id"] for m in status["modules"]} >= {"boundary", "fire_smoke", "crowd"}
    assert status["max_upload_mb"] == 0  # 10_000 bytes rounds down; the real default is 2048


def test_bad_file_type_and_empty_uploads_are_rejected(tmp_path):
    client = authed_client(build_app(tmp_path)[0])
    assert upload(client, "notes.txt").status_code == 400
    assert upload(client, "clip.mp4", body=b"").status_code == 400


def test_an_oversized_upload_is_refused_and_nothing_is_left_behind(tmp_path):
    app, store, _, _ = build_app(tmp_path, max_bytes=50)
    client = authed_client(app)
    assert upload(client, "big.mp4", body=b"x" * 500).status_code == 413
    assert list(store.directory.iterdir()) == []


def test_full_flow_start_poll_and_stop(tmp_path):
    app, store, manager, created = build_app(tmp_path)
    client = authed_client(app)
    video_id = upload(client).json()["id"]

    started = client.post("/api/test-video/session", json={
        "video_id": video_id, "modules": ["fire_smoke"], "loop": False})
    assert started.status_code == 200 and started.json()["modules"] == ["fire_smoke"]

    created[0].sink.publish({"kind": "fire", "severity": "high", "message": "Fire detected"})
    polled = client.get("/api/test-video/session").json()["session"]
    assert polled["alert_count"] == 1 and polled["alerts"][0]["kind"] == "fire"
    assert client.get("/api/test-video/session?since=1").json()["session"]["alerts"] == []

    assert client.delete("/api/test-video/session").json() == {"stopped": True}
    assert client.get("/api/test-video/session").json() == {"session": None}


def test_starting_with_an_unavailable_model_gives_a_clear_400(tmp_path):
    app, *_ = build_app(tmp_path)
    client = authed_client(app)
    video_id = upload(client).json()["id"]
    resp = client.post("/api/test-video/session", json={"video_id": video_id, "modules": ["ppe"]})
    assert resp.status_code == 400 and "unavailable" in resp.json()["detail"]


def test_deleting_the_video_being_played_stops_the_test(tmp_path):
    app, store, manager, created = build_app(tmp_path)
    client = authed_client(app)
    video_id = upload(client).json()["id"]
    client.post("/api/test-video/session", json={"video_id": video_id, "modules": ["boundary"]})
    assert client.delete(f"/api/test-video/uploads/{video_id}").status_code == 200
    assert created[0].stopped and manager.current() is None


def test_the_stream_route_needs_a_running_test(tmp_path):
    client = authed_client(build_app(tmp_path)[0])
    assert client.get("/api/test-video/session/stream").status_code == 404


def test_operators_cannot_use_the_test_section(tmp_path):
    app = build_app(tmp_path, with_operator=True)[0]
    client = authed_client(app, "op@test.local", "op-password")
    assert client.get("/api/test-video/status").status_code == 403
    assert upload(client).status_code == 403
    assert client.get("/api/app-settings").status_code == 200  # but every role may read the flag
    assert client.put("/api/app-settings", json={"video_test_enabled": False}).status_code == 403


def test_switching_the_section_off_hides_every_route(tmp_path):
    client = authed_client(build_app(tmp_path, enabled=False)[0])
    assert client.get("/api/app-settings").json()["video_test_enabled"] is False
    assert client.get("/api/test-video/status").status_code == 404
    assert upload(client).status_code == 404
    assert client.post("/api/test-video/session", json={"modules": ["boundary"]}).status_code == 404


def test_the_setting_can_be_flipped_and_switching_off_stops_a_running_test(tmp_path):
    app, store, manager, created = build_app(tmp_path)
    client = authed_client(app)
    video_id = upload(client).json()["id"]
    client.post("/api/test-video/session", json={"video_id": video_id, "modules": ["boundary"]})

    assert client.put("/api/app-settings", json={"video_test_enabled": False}).status_code == 200
    assert created[0].stopped  # the running test ended with it
    assert client.get("/api/test-video/status").status_code == 404

    client.put("/api/app-settings", json={"video_test_enabled": True})
    assert client.get("/api/test-video/status").status_code == 200
    assert client.put("/api/app-settings", json={"video_test_enabled": "yes"}).status_code == 400


def test_app_settings_persist_in_the_database_and_fall_back_to_the_default():
    collection = mongomock.MongoClient()["t"]["app_settings"]
    default = AppSettings(video_test_enabled=True)
    store = AppSettingsStore(collection, default, refresh_interval=0)
    assert store.get().video_test_enabled is True  # nothing saved yet: the app.yaml default

    store.save(AppSettings(video_test_enabled=False))
    reloaded = AppSettingsStore(collection, default, refresh_interval=0)
    assert reloaded.get().video_test_enabled is False


def test_admin_email_constant_is_used():
    assert ADMIN_EMAIL  # keeps the shared import honest


# -- boundaries drawn on the test page ---------------------------------------------------------


def drawn_zone(zone_id="polygon_1"):
    return {
        "id": zone_id, "type": "polygon", "name": "Drawn", "points": SQUARE,
        "severity": "medium", "detect": ["person"], "events": ["entry", "exit"],
    }


def test_drawn_zones_are_used_by_the_session_and_win_over_a_camera(tmp_path):
    manager, store, created = make_manager(tmp_path)
    video = add_video(store)

    manager.start(video.id, ["boundary"], zones_from="cam_01", zones=[drawn_zone()])

    view = created[0].zone_view
    assert [z.id for z in view.zones()] == ["polygon_1"]
    assert view.version == 1


def test_an_invalid_drawn_zone_is_refused_with_a_reason(tmp_path):
    manager, store, created = make_manager(tmp_path)
    video = add_video(store)
    bad = {**drawn_zone(), "points": [[0.1, 0.1], [0.2, 0.2]]}

    with pytest.raises(SessionError, match="Invalid boundary"):
        manager.start(video.id, ["boundary"], zones=[bad])
    assert created == []


def test_frame_jpeg_returns_a_still_from_a_real_video(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "real.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(20):
        writer.write(np.full((120, 160, 3), i * 10, np.uint8))
    writer.release()
    store = TestVideoStore(tmp_path / "videos", 10_000_000, 7)
    video_id, dest = store.new_upload("real.mp4")
    dest.write_bytes(path.read_bytes())
    video = store.finalize(video_id, dest, "real.mp4")

    jpeg = store.frame_jpeg(video.id)

    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
    assert store.frame_jpeg("0" * 32) is None


def test_frame_endpoint_is_admin_only(tmp_path):
    app = build_app(tmp_path)[0]
    client = authed_client(app)
    video_id = upload(client).json()["id"]

    assert client.get(f"/api/test-video/uploads/{'0' * 32}/frame").status_code == 404
    anonymous = TestClient(app)
    assert anonymous.get(f"/api/test-video/uploads/{video_id}/frame").status_code in (401, 403)
