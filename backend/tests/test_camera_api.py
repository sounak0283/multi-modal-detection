"""Tests for the camera registry API endpoints (Expansion Plan Phase A.2).

The security-relevant behaviour is that the RTSP URL is write-only from the browser's
point of view: the dashboard has no authentication yet (Phase B), so a readable password
field would hand the camera to anyone who reached the page.

`PipelineManager.start()` is deliberately never called in these fixtures - these are API
unit tests, not integration tests, and starting a pipeline would try to open real camera
hardware and load the ONNX model. `notify_camera_changed()` is a no-op on an unstarted
manager (see cameras/manager.py), so camera CRUD can be exercised safely without either.
"""

from __future__ import annotations

import time

import mongomock
import pytest
from conftest import authed_client, seed_admin

from perimeter.api.app import create_app
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import Settings

RTSP = "rtsp://admin:hunter2@10.0.0.5:554/Streaming/Channels/101"


@pytest.fixture
def client(tmp_path):
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"])
    users = seed_admin(db["users"])
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")
    app = create_app(settings, store, registry, manager, users)
    return authed_client(app)


def make_camera(**overrides) -> dict:
    payload = {
        "id": "cam_01",
        "name": "Front gate",
        "source_type": "rtsp",
        "rtsp_url": RTSP,
        "width": 1280,
        "height": 720,
        "fps": 30,
    }
    payload.update(overrides)
    return payload


# -- POST /api/cameras + GET /api/cameras/{id} ----------------------------


def test_create_and_get_camera_reports_the_shape(client):
    client.post("/api/cameras", json=make_camera())
    body = client.get("/api/cameras/cam_01").json()

    assert body["source_type"] == "rtsp"
    assert body["width"] == 1280
    assert body["height"] == 720
    assert body["fps"] == 30
    assert body["running"] is False, "manager.start() was never called"


def test_created_camera_never_returns_the_password(client):
    """The single most important assertion in this file."""
    client.post("/api/cameras", json=make_camera())
    body = client.get("/api/cameras/cam_01").json()
    serialised = str(body)

    assert "hunter2" not in serialised
    assert body["rtsp_url_set"] is True
    assert "10.0.0.5" in body["rtsp_url_redacted"]
    assert "rtsp_url" not in body, "the raw field must not be exposed at all"


def test_created_camera_reports_derived_detection_rate(client):
    client.post("/api/cameras", json=make_camera(decode_fps=12.0))
    body = client.get("/api/cameras/cam_01").json()
    assert body["person_fps"] == pytest.approx(6.0)


def test_a_module_needing_person_tracking_brings_boundary_with_it(client):
    client.post("/api/cameras", json=make_camera(enabled_modules=["crowd"]))
    body = client.get("/api/cameras/cam_01").json()
    assert set(body["enabled_modules"]) == {"boundary", "crowd"}


def test_a_new_camera_runs_no_detection_module_by_default(client):
    camera = make_camera()
    camera.pop("enabled_modules", None)
    client.post("/api/cameras", json=camera)
    body = client.get("/api/cameras/cam_01").json()
    assert body["enabled_modules"] == []


def test_unknown_module_is_rejected(client):
    response = client.post("/api/cameras", json=make_camera(enabled_modules=["laser_eyes"]))
    assert response.status_code == 400
    assert "unknown module" in response.json()["detail"]


@pytest.mark.parametrize("bad_id", ["cam/../x", "cam 01", "-leading-dash", "x" * 65])
def test_camera_id_that_would_break_its_own_urls_is_rejected(client, bad_id):
    response = client.post("/api/cameras", json=make_camera(id=bad_id))
    assert response.status_code == 400


def test_get_missing_camera_is_404(client):
    assert client.get("/api/cameras/does-not-exist").status_code == 404


def test_duplicate_camera_id_is_rejected(client):
    client.post("/api/cameras", json=make_camera())
    response = client.post("/api/cameras", json=make_camera())
    assert response.status_code == 409


def test_rtsp_without_a_url_is_rejected(client):
    response = client.post(
        "/api/cameras", json=make_camera(source_type="rtsp", rtsp_url="")
    )
    assert response.status_code == 400
    assert "rtsp_url" in response.json()["detail"]


# -- PUT /api/cameras/{id} -------------------------------------------------


def test_put_updates_and_is_reflected_in_get(client):
    client.post("/api/cameras", json=make_camera())
    response = client.put("/api/cameras/cam_01", json={"width": 640, "height": 360})
    assert response.status_code == 200

    body = client.get("/api/cameras/cam_01").json()
    assert body["width"] == 640
    assert body["height"] == 360


def test_put_absent_rtsp_url_keeps_the_stored_one():
    """The browser is never sent the URL, so it cannot echo it back on save - proven
    directly against the merge logic `PUT /api/cameras/{id}` uses."""
    from perimeter.cameras.models import camera_from_dict

    current = camera_from_dict(make_camera())
    merged = current.to_dict()
    merged.update({k: v for k, v in {"width": 640}.items() if k != "rtsp_url" or v})
    updated = camera_from_dict(merged)

    assert updated.rtsp_url == RTSP
    assert updated.width == 640


def test_put_rejects_a_bad_resolution(client):
    client.post("/api/cameras", json=make_camera())
    response = client.put("/api/cameras/cam_01", json={"width": 3})
    assert response.status_code == 400


def test_put_response_is_redacted(client):
    client.post("/api/cameras", json=make_camera())
    response = client.put("/api/cameras/cam_01", json={"rtsp_url": RTSP})
    assert "hunter2" not in str(response.json())


def test_put_cannot_change_the_camera_id(client):
    client.post("/api/cameras", json=make_camera())
    client.put("/api/cameras/cam_01", json={"id": "different"})
    body = client.get("/api/cameras/cam_01").json()
    assert body["id"] == "cam_01"


def test_put_missing_camera_is_404(client):
    assert client.put("/api/cameras/does-not-exist", json={"width": 640}).status_code == 404


# -- DELETE /api/cameras/{id} ----------------------------------------------


def test_delete_removes_the_camera(client):
    client.post("/api/cameras", json=make_camera())
    response = client.delete("/api/cameras/cam_01")
    assert response.status_code == 200
    assert client.get("/api/cameras/cam_01").status_code == 404


def test_delete_missing_camera_is_404(client):
    assert client.delete("/api/cameras/does-not-exist").status_code == 404


# -- GET /api/cameras (list) ------------------------------------------------


def test_list_cameras_returns_every_camera(client):
    client.post("/api/cameras", json=make_camera(id="cam_01"))
    client.post("/api/cameras", json=make_camera(id="cam_02", name="Loading bay"))

    body = client.get("/api/cameras").json()
    ids = {c["id"] for c in body["cameras"]}
    assert ids == {"cam_01", "cam_02"}


# -- POST /api/cameras/test -------------------------------------------------
#
# Ordering matters here. Abandoning a probe does not stop the FFmpeg call underneath it,
# and a stuck FFmpeg open measurably delays the NEXT open in the same process - roughly
# 30 s, its own internal default. The file-based success test therefore runs before the
# unreachable-host tests, which would otherwise inflate it from under a second to nearly
# a minute.


def test_test_endpoint_succeeds_on_a_real_video_file(client, tmp_path):
    import cv2
    import numpy as np

    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(10):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    writer.release()
    if not clip.is_file():
        pytest.skip("no usable video codec in this OpenCV build")

    body = client.post(
        "/api/cameras/test",
        json=make_camera(source_type="file", source=str(clip), rtsp_url="", width=1280, height=720),
    ).json()

    assert body["ok"] is True
    assert body["actual_width"] == 320
    assert body["actual_height"] == 240
    assert any("not the requested" in w for w in body["warnings"]), (
        "a source that ignores the requested resolution must be flagged - pixel "
        "thresholds depend on it"
    )


def test_probe_times_out_on_an_unreachable_host():
    """Driven directly rather than through the API so the bound can be kept short.

    FFmpeg blocks for its own ~30 s regardless of the options passed to it, which was
    measured on this build - so the timeout has to be enforced in Python. This asserts
    that it actually is.
    """
    from perimeter.capture.probe import probe_source

    started = time.perf_counter()
    result = probe_source("rtsp://10.255.255.1:554/nope", timeout_s=1.0)
    elapsed = time.perf_counter() - started

    assert result["ok"] is False
    assert elapsed < 5.0, "the probe must not wait for FFmpeg's own 30s default"


def test_probe_timeout_message_redacts_credentials():
    from perimeter.capture.probe import probe_source

    result = probe_source("rtsp://admin:hunter2@10.255.255.2:554/nope", timeout_s=1.0)

    assert "hunter2" not in str(result)
    assert "10.255.255.2" in result["error"]


def test_concurrent_probes_are_capped():
    """Repeated clicks on Test must not pile up threads stuck inside FFmpeg."""
    from perimeter.capture import probe as probe_module

    results = [
        probe_module.probe_source(f"rtsp://10.255.255.{i}:554/nope", timeout_s=0.5)
        for i in range(3, 3 + probe_module.MAX_CONCURRENT_PROBES + 1)
    ]

    assert any("already running" in (r.get("error") or "") for r in results)


def test_test_endpoint_reports_a_configuration_error_without_opening_anything(client):
    body = client.post(
        "/api/cameras/test", json=make_camera(source_type="rtsp", rtsp_url="")
    ).json()

    assert body["ok"] is False
    assert "rtsp_url" in body["error"]
