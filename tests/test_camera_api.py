"""Tests for the Camera settings endpoints.

The security-relevant behaviour is that the RTSP URL is write-only from the browser's
point of view: the dashboard has no authentication, so a readable password field would
hand the camera to anyone who reached the page.
"""

from __future__ import annotations

import os
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from fsbd.api.app import _camera_from_payload, create_app
from fsbd.boundary.zones import ZoneStore
from fsbd.settings import CameraSettings, Settings

RTSP = "rtsp://admin:hunter2@10.0.0.5:554/Streaming/Channels/101"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("FSBD_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def client(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump({"cameras": {"cam_01": {"zones": []}}}), encoding="utf-8")
    store = ZoneStore(path, camera_id="cam_01")
    settings = Settings(
        camera=CameraSettings(use_cctv=True, cctv_rtsp_url=RTSP),
        # Saves write back to the file the settings came from, so the test never
        # touches the developer's real .env.
        env_file=tmp_path / ".env",
    )
    return TestClient(create_app(settings, store, None))


# -- GET /api/camera ------------------------------------------------------


def test_get_camera_reports_the_switch_and_dimensions(client):
    body = client.get("/api/camera").json()

    assert body["use_cctv"] is True
    assert body["width"] == 1280
    assert body["height"] == 720
    assert body["fps"] == 30


def test_get_camera_never_returns_the_password(client):
    """The single most important assertion in this file."""
    body = client.get("/api/camera").json()
    serialised = str(body)

    assert "hunter2" not in serialised
    assert body["rtsp_url_set"] is True
    assert "10.0.0.5" in body["rtsp_url_redacted"]
    assert "rtsp_url" not in body, "the raw field must not be exposed at all"


def test_get_camera_reports_derived_detection_rate(client):
    body = client.get("/api/camera").json()
    assert body["person_fps"] == pytest.approx(6.0)


# -- payload parsing ------------------------------------------------------


def test_absent_rtsp_url_keeps_the_stored_one():
    """The browser is never sent the URL, so it cannot echo it back on save."""
    current = CameraSettings(use_cctv=True, cctv_rtsp_url=RTSP)
    updated = _camera_from_payload({"use_cctv": True, "width": 640}, current)

    assert updated.cctv_rtsp_url == RTSP
    assert updated.width == 640


def test_empty_rtsp_url_also_keeps_the_stored_one():
    current = CameraSettings(use_cctv=True, cctv_rtsp_url=RTSP)
    updated = _camera_from_payload({"use_cctv": True, "rtsp_url": ""}, current)
    assert updated.cctv_rtsp_url == RTSP


def test_a_new_rtsp_url_replaces_the_stored_one():
    current = CameraSettings(use_cctv=True, cctv_rtsp_url=RTSP)
    updated = _camera_from_payload({"use_cctv": True, "rtsp_url": "rtsp://new@1.2.3.4/s"}, current)
    assert updated.cctv_rtsp_url == "rtsp://new@1.2.3.4/s"


def test_clear_flag_erases_the_stored_url():
    current = CameraSettings(use_cctv=False, cctv_rtsp_url=RTSP)
    updated = _camera_from_payload({"use_cctv": False, "clear_rtsp_url": True}, current)
    assert updated.cctv_rtsp_url == ""


def test_selecting_cctv_without_any_url_is_rejected():
    current = CameraSettings(use_cctv=False, cctv_rtsp_url="")
    with pytest.raises(ValueError, match="no RTSP URL"):
        _camera_from_payload({"use_cctv": True}, current)


def test_unspecified_fields_keep_their_current_values():
    current = CameraSettings(id="bay", width=1920, height=1080, fps=25, decode_fps=8.0)
    updated = _camera_from_payload({"use_cctv": False}, current)

    assert (updated.id, updated.width, updated.height) == ("bay", 1920, 1080)
    assert updated.fps == 25
    assert updated.decode_fps == 8.0


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"width": 10}, "width"),
        ({"height": 99999}, "height"),
        ({"fps": 0}, "fps"),
        ({"decode_fps": 500}, "decode_fps"),
        ({"width": "wide"}, "width"),
    ],
)
def test_out_of_range_and_non_numeric_values_are_rejected(payload, match):
    with pytest.raises(ValueError, match=match):
        _camera_from_payload({"use_cctv": False, **payload}, CameraSettings())


def test_blank_source_falls_back_to_device_zero():
    updated = _camera_from_payload({"use_cctv": False, "source": "  "}, CameraSettings(source="3"))
    assert updated.source == "0"


# -- PUT /api/camera ------------------------------------------------------


def test_put_persists_and_is_reflected_in_get(client, tmp_path):
    response = client.put(
        "/api/camera",
        json={"use_cctv": False, "source": "0", "id": "bay_cam", "width": 640, "height": 360},
    )
    assert response.status_code == 200

    body = client.get("/api/camera").json()
    assert body["use_cctv"] is False
    assert body["id"] == "bay_cam"
    assert body["width"] == 640


def test_put_rejects_cctv_without_a_url(client):
    client.put("/api/camera", json={"use_cctv": False, "clear_rtsp_url": True})
    response = client.put("/api/camera", json={"use_cctv": True})

    assert response.status_code == 400
    assert "RTSP" in response.json()["detail"]


def test_put_rejects_a_bad_resolution(client):
    response = client.put("/api/camera", json={"use_cctv": False, "width": 3})
    assert response.status_code == 400


def test_put_response_is_redacted(client):
    response = client.put("/api/camera", json={"use_cctv": True, "rtsp_url": RTSP})
    assert "hunter2" not in str(response.json())


# -- POST /api/camera/test -----------------------------------------------
#
# Ordering matters here. Abandoning a probe does not stop the FFmpeg call underneath it,
# and a stuck FFmpeg open measurably delays the NEXT open in the same process - roughly
# 30 s, its own internal default. The file-based success test therefore runs before the
# unreachable-host tests, which would otherwise inflate it from under a second to nearly
# a minute. The same effect is visible in production: after testing an unreachable
# camera, the next Test click can be slow, which is why concurrent probes are capped.


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
        "/api/camera/test",
        json={"use_cctv": False, "source": str(clip), "width": 1280, "height": 720},
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
    from fsbd.capture.probe import probe_source

    started = time.perf_counter()
    result = probe_source("rtsp://10.255.255.1:554/nope", timeout_s=1.0)
    elapsed = time.perf_counter() - started

    assert result["ok"] is False
    assert elapsed < 5.0, "the probe must not wait for FFmpeg's own 30s default"


def test_probe_timeout_message_redacts_credentials():
    from fsbd.capture.probe import probe_source

    result = probe_source("rtsp://admin:hunter2@10.255.255.2:554/nope", timeout_s=1.0)

    assert "hunter2" not in str(result)
    assert "10.255.255.2" in result["error"]


def test_concurrent_probes_are_capped():
    """Repeated clicks on Test must not pile up threads stuck inside FFmpeg."""
    from fsbd.capture import probe as probe_module

    results = [
        probe_module.probe_source(f"rtsp://10.255.255.{i}:554/nope", timeout_s=0.5)
        for i in range(3, 3 + probe_module.MAX_CONCURRENT_PROBES + 1)
    ]

    assert any("already running" in (r.get("error") or "") for r in results)


def test_test_endpoint_reports_a_configuration_error_without_opening_anything(client):
    client.put("/api/camera", json={"use_cctv": False, "clear_rtsp_url": True})
    body = client.post("/api/camera/test", json={"use_cctv": True}).json()

    assert body["ok"] is False
    assert "RTSP" in body["error"]
