"""Tests for the dashboard API (PLAN.md section 6.6).

Runs without a camera: the pipeline is optional, so the zone-editing half of the API is
testable on its own. Endpoints that need frames are expected to return 503, not crash.
"""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from fsbd.api.app import create_app
from fsbd.boundary.zones import ZoneStore
from fsbd.settings import Settings

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


@pytest.fixture
def client(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump({"cameras": {"cam_01": {"zones": []}}}), encoding="utf-8")
    store = ZoneStore(path, camera_id="cam_01")
    return TestClient(create_app(Settings(), store, None))


def polygon(**overrides):
    return {"id": "bay", "type": "polygon", "points": SQUARE, **overrides}


# -- zones ----------------------------------------------------------------


def test_get_zones_starts_empty(client):
    body = client.get("/api/zones").json()
    assert body["zones"] == []
    assert body["camera_id"] == "cam_01"


def test_put_and_get_round_trip(client):
    response = client.put("/api/zones", json={"zones": [polygon(name="Loading Bay")]})
    assert response.status_code == 200
    assert response.json()["saved"] == 1

    zones = client.get("/api/zones").json()["zones"]
    assert zones[0]["name"] == "Loading Bay"


def test_put_rejects_a_self_intersecting_polygon_with_a_usable_message(client):
    """The editor shows this verbatim, so it must name the zone and the problem."""
    bowtie = polygon(points=[[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]])
    response = client.put("/api/zones", json={"zones": [bowtie]})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "bay" in detail and "self-intersecting" in detail


def test_put_rejects_duplicate_ids(client):
    response = client.put("/api/zones", json={"zones": [polygon(), polygon()]})
    assert response.status_code == 400
    assert "duplicate" in response.json()["detail"]


def test_put_rejects_a_malformed_body(client):
    assert client.put("/api/zones", json={"nope": []}).status_code == 400


def test_put_rejects_exclusion_without_applies_to(client):
    response = client.put(
        "/api/zones", json={"zones": [{"id": "x", "type": "exclusion", "points": SQUARE}]}
    )
    assert response.status_code == 400
    assert "applies_to" in response.json()["detail"]


def test_saving_bumps_the_version(client):
    before = client.get("/api/zones").json()["version"]
    client.put("/api/zones", json={"zones": [polygon()]})
    assert client.get("/api/zones").json()["version"] > before


# -- validation endpoint --------------------------------------------------


def test_validate_accepts_a_good_polygon(client):
    body = client.post("/api/zones/validate", json={"type": "polygon", "points": SQUARE}).json()
    assert body["valid"] is True
    assert body["errors"] == []


def test_validate_reports_errors_without_saving(client):
    body = client.post(
        "/api/zones/validate", json={"type": "polygon", "points": [[0.1, 0.1], [0.2, 0.2]]}
    ).json()

    assert body["valid"] is False
    assert client.get("/api/zones").json()["zones"] == [], "validation must not persist"


def test_validate_warns_about_a_full_frame_zone(client):
    body = client.post(
        "/api/zones/validate",
        json={"type": "polygon", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
    ).json()

    assert body["valid"] is True
    assert body["warnings"], "a zone covering the whole frame should be flagged"


def test_validate_rejects_an_unknown_type(client):
    assert client.post(
        "/api/zones/validate", json={"type": "hexagon", "points": SQUARE}
    ).status_code == 400


def test_validate_tripwire_length(client):
    body = client.post(
        "/api/zones/validate", json={"type": "tripwire", "points": [[0.5, 0.5], [0.51, 0.5]]}
    ).json()
    assert body["valid"] is False


# -- video endpoints without a pipeline -----------------------------------


def test_snapshot_without_a_pipeline_is_503(client):
    assert client.get("/api/snapshot").status_code == 503


def test_events_without_a_pipeline_is_503(client):
    assert client.get("/api/events").status_code == 503


# -- health ---------------------------------------------------------------


def test_health_reports_zone_count(client):
    client.put("/api/zones", json={"zones": [polygon()]})
    body = client.get("/api/health").json()

    assert body["zones"] == 1
    assert body["pipeline"] is False


def test_health_redacts_camera_credentials():
    """This endpoint ends up in screenshots and support bundles."""
    from fsbd.settings import CameraSettings

    settings = Settings(camera=CameraSettings(source="rtsp://admin:hunter2@10.0.0.5:554/s1"))
    store = ZoneStore("does-not-exist.yaml", camera_id="cam_01")
    client = TestClient(create_app(settings, store, None))

    source = client.get("/api/health").json()["source"]

    assert "hunter2" not in source
    assert "10.0.0.5" in source


# -- static files ---------------------------------------------------------


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Boundary Zones" in response.text


def test_editor_assets_are_served(client):
    assert client.get("/app.js").status_code == 200
    assert client.get("/app.css").status_code == 200


def test_path_traversal_is_blocked(client):
    assert client.get("/../../secrets.js").status_code in (404, 400)
