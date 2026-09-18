"""Tests for the dashboard API (Expansion Plan Phase A; PLAN.md section 6.6).

Runs without a running pipeline: `PipelineManager.start()` is never called, so the
zone-editing half of the API is testable on its own. Endpoints that need frames are
expected to return 503, not crash.
"""

from __future__ import annotations

import mongomock
import pytest
from conftest import authed_client, seed_admin

from perimeter.api.app import DIST_DIR, create_app
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.models import camera_from_dict
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import Settings

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
CAMERA_ID = "cam_01"


def build_app(settings: Settings | None = None, database=None):
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    registry.create(camera_from_dict({"id": CAMERA_ID, "source_type": "webcam", "source": 0}))
    store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    settings = settings or Settings()
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")
    app = create_app(settings, store, registry, manager, users, database=database)
    return app, registry, store


@pytest.fixture
def client():
    app, _registry, _store = build_app()
    return authed_client(app)


def polygon(**overrides):
    return {"id": "bay", "type": "polygon", "points": SQUARE, **overrides}


def zones_url(path: str = "") -> str:
    return f"/api/cameras/{CAMERA_ID}/zones{path}"


# -- zones ------------------------------------------------------------------


def test_get_zones_starts_empty(client):
    body = client.get(zones_url()).json()
    assert body["zones"] == []
    assert body["camera_id"] == CAMERA_ID


def test_put_and_get_round_trip(client):
    response = client.put(zones_url(), json={"zones": [polygon(name="Loading Bay")]})
    assert response.status_code == 200
    assert response.json()["saved"] == 1

    zones = client.get(zones_url()).json()["zones"]
    assert zones[0]["name"] == "Loading Bay"


def test_put_rejects_a_self_intersecting_polygon_with_a_usable_message(client):
    """The editor shows this verbatim, so it must name the zone and the problem."""
    bowtie = polygon(points=[[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]])
    response = client.put(zones_url(), json={"zones": [bowtie]})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "bay" in detail and "self-intersecting" in detail


def test_put_rejects_duplicate_ids(client):
    response = client.put(zones_url(), json={"zones": [polygon(), polygon()]})
    assert response.status_code == 400
    assert "duplicate" in response.json()["detail"]


def test_put_rejects_a_malformed_body(client):
    assert client.put(zones_url(), json={"nope": []}).status_code == 400


def test_put_rejects_exclusion_without_applies_to(client):
    response = client.put(
        zones_url(), json={"zones": [{"id": "x", "type": "exclusion", "points": SQUARE}]}
    )
    assert response.status_code == 400
    assert "applies_to" in response.json()["detail"]


def test_put_for_an_unknown_camera_is_404(client):
    response = client.put("/api/cameras/does-not-exist/zones", json={"zones": []})
    assert response.status_code == 404


def test_saving_bumps_the_version(client):
    before = client.get(zones_url()).json()["version"]
    client.put(zones_url(), json={"zones": [polygon()]})
    assert client.get(zones_url()).json()["version"] > before


# -- validation endpoint (camera-agnostic) -----------------------------------


def test_validate_accepts_a_good_polygon(client):
    body = client.post("/api/zones/validate", json={"type": "polygon", "points": SQUARE}).json()
    assert body["valid"] is True
    assert body["errors"] == []


def test_validate_reports_errors_without_saving(client):
    body = client.post(
        "/api/zones/validate", json={"type": "polygon", "points": [[0.1, 0.1], [0.2, 0.2]]}
    ).json()

    assert body["valid"] is False
    assert client.get(zones_url()).json()["zones"] == [], "validation must not persist"


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


@pytest.mark.parametrize("points", ["garbage", [["a", "b"], ["c", "d"], ["e", "f"]], [1, 2, 3]])
def test_validate_malformed_points_is_invalid_not_a_500(client, points):
    response = client.post("/api/zones/validate", json={"type": "polygon", "points": points})
    assert response.status_code == 200
    assert response.json()["valid"] is False


# -- video/event endpoints without a running pipeline/database --------------


def test_snapshot_without_a_pipeline_is_503(client):
    assert client.get(f"/api/cameras/{CAMERA_ID}/snapshot").status_code == 503


def test_snapshot_for_an_unknown_camera_is_404(client):
    assert client.get("/api/cameras/does-not-exist/snapshot").status_code == 404


def test_events_without_a_database_is_503(client):
    assert client.get("/api/events").status_code == 503


def test_events_live_without_an_alert_bus_is_empty(client):
    assert client.get("/api/events/live").json() == {"events": []}


# -- health -------------------------------------------------------------


def test_health_reports_cameras_and_their_running_state(client):
    body = client.get("/api/health").json()

    assert body["database"]["configured"] is False
    assert len(body["cameras"]) == 1
    assert body["cameras"][0]["id"] == CAMERA_ID
    assert body["cameras"][0]["running"] is False


def test_camera_detail_redacts_credentials():
    """This endpoint's `resolved` field ends up in screenshots and support bundles."""
    app, registry, _store = build_app()
    registry.update(
        CAMERA_ID,
        camera_from_dict(
            {
                "id": CAMERA_ID,
                "source_type": "rtsp",
                "rtsp_url": "rtsp://admin:hunter2@10.0.0.5:554/s1",
            }
        ),
    )
    client = authed_client(app)

    body = client.get(f"/api/cameras/{CAMERA_ID}").json()

    assert "hunter2" not in str(body)
    assert "10.0.0.5" in body["resolved"]


# -- static files ---------------------------------------------------------


BUILT = (DIST_DIR / "index.html").is_file()
needs_build = pytest.mark.skipif(BUILT is False, reason="frontend not built (npm run build)")


@needs_build
def test_spa_shell_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="root"' in response.text, "the React mount point should be present"


def test_missing_build_says_so_rather_than_showing_a_blank_page(client):
    """A white screen looks like a crashed backend. If the bundle is absent the server
    must say which command to run."""
    if BUILT:
        pytest.skip("frontend is built")
    response = client.get("/")
    assert response.status_code == 503
    assert "npm run build" in response.text


def test_unknown_api_paths_404_rather_than_returning_the_spa(client):
    """The SPA catch-all must not shadow the API. Otherwise a mistyped endpoint returns
    HTML with a 200 and the client fails on JSON parsing instead of a clean 404."""
    response = client.get("/api/definitely-not-an-endpoint")
    assert response.status_code == 404
    assert "<html" not in response.text.lower()


@needs_build
def test_traversal_cannot_read_files_outside_the_bundle(client):
    """An SPA answers unknown paths with index.html, which is correct - what must never
    happen is a file from outside web/dist coming back."""
    for attack in ("/../NOTICE.md", "/..%2fNOTICE.md", "/assets/../../NOTICE.md"):
        response = client.get(attack)
        assert response.status_code in (200, 404)
        assert "Third-party models" not in response.text, f"{attack} escaped the bundle"
