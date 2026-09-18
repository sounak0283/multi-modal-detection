"""Tests for GET /api/events/{id}/snapshot and /clip (Expansion Plan Phase C)."""

from __future__ import annotations

import mongomock
import pytest
from conftest import authed_client, seed_admin

from perimeter.api.app import create_app
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.evidence.store import LocalEvidenceStore
from perimeter.settings import Settings

EVENT_ID = "68f0000000000000000000aa"


class FakeDatabase:
    """Only what the evidence routes actually call - `get_event`."""

    def __init__(self, events: dict):
        self.events = events

    def get_event(self, event_id: str):
        return self.events.get(event_id)


def build(tmp_path, events: dict, database=..., evidence_store=...):
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")

    if database is ...:
        database = FakeDatabase(events)
    if evidence_store is ...:
        evidence_store = LocalEvidenceStore(
            root=tmp_path / "evidence", snapshot_dir="snapshots", clip_dir="clips"
        )

    app = create_app(
        settings, store, registry, manager, users, database=database, evidence_store=evidence_store
    )
    return authed_client(app), evidence_store


@pytest.fixture
def client_and_store(tmp_path):
    snapshot_key = "snapshots/cam_01/2026-01-01/" + EVENT_ID + ".jpg"
    clip_key = "clips/cam_01/2026-01-01/" + EVENT_ID + ".mp4"
    events = {
        EVENT_ID: {"id": EVENT_ID, "snapshot_path": snapshot_key, "clip_path": clip_key},
        "no-evidence": {"id": "no-evidence", "snapshot_path": None, "clip_path": None},
    }
    client, store = build(tmp_path, events)

    (store.root / snapshot_key).parent.mkdir(parents=True, exist_ok=True)
    (store.root / snapshot_key).write_bytes(b"\xff\xd8fake-jpeg")
    (store.root / clip_key).parent.mkdir(parents=True, exist_ok=True)
    (store.root / clip_key).write_bytes(b"fake-mp4-bytes")

    return client, store


# -- happy path -----------------------------------------------------------------


def test_snapshot_is_served_with_the_right_content_type(client_and_store):
    client, _store = client_and_store
    response = client.get(f"/api/events/{EVENT_ID}/snapshot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"\xff\xd8fake-jpeg"


def test_clip_is_served_with_the_right_content_type(client_and_store):
    client, _store = client_and_store
    response = client.get(f"/api/events/{EVENT_ID}/clip")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"fake-mp4-bytes"


def test_an_avi_fallback_clip_is_served_with_its_own_content_type(tmp_path):
    key = "clips/cam_01/2026-01-01/evt.avi"
    events = {"evt": {"id": "evt", "snapshot_path": None, "clip_path": key}}
    client, store = build(tmp_path, events)
    (store.root / key).parent.mkdir(parents=True, exist_ok=True)
    (store.root / key).write_bytes(b"fake-avi-bytes")

    response = client.get("/api/events/evt/clip")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/x-msvideo"


# -- not found / not configured --------------------------------------------------


def test_unknown_event_is_404(client_and_store):
    client, _store = client_and_store
    assert client.get("/api/events/does-not-exist/snapshot").status_code == 404
    assert client.get("/api/events/does-not-exist/clip").status_code == 404


def test_event_without_evidence_is_404(client_and_store):
    client, _store = client_and_store
    assert client.get("/api/events/no-evidence/snapshot").status_code == 404
    assert client.get("/api/events/no-evidence/clip").status_code == 404


def test_a_stored_key_whose_file_is_missing_is_404_not_500(tmp_path):
    key = "snapshots/cam_01/2026-01-01/evt.jpg"  # never actually written to disk
    events = {"evt": {"id": "evt", "snapshot_path": key, "clip_path": None}}
    client, _store = build(tmp_path, events)

    assert client.get("/api/events/evt/snapshot").status_code == 404


def test_without_a_database_configured_is_503(tmp_path):
    client, _store = build(tmp_path, events={}, database=None)
    assert client.get(f"/api/events/{EVENT_ID}/snapshot").status_code == 503


def test_without_evidence_storage_configured_is_503(tmp_path):
    client, _store = build(tmp_path, events={EVENT_ID: {"id": EVENT_ID}}, evidence_store=None)
    assert client.get(f"/api/events/{EVENT_ID}/snapshot").status_code == 503


# -- roles ------------------------------------------------------------------------


def test_operator_can_read_evidence(tmp_path):
    from perimeter.auth.models import Role, User
    from perimeter.auth.passwords import hash_password

    snapshot_key = "snapshots/cam_01/2026-01-01/evt.jpg"
    events = {"evt": {"id": "evt", "snapshot_path": snapshot_key, "clip_path": None}}
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    users.create(
        User(
            id="operator@test.local",
            email="operator@test.local",
            password_hash=hash_password("operator-password", rounds=4),
            role=Role.OPERATOR,
        )
    )
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")
    evidence_store = LocalEvidenceStore(
        root=tmp_path / "evidence", snapshot_dir="snapshots", clip_dir="clips"
    )
    (evidence_store.root / snapshot_key).parent.mkdir(parents=True, exist_ok=True)
    (evidence_store.root / snapshot_key).write_bytes(b"jpeg")
    app = create_app(
        settings, store, registry, manager, users,
        database=FakeDatabase(events), evidence_store=evidence_store,
    )
    client = authed_client(app, "operator@test.local", "operator-password")

    assert client.get("/api/events/evt/snapshot").status_code == 200
