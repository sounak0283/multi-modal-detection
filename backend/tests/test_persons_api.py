"""Tests for /api/persons (Expansion Plan Phase F).

`database` here is a minimal in-memory fake mirroring `store.db.Database`'s persons
methods - the same "only what the routes actually call" pattern `test_evidence_api.py`
uses, rather than standing up a real MongoDB-backed `Database`.
"""

from __future__ import annotations

import base64

import cv2
import mongomock
import numpy as np
from conftest import authed_client, seed_admin

from perimeter.api.app import create_app
from perimeter.auth.models import Role, User
from perimeter.auth.passwords import hash_password
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.identity.gallery import FaceGallery
from perimeter.identity.resolver import IdentityResolver
from perimeter.identity.yunet import FaceBox
from perimeter.settings import Settings

OPERATOR_EMAIL = "operator@test.local"
OPERATOR_PASSWORD = "operator-password"


class FakeDatabase:
    """Only what the persons routes actually call."""

    def __init__(self):
        self._persons: dict[str, dict] = {}

    def insert_person(self, person):
        self._persons[person["id"]] = {**person, "created_at": "now"}
        return person["id"]

    def list_persons(self, active_only: bool = False):
        values = list(self._persons.values())
        return [p for p in values if p["active"]] if active_only else values

    def get_person(self, person_id):
        return self._persons.get(person_id)

    def update_person(self, person_id, updates):
        if person_id in self._persons:
            self._persons[person_id].update(updates)

    def delete_person(self, person_id):
        self._persons.pop(person_id, None)


class FakeDetector:
    def __init__(self, found=True):
        self._found = found

    def largest(self, image_bgr):
        if not self._found:
            return None
        raw = np.zeros(15, dtype=np.float32)
        raw[2] = raw[3] = 40
        return FaceBox(raw=raw, score=0.9)


class FakeEmbedder:
    def embed(self, image_bgr, face):
        return np.ones(128, dtype=np.float32)


def jpeg_base64() -> str:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def all_poses() -> dict[str, str]:
    photo = jpeg_base64()
    return {"front": photo, "left": photo, "right": photo, "up": photo, "down": photo}


def build(tmp_path, identity_resolver=..., database=...):
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    users.create(
        User(
            id=OPERATOR_EMAIL, email=OPERATOR_EMAIL,
            password_hash=hash_password(OPERATOR_PASSWORD, rounds=4), role=Role.OPERATOR,
        )
    )
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")

    if database is ...:
        database = FakeDatabase()
    if identity_resolver is ...:
        identity_resolver = IdentityResolver(
            FakeDetector(), FakeEmbedder(), FaceGallery(threshold=0.5)
        )

    app = create_app(
        settings, store, registry, manager, users,
        database=database, identity_resolver=identity_resolver,
    )
    return app, database, identity_resolver


# -- GET --------------------------------------------------------------------


def test_list_is_empty_by_default(tmp_path):
    app, _db, _resolver = build(tmp_path)
    client = authed_client(app)

    response = client.get("/api/persons")

    assert response.status_code == 200
    assert response.json() == {"persons": []}


def test_operator_can_list(tmp_path):
    app, _db, _resolver = build(tmp_path)
    client = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)

    assert client.get("/api/persons").status_code == 200


def test_embedding_is_never_echoed_back(tmp_path):
    app, db, _resolver = build(tmp_path)
    db.insert_person(
        {"id": "p1", "name": "Alex", "external_id": "", "active": True, "embeddings": [[0.1] * 128]}
    )
    client = authed_client(app)

    body = client.get("/api/persons").json()

    assert "embeddings" not in body["persons"][0]


# -- POST (enroll) ------------------------------------------------------------


def test_admin_can_enrol_a_person_from_five_pose_photos(tmp_path):
    app, db, _resolver = build(tmp_path)
    client = authed_client(app)

    response = client.post(
        "/api/persons", json={"name": "Alex", "poses": all_poses()}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Alex"
    assert "embeddings" not in body
    stored = db.get_person(body["id"])
    assert len(stored["embeddings"]) == 5
    assert all(len(e) == 128 for e in stored["embeddings"])


def test_enrol_without_a_name_is_400(tmp_path):
    app, _db, _resolver = build(tmp_path)
    client = authed_client(app)

    response = client.post("/api/persons", json={"poses": all_poses()})

    assert response.status_code == 400


def test_enrol_with_a_missing_pose_is_400(tmp_path):
    app, _db, _resolver = build(tmp_path)
    client = authed_client(app)
    poses = all_poses()
    del poses["up"]

    response = client.post("/api/persons", json={"name": "Alex", "poses": poses})

    assert response.status_code == 400


def test_enrol_without_a_face_in_one_photo_is_400(tmp_path):
    app, _db, resolver = build(tmp_path)
    resolver.detector = FakeDetector(found=False)
    client = authed_client(app)

    response = client.post(
        "/api/persons", json={"name": "Alex", "poses": all_poses()}
    )

    assert response.status_code == 400
    assert "no face" in response.json()["detail"]


def test_enrol_when_identity_is_not_enabled_is_503(tmp_path):
    app, _db, _resolver = build(tmp_path, identity_resolver=None)
    client = authed_client(app)

    response = client.post(
        "/api/persons", json={"name": "Alex", "poses": all_poses()}
    )

    assert response.status_code == 503


def test_operator_cannot_enrol(tmp_path):
    app, _db, _resolver = build(tmp_path)
    client = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)

    response = client.post(
        "/api/persons", json={"name": "Alex", "poses": all_poses()}
    )

    assert response.status_code == 403


# -- PATCH / DELETE -----------------------------------------------------------


def test_admin_can_deactivate_a_person(tmp_path):
    app, db, _resolver = build(tmp_path)
    db.insert_person(
        {"id": "p1", "name": "Alex", "external_id": "", "active": True, "embeddings": [[0.1] * 128]}
    )
    client = authed_client(app)

    response = client.patch("/api/persons/p1", json={"active": False})

    assert response.status_code == 200
    assert db.get_person("p1")["active"] is False


def test_patch_missing_person_is_404(tmp_path):
    app, _db, _resolver = build(tmp_path)
    client = authed_client(app)

    assert client.patch("/api/persons/nope", json={"active": False}).status_code == 404


def test_admin_can_delete_a_person(tmp_path):
    app, db, _resolver = build(tmp_path)
    db.insert_person(
        {"id": "p1", "name": "Alex", "external_id": "", "active": True, "embeddings": [[0.1] * 128]}
    )
    client = authed_client(app)

    response = client.delete("/api/persons/p1")

    assert response.status_code == 200
    assert db.get_person("p1") is None


def test_delete_missing_person_is_404(tmp_path):
    app, _db, _resolver = build(tmp_path)
    client = authed_client(app)

    assert client.delete("/api/persons/nope").status_code == 404
