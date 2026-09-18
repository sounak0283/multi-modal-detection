"""Tests for GET/PUT /api/alert-config (Expansion Plan Phase D)."""

from __future__ import annotations

import mongomock
from conftest import authed_client, seed_admin

from perimeter.alerts.config_store import AlertConfigStore
from perimeter.api.app import create_app
from perimeter.auth.models import Role, User
from perimeter.auth.passwords import hash_password
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import Settings

OPERATOR_EMAIL = "operator@test.local"
OPERATOR_PASSWORD = "operator-password"


def build(tmp_path, alert_config_store=...):
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

    if alert_config_store is ...:
        alert_config_store = AlertConfigStore(db["alert_config"], refresh_interval=0)

    app = create_app(
        settings, store, registry, manager, users, alert_config_store=alert_config_store
    )
    return app, alert_config_store


# -- GET ------------------------------------------------------------------------


def test_get_returns_the_default_config(tmp_path):
    app, _store = build(tmp_path)
    client = authed_client(app)

    response = client.get("/api/alert-config")

    assert response.status_code == 200
    body = response.json()
    assert body["sinks"] == []
    assert body["sound_enabled"] is True


def test_get_without_a_store_configured_is_503(tmp_path):
    app, _store = build(tmp_path, alert_config_store=None)
    client = authed_client(app)

    assert client.get("/api/alert-config").status_code == 503


def test_operator_can_read_the_config(tmp_path):
    app, _store = build(tmp_path)
    client = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)

    assert client.get("/api/alert-config").status_code == 200


# -- PUT ------------------------------------------------------------------------


def test_admin_can_save_a_rule(tmp_path):
    app, store = build(tmp_path)
    client = authed_client(app)

    response = client.put(
        "/api/alert-config",
        json={
            "sinks": [
                {
                    "type": "smtp", "to": ["ops@example.com"],
                    "min_severity": "high", "cooldown_seconds": 60,
                }
            ],
            "sound_enabled": False,
            "sound_min_severity": "critical",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["sinks"]) == 1
    assert body["sinks"][0]["to"] == ["ops@example.com"]
    assert body["sound_enabled"] is False
    assert store.get().sound_min_severity.value == "critical"


def test_put_assigns_an_id_when_one_is_not_given(tmp_path):
    app, _store = build(tmp_path)
    client = authed_client(app)

    response = client.put(
        "/api/alert-config", json={"sinks": [{"type": "smtp", "to": ["ops@example.com"]}]}
    )

    assert response.json()["sinks"][0]["id"]


def test_put_rejects_an_invalid_rule_with_a_clean_400(tmp_path):
    app, _store = build(tmp_path)
    client = authed_client(app)

    response = client.put("/api/alert-config", json={"sinks": [{"type": "smtp", "to": []}]})

    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"]


def test_put_rejects_duplicate_rule_ids(tmp_path):
    app, _store = build(tmp_path)
    client = authed_client(app)

    response = client.put(
        "/api/alert-config",
        json={
            "sinks": [
                {"id": "same", "type": "smtp", "to": ["a@b.com"]},
                {"id": "same", "type": "smtp", "to": ["c@d.com"]},
            ]
        },
    )

    assert response.status_code == 400


def test_operator_cannot_save(tmp_path):
    app, _store = build(tmp_path)
    client = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)

    response = client.put(
        "/api/alert-config", json={"sinks": [{"type": "smtp", "to": ["ops@example.com"]}]}
    )

    assert response.status_code == 403


def test_put_without_a_store_configured_is_503(tmp_path):
    app, _store = build(tmp_path, alert_config_store=None)
    client = authed_client(app)

    response = client.put(
        "/api/alert-config", json={"sinks": [{"type": "smtp", "to": ["ops@example.com"]}]}
    )

    assert response.status_code == 503
