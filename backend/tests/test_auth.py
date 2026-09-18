"""Tests for session auth and RBAC (Expansion Plan Phase B).

Builds its own minimal app per test (no camera/zone fixtures needed) since these tests
are about the auth layer itself, not the domain routes it protects - `test_camera_api.py`
and `test_api.py` cover "does an authenticated admin still get the same behaviour as
before" for those.
"""

from __future__ import annotations

import time

import mongomock
import pytest
from conftest import ADMIN_EMAIL, ADMIN_PASSWORD, authed_client, seed_admin
from fastapi.testclient import TestClient

from perimeter.api.app import create_app
from perimeter.auth.models import Role, User
from perimeter.auth.passwords import hash_password
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import AuthSettings, Settings

OPERATOR_EMAIL = "operator@test.local"
OPERATOR_PASSWORD = "operator-password"


def build(tmp_path):
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")
    app = create_app(settings, store, registry, manager, users)
    return app, registry, users


# -- login / logout / me -----------------------------------------------------


def test_login_succeeds_and_sets_a_cookie(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )

    assert response.status_code == 200
    assert response.json()["email"] == ADMIN_EMAIL
    assert response.json()["role"] == "admin"
    assert "password_hash" not in response.json()
    assert "perimeter_session" in response.cookies


def test_login_rejects_wrong_password_without_leaking_which_field(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/auth/login", json={"email": ADMIN_EMAIL, "password": "not-it"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid email or password"


def test_login_rejects_unknown_email_with_the_same_message(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/auth/login", json={"email": "nobody@test.local", "password": "anything"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid email or password"


def test_login_rejects_a_deactivated_account(tmp_path):
    app, _registry, users = build(tmp_path)
    users.update(
        ADMIN_EMAIL,
        User(
            id=ADMIN_EMAIL,
            email=ADMIN_EMAIL,
            password_hash=hash_password(ADMIN_PASSWORD, rounds=4),
            role=Role.ADMIN,
            active=False,
        ),
    )
    client = TestClient(app)

    response = client.post(
        "/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )

    assert response.status_code == 401


def test_protected_route_without_a_cookie_is_401(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = TestClient(app)

    assert client.get("/api/cameras").status_code == 401
    assert client.get("/api/health").status_code == 401


def test_me_reflects_the_logged_in_user(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = authed_client(app)

    response = client.get("/api/auth/me")

    assert response.status_code == 200
    assert response.json()["email"] == ADMIN_EMAIL


def test_logout_clears_the_session(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = authed_client(app)
    assert client.get("/api/auth/me").status_code == 200

    logout = client.post("/api/auth/logout")

    assert logout.status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_a_session_token_copied_before_logout_is_rejected_after_it(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = authed_client(app)
    copied = dict(client.cookies)

    client.post("/api/auth/logout")

    replay = TestClient(app)
    replay.cookies.update(copied)
    assert replay.get("/api/auth/me").status_code == 401


def test_logout_only_ends_that_session_not_the_same_account_elsewhere(tmp_path):
    app, _registry, _users = build(tmp_path)
    control_room_screen = authed_client(app)
    laptop = authed_client(app)

    laptop.post("/api/auth/logout")

    assert laptop.get("/api/auth/me").status_code == 401
    assert control_room_screen.get("/api/auth/me").status_code == 200


def test_logging_in_again_after_logout_works(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = authed_client(app)
    client.post("/api/auth/logout")

    again = authed_client(app)

    assert again.get("/api/auth/me").status_code == 200


def test_an_invalid_cookie_value_is_401_not_a_crash(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = TestClient(app)
    client.cookies.set("perimeter_session", "garbage-not-a-real-token")

    assert client.get("/api/auth/me").status_code == 401


def test_an_expired_session_is_401(tmp_path):
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    settings = Settings(
        env_file=tmp_path / ".env",
        auth=AuthSettings(session_secret="x", session_max_age_s=1),
    )
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")
    app = create_app(settings, store, registry, manager, users)
    client = authed_client(app)
    assert client.get("/api/auth/me").status_code == 200

    time.sleep(1.5)

    assert client.get("/api/auth/me").status_code == 401


# -- role enforcement ---------------------------------------------------------


def _create_operator(users) -> None:
    users.create(
        User(
            id=OPERATOR_EMAIL,
            email=OPERATOR_EMAIL,
            password_hash=hash_password(OPERATOR_PASSWORD, rounds=4),
            role=Role.OPERATOR,
        )
    )


def test_operator_can_read_cameras_but_not_create_one(tmp_path):
    app, _registry, users = build(tmp_path)
    _create_operator(users)
    client = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)

    assert client.get("/api/cameras").status_code == 200
    create = client.post(
        "/api/cameras", json={"id": "cam_01", "source_type": "webcam", "source": 0}
    )
    assert create.status_code == 403


def test_operator_cannot_edit_zones_or_run_dev_tools(tmp_path):
    app, registry, users = build(tmp_path)
    _create_operator(users)
    from perimeter.cameras.models import camera_from_dict

    registry.create(camera_from_dict({"id": "cam_01", "source_type": "webcam", "source": 0}))
    client = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)

    assert client.put("/api/cameras/cam_01/zones", json={"zones": []}).status_code == 403
    assert client.post("/api/dev/simulate-alert", json={"kind": "fire"}).status_code == 403


def test_operator_cannot_manage_users(tmp_path):
    app, _registry, users = build(tmp_path)
    _create_operator(users)
    client = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)

    assert client.get("/api/users").status_code == 403
    create = client.post("/api/users", json={"email": "x@y.com", "password": "abcdefgh"})
    assert create.status_code == 403


# -- user management (admin) --------------------------------------------------


def test_admin_can_create_list_and_delete_an_operator(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = authed_client(app)

    created = client.post(
        "/api/users",
        json={"email": OPERATOR_EMAIL, "password": OPERATOR_PASSWORD, "role": "operator"},
    )
    assert created.status_code == 200
    assert created.json()["role"] == "operator"
    assert "password_hash" not in created.json()

    listed = client.get("/api/users").json()["users"]
    assert {u["email"] for u in listed} == {ADMIN_EMAIL, OPERATOR_EMAIL}

    deleted = client.delete(f"/api/users/{OPERATOR_EMAIL}")
    assert deleted.status_code == 200


def test_short_password_is_rejected(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = authed_client(app)

    response = client.post(
        "/api/users", json={"email": "short@test.local", "password": "short", "role": "operator"}
    )

    assert response.status_code == 400


def test_cannot_create_a_duplicate_email(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = authed_client(app)

    response = client.post(
        "/api/users", json={"email": ADMIN_EMAIL, "password": "another-password"}
    )

    assert response.status_code == 409


def test_cannot_delete_the_last_active_admin(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = authed_client(app)

    response = client.delete(f"/api/users/{ADMIN_EMAIL}")

    assert response.status_code == 409


def test_cannot_demote_the_last_active_admin(tmp_path):
    app, _registry, _users = build(tmp_path)
    client = authed_client(app)

    response = client.put(f"/api/users/{ADMIN_EMAIL}", json={"role": "operator"})

    assert response.status_code == 409


def test_can_demote_an_admin_when_another_active_admin_remains(tmp_path):
    app, _registry, users = build(tmp_path)
    second_admin_email = "second-admin@test.local"
    users.create(
        User(
            id=second_admin_email,
            email=second_admin_email,
            password_hash=hash_password("second-admin-pw", rounds=4),
            role=Role.ADMIN,
        )
    )
    client = authed_client(app)

    response = client.put(f"/api/users/{ADMIN_EMAIL}", json={"role": "operator"})

    assert response.status_code == 200
    assert response.json()["role"] == "operator"


def test_role_change_takes_effect_on_the_very_next_request(tmp_path):
    """The cookie carries only a user id - the role is re-read from storage every
    request, so a demotion made from one session is visible immediately on another."""
    app, _registry, users = build(tmp_path)
    _create_operator(users)
    operator_client = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)
    admin_client = authed_client(app)
    assert operator_client.get("/api/cameras").status_code == 200

    promote = admin_client.put(f"/api/users/{OPERATOR_EMAIL}", json={"role": "admin"})
    assert promote.status_code == 200

    assert operator_client.get("/api/users").status_code == 200


def test_deactivating_a_user_logs_them_out_on_their_next_request(tmp_path):
    app, _registry, users = build(tmp_path)
    _create_operator(users)
    operator_client = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)
    admin_client = authed_client(app)
    assert operator_client.get("/api/cameras").status_code == 200

    deactivate = admin_client.put(f"/api/users/{OPERATOR_EMAIL}", json={"active": False})
    assert deactivate.status_code == 200

    assert operator_client.get("/api/cameras").status_code == 401


def test_updating_a_user_can_reset_their_password(tmp_path):
    app, _registry, users = build(tmp_path)
    _create_operator(users)
    admin_client = authed_client(app)

    reset = admin_client.put(f"/api/users/{OPERATOR_EMAIL}", json={"password": "brand-new-pw"})
    assert reset.status_code == 200

    login_old = TestClient(app).post(
        "/api/auth/login", json={"email": OPERATOR_EMAIL, "password": OPERATOR_PASSWORD}
    )
    assert login_old.status_code == 401

    login_new = TestClient(app).post(
        "/api/auth/login", json={"email": OPERATOR_EMAIL, "password": "brand-new-pw"}
    )
    assert login_new.status_code == 200


def test_a_password_reset_signs_out_existing_sessions(tmp_path):
    app, _registry, users = build(tmp_path)
    _create_operator(users)
    operator = authed_client(app, OPERATOR_EMAIL, OPERATOR_PASSWORD)
    admin_client = authed_client(app)

    admin_client.put(f"/api/users/{OPERATOR_EMAIL}", json={"password": "brand-new-pw"})

    assert operator.get("/api/auth/me").status_code == 401


@pytest.mark.parametrize("missing_field", ["email", "password"])
def test_login_with_a_malformed_body_is_a_clean_error_not_a_500(tmp_path, missing_field):
    app, _registry, _users = build(tmp_path)
    client = TestClient(app)
    payload = {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    payload.pop(missing_field)

    response = client.post("/api/auth/login", json=payload)

    assert response.status_code == 401
