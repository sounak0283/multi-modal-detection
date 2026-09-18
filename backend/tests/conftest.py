"""Shared test auth helpers (Expansion Plan Phase B).

Every route now requires a logged-in session, so any test that builds a
`TestClient(create_app(...))` needs an authenticated client. Rather than repeat the
login dance in every file, `authed_client()` seeds a fixed test admin and logs in once;
`TestClient` persists the resulting `Set-Cookie` across subsequent requests on the same
client automatically, so callers use it exactly like a bare `TestClient`.

`rounds=4` (vs. the production default of 12) keeps bcrypt hashing sub-millisecond -
at ~200-300ms/hash-at-12, hashing once per test-file fixture instantiation across the
suite would meaningfully slow it down for no reason the cost factor is actually testing.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from pymongo.collection import Collection

from perimeter.auth.models import Role, User
from perimeter.auth.passwords import hash_password
from perimeter.auth.registry import UserRegistry

TEST_ROUNDS = 4
ADMIN_EMAIL = "admin@test.local"
ADMIN_PASSWORD = "admin-password"


def seed_admin(
    users_collection: Collection, email: str = ADMIN_EMAIL, password: str = ADMIN_PASSWORD
) -> UserRegistry:
    """Build a `UserRegistry` on `users_collection` with one active admin already in it."""
    registry = UserRegistry(users_collection)
    registry.create(
        User(
            id=email,
            email=email,
            password_hash=hash_password(password, rounds=TEST_ROUNDS),
            role=Role.ADMIN,
        )
    )
    return registry


def authed_client(app, email: str = ADMIN_EMAIL, password: str = ADMIN_PASSWORD) -> TestClient:
    """A `TestClient` already logged in as the seeded test admin."""
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return client
