"""User registry - MongoDB-backed (Expansion Plan, Phase B).

Structurally identical to `cameras/registry.py:CameraRegistry` - CRUD over one
collection, keyed by an immutable `id`, no separate version counter needed here since
nothing polls this collection for hot-reload the way `PipelineManager` polls cameras.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

from pymongo.collection import Collection

from perimeter.auth.models import User, user_from_dict

log = logging.getLogger("perimeter.auth.registry")

COLLECTION = "users"
REVOKED_SESSIONS = "revoked_sessions"


class UserRegistry:
    """Thread-safe CRUD over the `users` collection."""

    def __init__(self, collection: Collection) -> None:
        self._collection = collection
        self._lock = threading.Lock()
        self._collection.create_index("id", unique=True)
        # Revoked session ids only need remembering until the token would have expired
        # anyway - the TTL index makes MongoDB purge them after `expires_at`.
        self._revoked = collection.database[REVOKED_SESSIONS]
        self._revoked.create_index("sid", unique=True)
        self._revoked.create_index("expires_at", expireAfterSeconds=0)

    def revoke_session(self, session_id: str, max_age_s: int) -> None:
        expires_at = datetime.now(UTC) + timedelta(seconds=max_age_s)
        self._revoked.update_one(
            {"sid": session_id}, {"$set": {"sid": session_id, "expires_at": expires_at}},
            upsert=True,
        )

    def is_session_revoked(self, session_id: str) -> bool:
        return self._revoked.find_one({"sid": session_id}, {"_id": 1}) is not None

    def list(self) -> list[User]:
        with self._lock:
            return [user_from_dict(doc) for doc in self._collection.find({}, {"_id": 0})]

    def get(self, user_id: str) -> User | None:
        with self._lock:
            doc = self._collection.find_one({"id": user_id}, {"_id": 0})
        return user_from_dict(doc) if doc else None

    def get_by_email(self, email: str) -> User | None:
        return self.get(email.strip().lower())

    def count_active_admins(self) -> int:
        from perimeter.auth.models import Role

        with self._lock:
            return self._collection.count_documents(
                {"role": Role.ADMIN.value, "active": True}
            )

    def create(self, user: User) -> User:
        with self._lock:
            if self._collection.find_one({"id": user.id}, {"_id": 1}):
                raise ValueError(f"an account with email {user.email!r} already exists")
            self._collection.insert_one(user.to_dict())
        log.info("user created: %s (%s)", user.email, user.role.value)
        return user

    def update(self, user_id: str, user: User) -> User:
        if user.id != user_id:
            raise ValueError("user id (email) cannot be changed")
        with self._lock:
            result = self._collection.replace_one({"id": user_id}, user.to_dict())
        if result.matched_count == 0:
            raise KeyError(f"user {user_id!r} not found")
        log.info("user updated: %s (%s, active=%s)", user.email, user.role.value, user.active)
        return user

    def delete(self, user_id: str) -> None:
        with self._lock:
            result = self._collection.delete_one({"id": user_id})
        if result.deleted_count == 0:
            raise KeyError(f"user {user_id!r} not found")
        log.info("user deleted: %s", user_id)

    def upsert(self, user: User) -> User:
        """Used only by the admin-bootstrap step in `main.py`."""
        with self._lock:
            self._collection.replace_one({"id": user.id}, user.to_dict(), upsert=True)
        return user
