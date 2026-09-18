"""User domain model (Expansion Plan, Phase B).

One document per account in MongoDB's `users` collection. Mirrors
`cameras/models.py`'s shape deliberately: a frozen dataclass, a `*_from_dict` validator
that raises actionable `ValueError`s, and a `to_public_dict()` that never echoes the
secret field (`password_hash` here, `rtsp_url` there).

`id` is the lowercased, trimmed email address - immutable via the API once created, the
same rule `api/app.py` already applies to a camera's `id`. Using the email as the key
avoids inventing a second identifier a human has to remember alongside their login.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# bcrypt silently truncates/rejects beyond 72 bytes - reject up front with a clear
# message rather than let a long password be accepted and then never match again.
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 72


class Role(Enum):
    ADMIN = "admin"
    OPERATOR = "operator"


@dataclass(frozen=True)
class User:
    """An account and the role that gates what it can do.

    `password_hash` is a bcrypt hash, never the plaintext password - nothing in this
    codebase ever holds a plaintext password past the request that receives it (see
    `auth/passwords.py`).
    """

    id: str
    email: str
    password_hash: str
    role: Role = Role.OPERATOR
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Embedded in every session token and compared on each request. Bumping it (logout,
    # password change) invalidates every outstanding token for this account at once -
    # a signed cookie is otherwise valid until it expires, even after logout.
    session_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Full representation for internal use (Mongo storage) - includes the hash."""
        return {
            "id": self.id,
            "email": self.email,
            "password_hash": self.password_hash,
            "role": self.role.value,
            "active": self.active,
            "created_at": self.created_at,
            "session_version": self.session_version,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """API-facing representation - the password hash is never echoed at all."""
        payload = self.to_dict()
        payload.pop("password_hash")
        payload.pop("session_version")
        created_at = payload["created_at"]
        if isinstance(created_at, datetime):
            payload["created_at"] = created_at.isoformat()
        return payload


def normalise_email(email: str) -> str:
    return str(email or "").strip().lower()


def validate_email(email: str) -> str:
    normalised = normalise_email(email)
    if not _EMAIL_PATTERN.match(normalised):
        raise ValueError(f"{email!r} is not a valid email address")
    return normalised


def validate_password(password: str) -> str:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"password must be at most {MAX_PASSWORD_LENGTH} characters")
    return password


def user_from_dict(data: dict[str, Any]) -> User:
    """Parse and validate one user document already carrying a `password_hash`.

    Does not hash a password itself - callers (the API routes) hash a plaintext
    password with `auth/passwords.py:hash_password` first, so this function has one job
    (shape validation) matching `cameras/models.py:camera_from_dict`'s contract.
    """
    if not isinstance(data, dict):
        raise ValueError("user must be a mapping")

    email = validate_email(data.get("email", ""))

    password_hash = str(data.get("password_hash", "")).strip()
    if not password_hash:
        raise ValueError(f"user {email!r} is missing a password_hash")

    try:
        role = Role(str(data.get("role", Role.OPERATOR.value)).strip().lower())
    except ValueError:
        valid = ", ".join(r.value for r in Role)
        raise ValueError(
            f"user {email!r}: invalid role {data.get('role')!r} (expected: {valid})"
        ) from None

    created_at = data.get("created_at") or datetime.now(UTC)
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    return User(
        id=email,
        email=email,
        password_hash=password_hash,
        role=role,
        active=bool(data.get("active", True)),
        created_at=created_at,
        session_version=int(data.get("session_version", 0) or 0),
    )
