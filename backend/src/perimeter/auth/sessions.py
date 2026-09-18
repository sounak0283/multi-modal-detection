"""Signed session cookies (Expansion Plan, Phase B).

`itsdangerous`, not a JWT - the cookie's payload is a user id, the account's
`session_version`, and a random per-login session id; the trust boundary is the
server-held secret verifying an unmodified, non-expired token. Deliberately does not embed
the role: a role change or deactivation must take effect on the *next request*, not the
next login, so `api/app.py`'s `get_current_user` re-reads the user from `UserRegistry`
every time instead of trusting a cached claim.

A signed token is otherwise valid until it expires, so two revocation paths exist:
logout revokes just that one session id, and bumping `User.session_version` (password
reset) invalidates every session for the account at once.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SALT = "perimeter.session"


@dataclass(frozen=True)
class Session:
    user_id: str
    session_version: int
    session_id: str


class SessionManager:
    def __init__(self, secret: str, max_age_s: int) -> None:
        self._serializer = URLSafeTimedSerializer(secret, salt=SALT)
        self.max_age_s = max_age_s

    def issue(self, user_id: str, session_version: int = 0) -> str:
        return self._serializer.dumps([user_id, session_version, secrets.token_urlsafe(16)])

    def resolve(self, token: str | None) -> Session | None:
        """Verify signature and expiry. Never raises - any failure just means "not logged in"."""
        if not token:
            return None
        try:
            payload = self._serializer.loads(token, max_age=self.max_age_s)
        except (BadSignature, SignatureExpired):
            return None
        if (
            not isinstance(payload, list)
            or len(payload) != 3
            or not isinstance(payload[0], str)
            or not isinstance(payload[1], int)
            or not isinstance(payload[2], str)
        ):
            return None
        return Session(*payload)
