"""Password hashing (Expansion Plan, Phase B).

A thin `bcrypt` wrapper so nothing else in the codebase imports `bcrypt` directly. `rounds`
is overridable because the production default (12) costs ~200-300ms per hash by design -
appropriate for a login, and a real cost the test suite should not pay on every fixture
instantiation, so tests pass a cheap `rounds=4` instead (see `tests/conftest.py`).
"""

from __future__ import annotations

import bcrypt

DEFAULT_ROUNDS = 12


def hash_password(password: str, rounds: int = DEFAULT_ROUNDS) -> str:
    salt = bcrypt.gensalt(rounds=rounds)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # A malformed/foreign hash must fail closed, not raise into a 500.
        return False
