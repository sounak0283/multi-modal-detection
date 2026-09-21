"""Which alerts get a video clip, and which are urgent (evidence/writer.py uses this).

The trigger is the alert itself - the camera's ring buffer runs all the time, and a clip is
only cut when an alert is recorded. This module decides, per alert:

* whether to cut a clip at all. A registered (recognised) person walking into a zone is
  routine and is not recorded; an unregistered person is. Fire, smoke, crowd and
  unauthorized-access alerts are always recorded.
* whether the alert is urgent: urgent alerts are emailed at once (the video follows as a
  second email) and use a much shorter post-roll, so fire is never queued behind a slow
  upload.

The rules live in `storage.clip_triggers` in config/app.yaml, so changing them is a config
edit, not a code change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Boundary alerts: `all` records every one, `unregistered` skips a recognised person,
# `off` records none. Other kinds: `on` / `off`.
BOUNDARY_MODES = frozenset({"all", "unregistered", "off"})
SIMPLE_MODES = frozenset({"on", "off"})

DEFAULT_TRIGGERS: dict[str, str] = {
    "boundary_entry": "unregistered",
    "boundary_exit": "off",
    "crowd": "on",
    "fire": "on",
    "smoke": "on",
    "access": "on",
    "ppe": "on",
    "welding": "off",
}
DEFAULT_URGENT_KINDS = frozenset({"fire", "smoke", "crowd", "access"})


@dataclass(frozen=True)
class ClipDecision:
    record: bool
    urgent: bool
    reason: str


@dataclass(frozen=True)
class ClipPolicy:
    triggers: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_TRIGGERS))
    urgent_kinds: frozenset[str] = DEFAULT_URGENT_KINDS

    def decide(self, payload: dict[str, Any]) -> ClipDecision:
        kind = str(payload.get("kind") or "")
        urgent = kind in self.urgent_kinds

        if kind == "boundary":
            subtype = str(payload.get("subtype") or "")
            key = f"boundary_{subtype}"
            mode = self.triggers.get(key, DEFAULT_TRIGGERS.get(key, "unregistered"))
            if mode == "off":
                return ClipDecision(False, urgent, f"{key} is off")
            if mode == "unregistered" and payload.get("identity_status") == "known":
                return ClipDecision(False, urgent, "registered person")
            return ClipDecision(True, urgent, mode)

        mode = self.triggers.get(kind, "on")
        if mode == "off":
            return ClipDecision(False, urgent, f"{kind} is off")
        return ClipDecision(True, urgent, mode)


def policy_from_settings(raw_triggers: Mapping[str, Any] | None, urgent_kinds: Any) -> ClipPolicy:
    """Parse and validate `storage.clip_triggers` / `storage.urgent_kinds`. Unknown keys and
    bad values raise, so a typo fails at startup rather than silently recording nothing."""
    triggers = dict(DEFAULT_TRIGGERS)
    for key, value in (raw_triggers or {}).items():
        key = str(key).strip().lower()
        if isinstance(value, bool):
            # YAML 1.1 reads an unquoted `off` / `on` as False / True, so `boundary_exit: off`
            # arrives here as a bool. Accept it rather than make people quote it.
            value = ("all" if key.startswith("boundary_") else "on") if value else "off"
        mode = str(value).strip().lower()
        if key not in DEFAULT_TRIGGERS:
            raise ValueError(
                f"storage.clip_triggers: unknown trigger {key!r} "
                f"(expected: {', '.join(sorted(DEFAULT_TRIGGERS))})"
            )
        allowed = BOUNDARY_MODES if key.startswith("boundary_") else SIMPLE_MODES
        if mode not in allowed:
            raise ValueError(
                f"storage.clip_triggers.{key}: {value!r} is not valid "
                f"(expected: {', '.join(sorted(allowed))})"
            )
        triggers[key] = mode

    if urgent_kinds is None:
        urgent = DEFAULT_URGENT_KINDS
    else:
        if not isinstance(urgent_kinds, list | tuple | set | frozenset):
            raise ValueError("storage.urgent_kinds must be a list of alert kinds")
        urgent = frozenset(str(k).strip().lower() for k in urgent_kinds)
    return ClipPolicy(triggers=triggers, urgent_kinds=urgent)
