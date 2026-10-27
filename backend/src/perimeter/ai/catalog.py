"""What each alert type means, for AI prompts.

Built from `ALERT_KINDS` - the list every module's alerts are already validated against - so
a new detection module's alerts appear here automatically. A description is optional: an
alert type without one gets a generic description and still works everywhere.
"""

from __future__ import annotations

from perimeter.alerts.cooldown import ALERT_KINDS

DESCRIPTIONS = {
    "boundary": "A person crossed into (entry) or out of (exit) a drawn zone or line.",
    "crowd": "People clustered together in a zone (formed) or the cluster broke up (dispersed).",
    "ppe": "A person in a zone that requires protective equipment was missing an item "
           "(e.g. helmet).",
    "access": "Someone entered a restricted zone without being on its allow-list.",
    "fire": "The fire detector saw flames.",
    "smoke": "The fire detector saw smoke.",
    "welding": "Bright welding sparks were seen - a soft alert, not a fire.",
    "phone": "A person was using a phone where that is not allowed.",
}


def describe_kind(kind: str) -> str:
    return DESCRIPTIONS.get(kind, f"Alerts of type '{kind}' from a detection module.")


def known_kinds() -> list[str]:
    return sorted(ALERT_KINDS)


def catalog() -> list[dict[str, str]]:
    return [{"kind": kind, "description": describe_kind(kind)} for kind in known_kinds()]
