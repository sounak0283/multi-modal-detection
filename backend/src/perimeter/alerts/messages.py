"""Operator-facing alert wording.

Kept in one place because these strings are the product's actual output - the thing a
customer reads at 3am - and because they are the natural extension point for the
identity layer. Today every person is "Someone"; once YuNet + SFace ship, only
`subject_for` changes and every message follows.

The rendered sentence is stored on the event row rather than rebuilt at read time, so
alert history always shows what was actually sent even after the wording changes.
"""

from __future__ import annotations

from perimeter.boundary.zones import EventKind

# Wording is configurable because it is customer-visible and occasionally
# site-specific ("entered the substation" reads better than "entered Zone 3").
ENTRY_VERB = "entered"
EXIT_VERB = "left"

ANONYMOUS_SUBJECT = "Someone"
UNKNOWN_SUBJECT = "An unrecognised person"


def subject_for(identity_status: str | None = None, identity_name: str | None = None) -> str:
    """Who the alert is about.

    Three outcomes, all of which must stay first-class when the face layer lands
    (PLAN.md section 7):

    * a confident match  -> the person's name
    * a face, no match   -> "An unrecognised person"
    * no usable face     -> "Someone"

    `no_face` deliberately reads the same as having no identity layer at all. Outdoors
    at a gate it will be the common case, and rendering it as "intruder" would turn an
    abstention into an accusation.
    """
    if identity_status == "known" and identity_name:
        return identity_name
    if identity_status == "unknown_face":
        return UNKNOWN_SUBJECT
    return ANONYMOUS_SUBJECT


def boundary_message(
    kind: EventKind,
    zone_name: str,
    identity_status: str | None = None,
    identity_name: str | None = None,
) -> str:
    """e.g. "Someone entered Loading Bay"."""
    subject = subject_for(identity_status, identity_name)
    verb = ENTRY_VERB if kind is EventKind.ENTRY else EXIT_VERB
    return f"{subject} {verb} {zone_name}" if zone_name else f"{subject} {verb}"


def health_message(subtype: str, camera_id: str) -> str:
    return {
        "feed_lost": f"Camera {camera_id} stopped responding",
        "feed_restored": f"Camera {camera_id} is back online",
        "lens_obscured": f"Camera {camera_id} may be covered or blinded",
    }.get(subtype, f"Camera {camera_id}: {subtype}")


def firesmoke_message(klass: str, zone_name: str | None = None) -> str:
    label = "Fire" if klass == "fire" else "Smoke"
    return f"{label} detected in {zone_name}" if zone_name else f"{label} detected"


def crowd_message(zone_name: str, cluster_size: int) -> str:
    """e.g. "Crowd forming in Loading Bay (6 people clustered)" (Expansion Plan
    Phase G)."""
    people = "person" if cluster_size == 1 else "people"
    if zone_name:
        return f"Crowd forming in {zone_name} ({cluster_size} {people} clustered)"
    return f"Crowd forming ({cluster_size} {people} clustered)"


def ppe_message(zone_name: str, item: str) -> str:
    """e.g. "Missing helmet in Loading Bay" (Expansion Plan Phase H)."""
    if zone_name:
        return f"Missing {item} in {zone_name}"
    return f"Missing {item}"


def unauthorized_access_message(
    zone_name: str, identity_status: str | None = None, identity_name: str | None = None
) -> str:
    """A restricted zone's allow-list check failed (Expansion Plan Phase F.1) - fired
    alongside, not instead of, the normal boundary_message for the same ENTRY."""
    subject = subject_for(identity_status, identity_name)
    if zone_name:
        return f"Unauthorised access: {subject} entered {zone_name}"
    return f"Unauthorised access: {subject} entered a restricted zone"
