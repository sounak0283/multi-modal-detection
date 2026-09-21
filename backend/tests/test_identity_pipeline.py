"""Tests for Pipeline._publish's identity wiring and the restricted-zone access-control
check (Expansion Plan Phase F / F.1).

Constructing a real `Pipeline` needs the real person-detection ONNX weights, same
`pytest.mark.skipif` pattern `test_firesmoke_pipeline.py` uses. `_publish` and
`_check_access_control` are exercised directly with a hand-built `BoundaryEvent` and a
fake `IdentityResult`, bypassing the full capture/detect/track path - identity resolution
itself is covered separately in test_identity_resolver.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from perimeter.boundary.engine import BoundaryEvent
from perimeter.boundary.zones import EventKind, Severity, ZoneStore, zone_from_dict
from perimeter.cameras.models import camera_from_dict
from perimeter.identity.resolver import IdentityResult
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings

PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")

pytestmark = pytest.mark.skipif(
    not PERSON_MODEL.is_file(),
    reason="yolox_nano.onnx not present - see README for the download step",
)

SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


class RecordingAlertBus:
    def __init__(self):
        self.published = []

    def publish(self, event):
        self.published.append(event)


def make_store():
    import mongomock

    db = mongomock.MongoClient()["perimeter_test"]
    return ZoneStore(db["zones"], refresh_interval=0)


def make_pipeline():
    camera = camera_from_dict(
        {"id": "cam_01", "source_type": "webcam", "source": 0, "enabled_modules": ["identity"]}
    )
    alerts = RecordingAlertBus()
    pipeline = Pipeline(Settings(), camera, make_store(), str(PERSON_MODEL), alerts=alerts)
    return pipeline, alerts


def make_event(zone_id="restricted", kind=EventKind.ENTRY, track_id=1):
    return BoundaryEvent(
        ts=0.0, camera_id="cam_01", zone_id=zone_id, zone_name="Server Room", kind=kind,
        track_id=track_id, severity=Severity.MEDIUM,
        bbox=(0.1, 0.1, 0.2, 0.4), foot_point=(0.15, 0.4),
    )


# -- _publish carries identity through --------------------------------------------


def test_publish_includes_identity_fields_when_resolved():
    pipeline, alerts = make_pipeline()
    identity = IdentityResult(status="known", person_id="p1", name="Alex", confidence=0.812)

    pipeline._publish(make_event(zone_id="unrestricted"), identity)

    assert len(alerts.published) == 1
    payload = alerts.published[0]
    assert payload["identity_status"] == "known"
    assert payload["identity_name"] == "Alex"
    assert payload["identity_confidence"] == 0.812
    assert "Alex" in payload["message"]


def test_publish_with_no_identity_keeps_the_old_none_behaviour():
    pipeline, alerts = make_pipeline()

    pipeline._publish(make_event(zone_id="unrestricted"))

    payload = alerts.published[0]
    assert payload["identity_status"] is None
    assert payload["identity_name"] is None


# -- access control (Phase F.1) ----------------------------------------------------


def test_silent_when_the_zone_has_no_allow_list():
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [zone_from_dict({"id": "restricted", "type": "polygon", "points": SQUARE})], "cam_01"
    )

    pipeline._publish(make_event(), IdentityResult(status="unknown_face"))

    assert len(alerts.published) == 1  # only the normal ENTRY alert


def test_fires_a_critical_alert_for_an_unrecognised_person():
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [
            zone_from_dict(
                {
                    "id": "restricted", "type": "polygon", "points": SQUARE,
                    "authorized_person_ids": ["p_authorized"],
                }
            )
        ],
        "cam_01",
    )

    pipeline._publish(make_event(), IdentityResult(status="unknown_face"))

    assert len(alerts.published) == 2
    access = alerts.published[1]
    assert access["kind"] == "access"
    assert access["subtype"] == "unauthorized"
    assert access["severity"] == "critical"
    assert "Unauthorised access" in access["message"]


def test_fires_for_a_known_but_unlisted_person():
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [
            zone_from_dict(
                {
                    "id": "restricted", "type": "polygon", "points": SQUARE,
                    "authorized_person_ids": ["p_authorized"],
                }
            )
        ],
        "cam_01",
    )

    pipeline._publish(make_event(), IdentityResult(status="known", person_id="p_other", name="Sam"))

    assert len(alerts.published) == 2
    assert alerts.published[1]["identity_name"] == "Sam"


def test_stays_silent_for_an_authorized_person():
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [
            zone_from_dict(
                {
                    "id": "restricted", "type": "polygon", "points": SQUARE,
                    "authorized_person_ids": ["p_authorized"],
                }
            )
        ],
        "cam_01",
    )

    pipeline._publish(
        make_event(), IdentityResult(status="known", person_id="p_authorized", name="Alex")
    )

    assert len(alerts.published) == 1  # no second, critical alert


def test_skipped_for_exit_events():
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [
            zone_from_dict(
                {
                    "id": "restricted", "type": "polygon", "points": SQUARE,
                    "authorized_person_ids": ["p_authorized"],
                }
            )
        ],
        "cam_01",
    )

    pipeline._publish(make_event(kind=EventKind.EXIT), IdentityResult(status="unknown_face"))

    assert len(alerts.published) == 1


class FakeIdentityResolver:
    """A stub with the one method `_resolve_identity_for_event` actually calls - no need
    for real YuNet/SFace models to test the ENTRY/EXIT wiring itself."""

    def __init__(self, result: IdentityResult):
        self.result = result
        self.calls = 0

    def resolve(self, frame_bgr, person_box_px):
        self.calls += 1
        return self.result


# -- ENTRY-resolved identity is reused for the matching EXIT -----------------------


def test_exit_reuses_the_identity_resolved_at_entry_without_re_resolving():
    """A person has usually turned away by the time they exit - re-running face
    detection there would mostly just fail. The EXIT for the same track id should get
    the same identity ENTRY found, without calling the resolver a second time."""
    pipeline, _alerts = make_pipeline()
    pipeline.identity_resolver = FakeIdentityResolver(
        IdentityResult(status="known", person_id="p1", name="Alex")
    )
    track_boxes = {7: (0.0, 0.0, 100.0, 200.0)}

    entry_identity = pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.ENTRY, track_id=7),
        frame_image=None, frame_ts=0.0, track_boxes=track_boxes,
    )
    exit_identity = pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.EXIT, track_id=7),
        frame_image=None, frame_ts=1.0, track_boxes={},
    )

    assert entry_identity.name == "Alex"
    assert exit_identity.name == "Alex"
    assert pipeline.identity_resolver.calls == 1  # not called again for the EXIT


def test_exit_for_a_track_with_no_prior_entry_gets_no_identity():
    """E.g. a track born already inside the zone (§6.3's "born settled" rule) never
    fires an ENTRY, so there is nothing to remember for its eventual EXIT - must not
    crash, and must not fabricate an identity."""
    pipeline, _alerts = make_pipeline()
    pipeline.identity_resolver = FakeIdentityResolver(IdentityResult(status="unknown_face"))

    exit_identity = pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.EXIT, track_id=99),
        frame_image=None, frame_ts=0.0, track_boxes={},
    )

    assert exit_identity is None
    assert pipeline.identity_resolver.calls == 0


def test_different_tracks_do_not_share_a_remembered_identity():
    pipeline, _alerts = make_pipeline()
    pipeline.identity_resolver = FakeIdentityResolver(
        IdentityResult(status="known", person_id="p1", name="Alex")
    )
    pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.ENTRY, track_id=1),
        frame_image=None, frame_ts=0.0, track_boxes={1: (0.0, 0.0, 100.0, 200.0)},
    )

    exit_identity = pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.EXIT, track_id=2),
        frame_image=None, frame_ts=0.0, track_boxes={},
    )

    assert exit_identity is None


# -- pruning: time-based, not "missing from this frame's tracked set" --------------


def test_prune_keeps_an_identity_still_within_the_occlusion_tolerance():
    pipeline, _alerts = make_pipeline()
    pipeline.identity_resolver = FakeIdentityResolver(
        IdentityResult(status="known", person_id="p1", name="Alex")
    )
    pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.ENTRY, track_id=3),
        frame_image=None, frame_ts=100.0, track_boxes={3: (0.0, 0.0, 100.0, 200.0)},
    )

    # A track PersonTracker just doesn't return for a frame or two (mid-occlusion, a
    # missed detection) must not lose the remembered identity - this is the exact bug
    # a presence-based prune caused: evicting well before the real EXIT arrived.
    ttl = pipeline.settings.boundary.lost_track_seconds
    pipeline._prune_stale_identities(now_ts=100.0 + ttl - 0.1)

    exit_identity = pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.EXIT, track_id=3),
        frame_image=None, frame_ts=100.0 + ttl - 0.1, track_boxes={},
    )
    assert exit_identity.name == "Alex"


def test_a_visit_longer_than_the_ttl_still_keeps_its_identity_if_continuously_seen():
    """The exact bug this whole mechanism exists to avoid: a person who simply stays in
    the zone longer than lost_track_seconds (completely normal) must not lose their
    identity by the time they exit, as long as their track keeps being seen."""
    pipeline, _alerts = make_pipeline()
    pipeline.identity_resolver = FakeIdentityResolver(
        IdentityResult(status="known", person_id="p1", name="Alex")
    )
    pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.ENTRY, track_id=5),
        frame_image=None, frame_ts=0.0, track_boxes={5: (0.0, 0.0, 100.0, 200.0)},
    )

    ttl = pipeline.settings.boundary.lost_track_seconds
    # Seen every frame, well past what the raw TTL from ENTRY alone would allow.
    for tick in range(1, int(ttl * 3)):
        pipeline._touch_live_identities([5], now_ts=float(tick))
        pipeline._prune_stale_identities(now_ts=float(tick))

    exit_identity = pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.EXIT, track_id=5),
        frame_image=None, frame_ts=float(int(ttl * 3)), track_boxes={},
    )
    assert exit_identity.name == "Alex"


def test_a_genuine_gap_past_the_ttl_still_evicts_even_with_touching_elsewhere():
    pipeline, _alerts = make_pipeline()
    pipeline.identity_resolver = FakeIdentityResolver(
        IdentityResult(status="known", person_id="p1", name="Alex")
    )
    pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.ENTRY, track_id=5),
        frame_image=None, frame_ts=0.0, track_boxes={5: (0.0, 0.0, 100.0, 200.0)},
    )

    ttl = pipeline.settings.boundary.lost_track_seconds
    # Track 5 is never touched again (genuinely gone) - only an unrelated track 6 is,
    # so this isn't simply "nothing ever gets pruned".
    pipeline._touch_live_identities([6], now_ts=ttl + 5.0)
    pipeline._prune_stale_identities(now_ts=ttl + 5.0)

    exit_identity = pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.EXIT, track_id=5),
        frame_image=None, frame_ts=ttl + 5.0, track_boxes={},
    )
    assert exit_identity is None


def test_prune_evicts_an_identity_past_the_occlusion_tolerance():
    pipeline, _alerts = make_pipeline()
    pipeline.identity_resolver = FakeIdentityResolver(
        IdentityResult(status="known", person_id="p1", name="Alex")
    )
    pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.ENTRY, track_id=3),
        frame_image=None, frame_ts=100.0, track_boxes={3: (0.0, 0.0, 100.0, 200.0)},
    )

    ttl = pipeline.settings.boundary.lost_track_seconds
    pipeline._prune_stale_identities(now_ts=100.0 + ttl + 1.0)

    exit_identity = pipeline._resolve_identity_for_event(
        make_event(kind=EventKind.EXIT, track_id=3),
        frame_image=None, frame_ts=100.0 + ttl + 1.0, track_boxes={},
    )
    assert exit_identity is None


def test_skipped_when_identity_was_never_resolved():
    """No identity module active on this camera (or the resolver returned nothing) -
    access control cannot run without a result to check."""
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [
            zone_from_dict(
                {
                    "id": "restricted", "type": "polygon", "points": SQUARE,
                    "authorized_person_ids": ["p_authorized"],
                }
            )
        ],
        "cam_01",
    )

    pipeline._publish(make_event())  # identity=None

    assert len(alerts.published) == 1
