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
    identity = IdentityResult(status="known", person_id="p1", name="Alex")

    pipeline._publish(make_event(zone_id="unrestricted"), identity)

    assert len(alerts.published) == 1
    payload = alerts.published[0]
    assert payload["identity_status"] == "known"
    assert payload["identity_name"] == "Alex"
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
