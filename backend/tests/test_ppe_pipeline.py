"""Tests for Pipeline's PPE wiring (Expansion Plan Phase H).

Constructing a real `Pipeline` needs the real person-detection ONNX weights, same
`pytest.mark.skipif` pattern the other module-wiring tests use. `PPEClassifier` is not
built on `YoloxOnnx` (see `detect/ppe.py`'s docstring), so - unlike fire/smoke - the
person weights cannot stand in for a real PPE model; construction/graceful-degradation
is tested via the model-file-path checks, and the zone-membership/hysteresis/publish
wiring is tested by injecting a fake classifier directly onto a real `Pipeline`, mirroring
`test_identity_pipeline.py`'s approach of calling `_publish` directly with a fake result
rather than a real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perimeter.boundary.zones import ZoneStore, zone_from_dict
from perimeter.cameras.models import camera_from_dict
from perimeter.capture.source import Frame
from perimeter.detect.ppe import PPEMonitor
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings
from perimeter.track.tracker import TrackedDetections

PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")
MISSING_MODEL = Path("models/ppe/does-not-exist.onnx")

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


class FakeClassifier:
    def __init__(self, probabilities: dict[str, float]):
        self.probabilities = probabilities
        self.calls = 0

    def classify(self, person_crop_bgr):
        self.calls += 1
        return self.probabilities


def make_store():
    import mongomock

    db = mongomock.MongoClient()["perimeter_test"]
    return ZoneStore(db["zones"], refresh_interval=0)


def make_camera(enabled_modules):
    return camera_from_dict(
        {"id": "cam_01", "source_type": "webcam", "source": 0, "enabled_modules": enabled_modules}
    )


def make_pipeline(enabled_modules=("boundary",), ppe_model_path=None):
    alerts = RecordingAlertBus()
    pipeline = Pipeline(
        Settings(), make_camera(list(enabled_modules)), make_store(), str(PERSON_MODEL),
        alerts=alerts, ppe_model_path=ppe_model_path,
    )
    return pipeline, alerts


def make_tracked(box=(100.0, 100.0, 150.0, 300.0)) -> TrackedDetections:
    return TrackedDetections(
        xyxy=np.array([box], dtype=np.float32),
        scores=np.ones(1, np.float32),
        class_ids=np.zeros(1, np.int32),
        track_ids=np.array([7], dtype=np.int32),
    )


def make_frame() -> Frame:
    return Frame(seq=0, ts=0.0, image=np.zeros((480, 640, 3), dtype=np.uint8))


# -- construction / graceful degradation ---------------------------------------------


def test_ppe_stays_inactive_when_the_module_is_not_enabled():
    pipeline, _alerts = make_pipeline(
        enabled_modules=["boundary"], ppe_model_path=str(PERSON_MODEL)
    )
    assert pipeline.ppe_classifier is None


def test_ppe_stays_inactive_when_no_model_path_was_given():
    pipeline, _alerts = make_pipeline(enabled_modules=["boundary", "ppe"])
    assert pipeline.ppe_classifier is None


def test_ppe_degrades_gracefully_when_the_model_file_is_missing():
    pipeline, _alerts = make_pipeline(
        enabled_modules=["boundary", "ppe"], ppe_model_path=str(MISSING_MODEL)
    )
    assert pipeline.ppe_classifier is None
    assert pipeline.ppe_monitor is None


# -- zone-membership / hysteresis / publish wiring -------------------------------------


def test_no_check_without_ppe_required_on_the_zone():
    pipeline, alerts = make_pipeline(enabled_modules=["boundary", "ppe"])
    pipeline.store.save(
        [zone_from_dict({"id": "z1", "type": "polygon", "points": SQUARE})], "cam_01"
    )
    pipeline.ppe_classifier = FakeClassifier({"helmet": 0.1})
    pipeline.ppe_monitor = PPEMonitor()

    tracked = make_tracked()
    masked = np.zeros(1, dtype=bool)
    pipeline._process_ppe(make_frame(), tracked, masked, 640, 480)

    assert pipeline.ppe_classifier.calls == 0
    assert alerts.published == []


def test_fires_once_absence_is_confirmed():
    pipeline, alerts = make_pipeline(enabled_modules=["boundary", "ppe"])
    pipeline.store.save(
        [
            zone_from_dict(
                {"id": "z1", "type": "polygon", "points": SQUARE, "ppe_required": ["helmet"]}
            )
        ],
        "cam_01",
    )
    pipeline.ppe_classifier = FakeClassifier({"helmet": 0.05})  # confidently absent
    pipeline.ppe_monitor = PPEMonitor()

    tracked = make_tracked()
    masked = np.zeros(1, dtype=bool)
    for _ in range(3):
        pipeline._process_ppe(make_frame(), tracked, masked, 640, 480)

    assert len(alerts.published) == 1
    payload = alerts.published[0]
    assert payload["kind"] == "ppe"
    assert payload["subtype"] == "helmet"
    assert payload["zone_id"] == "z1"
    assert payload["track_id"] == 7
    assert "helmet" in payload["message"]


def test_no_violation_while_the_item_reads_present():
    pipeline, alerts = make_pipeline(enabled_modules=["boundary", "ppe"])
    pipeline.store.save(
        [
            zone_from_dict(
                {"id": "z1", "type": "polygon", "points": SQUARE, "ppe_required": ["helmet"]}
            )
        ],
        "cam_01",
    )
    pipeline.ppe_classifier = FakeClassifier({"helmet": 0.95})  # confidently present
    pipeline.ppe_monitor = PPEMonitor()

    tracked = make_tracked()
    masked = np.zeros(1, dtype=bool)
    for _ in range(5):
        pipeline._process_ppe(make_frame(), tracked, masked, 640, 480)

    assert alerts.published == []


def test_excluded_people_are_never_checked():
    pipeline, alerts = make_pipeline(enabled_modules=["boundary", "ppe"])
    pipeline.store.save(
        [
            zone_from_dict(
                {"id": "z1", "type": "polygon", "points": SQUARE, "ppe_required": ["helmet"]}
            )
        ],
        "cam_01",
    )
    pipeline.ppe_classifier = FakeClassifier({"helmet": 0.05})
    pipeline.ppe_monitor = PPEMonitor()

    tracked = make_tracked()
    masked = np.ones(1, dtype=bool)  # this person is exclusion-masked
    pipeline._process_ppe(make_frame(), tracked, masked, 640, 480)

    assert pipeline.ppe_classifier.calls == 0
    assert alerts.published == []
