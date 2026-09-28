"""Pipeline wiring for the PPE helmet detector backend: backend selection, items the
model cannot see, and the live overlay. Reuses test_ppe_pipeline.py's fixtures."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from perimeter.boundary.zones import zone_from_dict
from perimeter.detect.ppe import PPE_ITEMS, PPEClassifier, PPEMonitor, PPEResult
from perimeter.detect.ppe_helmet import PPEHelmetDetector
from perimeter.pipeline import Pipeline
from perimeter.settings import PPESettings, Settings
from tests.test_ppe_pipeline import (
    PERSON_MODEL,
    SQUARE,
    RecordingAlertBus,
    make_camera,
    make_frame,
    make_store,
    make_tracked,
)

HELMET_MODEL = Path("models/ppe/ppe_final_C_v2_320.onnx")

pytestmark = pytest.mark.skipif(
    not PERSON_MODEL.is_file(), reason="yolox_nano.onnx not present"
)


class FakePersonChecker:
    """Duck-types the helmet detector's classify_person()."""

    supported_items = frozenset({"helmet"})

    def __init__(self, helmet: float, head_box=(110.0, 100.0, 140.0, 130.0), label="NO HELMET"):
        self.helmet, self.head_box, self.label = helmet, head_box, label
        self.calls = []

    def classify_person(self, frame, box):
        self.calls.append(box)
        probabilities = dict.fromkeys(PPE_ITEMS, 0.5)
        probabilities["helmet"] = self.helmet
        return PPEResult(probabilities, head_box=self.head_box, label=self.label)


def make_pipeline(settings=None, ppe_model_path=None):
    alerts = RecordingAlertBus()
    pipeline = Pipeline(
        settings or Settings(), make_camera(["boundary", "ppe"]), make_store(),
        str(PERSON_MODEL), alerts=alerts, ppe_model_path=ppe_model_path,
    )
    return pipeline, alerts


def save_zone(pipeline, ppe_required):
    pipeline.store.save(
        [zone_from_dict({"id": "z1", "type": "polygon", "points": SQUARE,
                         "ppe_required": ppe_required})],
        "cam_01",
    )


@pytest.mark.skipif(not HELMET_MODEL.is_file(), reason="PPE weights not installed")
def test_default_backend_builds_the_helmet_detector():
    pipeline, _ = make_pipeline(ppe_model_path=str(HELMET_MODEL))
    assert isinstance(pipeline.ppe_classifier, PPEHelmetDetector)
    assert isinstance(pipeline.ppe_monitor, PPEMonitor)


@pytest.mark.skipif(not HELMET_MODEL.is_file(), reason="PPE weights not installed")
def test_classifier_backend_switch_builds_the_old_classifier():
    settings = replace(Settings(), ppe=PPESettings(backend="classifier"))
    pipeline, _ = make_pipeline(settings, ppe_model_path=str(HELMET_MODEL))
    assert isinstance(pipeline.ppe_classifier, PPEClassifier)


def test_a_bare_head_confirmed_over_three_checks_raises_one_helmet_alert():
    pipeline, alerts = make_pipeline()
    save_zone(pipeline, ["helmet"])
    pipeline.ppe_classifier = FakePersonChecker(helmet=0.1)
    pipeline.ppe_monitor = PPEMonitor()

    for _ in range(5):
        pipeline._process_ppe(make_frame(), make_tracked(), np.zeros(1, bool), 640, 480)

    assert [a["subtype"] for a in alerts.published] == ["helmet"]
    assert pipeline.ppe_classifier.calls[0] == (100, 100, 150, 300)


def test_items_the_model_cannot_see_never_alert_and_warn_once(caplog):
    pipeline, alerts = make_pipeline()
    save_zone(pipeline, ["vest"])
    checker = FakePersonChecker(helmet=0.1)
    pipeline.ppe_classifier = checker
    pipeline.ppe_monitor = PPEMonitor()

    with caplog.at_level(logging.WARNING, logger="perimeter.pipeline"):
        for _ in range(5):
            pipeline._process_ppe(make_frame(), make_tracked(), np.zeros(1, bool), 640, 480)

    assert alerts.published == []
    assert checker.calls == []  # nothing this model can check, so it is not even run
    assert sum("'vest'" in r.getMessage() for r in caplog.records) == 1


def test_overlay_marks_the_head_and_flags_it_once_confirmed():
    pipeline, _ = make_pipeline()
    save_zone(pipeline, ["helmet"])
    pipeline.ppe_classifier = FakePersonChecker(helmet=0.1)
    pipeline.ppe_monitor = PPEMonitor()
    pipeline._tracked = make_tracked()
    pipeline._masked = np.zeros(1, bool)

    pipeline._process_ppe(make_frame(), make_tracked(), np.zeros(1, bool), 640, 480)
    _box, label, violating, _expiry = pipeline._ppe_marks[7]
    assert (label, violating) == ("NO HELMET", False)

    for _ in range(2):
        pipeline._process_ppe(make_frame(), make_tracked(), np.zeros(1, bool), 640, 480)
    assert pipeline._ppe_marks[7][2] is True

    canvas = pipeline._annotate(make_frame().image)
    # Red (BGR 0,0,255) rectangle edge on the head box's top-left corner.
    assert tuple(canvas[100, 110]) == (0, 0, 255)
