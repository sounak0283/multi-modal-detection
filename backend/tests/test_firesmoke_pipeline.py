"""Tests for Pipeline's fire/smoke module wiring (Expansion Plan Phase E).

Constructing a real `Pipeline` needs the real person-detection ONNX weights (the person
detector is not optional), so these are skipped on a checkout without them - same
`pytest.mark.skipif` pattern `test_detect_integration.py` already uses. `Pipeline.__init__`
never opens camera hardware itself (only `.start()` does), so this is safe to construct
directly without a real camera.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.models import camera_from_dict
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings

PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")
MISSING_MODEL = Path("models/firesmoke/does-not-exist.onnx")

pytestmark = pytest.mark.skipif(
    not PERSON_MODEL.is_file(),
    reason="yolox_nano.onnx not present - see README for the download step",
)


def make_camera(enabled_modules):
    return camera_from_dict(
        {"id": "cam_01", "source_type": "webcam", "source": 0, "enabled_modules": enabled_modules}
    )


def make_store():
    import mongomock

    db = mongomock.MongoClient()["perimeter_test"]
    return ZoneStore(db["zones"], refresh_interval=0)


def test_firesmoke_stays_inactive_when_the_module_is_not_enabled():
    camera = make_camera(["boundary"])
    pipeline = Pipeline(
        Settings(), camera, make_store(), str(PERSON_MODEL),
        firesmoke_model_path=str(PERSON_MODEL),  # a real, loadable model - shouldn't matter
    )
    assert pipeline.firesmoke_detector is None
    assert pipeline.firesmoke_gate is None


def test_firesmoke_stays_inactive_when_no_model_path_was_given():
    camera = make_camera(["boundary", "fire_smoke"])
    pipeline = Pipeline(Settings(), camera, make_store(), str(PERSON_MODEL))
    assert pipeline.firesmoke_detector is None


def test_firesmoke_degrades_gracefully_when_the_model_file_is_missing():
    """The one behaviour actually exercisable without real fire/smoke weights: the
    pipeline must not raise, just log and leave the module inactive."""
    camera = make_camera(["boundary", "fire_smoke"])
    pipeline = Pipeline(
        Settings(), camera, make_store(), str(PERSON_MODEL),
        firesmoke_model_path=str(MISSING_MODEL),
    )
    assert pipeline.firesmoke_detector is None
    assert pipeline.firesmoke_gate is None


def test_firesmoke_activates_when_enabled_and_a_loadable_model_is_given():
    """Stands in for a real trained model with the person-detection weights - same
    architecture/wrapper, per detect/yolox_onnx.py's own docstring, so this proves the
    construction wiring works end to end without needing real fire/smoke weights."""
    camera = make_camera(["boundary", "fire_smoke"])
    pipeline = Pipeline(
        Settings(), camera, make_store(), str(PERSON_MODEL),
        firesmoke_model_path=str(PERSON_MODEL),
    )
    assert pipeline.firesmoke_detector is not None
    assert pipeline.firesmoke_gate is not None
