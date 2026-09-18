"""Every detection module is opt-in, boundary included."""

from __future__ import annotations

from pathlib import Path

import mongomock
import pytest

from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.models import camera_from_dict
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings

PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")
pytestmark = pytest.mark.skipif(not PERSON_MODEL.is_file(), reason="yolox_nano.onnx not present")


def make_pipeline(modules):
    camera = camera_from_dict(
        {"id": "cam_01", "source_type": "webcam", "source": 0, "enabled_modules": modules}
    )
    store = ZoneStore(mongomock.MongoClient()["t"]["zones"], refresh_interval=0)
    return Pipeline(Settings(), camera, store, str(PERSON_MODEL))


def test_a_camera_with_no_modules_loads_no_person_model():
    pipeline = make_pipeline([])

    assert pipeline.detector is None
    assert pipeline.firesmoke_detector is None
    assert pipeline.crowd_monitor is None


def test_enabling_boundary_loads_the_person_model():
    assert make_pipeline(["boundary"]).detector is not None


class IdleCapture:
    """Stands in for CaptureThread so reconfigure() never opens real camera hardware."""

    def start(self):
        pass

    def stop(self, timeout=None):
        pass


def test_enabling_boundary_later_loads_the_model_and_disabling_drops_it(monkeypatch):
    pipeline = make_pipeline([])
    pipeline._capture = IdleCapture()
    monkeypatch.setattr(pipeline, "_build_capture", lambda camera: IdleCapture())

    pipeline.reconfigure(camera_from_dict({"id": "cam_01", "enabled_modules": ["boundary"]}))
    assert pipeline.detector is not None

    pipeline.reconfigure(camera_from_dict({"id": "cam_01", "enabled_modules": []}))
    assert pipeline.detector is None
