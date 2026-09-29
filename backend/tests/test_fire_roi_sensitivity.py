"""A fire_roi's negative conf_delta must actually lower the bar.

Regression: FireSmokeDetector used to decode at min(conf_fire, conf_smoke), so any box a
negative delta was meant to accept was discarded before the zone policy ever saw it - the
delta had no effect at all. These tests inject a RAW model output (not a stubbed
`detect()`), so the real decode threshold is exercised.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mongomock
import numpy as np
import pytest

from perimeter.boundary.zones import FIRE_ROI_MAX_SENSITIVITY, ZoneStore, zone_from_dict
from perimeter.cameras.models import camera_from_dict
from perimeter.capture.source import Frame
from perimeter.detect.firesmoke import FireSmokeDetector
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings

PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")
FIRE_MODEL = Path("models/firesmoke/model.onnx")

pytestmark = pytest.mark.skipif(
    not (PERSON_MODEL.is_file() and FIRE_MODEL.is_file()),
    reason="person and fire/smoke model files are required",
)

W, H = 640, 480
RIGHT = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]


class Bus:
    def __init__(self):
        self.published = []

    def publish(self, event):
        self.published.append(event)


def settings():
    """Both classes at the same bar, as in config/app.yaml when the bug was found (0.70 /
    0.70). With the code defaults (fire 0.45, smoke 0.35) the old decode threshold,
    min(fire, smoke), happened to sit low enough to hide the bug."""
    base = Settings()
    return replace(base, firesmoke=replace(base.firesmoke, conf_fire=0.60, conf_smoke=0.60))


def build(conf_delta):
    store = ZoneStore(mongomock.MongoClient()["t"]["zones"], refresh_interval=0)
    zones = []
    if conf_delta is not None:
        zones.append({"id": "hot", "type": "fire_roi", "points": RIGHT,
                      "applies_to": ["fire", "smoke"], "conf_delta": conf_delta})
    store.save([zone_from_dict(z) for z in zones], "cam_01")
    camera = camera_from_dict({"id": "cam_01", "source_type": "webcam", "source": 0,
                               "enabled_modules": ["fire_smoke"]})
    bus = Bus()
    pipeline = Pipeline(settings(), camera, store, str(PERSON_MODEL),
                        firesmoke_model_path=str(FIRE_MODEL), alerts=bus)
    return pipeline, bus


def raw_fire(model, frame_xy, score):
    """A raw YOLOX output with one fire anchor of `score` centred at frame_xy."""
    ratio = min(model.input_size[0] / H, model.input_size[1] / W)
    target = np.array(frame_xy, np.float32) * ratio
    centres = (model._grids[0] + 0.5) * model._strides[0]
    index = int(np.argmin(np.linalg.norm(centres - target, axis=1)))
    raw = np.zeros((1, centres.shape[0], 7), np.float32)
    stride = model._strides[0][index, 0]
    raw[0, index, 0:2] = target / stride - model._grids[0][index]
    raw[0, index, 2:4] = np.log(40.0 * ratio / stride)
    raw[0, index, 4] = 1.0  # objectness
    raw[0, index, 5] = score  # fire class
    return raw


def feed(pipeline, raw, times=10):
    pipeline.firesmoke_detector.model.forward = lambda blob: raw
    image = np.zeros((H, W, 3), np.uint8)
    for i in range(times):
        pipeline._process_firesmoke(Frame(seq=i, ts=1.0 + i, image=image))


def fires(bus):
    return [e for e in bus.published if e["kind"] == "fire"]


def test_detector_decodes_down_to_the_strongest_roi_sensitivity():
    detector = FireSmokeDetector(FIRE_MODEL, conf_fire=0.70, conf_smoke=0.70,
                                 max_roi_sensitivity=0.5)
    assert detector.decode_threshold == pytest.approx(0.20)
    assert detector.model.conf_threshold == pytest.approx(0.20)


def test_detector_without_sensitivity_keeps_the_old_decode_threshold():
    detector = FireSmokeDetector(FIRE_MODEL, conf_fire=0.45, conf_smoke=0.70)
    assert detector.model.conf_threshold == pytest.approx(0.45)


def test_a_negative_roi_delta_accepts_a_fire_below_the_base_threshold():
    pipeline, bus = build(conf_delta=-0.10)
    base = pipeline.settings.firesmoke.conf_fire
    feed(pipeline, raw_fire(pipeline.firesmoke_detector.model, (480, 240), base - 0.05))
    assert fires(bus) and fires(bus)[0]["zone_id"] == "hot"


def test_the_same_score_without_an_roi_does_not_alert():
    pipeline, bus = build(conf_delta=None)
    base = pipeline.settings.firesmoke.conf_fire
    feed(pipeline, raw_fire(pipeline.firesmoke_detector.model, (480, 240), base - 0.05))
    assert fires(bus) == []


def test_the_delta_is_clamped_to_the_decode_floor():
    pipeline, _bus = build(conf_delta=-0.9)  # beyond the dashboard's -0.5 limit
    delta = pipeline.engine.confidence_delta((480.0, 240.0), W, H)
    assert delta == pytest.approx(-FIRE_ROI_MAX_SENSITIVITY)


def test_weak_boxes_far_below_the_bar_are_not_drawn_on_the_overlay():
    pipeline, _bus = build(conf_delta=None)
    base = pipeline.settings.firesmoke.conf_fire
    model = pipeline.firesmoke_detector.model

    feed(pipeline, raw_fire(model, (480, 240), base - 0.30), times=1)
    assert pipeline._fire_boxes[0] == []  # decoded, but not a near miss: not drawn

    feed(pipeline, raw_fire(model, (480, 240), base - 0.05), times=1)
    assert [counted for *_rest, counted in pipeline._fire_boxes[0]] == [False]  # grey
