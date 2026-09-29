"""Fire/smoke alarm timing: alarm on the first detection, then at most one alarm per
class per `alert_gap_seconds` (30 s) while it keeps being detected. `alert_gap_seconds: 0`
restores the K-of-N gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mongomock
import numpy as np
import pytest

from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.models import camera_from_dict
from perimeter.capture.source import Frame
from perimeter.detect.yolox_onnx import Detections
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings

PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")
FIRE_MODEL = Path("models/firesmoke/model.onnx")

pytestmark = pytest.mark.skipif(
    not (PERSON_MODEL.is_file() and FIRE_MODEL.is_file()),
    reason="person and fire/smoke model files are required",
)

CHECK_EVERY_S = 8 / 12.0  # firesmoke_every_n / decode_fps
FIRE, SMOKE = 0, 1


class Bus:
    def __init__(self):
        self.published = []

    def publish(self, event):
        self.published.append(event)


def build(gap=30.0):
    base = Settings()
    settings = replace(base, firesmoke=replace(base.firesmoke, alert_gap_seconds=gap))
    store = ZoneStore(mongomock.MongoClient()["t"]["zones"], refresh_interval=0)
    camera = camera_from_dict({"id": "cam_01", "source_type": "webcam", "source": 0,
                               "enabled_modules": ["fire_smoke"]})
    bus = Bus()
    pipeline = Pipeline(settings, camera, store, str(PERSON_MODEL),
                        firesmoke_model_path=str(FIRE_MODEL), alerts=bus)
    return pipeline, bus


def dets(*items):
    """items: (class_id, score, size) - boxes centred at (300, 240)."""
    if not items:
        return Detections.empty()
    boxes = [[300 - s, 240 - s, 300 + s, 240 + s] for _c, _sc, s in items]
    return Detections(np.array(boxes, np.float32), np.array([sc for _c, sc, _s in items],
                      np.float32), np.array([c for c, _sc, _s in items], np.int32))


def check(pipeline, detections, t, video_pos_s=None):
    pipeline.firesmoke_detector.detect = lambda image: detections
    pipeline._process_firesmoke(Frame(seq=0, ts=t, image=np.zeros((480, 640, 3), np.uint8),
                                      video_pos_s=video_pos_s))


def alarms(bus, kind=None):
    return [e for e in bus.published if kind is None or e["kind"] == kind]


def test_the_first_detection_alarms_immediately():
    pipeline, bus = build()
    check(pipeline, dets((FIRE, 0.9, 30)), t=100.0)
    assert len(alarms(bus, "fire")) == 1


def test_a_continuous_fire_alarms_once_every_30_seconds():
    pipeline, bus = build()
    t = 0.0
    while t < 70.0:  # 70 s of fire, checked at the real cadence
        check(pipeline, dets((FIRE, 0.9, 30)), t)
        t += CHECK_EVERY_S
    times = [e["ts"] for e in alarms(bus, "fire")]
    assert len(times) == 3  # at ~0 s, ~30 s, ~60 s
    assert all(b - a >= 30.0 for a, b in zip(times, times[1:], strict=False))


def test_nothing_between_alarms_even_if_the_fire_flickers():
    pipeline, bus = build()
    check(pipeline, dets((FIRE, 0.9, 30)), 0.0)
    for i in range(1, 40):  # ~26 s of flickering detections
        check(pipeline, dets((FIRE, 0.9, 30)) if i % 2 else dets(), i * CHECK_EVERY_S)
    assert len(alarms(bus, "fire")) == 1


def test_fire_and_smoke_have_separate_timers():
    pipeline, bus = build()
    check(pipeline, dets((FIRE, 0.9, 30)), 0.0)
    check(pipeline, dets((FIRE, 0.9, 30), (SMOKE, 0.9, 60)), 5.0)
    assert len(alarms(bus, "fire")) == 1 and len(alarms(bus, "smoke")) == 1


def test_detections_below_the_threshold_never_alarm():
    pipeline, bus = build()
    bar = pipeline.settings.firesmoke.conf_fire
    for i in range(20):
        check(pipeline, dets((FIRE, bar - 0.05, 30)), i * CHECK_EVERY_S)
    assert alarms(bus) == []


def test_the_gap_is_measured_in_video_time_for_file_sources():
    """The Video test page runs faster than real time: 30 s of video can pass in a few
    seconds of wall clock, and must still space alarms 30 s apart in the video."""
    pipeline, bus = build()
    for i in range(100):  # 100 checks ~ 66 s of video, but only 10 s of wall clock
        check(pipeline, dets((FIRE, 0.9, 30)), t=i * 0.1, video_pos_s=i * CHECK_EVERY_S)
    assert len(alarms(bus, "fire")) == 3


def test_a_growing_fire_is_critical():
    pipeline, bus = build()
    for i, size in enumerate((20, 30, 40, 50)):
        check(pipeline, dets((FIRE, 0.9, size)), i * 40.0)  # 40 s apart: each alarms
    assert alarms(bus, "fire")[0]["severity"] == "high"
    assert alarms(bus, "fire")[-1]["severity"] == "critical"


def test_gap_zero_restores_the_k_of_n_gate():
    pipeline, bus = build(gap=0.0)
    k = pipeline.settings.firesmoke.gate_k
    for i in range(k - 1):
        check(pipeline, dets((FIRE, 0.9, 30)), i * CHECK_EVERY_S)
    assert alarms(bus) == []
    check(pipeline, dets((FIRE, 0.9, 30)), k * CHECK_EVERY_S)
    assert len(alarms(bus, "fire")) == 1
