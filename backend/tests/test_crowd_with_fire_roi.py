"""Crowd formation and a fire ROI on the same camera: both must work, independently.

Drives the real `Pipeline` (person model + fire model files present) with hand-built
detections, the same way test_crowd_pipeline.py and test_firesmoke_pipeline.py do.
"""

from __future__ import annotations

from pathlib import Path

import mongomock
import numpy as np
import pytest

from perimeter.boundary.zones import ZoneStore, zone_from_dict
from perimeter.cameras.models import camera_from_dict
from perimeter.capture.source import Frame
from perimeter.detect.yolox_onnx import Detections
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings
from perimeter.track.tracker import TrackedDetections

PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")
FIRE_MODEL = Path("models/firesmoke/model.onnx")

pytestmark = pytest.mark.skipif(
    not (PERSON_MODEL.is_file() and FIRE_MODEL.is_file()),
    reason="person and fire/smoke model files are required",
)

W, H = 640, 480
LEFT = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
RIGHT = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
CROWD = [(100, 200), (110, 205), (95, 210), (105, 195), (100, 190)]  # left half


class Bus:
    def __init__(self):
        self.published = []

    def publish(self, event):
        self.published.append(event)


def build(zones):
    store = ZoneStore(mongomock.MongoClient()["t"]["zones"], refresh_interval=0)
    store.save([zone_from_dict(z) for z in zones], "cam_01")
    camera = camera_from_dict({
        "id": "cam_01", "source_type": "webcam", "source": 0,
        "enabled_modules": ["boundary", "crowd", "fire_smoke"],
    })
    bus = Bus()
    pipeline = Pipeline(
        Settings(), camera, store, str(PERSON_MODEL),
        firesmoke_model_path=str(FIRE_MODEL), alerts=bus,
    )
    return pipeline, bus


def tracked_people(points):
    boxes = np.array([[x - 10, y - 50, x + 10, y] for x, y in points], np.float32)
    n = len(points)
    return TrackedDetections(
        xyxy=boxes, scores=np.ones(n, np.float32),
        class_ids=np.zeros(n, np.int32), track_ids=np.arange(n, dtype=np.int32),
    )


def fire_at(cx, cy, score):
    return Detections(
        xyxy=np.array([[cx - 30, cy - 30, cx + 30, cy + 30]], np.float32),
        scores=np.array([score], np.float32),
        class_ids=np.array([0], np.int32),  # fire
    )


ZONES = [
    {"id": "crowd_area", "type": "polygon", "points": LEFT, "crowd_threshold": 4,
     "crowd_min_frames": 2, "detect": ["person"], "events": ["entry", "exit"]},
    {"id": "hot_spot", "type": "fire_roi", "points": RIGHT, "applies_to": ["fire", "smoke"],
     "conf_delta": -0.10},
]


def run_fire(pipeline, detection, times=10):
    frame = Frame(seq=0, ts=1.0, image=np.zeros((H, W, 3), np.uint8))
    pipeline.firesmoke_detector.detect = lambda image: detection
    for i in range(times):
        frame = Frame(seq=i, ts=1.0 + i, image=frame.image)
        pipeline._process_firesmoke(frame)


def test_crowd_alert_still_fires_when_a_fire_roi_is_present():
    pipeline, bus = build(ZONES)
    masked = np.zeros(len(CROWD), dtype=bool)
    for _ in range(3):
        pipeline._process_crowd(tracked_people(CROWD), masked, W, H, ts=0.0)

    crowd = [e for e in bus.published if e["kind"] == "crowd"]
    assert len(crowd) == 1 and crowd[0]["zone_id"] == "crowd_area"


def test_fire_inside_the_roi_alerts_at_a_lower_score_than_the_base_threshold():
    pipeline, bus = build(ZONES)
    base = pipeline.settings.firesmoke.conf_fire
    run_fire(pipeline, fire_at(480, 240, base - 0.05))  # right half = inside the ROI

    fire = [e for e in bus.published if e["kind"] == "fire"]
    assert fire and fire[0]["zone_id"] == "hot_spot"


def test_fire_outside_the_roi_still_needs_the_normal_threshold():
    pipeline, bus = build(ZONES)
    base = pipeline.settings.firesmoke.conf_fire
    run_fire(pipeline, fire_at(150, 240, base - 0.05))  # left half, no ROI here
    assert [e for e in bus.published if e["kind"] == "fire"] == []

    run_fire(pipeline, fire_at(150, 240, base + 0.05))  # above the normal bar: alerts
    assert [e for e in bus.published if e["kind"] == "fire"]


def test_both_alert_together_and_neither_blocks_the_other():
    pipeline, bus = build(ZONES)
    masked = np.zeros(len(CROWD), dtype=bool)
    base = pipeline.settings.firesmoke.conf_fire

    for _ in range(3):
        pipeline._process_crowd(tracked_people(CROWD), masked, W, H, ts=0.0)
    run_fire(pipeline, fire_at(480, 240, base + 0.1))

    kinds = sorted(e["kind"] for e in bus.published)
    assert "crowd" in kinds and "fire" in kinds


def test_a_crowd_inside_the_fire_roi_is_not_a_crowd_alert():
    """The ROI only tunes fire sensitivity: it is not a crowd zone, so people clustered in
    it raise nothing."""
    pipeline, bus = build(ZONES)
    right_crowd = [(x + 400, y) for x, y in CROWD]
    masked = np.zeros(len(right_crowd), dtype=bool)
    for _ in range(3):
        pipeline._process_crowd(tracked_people(right_crowd), masked, W, H, ts=0.0)
    assert [e for e in bus.published if e["kind"] == "crowd"] == []
