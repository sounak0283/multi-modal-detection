"""End-to-end detector checks against the real ONNX weights.

Skipped when the weights are absent, so a clean checkout still passes CI. These are the
tests that would catch a regression in the decode maths, which the unit tests cannot:
they check that real pixels produce real boxes in the right places.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perimeter.detect.person import COCO_PERSON_CLASS_ID, PersonDetector, foot_points
from perimeter.detect.yolox_onnx import Detections

MODEL = Path("models/yolox_person/yolox_nano.onnx")

pytestmark = pytest.mark.skipif(
    not MODEL.is_file(), reason="yolox_nano.onnx not present - see README for the download step"
)


@pytest.fixture(scope="module")
def detector() -> PersonDetector:
    return PersonDetector(MODEL)


def test_input_size_is_the_planned_416(detector):
    assert detector.input_size == (416, 416)


def test_colour_order_is_bgr_from_manifest(detector):
    """Megvii exports expect BGR. Regressing this silently costs ~25% recall."""
    assert detector.model.to_rgb is False


def test_blank_frame_yields_no_people(detector):
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert len(detector.detect(blank)) == 0


def test_detects_a_synthetic_figure_or_stays_quiet(detector):
    """Never asserts a false positive: a grey box is not a person."""
    frame = np.full((480, 640, 3), 200, np.uint8)
    frame[150:400, 280:360] = 60
    detections = detector.detect(frame)
    assert all(c == COCO_PERSON_CLASS_ID for c in detections.class_ids)


def test_boxes_are_clipped_to_frame(detector):
    frame = np.random.default_rng(1).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    det = detector.detect(frame)

    if len(det):
        assert det.xyxy[:, 0].min() >= 0
        assert det.xyxy[:, 1].min() >= 0
        assert det.xyxy[:, 2].max() <= 639
        assert det.xyxy[:, 3].max() <= 359


def test_min_box_height_filter_drops_small_boxes():
    detector = PersonDetector(MODEL, min_box_height_px=10_000)
    frame = np.random.default_rng(2).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    assert len(detector.detect(frame)) == 0


def test_foot_point_is_bottom_centre_not_centroid():
    """PLAN.md section 6.1: centroids break under perspective."""
    det = Detections(
        xyxy=np.array([[100.0, 50.0, 200.0, 450.0]], np.float32),
        scores=np.array([0.9], np.float32),
        class_ids=np.array([0], np.int32),
    )
    points = foot_points(det)

    assert points.tolist() == [[150.0, 450.0]]


def test_foot_points_empty_input():
    assert foot_points(Detections.empty()).shape == (0, 2)
