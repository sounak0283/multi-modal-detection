"""Tests for PPEHelmetDetector - the adapter mapping demo_package's head/helmet detector
onto the PPE classifier contract (`{item: probability}` -> `classify_state`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perimeter.detect.ppe import PPE_ITEMS, classify_state
from perimeter.detect.ppe_helmet import (
    HEAD,
    HELMET,
    PPEHelmetDetector,
    helmet_probability,
)
from perimeter.detect.yolox_onnx import Detections

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
PERSON = (200.0, 100.0, 300.0, 400.0)  # 100x300; padded crop origin = (190, 64)
ORIGIN = np.array([190.0, 64.0, 190.0, 64.0], dtype=np.float32)


class FakeModel:
    """Returns fixed detections, given in FRAME coordinates for readability."""

    def __init__(self, dets=()):
        self.dets = list(dets)
        self.calls = 0

    def detect(self, crop):
        self.calls += 1
        if not self.dets:
            return Detections.empty()
        boxes = np.array([d[0] for d in self.dets], dtype=np.float32) - ORIGIN
        return Detections(
            xyxy=boxes,
            scores=np.array([d[1] for d in self.dets], dtype=np.float32),
            class_ids=np.array([d[2] for d in self.dets], dtype=np.int32),
        )


def detector(*dets):
    return PPEHelmetDetector("unused.onnx", model=FakeModel(dets))


HEAD_BOX = (230.0, 100.0, 270.0, 140.0)  # top-centre of PERSON


# -- probability mapping ---------------------------------------------------------------


def test_helmet_maps_into_the_present_band():
    assert classify_state(helmet_probability(HELMET, 0.55, 0.55)) == "present"
    assert classify_state(helmet_probability(HELMET, 0.99, 0.55)) == "present"


def test_bare_head_maps_into_the_absent_band():
    assert classify_state(helmet_probability(HEAD, 0.45, 0.45)) == "absent"
    assert classify_state(helmet_probability(HEAD, 0.99, 0.45)) == "absent"


def test_more_confident_detections_move_further_from_the_gap():
    assert helmet_probability(HELMET, 0.9, 0.55) > helmet_probability(HELMET, 0.6, 0.55)
    assert helmet_probability(HEAD, 0.9, 0.45) < helmet_probability(HEAD, 0.5, 0.45)


# -- per-person decision -----------------------------------------------------------------


def test_helmet_on_the_head_reads_present_with_a_frame_space_box():
    result = detector((HEAD_BOX, 0.9, HELMET)).classify_person(FRAME, PERSON)
    assert classify_state(result.probabilities["helmet"]) == "present"
    assert result.label == "HELMET"
    assert result.head_box == pytest.approx(HEAD_BOX)


def test_bare_head_reads_absent():
    result = detector((HEAD_BOX, 0.8, HEAD)).classify_person(FRAME, PERSON)
    assert classify_state(result.probabilities["helmet"]) == "absent"
    assert result.label == "NO HELMET"


def test_nothing_found_is_indeterminate_not_a_violation():
    result = detector().classify_person(FRAME, PERSON)
    assert classify_state(result.probabilities["helmet"]) == "indeterminate"
    assert result.head_box is None


def test_people_below_the_trained_minimum_height_are_not_checked():
    fake = FakeModel([(HEAD_BOX, 0.9, HEAD)])
    small = PPEHelmetDetector("unused.onnx", model=fake, min_person_height_px=96)
    result = small.classify_person(FRAME, (200.0, 100.0, 240.0, 190.0))  # 90 px tall
    assert classify_state(result.probabilities["helmet"]) == "indeterminate"
    assert fake.calls == 0


def test_per_class_thresholds_apply():
    below_helmet = detector((HEAD_BOX, 0.50, HELMET))  # helmet threshold is 0.55
    assert classify_state(below_helmet.classify_person(FRAME, PERSON).probabilities["helmet"]) == (
        "indeterminate"
    )


def test_a_head_covered_by_a_stronger_helmet_box_is_a_helmet():
    result = detector((HEAD_BOX, 0.6, HEAD), (HEAD_BOX, 0.9, HELMET)).classify_person(
        FRAME, PERSON
    )
    assert result.label == "HELMET"


def test_a_weak_helmet_over_a_stronger_bare_head_is_a_bare_head():
    result = detector((HEAD_BOX, 0.9, HEAD), (HEAD_BOX, 0.6, HELMET)).classify_person(
        FRAME, PERSON
    )
    assert result.label == "NO HELMET"


def test_a_neighbours_head_inside_the_padded_crop_is_ignored():
    neighbour_head = (192.0, 90.0, 198.0, 120.0)  # centre x=195, left of PERSON's x1=200
    result = detector((neighbour_head, 0.95, HEAD)).classify_person(FRAME, PERSON)
    assert classify_state(result.probabilities["helmet"]) == "indeterminate"


def test_a_head_shaped_detection_low_on_the_body_is_ignored():
    low = (230.0, 300.0, 270.0, 340.0)  # centre y=320, lower half of the person
    result = detector((low, 0.95, HEAD)).classify_person(FRAME, PERSON)
    assert classify_state(result.probabilities["helmet"]) == "indeterminate"


def test_the_head_nearest_the_persons_centre_decides():
    off_centre_helmet = (201.0, 100.0, 221.0, 130.0)
    result = detector((off_centre_helmet, 0.99, HELMET), (HEAD_BOX, 0.7, HEAD)).classify_person(
        FRAME, PERSON
    )
    assert result.label == "NO HELMET"


def test_items_the_model_cannot_see_are_always_indeterminate():
    result = detector((HEAD_BOX, 0.9, HEAD)).classify_person(FRAME, PERSON)
    for item in PPE_ITEMS:
        if item != "helmet":
            assert classify_state(result.probabilities[item]) == "indeterminate"
    assert PPEHelmetDetector.supported_items == frozenset({"helmet"})


def test_legacy_classify_treats_the_whole_crop_as_the_person():
    head_in_crop = np.array([[40.0, 0.0, 80.0, 40.0]], np.float32)

    class CropModel:
        def detect(self, crop):
            scores, classes = np.array([0.9], np.float32), np.array([HELMET], np.int32)
            return Detections(head_in_crop, scores, classes)

    crop = np.zeros((300, 100, 3), dtype=np.uint8)
    probabilities = PPEHelmetDetector("unused.onnx", model=CropModel()).classify(crop)
    assert classify_state(probabilities["helmet"]) == "present"


# -- the real model, when installed ------------------------------------------------------

REAL_MODEL = Path("models/ppe/ppe_final_C_v2_320.onnx")
PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")
SAMPLE = Path("../third_party/demo_package/samples/test_input.mp4")


@pytest.mark.skipif(
    not (REAL_MODEL.is_file() and PERSON_MODEL.is_file()),
    reason="PPE weights not installed - run tools/install_ppe_model.py",
)
def test_real_model_loads_and_returns_the_classifier_shape():
    real = PPEHelmetDetector(REAL_MODEL)
    result = real.classify_person(np.zeros((480, 640, 3), np.uint8), PERSON)
    assert set(result.probabilities) == set(PPE_ITEMS)
    assert classify_state(result.probabilities["helmet"]) == "indeterminate"  # blank frame


@pytest.mark.skipif(
    not (REAL_MODEL.is_file() and PERSON_MODEL.is_file()),
    reason="PPE weights not installed - run tools/install_ppe_model.py",
)
def test_real_model_never_modifies_the_shared_frame():
    """The crop is a view into the frame every other module reads (fire, sparks,
    identity, evidence); inference must not write through it."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    before = frame.copy()
    PPEHelmetDetector(REAL_MODEL).classify_person(frame, PERSON)
    assert np.array_equal(frame, before)


@pytest.mark.skipif(
    not (REAL_MODEL.is_file() and PERSON_MODEL.is_file() and SAMPLE.is_file()),
    reason="PPE weights or demo_package sample video not present",
)
def test_real_model_on_the_package_sample_finds_helmets_and_bare_heads():
    import cv2

    from perimeter.detect.person import PersonDetector

    people = PersonDetector(PERSON_MODEL)
    real = PPEHelmetDetector(REAL_MODEL)
    cap = cv2.VideoCapture(str(SAMPLE))
    labels = []
    for index in range(0, 600, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        assert ok
        for box in people.detect(frame).xyxy:
            labels.append(real.classify_person(frame, box).label)
    cap.release()
    assert labels.count("HELMET") > 10
    assert labels.count("NO HELMET") > 10
