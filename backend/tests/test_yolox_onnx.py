"""Tests for the YOLOX pre/post-processing pipeline.

These stages fail silently rather than loudly - a wrong letterbox ratio or a mis-built
anchor grid produces plausible boxes in the wrong places, not an exception. Each stage
is therefore checked against values derived by hand.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from perimeter.detect.yolox_onnx import (
    PAD_VALUE,
    Detections,
    YoloxOnnx,
    build_grids,
    class_aware_nms,
    default_intra_op_threads,
    letterbox,
    nms,
)

# -- letterbox ------------------------------------------------------------


def test_letterbox_preserves_aspect_ratio_of_wide_image():
    image = np.full((360, 1280, 3), 200, np.uint8)
    padded, ratio = letterbox(image, (416, 416))

    assert padded.shape == (416, 416, 3)
    assert ratio == pytest.approx(416 / 1280)
    # 360 * (416/1280) = 117 rows of content, the rest padding.
    assert padded[116, 0, 0] == 200
    assert padded[130, 0, 0] == PAD_VALUE


def test_letterbox_places_image_top_left():
    """Top-left placement, not centred. Centring would shift every box by half the pad.

    Uses a 2:1 image so padding actually exists - a square image scales to fill the
    canvas exactly and would assert nothing.
    """
    image = np.full((100, 200, 3), 50, np.uint8)
    padded, ratio = letterbox(image, (416, 416))

    assert ratio == pytest.approx(416 / 200)
    assert padded[0, 0, 0] == 50, "content starts at the origin"
    assert padded[207, 415, 0] == 50, "content fills the full width of the top band"
    assert padded[415, 415, 0] == PAD_VALUE, "padding is at the bottom, not the top"
    assert padded[209, 0, 0] == PAD_VALUE


def test_letterbox_square_image_needs_no_padding():
    image = np.full((500, 500, 3), 77, np.uint8)
    padded, ratio = letterbox(image, (416, 416))

    assert ratio == pytest.approx(416 / 500)
    assert (padded == 77).all()


def test_letterbox_ratio_maps_boxes_back():
    """A box in letterboxed space divided by ratio must land on the original pixel."""
    image = np.zeros((720, 1280, 3), np.uint8)
    _, ratio = letterbox(image, (416, 416))

    assert (100.0 / ratio) == pytest.approx(100 * 1280 / 416)


# -- anchor grid ----------------------------------------------------------


def test_build_grids_anchor_count_matches_model_output():
    """416 -> 52^2 + 26^2 + 13^2 = 3549, which is what yolox_nano.onnx emits."""
    grids, strides = build_grids((416, 416))
    assert grids.shape == (1, 3549, 2)
    assert strides.shape == (1, 3549, 1)


def test_build_grids_640_matches_zoo_model():
    grids, _ = build_grids((640, 640))
    assert grids.shape == (1, 8400, 2)


def test_build_grids_strides_are_ordered_8_16_32():
    _, strides = build_grids((416, 416))
    flat = strides[0, :, 0]
    assert flat[0] == 8
    assert flat[52 * 52] == 16
    assert flat[52 * 52 + 26 * 26] == 32


def test_build_grids_first_anchor_is_origin():
    grids, _ = build_grids((416, 416))
    assert tuple(grids[0, 0]) == (0.0, 0.0)


# -- NMS ------------------------------------------------------------------


def test_nms_suppresses_heavy_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    assert nms(boxes, scores, 0.5) == [0]


def test_nms_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    assert sorted(nms(boxes, scores, 0.5)) == [0, 1]


def test_nms_keeps_highest_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32)
    scores = np.array([0.3, 0.95], np.float32)
    assert nms(boxes, scores, 0.5) == [1]


def test_nms_on_empty_input():
    assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.5) == []


def test_class_aware_nms_does_not_suppress_across_classes():
    """A person standing in front of a car must survive - overlapping, different class."""
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    class_ids = np.array([0, 2], np.int32)

    assert class_aware_nms(boxes, scores, class_ids, 0.5) == [0, 1]


def test_class_aware_nms_still_suppresses_within_a_class():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    class_ids = np.array([0, 0], np.int32)

    assert class_aware_nms(boxes, scores, class_ids, 0.5) == [0]


# -- Detections -----------------------------------------------------------


def test_detections_empty():
    empty = Detections.empty()
    assert len(empty) == 0
    assert empty.xyxy.shape == (0, 4)


def test_filter_classes_keeps_only_requested():
    det = Detections(
        xyxy=np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], np.float32),
        scores=np.array([0.9, 0.8, 0.7], np.float32),
        class_ids=np.array([0, 2, 0], np.int32),
    )
    persons = det.filter_classes({0})

    assert len(persons) == 2
    assert (persons.class_ids == 0).all()
    assert persons.scores.tolist() == pytest.approx([0.9, 0.7])


# -- threading heuristic --------------------------------------------------


def test_default_threads_leaves_a_core_free():
    """Oversubscribing ORT against OpenCV is the classic 'optimisation' regression."""
    threads = default_intra_op_threads()
    assert threads >= 1
    assert threads < (__import__("os").cpu_count() or 2)


# -- colour order resolution ---------------------------------------------


def test_colour_order_read_from_manifest_per_file(tmp_path):
    model = tmp_path / "yolox_nano.onnx"
    model.write_bytes(b"not a real model")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": [{"file": "yolox_nano.onnx", "to_rgb": False}]}), encoding="utf-8"
    )
    assert YoloxOnnx._resolve_colour_order(model, None) is False


def test_colour_order_explicit_argument_wins(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": [{"file": "m.onnx", "to_rgb": False}]}), encoding="utf-8"
    )
    assert YoloxOnnx._resolve_colour_order(model, True) is True


def test_colour_order_without_manifest_raises(tmp_path):
    """Silent guessing costs ~25% recall, so an unrecorded model must fail loudly."""
    model = tmp_path / "mystery.onnx"
    model.write_bytes(b"x")

    with pytest.raises(ValueError, match="Colour order"):
        YoloxOnnx._resolve_colour_order(model, None)


def test_missing_model_file_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="NOTICE.md"):
        YoloxOnnx(tmp_path / "absent.onnx")
