"""Tests for the tracking wrapper.

The boundary state machine keys entry/exit state on track id (PLAN.md section 6.3), so
anything that lets two people share an id, or that silently shortens occlusion
tolerance, corrupts events rather than merely degrading tracking quality. Those two
failure modes are what these tests pin down.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from fsbd.detect.yolox_onnx import Detections
from fsbd.track.tracker import (
    PersonTracker,
    TrackedDetections,
    from_supervision,
    to_supervision,
)


def make_sv(xyxy, tracker_id, confidence=None, class_id=None) -> sv.Detections:
    n = len(xyxy)
    return sv.Detections(
        xyxy=np.asarray(xyxy, np.float64).reshape(n, 4),
        confidence=np.asarray(confidence if confidence is not None else [0.9] * n, np.float64),
        class_id=np.asarray(class_id if class_id is not None else [0] * n, int),
        tracker_id=np.asarray(tracker_id, int),
    )


# -- the -1 sentinel ------------------------------------------------------


def test_unconfirmed_detections_are_dropped():
    """Unconfirmed tracks come back as -1, NOT as None - tracker_id is an int array.

    Regression test: filtering on `!= None` silently keeps everything, because a numpy
    integer array is never None elementwise.
    """
    tracked = from_supervision(
        make_sv([[0, 0, 10, 10], [20, 20, 30, 30]], tracker_id=[-1, 7])
    )

    assert len(tracked) == 1
    assert tracked.track_ids.tolist() == [7]


def test_several_unconfirmed_detections_do_not_collapse_into_one_identity():
    """Four unconfirmed people all carry id -1. Passing them through would make the
    boundary engine treat four strangers as a single track."""
    tracked = from_supervision(
        make_sv([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5], [6, 6, 7, 7]], [-1, -1, -1, -1])
    )

    assert len(tracked) == 0


def test_zero_is_a_valid_track_id_not_a_sentinel():
    """trackers 2.6.0 issues 0 as the first track id, unlike the original ByteTrack
    which starts at 1. Filtering on `> 0` would silently drop the first person to
    appear on every camera start."""
    tracked = from_supervision(make_sv([[0, 0, 1, 1], [2, 2, 3, 3]], [0, 3]))
    assert tracked.track_ids.tolist() == [0, 3]


def test_confirmed_track_ids_are_unique_within_a_frame():
    tracked = from_supervision(make_sv([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], [1, 2, 3]))
    assert len(set(tracked.track_ids.tolist())) == len(tracked)


def test_empty_and_untracked_inputs():
    assert len(from_supervision(sv.Detections.empty())) == 0
    assert len(from_supervision(make_sv([[0, 0, 1, 1]], [-1]))) == 0


# -- conversion round trip ------------------------------------------------


def test_to_supervision_preserves_values():
    det = Detections(
        xyxy=np.array([[1, 2, 3, 4]], np.float32),
        scores=np.array([0.75], np.float32),
        class_ids=np.array([0], np.int32),
    )
    converted = to_supervision(det)

    assert converted.xyxy.tolist() == [[1, 2, 3, 4]]
    assert converted.confidence[0] == pytest.approx(0.75)


def test_to_supervision_on_empty():
    assert len(to_supervision(Detections.empty())) == 0


# -- occlusion tolerance in seconds --------------------------------------


def test_lost_track_buffer_scales_with_detection_rate():
    """The buffer counts UPDATE CALLS. We update at the detection cadence, not the
    camera rate, so leaving the library default would quarter the real tolerance."""
    slow = PersonTracker(detection_fps=7.5, lost_track_seconds=5.0)
    fast = PersonTracker(detection_fps=30.0, lost_track_seconds=5.0)

    assert slow.lost_track_buffer == 38
    assert fast.lost_track_buffer == 150
    assert fast.lost_track_buffer > slow.lost_track_buffer


def test_lost_track_buffer_is_at_least_one_update():
    tracker = PersonTracker(detection_fps=1.0, lost_track_seconds=0.01)
    assert tracker.lost_track_buffer >= 1


def test_zero_detection_fps_is_rejected():
    with pytest.raises(ValueError, match="detection_fps"):
        PersonTracker(detection_fps=0)


# -- tracking behaviour ---------------------------------------------------


def test_track_id_is_stable_across_frames():
    tracker = PersonTracker(detection_fps=7.5, min_consecutive_frames=1)
    ids = []
    for step in range(6):
        det = Detections(
            xyxy=np.array([[100 + step * 4, 100, 160 + step * 4, 300]], np.float32),
            scores=np.array([0.9], np.float32),
            class_ids=np.array([0], np.int32),
        )
        tracked = tracker.update(det)
        if len(tracked):
            ids.append(int(tracked.track_ids[0]))

    assert ids, "a steadily moving box should produce a confirmed track"
    assert len(set(ids)) == 1, f"track id changed across frames: {ids}"


def test_reset_clears_state():
    tracker = PersonTracker(detection_fps=7.5, min_consecutive_frames=1)
    det = Detections(
        xyxy=np.array([[10, 10, 60, 200]], np.float32),
        scores=np.array([0.9], np.float32),
        class_ids=np.array([0], np.int32),
    )
    for _ in range(4):
        tracker.update(det)
    tracker.reset()
    tracker.update(det)  # must not raise


# -- foot points ----------------------------------------------------------


def test_foot_point_is_bottom_centre():
    tracked = TrackedDetections(
        xyxy=np.array([[100.0, 50.0, 200.0, 450.0]], np.float32),
        scores=np.array([0.9], np.float32),
        class_ids=np.array([0], np.int32),
        track_ids=np.array([1], np.int32),
    )
    assert tracked.foot_points().tolist() == [[150.0, 450.0]]


def test_foot_points_on_empty():
    assert TrackedDetections.empty().foot_points().shape == (0, 2)
