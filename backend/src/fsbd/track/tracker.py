"""Person tracking via ByteTrack (PLAN.md sections 3, 4 and 6.3).

Why tracking is not optional
----------------------------
"Crossed the boundary" is only definable per persistent identity. Without stable track
IDs there is no previous state to compare against, and the entry/exit state machine in
section 6.3 has nothing to key on.

Two things this wrapper exists to get right
-------------------------------------------
1. **Detection-frame cadence.** The tracker is driven by detections, and person
   inference runs every 2nd frame. On skip frames we simply do not call it - calling
   with an empty detection set does not interpolate, it ages tracks toward deletion via
   the lost-track buffer and actively degrades tracking. See PLAN.md section 4.

2. **Occlusion tolerance expressed in seconds, not frames.** ByteTrack's
   `lost_track_buffer` counts *update calls*, and we update at ~6-8 Hz rather than the
   library's default assumption of 30. Leaving the defaults alone would silently cut
   occlusion tolerance to a quarter of what it appears to be. Track ID churn is the
   direct cause of phantom ENTRY events (section 6.3, bug 2), so this conversion is
   load-bearing rather than cosmetic.

Licence note: `trackers` (Roboflow) is Apache-2.0 and a clean-room reimplementation.
It replaces `supervision.ByteTrack`, which is deprecated and removed in supervision
v0.31. See NOTICE.md section 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from fsbd.detect.yolox_onnx import Detections

log = logging.getLogger("fsbd.track")

# Occlusion tolerance. A person walking behind warehouse racking is out of sight for a
# second or two; beyond ~5 s they are better treated as a new track than as the same one.
DEFAULT_LOST_TRACK_SECONDS = 5.0


@dataclass(frozen=True)
class TrackedDetections:
    """Detections carrying a persistent track ID.

    Same column layout as Detections, plus track_ids, so the boundary engine can key
    per-track state without a second lookup.
    """

    xyxy: np.ndarray  # (N, 4) float32
    scores: np.ndarray  # (N,)  float32
    class_ids: np.ndarray  # (N,)  int32
    track_ids: np.ndarray  # (N,)  int32

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @classmethod
    def empty(cls) -> TrackedDetections:
        return cls(
            xyxy=np.zeros((0, 4), np.float32),
            scores=np.zeros((0,), np.float32),
            class_ids=np.zeros((0,), np.int32),
            track_ids=np.zeros((0,), np.int32),
        )

    def foot_points(self) -> np.ndarray:
        """Bottom-centre of each box - the point the boundary engine tests.

        Not the centroid: under perspective a tall person standing outside a floor zone
        has a centroid inside it (PLAN.md section 6.1).
        """
        if len(self) == 0:
            return np.zeros((0, 2), np.float32)
        xs = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        return np.stack([xs, self.xyxy[:, 3]], axis=1).astype(np.float32)


def to_supervision(detections: Detections) -> sv.Detections:
    if len(detections) == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=detections.xyxy.astype(np.float64),
        confidence=detections.scores.astype(np.float64),
        class_id=detections.class_ids.astype(int),
    )


def from_supervision(tracked: sv.Detections) -> TrackedDetections:
    if len(tracked) == 0 or tracked.tracker_id is None:
        return TrackedDetections.empty()

    # Detections the tracker has seen but not yet CONFIRMED come back with the sentinel
    # id -1, not with None - tracker_id is an integer array, so it cannot hold None.
    #
    # Dropping them is not a tidiness measure. Every unconfirmed detection in a frame
    # carries the same -1, so passing them through would collapse several different
    # people into a single track identity. The boundary engine keys its entry/exit state
    # machine on track id (PLAN.md section 6.3), so two strangers walking into a zone
    # would share one state and silently cancel each other's events.
    #
    # An unconfirmed box must not be able to open a boundary event in any case: waiting
    # for confirmation is what min_consecutive_frames is for.
    #
    # Note the comparison is >= 0, not > 0: this implementation issues 0 as a valid
    # first track id, unlike the original ByteTrack which starts at 1. Only -1 is the
    # sentinel. Measured against trackers 2.6.0.
    keep = np.asarray(tracked.tracker_id) >= 0
    if not keep.any():
        return TrackedDetections.empty()

    xyxy = np.asarray(tracked.xyxy, np.float32)[keep]
    scores = (
        np.asarray(tracked.confidence, np.float32)[keep]
        if tracked.confidence is not None
        else np.ones(int(keep.sum()), np.float32)
    )
    class_ids = (
        np.asarray(tracked.class_id, np.int32)[keep]
        if tracked.class_id is not None
        else np.zeros(int(keep.sum()), np.int32)
    )
    track_ids = np.asarray(tracked.tracker_id, np.int32)[keep]
    return TrackedDetections(xyxy, scores, class_ids, track_ids)


class PersonTracker:
    """ByteTrack driven at the person-detection cadence.

    Args:
        detection_fps: how often `update` is actually called - the person-inference
            rate, NOT the camera frame rate. Everything else scales from this.
        lost_track_seconds: how long a track survives without a matching detection.
        activation_threshold: confidence needed to start a new track.
        min_consecutive_frames: detections needed before a track is confirmed and
            given an ID. Raising this trades responsiveness for fewer spurious tracks.
    """

    def __init__(
        self,
        detection_fps: float = 7.0,
        lost_track_seconds: float = DEFAULT_LOST_TRACK_SECONDS,
        activation_threshold: float = 0.5,
        min_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.2,
    ) -> None:
        if detection_fps <= 0:
            raise ValueError("detection_fps must be positive")

        self.detection_fps = detection_fps
        self.lost_track_seconds = lost_track_seconds
        # The conversion this class exists for: seconds -> update calls.
        self.lost_track_buffer = max(1, round(lost_track_seconds * detection_fps))

        self._tracker = ByteTrackTracker(
            lost_track_buffer=self.lost_track_buffer,
            frame_rate=detection_fps,
            track_activation_threshold=activation_threshold,
            minimum_consecutive_frames=min_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
        )

        log.info(
            "tracker: %.1f detection fps, %.1fs occlusion tolerance (%d updates)",
            detection_fps,
            lost_track_seconds,
            self.lost_track_buffer,
        )

    def update(self, detections: Detections) -> TrackedDetections:
        """Advance the tracker by one DETECTION frame.

        Call only on frames where detection actually ran. Skipping a call is correct;
        calling with an empty set to represent a skipped frame is not - that tells the
        tracker every track just disappeared.
        """
        return from_supervision(self._tracker.update(to_supervision(detections)))

    def reset(self) -> None:
        self._tracker.reset()
