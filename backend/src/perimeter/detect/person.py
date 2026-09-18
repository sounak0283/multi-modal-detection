"""Person detector (PLAN.md section 4).

A thin policy layer over YoloxOnnx: COCO-pretrained weights, everything except the
person class discarded. Separate from the wrapper because the fire/smoke detector will
use the same wrapper with different weights and a different class policy.
"""

from __future__ import annotations

from pathlib import Path

from perimeter.detect.yolox_onnx import Detections, YoloxOnnx

COCO_PERSON_CLASS_ID = 0

# A person crossing a boundary is worth a lower bar than a general-purpose detection:
# a missed crossing is a security failure, while a spurious box is damped by the
# 4-frame hysteresis in the boundary state machine (PLAN.md section 6.3).
DEFAULT_CONF = 0.30
DEFAULT_NMS = 0.45

# Boxes below this height in the original frame are too small for the identity layer
# to do anything with (PLAN.md section 7) and are usually far-field noise. Detection
# still reports them; this is exposed for consumers that need the cutoff.
MIN_IDENTITY_BOX_HEIGHT_PX = 120


class PersonDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = DEFAULT_CONF,
        nms_threshold: float = DEFAULT_NMS,
        intra_op_threads: int | None = None,
        min_box_height_px: int = 0,
    ) -> None:
        self.model = YoloxOnnx(
            model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            intra_op_threads=intra_op_threads,
        )
        self.min_box_height_px = min_box_height_px

    @property
    def input_size(self) -> tuple[int, int]:
        return self.model.input_size

    def detect(self, frame_bgr) -> Detections:
        detections = self.model.detect(frame_bgr).filter_classes({COCO_PERSON_CLASS_ID})
        if self.min_box_height_px and len(detections):
            heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            mask = heights >= self.min_box_height_px
            detections = Detections(
                detections.xyxy[mask], detections.scores[mask], detections.class_ids[mask]
            )
        return detections


def foot_points(detections: Detections):
    """Bottom-centre of each box - the point the boundary engine tests.

    Not the centroid. Under perspective a tall person standing outside a floor zone has
    a centroid inside it, which is the classic false-positive in tripwire systems
    (PLAN.md section 6.1). Provided here so detector and boundary engine cannot disagree
    about where a person "is".
    """
    import numpy as np

    if len(detections) == 0:
        return np.zeros((0, 2), np.float32)
    xs = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2.0
    ys = detections.xyxy[:, 3]
    return np.stack([xs, ys], axis=1).astype(np.float32)
