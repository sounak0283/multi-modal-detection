"""YuNet face detection (OpenCV Zoo, MIT; Expansion Plan Phase F; PLAN.md section 7).

Wraps `cv2.FaceDetectorYN`, which already decodes YuNet's raw ONNX output (three
detection heads, each producing separate cls/obj/bbox/kps tensors across three scales)
and runs NMS. Hand-decoding that output over raw onnxruntime was evaluated and rejected
during this phase's research: OpenCV ships the reference decoder for exactly this export,
and there is no reason to reimplement it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Small crops (a face region cropped from the top of a person box) don't need YuNet's
# default 320x320 working size to be any larger - setInputSize is called per-crop below
# since person boxes vary in size frame to frame.
DEFAULT_INPUT_SIZE = (320, 320)
DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_TOP_K = 20


@dataclass(frozen=True)
class FaceBox:
    """One detected face. `raw` is the verbatim YuNet output row (bbox + 5 landmark
    pairs + score) - `cv2.FaceRecognizerSF.alignCrop()` expects exactly this shape, so it
    is kept intact rather than repackaged into separate fields."""

    raw: np.ndarray  # (15,) float32: x, y, w, h, then 5 (x, y) landmark pairs, then score
    score: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.raw[0:4])


class FaceDetector:
    """YuNet, invoked only on a small crop from a confirmed boundary ENTRY - never on a
    full frame, and never on every frame (PLAN.md section 7)."""

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._detector = cv2.FaceDetectorYN_create(
            str(model_path), "", DEFAULT_INPUT_SIZE, score_threshold, nms_threshold, top_k
        )

    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        height, width = image_bgr.shape[:2]
        if height < 1 or width < 1:
            return []
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image_bgr)
        if faces is None:
            return []
        return [
            FaceBox(raw=row.astype(np.float32), score=float(row[14]))
            for row in faces
        ]

    def largest(self, image_bgr: np.ndarray) -> FaceBox | None:
        """The largest face by box area - PLAN.md section 7 step 3: 'take the largest
        face' when more than one is visible in the crop."""
        faces = self.detect(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.box[2] * f.box[3])
