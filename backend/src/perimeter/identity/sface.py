"""SFace face recognition (OpenCV Zoo, Apache-2.0; Expansion Plan Phase F; PLAN.md section 7).

Wraps `cv2.FaceRecognizerSF`. `alignCrop()` applies the 5-point-landmark alignment SFace's
embedding was trained against (PLAN.md section 7 step 4) - skipping it measurably degrades
match quality, so every embed call runs it, not just enrollment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from cv2 import FaceRecognizerSF_create

from perimeter.identity.yunet import FaceBox


class FaceEmbedder:
    def __init__(self, model_path: str | Path) -> None:
        self._recognizer = FaceRecognizerSF_create(str(model_path), "")

    def embed(self, image_bgr: np.ndarray, face: FaceBox) -> np.ndarray:
        aligned = self._recognizer.alignCrop(image_bgr, face.raw)
        feature = self._recognizer.feature(aligned)
        return np.asarray(feature, dtype=np.float32).reshape(-1)
