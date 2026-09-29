"""PPE helmet detector - adapter for demo_package's `final_C_v2` model (Phase H).

The package ships a YOLOX-nano detector that finds **head** (bare head) and **helmet**
(a head wearing a hard hat) inside a padded person crop. The rest of the pipeline was
written against a different model shape - a whole-crop multi-label classifier that
returns one presence probability per `PPE_ITEMS` entry (`detect/ppe.py:PPEClassifier`).
This adapter keeps that contract: `classify()` / `classify_person()` return the same
`{item: probability}` dict, with the detector's result mapped into the bands
`classify_state` already uses, so `PPEMonitor`, zone `ppe_required`, alerts, evidence and
e-mail all work unchanged.

Mapping (per person):
  * helmet chosen  -> helmet probability in [0.65, 1.0]  -> "present"
  * bare head      -> helmet probability in [0.0, 0.35]  -> "absent" (a violation, after
                      PPEMonitor's hysteresis)
  * neither, or the person is shorter than `min_person_height_px` -> 0.5 -> "indeterminate"
  * vest / gloves / shoes / glasses -> always 0.5: this model cannot see them, and an
    unknown must never be reported as "absent" (the same principle as identity's
    `no_face`).

Preprocessing is the model's own contract, pinned by `models/ppe/manifest.json` and
reproduced exactly by `YoloxOnnx` (letterbox pad 114 top-left, raw 0-255 BGR, NCHW) -
verified box-for-box against the package's own runner before this was written.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from perimeter.detect.ppe import PPE_ITEMS, PPEResult, person_crop
from perimeter.detect.yolox_onnx import Detections, YoloxOnnx, preferred_providers

log = logging.getLogger("perimeter.detect.ppe_helmet")

CLASSES: tuple[str, ...] = ("head", "helmet")  # model output order (config.json)
HEAD, HELMET = 0, 1

# Person crop geometry the model was trained on (demo.py / perimeter/detect/ppe_crop.py).
CROP_PAD = (0.10, 0.12, 0.04)  # x, top, bottom - fractions of the person box
NMS_IOU = 0.45
# A head and a helmet box overlapping this much are the same head: keep the stronger.
SAME_HEAD_IOU = 0.5
# Heads are looked for in the upper part of the person box, so a neighbour's head that
# falls inside this person's padded crop is not taken as theirs.
HEAD_REGION_TOP_FRACTION = 0.5

PRESENT_FLOOR = 0.65  # classify_state's default "present" threshold
ABSENT_CEILING = 0.35  # classify_state's default "absent" threshold
INDETERMINATE = 0.5


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def helmet_probability(class_id: int, score: float, threshold: float) -> float:
    """Map one confident detection onto the classifier's probability bands."""
    headroom = max(1e-6, 1.0 - threshold)
    confidence = min(1.0, max(0.0, (score - threshold) / headroom))
    if class_id == HELMET:
        return PRESENT_FLOOR + (1.0 - PRESENT_FLOOR) * confidence
    return ABSENT_CEILING * (1.0 - confidence)


def _indeterminate() -> dict[str, float]:
    return dict.fromkeys(PPE_ITEMS, INDETERMINATE)


class PPEHelmetDetector:
    """Loads once, reused for every frame and every person."""

    supported_items: frozenset[str] = frozenset({"helmet"})

    def __init__(
        self,
        model_path: str | Path,
        head_threshold: float = 0.45,
        helmet_threshold: float = 0.55,
        min_person_height_px: int = 96,
        intra_op_threads: int | None = None,
        model=None,
    ) -> None:
        self.thresholds = {HEAD: float(head_threshold), HELMET: float(helmet_threshold)}
        self.min_person_height_px = int(min_person_height_px)
        # `model` is injectable so selection logic is testable without weights.
        self.model = model or YoloxOnnx(
            model_path,
            conf_threshold=min(self.thresholds.values()),
            nms_threshold=NMS_IOU,
            intra_op_threads=intra_op_threads,
            providers=preferred_providers(),
        )
        providers = getattr(self.model, "providers", None)
        if providers is not None:
            log.info("PPE helmet detector ready on %s", providers[0])

    def classify(self, person_crop_bgr: np.ndarray) -> dict[str, float]:
        """Legacy whole-crop entry point: treats the entire crop as the person."""
        h, w = person_crop_bgr.shape[:2]
        return self.classify_person(person_crop_bgr, (0.0, 0.0, float(w), float(h))).probabilities

    def classify_person(self, frame_bgr: np.ndarray, box_xyxy) -> PPEResult:
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        if y2 - y1 < self.min_person_height_px:
            return PPEResult(_indeterminate(), label="too small")

        crop, (ox, oy) = person_crop(frame_bgr, box_xyxy, CROP_PAD)
        if crop.size == 0:
            return PPEResult(_indeterminate())
        detections: Detections = self.model.detect(crop)

        candidates = []
        head_limit = y1 + HEAD_REGION_TOP_FRACTION * (y2 - y1)
        for box, score, cls in zip(
            detections.xyxy, detections.scores, detections.class_ids, strict=True
        ):
            cls = int(cls)
            if cls not in self.thresholds or score < self.thresholds[cls]:
                continue
            fx1, fy1, fx2, fy2 = box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy
            cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            if not (x1 <= cx <= x2 and cy <= head_limit):
                continue
            candidates.append(((fx1, fy1, fx2, fy2), float(score), cls))

        # A head also covered by a helmet box is one head: keep the stronger label.
        kept = [
            c for c in candidates
            if all(
                c[1] >= o[1]
                for o in candidates
                if o is not c and o[2] != c[2] and _iou(o[0], c[0]) > SAME_HEAD_IOU
            )
        ]
        if not kept:
            return PPEResult(_indeterminate(), label="checking")

        person_cx = (x1 + x2) / 2.0
        box, score, cls = min(
            kept, key=lambda c: (abs((c[0][0] + c[0][2]) / 2.0 - person_cx), -c[1])
        )
        probabilities = _indeterminate()
        probabilities["helmet"] = helmet_probability(cls, score, self.thresholds[cls])
        return PPEResult(
            probabilities,
            head_box=tuple(float(v) for v in box),
            label="HELMET" if cls == HELMET else "NO HELMET",
        )
