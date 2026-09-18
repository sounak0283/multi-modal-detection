"""Fire/smoke detector (Expansion Plan Phase E; PLAN.md §5).

A thin policy layer over `YoloxOnnx`, mirroring `detect/person.py:PersonDetector`'s
shape exactly - same wrapper, different weights and class policy, as that module's own
docstring already anticipated. Class ids match `PLAN.md` §5.3's training recipe
(`{0: fire, 1: smoke}`).

Deliberately does NOT apply the per-class confidence thresholds itself - `PersonDetector`
doesn't know about zones either, so the zone-aware policy (per-class threshold, a
fire_roi's sensitivity delta) belongs in `Pipeline`, which has the `BoundaryEngine`
reference this detector doesn't. `.detect()` returns raw, class-filtered, NMS'd
detections only.
"""

from __future__ import annotations

from pathlib import Path

from perimeter.detect.yolox_onnx import Detections, YoloxOnnx

FIRE_CLASS_ID = 0
SMOKE_CLASS_ID = 1
FIRE_SMOKE_CLASS_IDS = {FIRE_CLASS_ID, SMOKE_CLASS_ID}

DEFAULT_NMS = 0.5


class FireSmokeDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_fire: float,
        conf_smoke: float,
        nms_threshold: float = DEFAULT_NMS,
        intra_op_threads: int | None = None,
    ) -> None:
        # The ONNX-decode threshold is the lower of the two class thresholds so neither
        # class is discarded before Pipeline gets a chance to apply the sharper,
        # per-class (and per-zone, via BoundaryEngine.confidence_delta) bar.
        self.model = YoloxOnnx(
            model_path,
            conf_threshold=min(conf_fire, conf_smoke),
            nms_threshold=nms_threshold,
            intra_op_threads=intra_op_threads,
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self.model.input_size

    def detect(self, frame_bgr) -> Detections:
        return self.model.detect(frame_bgr).filter_classes(FIRE_SMOKE_CLASS_IDS)


def class_name(class_id: int) -> str:
    return "fire" if class_id == FIRE_CLASS_ID else "smoke"
