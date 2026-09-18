"""End-to-end fire/smoke detector checks against real trained weights (Expansion Plan
Phase E). Skipped when the weights are absent - see `models/firesmoke/README.md` - which
is expected for every checkout of this repo today, no GPU/dataset workstream has run yet.
Mirrors `test_detect_integration.py`'s pattern exactly so this activates automatically,
with no code change, the moment a real `model.onnx` is placed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perimeter.detect.firesmoke import FireSmokeDetector

MODEL = Path("models/firesmoke/model.onnx")

pytestmark = pytest.mark.skipif(
    not MODEL.is_file(),
    reason="models/firesmoke/model.onnx not present - see models/firesmoke/README.md",
)


@pytest.fixture(scope="module")
def detector() -> FireSmokeDetector:
    return FireSmokeDetector(MODEL, conf_fire=0.45, conf_smoke=0.35)


def test_input_size_is_the_planned_416(detector):
    assert detector.input_size == (416, 416)


def test_blank_frame_yields_no_fire_or_smoke(detector):
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert len(detector.detect(blank)) == 0
