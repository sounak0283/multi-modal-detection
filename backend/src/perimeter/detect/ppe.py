"""PPE compliance checking (Expansion Plan Phase H; PLATFORM_EXPANSION_PLAN.md section 5).

Software integration only - no trained weights ship in this repo. The model this wraps
is a "new lightweight, person-crop, region-constrained multi-label model... helmet/vest/
gloves/shoes/glasses per zone requirement, with an explicit 'indeterminate' state"
(§5's own description) - a genuinely different architecture from the YOLOX detector the
person/fire-smoke modules share, so `PPEClassifier` is a plain `onnxruntime` wrapper, not
built on `detect/yolox_onnx.py:YoloxOnnx` (that wrapper's own docstring is explicit that
it decodes YOLOX's specific anchor grid output - a multi-label classifier over a whole
person crop has nothing in common with that shape).

See `backend/models/ppe/README.md` for the expected model I/O once trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

# The fixed, ordered item set the classifier's output vector indexes into. Matches
# PLATFORM_EXPANSION_PLAN.md section 5's own list verbatim. Declared here as the
# detector's canonical item order; boundary/zones.py declares its own PPE_ITEMS
# constant for `ppe_required` validation, kept in sync by convention - the same
# "zone-domain module doesn't import from detect/" precedent DETECT_CLASSES already
# established there for person/fire/smoke.
PPE_ITEMS: tuple[str, ...] = ("helmet", "vest", "gloves", "shoes", "glasses")

DEFAULT_INPUT_SIZE = (128, 128)

# Confidence gap between these two is the explicit "indeterminate" state
# PLATFORM_EXPANSION_PLAN.md section 5 calls for - the same principle Phase F's
# `no_face` established for identity: an abstention must never be read as a violation.
DEFAULT_PRESENT_THRESHOLD = 0.65
DEFAULT_ABSENT_THRESHOLD = 0.35


def classify_state(
    probability: float,
    present: float = DEFAULT_PRESENT_THRESHOLD,
    absent: float = DEFAULT_ABSENT_THRESHOLD,
) -> str:
    """One item's presence probability -> "present" | "absent" | "indeterminate"."""
    if probability >= present:
        return "present"
    if probability <= absent:
        return "absent"
    return "indeterminate"


class PPEClassifier:
    """Wraps a multi-label ONNX classifier: one whole-person crop in, one sigmoid
    presence probability per `PPE_ITEMS` entry out."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        intra_op_threads: int | None = None,
    ) -> None:
        options = onnxruntime.SessionOptions()
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        self._session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.input_size = input_size

    def classify(self, person_crop_bgr: np.ndarray) -> dict[str, float]:
        if person_crop_bgr.size == 0:
            return dict.fromkeys(PPE_ITEMS, 0.0)
        resized = cv2.resize(person_crop_bgr, self.input_size)
        blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        (output,) = self._session.run(None, {self._input_name: blob})
        probabilities = np.asarray(output, dtype=np.float32).reshape(-1)
        return dict(zip(PPE_ITEMS, (float(p) for p in probabilities), strict=False))


@dataclass
class _ItemState:
    age: int = 0
    confirmed: bool = False
    last_seen: int = 0


class PPEMonitor:
    """Per-(zone, track, item) hysteresis, same confirmed/age shape
    `boundary/crowd.py:CrowdMonitor` established (Phase G): a violation fires once
    absence has been observed for `min_frames` consecutive checks, then does not
    re-fire until the item is seen "present" again. "indeterminate" neither advances
    nor resets progress - a single unclear frame should not cost several frames of
    hysteresis, nor count toward confirming a violation.
    """

    def __init__(self) -> None:
        self._states: dict[tuple[str, int, str], _ItemState] = {}
        self._calls = 0

    def update(
        self, zone_id: str, track_id: int, item: str, state: str, min_frames: int = 3
    ) -> bool:
        """Advance one (zone, track, item) by one check. Returns True the moment a
        violation is newly confirmed."""
        self._calls += 1
        key = (zone_id, track_id, item)
        entry = self._states.setdefault(key, _ItemState())
        entry.last_seen = self._calls

        if state == "present":
            entry.age = 0
            entry.confirmed = False
            return False
        if state == "indeterminate":
            return False

        # state == "absent"
        entry.age += 1
        if entry.confirmed or entry.age < min_frames:
            return False
        entry.confirmed = True
        return True

    def evict_stale(self, max_age_calls: int = 300) -> None:
        """Drop entries for tracks that stopped being checked a while ago (left the
        zone, left the frame, or the camera was reconfigured) - mirrors
        `BoundaryEngine._evict_stale`'s call-counter-based cleanup, so a long-running
        process doesn't accumulate state for every person who ever walked through."""
        cutoff = self._calls - max_age_calls
        if cutoff <= 0:
            return
        stale = [key for key, entry in self._states.items() if entry.last_seen < cutoff]
        for key in stale:
            del self._states[key]
