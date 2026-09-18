"""K-of-N temporal confirmation gate for fire/smoke (Expansion Plan Phase E; PLAN.md §5.5).

Per-frame detections are noisy - a raw per-frame confidence firing an alert is how false
alarms are made. This maintains "candidate" blobs across processed frames and only
confirms (emits once) a candidate that has been detected in at least `k` of its last `n`
processed frames, with IoU >= `iou_threshold` linking a new detection to an existing
candidate rather than starting a new one every frame. Pure logic, no cv2/ONNX dependency,
so it is fully unit-testable without a real model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class ConfirmedEvent:
    klass: str
    bbox: BBox
    escalate: bool  # box area grew monotonically over the confirming hits


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


@dataclass
class _Candidate:
    klass: str
    last_box: BBox
    hits: deque[bool]
    areas: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    confirmed: bool = False

    @property
    def hit_count(self) -> int:
        return sum(self.hits)


class TemporalGate:
    def __init__(self, k: int, n: int, iou_threshold: float) -> None:
        self.k = k
        self.n = n
        self.iou_threshold = iou_threshold
        self._candidates: list[_Candidate] = []

    def update(self, detections: list[tuple[str, BBox]]) -> list[ConfirmedEvent]:
        """One processed-frame tick. `detections` is `(klass, bbox)` for whatever passed
        the caller's per-class/per-zone confidence check this frame. Returns the
        candidates that newly reached `k` hits this tick - never re-emits an already-
        confirmed candidate."""
        unmatched = list(detections)
        events: list[ConfirmedEvent] = []

        for candidate in self._candidates:
            match_index = self._best_match(candidate, unmatched)
            if match_index is not None:
                klass, box = unmatched.pop(match_index)
                candidate.last_box = box
                candidate.hits.append(True)
                candidate.areas.append(_area(box))
            else:
                candidate.hits.append(False)

            if not candidate.confirmed and candidate.hit_count >= self.k:
                candidate.confirmed = True
                events.append(
                    ConfirmedEvent(
                        klass=candidate.klass,
                        bbox=candidate.last_box,
                        escalate=_areas_growing(candidate.areas),
                    )
                )

        # Anything left unmatched starts a brand new candidate for the next tick.
        for klass, box in unmatched:
            self._candidates.append(
                _Candidate(
                    klass=klass, last_box=box,
                    hits=deque([True], maxlen=self.n), areas=deque([_area(box)], maxlen=8),
                )
            )

        # Drop candidates with zero hits anywhere in the window - fully aged out, no
        # point carrying them forever and leaking memory on a long-running process.
        self._candidates = [c for c in self._candidates if c.hit_count > 0]

        return events

    def _best_match(self, candidate: _Candidate, unmatched: list[tuple[str, BBox]]) -> int | None:
        best_index, best_iou = None, self.iou_threshold
        for index, (klass, box) in enumerate(unmatched):
            if klass != candidate.klass:
                continue
            iou = _iou(candidate.last_box, box)
            if iou >= best_iou:
                best_index, best_iou = index, iou
        return best_index


def _areas_growing(areas: deque[float]) -> bool:
    """True if the last >= 3 hit areas are strictly increasing - PLAN.md §5.5: "fire
    grows, brake lights and sunsets do not"."""
    if len(areas) < 3:
        return False
    recent = list(areas)[-3:]
    return recent[0] < recent[1] < recent[2]
