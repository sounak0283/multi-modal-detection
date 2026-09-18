"""Crowd formation monitor (Expansion Plan Phase G; PLATFORM_EXPANSION_PLAN.md
sections 4.2 and 5).

No model - DBSCAN over `scipy.spatial.cKDTree`, hand-rolled rather than pulling in
scikit-learn for one algorithm (the same minimal-dependency reasoning that ruled out
Ultralytics/AGPL tooling in PLAN.md). Reading `BoundaryEngine.zone_occupancy()`'s
headcount alone cannot tell a genuine huddle from people incidentally spread across a
wide zone - clustering the foot points is what makes that distinction.

Hysteresis mirrors `BoundaryEngine`'s confirmed/candidate design (boundary/engine.py),
simplified to a size-only state: a per-zone state machine fires once when the largest
cluster reaches `threshold` and holds for `min_frames` consecutive detection frames, then
requires the cluster to drop below threshold again before it can re-fire - so one
persistent crowd does not spam an alert every subsequent frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# Neighbour radius for clustering, in pixels of the frame the boundary engine already
# works in. Foot points, not body centres, are what's clustered - perspective makes a
# near-camera person's foot point spread further from their neighbours than a
# far-camera person's for the same real-world distance, so this is deliberately generous.
DEFAULT_EPS_PX = 80.0
DEFAULT_MIN_SAMPLES = 2  # points needed to form a cluster (DBSCAN's core-point rule)


def _dbscan_cluster_sizes(points: np.ndarray, eps: float, min_samples: int) -> list[int]:
    """Minimal DBSCAN: the size of each density-connected cluster found. A point with
    fewer than `min_samples` neighbours within `eps` (including itself) is noise and
    excluded from every cluster."""
    n = len(points)
    if n == 0:
        return []

    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, r=eps)
    core = [i for i in range(n) if len(neighbours[i]) >= min_samples]
    if not core:
        return []
    core_set = set(core)

    labels = [-1] * n
    next_label = 0
    for start in core:
        if labels[start] != -1:
            continue
        label = next_label
        next_label += 1
        stack = [start]
        labels[start] = label
        while stack:
            current = stack.pop()
            if current not in core_set:
                continue
            for neighbour in neighbours[current]:
                if labels[neighbour] == -1:
                    labels[neighbour] = label
                    stack.append(neighbour)

    sizes: dict[int, int] = {}
    for label in labels:
        if label != -1:
            sizes[label] = sizes.get(label, 0) + 1
    return list(sizes.values())


@dataclass
class _ZoneCrowdState:
    age: int = 0
    confirmed: bool = False


@dataclass(frozen=True)
class CrowdEvent:
    zone_id: str
    cluster_size: int


class CrowdMonitor:
    def __init__(
        self, eps_px: float = DEFAULT_EPS_PX, min_samples: int = DEFAULT_MIN_SAMPLES
    ) -> None:
        self.eps_px = eps_px
        self.min_samples = min_samples
        self._states: dict[str, _ZoneCrowdState] = {}

    def update(
        self, zone_id: str, foot_points_px: np.ndarray, threshold: int, min_frames: int
    ) -> CrowdEvent | None:
        """Advance one zone by one detection frame. Call once per zone per detection
        frame, with only the foot points already known to be inside that zone."""
        sizes = _dbscan_cluster_sizes(foot_points_px, self.eps_px, self.min_samples)
        largest = max(sizes, default=0)
        state = self._states.setdefault(zone_id, _ZoneCrowdState())

        # Hysteresis applies in both directions. Releasing on a single below-threshold
        # frame meant one dropped detection re-armed the zone, so a crowd that never
        # dispersed fired again every time the detector flickered.
        observed = largest >= threshold
        if observed == state.confirmed:
            state.age = 0
            return None

        state.age += 1
        if state.age < min_frames:
            return None

        state.confirmed = observed
        state.age = 0
        return CrowdEvent(zone_id=zone_id, cluster_size=largest) if observed else None


# Re-exported for tests that want to exercise the clustering primitive in isolation
# without going through the per-zone hysteresis state machine.
dbscan_cluster_sizes = _dbscan_cluster_sizes
