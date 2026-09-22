"""Welding-spark detector: a rule-based "soft" alert, no model.

The fire model was trained on flames and smoke and scores welding sparks near zero (measured
on a real weld-shop video: max fire score 0.08), so sparks get their own cheap check on
what makes them distinctive: many tiny, very bright specks that are NEW since the previous
frame, clustered around one point. Static bright things - ceiling lights, windows, a lit
wall - are in the previous frame too, so they cancel out; large flashes and whole-frame
brightening fail the speck-size test.

Deliberately separate from fire/smoke and never escalated: it tells a person "this looks
like welding", it does not claim a fire.

LIMITATION, stated plainly: an electrical short or arcing fault can look a lot like this -
also many small very bright specks/flashes. The one extra signal used here is that welding
sparks fall under gravity (the brightest cluster's position drifts down, frame to frame),
so an abrupt jump *upward* between frames is discounted rather than counted as evidence
(`upward_jump_px`). That is a real but weak signal, not a trained distinction between "this
is welding" and "this is an electrical fault" - it was built without any labelled footage of
electrical sparking to check against. If that footage becomes available, this should be
replaced with an actual classifier trained on both classes, not extended with more heuristics.

SECOND LIMITATION, found live (2026-09-22): a shiny/oily face under a bright ceiling light
briefly raised a false "welding" alert. Unlike a matte surface, a specular highlight on skin
shifts with tiny head movement (blinking, breathing), so it can look like scattered new bright
specks the same way sparks do. `brightness` and `warm_min` were raised (225->240, 30->60,
both still comfortably below every real spark measurement on file - see tests/test_welding.py
and the offline scan in this module's git history) as a reasoned tightening toward what molten
metal actually looks like versus typical indoor lighting on skin - not a confirmed fix, since
the exact triggering frame was not captured to test against directly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

_GROW = np.ones((5, 5), np.uint8)


@dataclass(frozen=True)
class SparkEvent:
    bbox: tuple[float, float, float, float]  # original-image pixels
    specks: int
    warmth: int = 0


class SparkDetector:
    def __init__(
        self,
        *,
        work_width: int = 640,
        brightness: int = 240,
        min_specks: int = 8,
        max_speck_area: int = 150,
        cluster_radius: int = 90,
        warm_min: int = 60,
        upward_jump_px: int = 25,
        confirm_k: int = 3,
        confirm_n: int = 10,
        rearm_seconds: float = 4.0,
    ) -> None:
        self.work_width = work_width
        self.brightness = brightness
        self.min_specks = min_specks
        self.max_speck_area = max_speck_area
        self.cluster_radius = cluster_radius
        self.warm_min = warm_min
        self.upward_jump_px = upward_jump_px
        self.confirm_k = confirm_k
        self.rearm_seconds = rearm_seconds
        self._recent: deque[bool] = deque(maxlen=confirm_n)
        self._previous: np.ndarray | None = None
        self._last_spark_ts = float("-inf")
        self._last_centroid: tuple[float, float] | None = None
        self._latched = False

    def _bright_mask(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        height, width = image.shape[:2]
        scale = self.work_width / width if width > self.work_width else 1.0
        size = (int(width * scale), int(height * scale))
        small = (
            cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale != 1.0 else image
        )
        value = small.max(axis=2)  # V of HSV without the full colour conversion
        return value >= self.brightness, small, scale

    def _cluster(self, fresh: np.ndarray) -> tuple[int, tuple[int, int, int, int]] | None:
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            fresh.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = (areas >= 2) & (areas <= self.max_speck_area)
        if int(keep.sum()) < self.min_specks:
            return None
        points = centroids[1:][keep]
        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        neighbours = dist <= self.cluster_radius
        sizes = neighbours.sum(axis=1)
        best = int(sizes.argmax())
        if int(sizes[best]) < self.min_specks:
            return None
        group = points[neighbours[best]]
        x0, y0 = group.min(axis=0)
        x1, y1 = group.max(axis=0)
        return int(sizes[best]), (int(x0), int(y0), int(x1), int(y1))

    @staticmethod
    def _warmth(small: np.ndarray, box: tuple[int, int, int, int]) -> int:
        """Pixels around the sparks that glow orange/yellow (red well above blue). White
        clothing and lamps are neutral, so this is what tells sparks from bright fabric."""
        x0, y0, x1, y1 = box
        pad = 25
        region = small[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad].astype(np.int16)
        blue, red = region[..., 0], region[..., 2]
        return int(((red >= 200) & (red - blue >= 70)).sum())

    def update(self, image: np.ndarray, ts: float) -> SparkEvent | None:
        """Feed consecutive frames. Returns an event once per welding episode - when sparks
        have persisted (k of the last n frames) - and again once a new episode starts, after
        `rearm_seconds` with no spark activity at all (so several strikes in one weld job,
        each separated by a calm gap, each raise their own event/clip; a continuous shower
        with only brief gaps stays one episode)."""
        mask, small, scale = self._bright_mask(image)
        previous, self._previous = self._previous, mask
        if previous is None or previous.shape != mask.shape:
            return None

        fresh = mask & ~cv2.dilate(previous.astype(np.uint8), _GROW).astype(bool)
        found = self._cluster(fresh)
        warmth = 0
        centroid = None
        if found is not None:
            warmth = self._warmth(small, found[1])
            if warmth < self.warm_min:
                found = None
        if found is not None:
            x0, y0, x1, y1 = found[1]
            centroid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

        # This frame's cluster still keeps the episode alive (`_last_spark_ts` below), but
        # does not itself build confidence if it jumped sharply upward since the last one -
        # see the module docstring's limitation note.
        counts = found is not None
        if counts and centroid is not None and self._last_centroid is not None:
            dy = centroid[1] - self._last_centroid[1]
            if dy < -self.upward_jump_px:
                counts = False
        if centroid is not None:
            self._last_centroid = centroid

        self._recent.append(counts)
        if found is not None:
            self._last_spark_ts = ts

        if self._latched and ts - self._last_spark_ts >= self.rearm_seconds:
            self._latched = False
            self._recent.clear()
            self._last_centroid = None

        if found is None or self._latched or sum(self._recent) < self.confirm_k:
            return None
        self._latched = True
        specks, (x0, y0, x1, y1) = found
        pad = 12
        bbox = tuple(float(v / scale) for v in (x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        return SparkEvent(bbox=bbox, specks=specks, warmth=warmth)  # type: ignore[arg-type]
