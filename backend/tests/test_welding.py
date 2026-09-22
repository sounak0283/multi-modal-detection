"""Welding-spark detector: soft alert, no model, independent of the boundary."""

from __future__ import annotations

import numpy as np
import pytest

from perimeter.alerts.cooldown import ALERT_KINDS
from perimeter.cameras.models import ALL_MODULES, REQUIRES_BOUNDARY
from perimeter.detect.welding import SparkDetector
from perimeter.evidence.policy import ClipPolicy

W, H = 640, 360


def dark():
    return np.full((H, W, 3), 40, np.uint8)


ORANGE = (0, 140, 255)  # BGR: glowing orange-white, like real sparks
WHITE = (255, 255, 255)


def spray(rng, cx=300, cy=200, n=14, colour=ORANGE):
    """A burst of tiny, very bright specks around one point - different every frame.
    4x4 (not 1-2px) so the synthetic warm-pixel count is in the same ballpark as a real
    spark's rendered size, not an order of magnitude below it."""
    frame = dark()
    for _ in range(n):
        x = int(cx + rng.integers(-50, 50))
        y = int(cy + rng.integers(-40, 40))
        frame[y : y + 4, x : x + 4] = colour
    return frame


def run(detector, frames, fps=12.0):
    return [detector.update(f, i / fps) for i, f in enumerate(frames)]


def test_a_persistent_spray_of_new_specks_raises_one_event():
    rng = np.random.default_rng(1)
    events = [e for e in run(SparkDetector(), [spray(rng) for _ in range(30)]) if e]

    assert len(events) == 1
    x0, y0, x1, y1 = events[0].bbox
    assert 200 < x0 < x1 < 420 and 120 < y0 < y1 < 300


def test_static_bright_lights_never_alert():
    frame = dark()
    frame[20:40, 100:140] = 255  # a lamp: bright in every frame, so never "new"
    assert not any(run(SparkDetector(), [frame.copy() for _ in range(40)]))


def test_a_large_flash_is_not_a_spark():
    frames = [dark() if i % 2 else np.full((H, W, 3), 255, np.uint8) for i in range(30)]
    assert not any(run(SparkDetector(), frames))


def test_scattered_specks_far_apart_are_not_a_cluster():
    rng = np.random.default_rng(2)
    frames = []
    for _ in range(30):
        f = dark()
        for cx in (60, 320, 580):
            x, y = cx, int(rng.integers(20, 340))
            f[y : y + 2, x : x + 2] = 255
        frames.append(f)
    assert not any(run(SparkDetector(min_specks=4), frames))


def test_one_episode_alerts_once_and_rearms_after_calm():
    rng = np.random.default_rng(3)
    detector = SparkDetector(rearm_seconds=5.0)
    first = [e for e in run(detector, [spray(rng) for _ in range(30)]) if e]
    calm = [detector.update(dark(), 3 + i / 12) for i in range(80)]  # ~7 s with no sparks
    later = [detector.update(spray(rng), 10 + i / 12) for i in range(30)]

    assert len(first) == 1 and not any(calm) and any(later)


def test_bright_white_specks_are_not_sparks_such_as_moving_white_clothing():
    rng = np.random.default_rng(5)
    assert not any(run(SparkDetector(), [spray(rng, colour=WHITE) for _ in range(30)]))


def test_a_single_flicker_is_not_enough():
    rng = np.random.default_rng(4)
    frames = [dark()] * 5 + [spray(rng)] + [dark()] * 10
    assert not any(run(SparkDetector(), frames))


def test_distinct_bursts_several_seconds_apart_each_raise_their_own_event():
    """Shorter default re-arm than a single weld job's natural pauses: separate strikes
    each get their own event (and so their own clip/history entry) instead of being
    folded into one. Email pacing across these is a router-level cooldown, not this."""
    rng = np.random.default_rng(6)
    detector = SparkDetector()
    frames = (
        [(i / 12.0, spray(rng)) for i in range(30)]
        + [(2.5 + i / 12.0, dark()) for i in range(80)]  # ~6.7s calm > default 4s re-arm
        + [(9.5 + i / 12.0, spray(rng)) for i in range(30)]
    )
    events = [e for ts, frame in frames if (e := detector.update(frame, ts))]
    assert len(events) == 2


def test_an_abrupt_upward_jump_between_frames_never_confirms():
    """Real sparks fall under gravity; a cluster that keeps relocating sharply upward
    frame to frame looks more like an electrical arc restriking than a shower of embers
    (see the module docstring's limitation note) and must not confirm."""
    detector = SparkDetector()
    rng = np.random.default_rng(7)
    # A big jump (not just over the threshold) so cluster-centroid jitter from the random
    # point scatter can never accidentally read as a small, undiscounted rise; and cy must
    # stay within the frame, since a negative value wraps via numpy indexing.
    frames = [spray(rng, cx=300, cy=350 - i * 80) for i in range(5)]
    assert not any(run(detector, frames))


def test_a_gently_falling_cluster_still_confirms():
    """Regression: the upward-jump discount must not suppress the ordinary case of a
    spark cluster drifting down (or holding roughly still) frame to frame."""
    detector = SparkDetector()
    rng = np.random.default_rng(8)
    frames = [spray(rng, cx=300, cy=150 + i * 5) for i in range(10)]  # falls 5px/frame
    assert any(run(detector, frames))


def test_welding_is_a_module_that_does_not_need_the_boundary():
    assert "welding" in ALL_MODULES and "welding" not in REQUIRES_BOUNDARY
    assert "welding" in ALERT_KINDS


@pytest.mark.parametrize("kind,expected", [("welding", False), ("fire", True)])
def test_welding_alerts_do_not_record_clips_by_default(kind, expected):
    decision = ClipPolicy().decide({"kind": kind, "severity": "low"})
    assert decision.record is expected
