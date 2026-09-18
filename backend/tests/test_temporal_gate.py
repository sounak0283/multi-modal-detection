"""Tests for the K-of-N temporal confirmation gate (Expansion Plan Phase E; PLAN.md §5.5)."""

from __future__ import annotations

from perimeter.detect.temporal_gate import TemporalGate

BOX = (100.0, 100.0, 150.0, 150.0)


def shifted(box, dx=0.0, dy=0.0):
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def grown(box, factor):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor / 2, (y2 - y1) * factor / 2
    return (cx - w, cy - h, cx + w, cy + h)


# -- basic K-of-N confirmation ------------------------------------------------


def test_confirms_after_k_hits_within_the_window():
    gate = TemporalGate(k=3, n=10, iou_threshold=0.3)
    for _ in range(2):
        assert gate.update([("fire", BOX)]) == []
    events = gate.update([("fire", BOX)])
    assert len(events) == 1
    assert events[0].klass == "fire"


def test_does_not_confirm_below_k_hits():
    gate = TemporalGate(k=6, n=10, iou_threshold=0.3)
    events = []
    for _ in range(5):
        events = gate.update([("fire", BOX)])
    assert events == []


def test_a_confirmed_candidate_never_re_emits():
    gate = TemporalGate(k=2, n=10, iou_threshold=0.3)
    gate.update([("fire", BOX)])
    first = gate.update([("fire", BOX)])
    assert len(first) == 1
    for _ in range(5):
        assert gate.update([("fire", BOX)]) == []


def test_hits_outside_the_window_no_longer_count():
    """3 hits then 3 misses then 1 more hit: with n=4 and k=4, the old hits have
    scrolled out of the window by the time the 4th real hit lands - a real fire/smoke
    detection would keep re-hitting, not go quiet for a while and come back."""
    gate = TemporalGate(k=4, n=4, iou_threshold=0.3)
    for _ in range(3):
        gate.update([("fire", BOX)])
    for _ in range(3):
        gate.update([])  # gap - candidate stays alive with hit_count > 0 until fully aged out
    events = gate.update([("fire", BOX)])
    assert events == []  # only 1 hit inside the current 4-frame window


# -- IoU matching / candidate identity ---------------------------------------


def test_a_nearby_box_joins_the_same_candidate():
    gate = TemporalGate(k=2, n=10, iou_threshold=0.3)
    gate.update([("fire", BOX)])
    events = gate.update([("fire", shifted(BOX, dx=5, dy=5))])
    assert len(events) == 1


def test_a_far_away_box_starts_a_new_candidate_not_joins():
    gate = TemporalGate(k=2, n=10, iou_threshold=0.3)
    gate.update([("fire", BOX)])
    far_box = shifted(BOX, dx=500, dy=500)
    events = gate.update([("fire", far_box)])
    assert events == []  # each candidate only has 1 hit so far, k=2 not reached

    events = gate.update([("fire", BOX), ("fire", far_box)])
    assert len(events) == 2  # both candidates independently reach k=2 this tick


def test_fire_and_smoke_never_share_a_candidate_even_at_the_same_location():
    gate = TemporalGate(k=2, n=10, iou_threshold=0.3)
    gate.update([("fire", BOX)])
    gate.update([("smoke", BOX)])  # same box, different class
    events = gate.update([("fire", BOX)])
    assert len(events) == 1
    assert events[0].klass == "fire"


# -- area-growth escalation ---------------------------------------------------


def test_monotonically_growing_area_escalates():
    gate = TemporalGate(k=3, n=10, iou_threshold=0.3)
    gate.update([("fire", BOX)])
    gate.update([("fire", grown(BOX, 1.3))])
    events = gate.update([("fire", grown(BOX, 1.6))])
    assert len(events) == 1
    assert events[0].escalate is True


def test_stable_area_does_not_escalate():
    gate = TemporalGate(k=3, n=10, iou_threshold=0.3)
    for _ in range(3):
        events = gate.update([("fire", BOX)])
    assert len(events) == 1
    assert events[0].escalate is False


def test_shrinking_area_does_not_escalate():
    gate = TemporalGate(k=3, n=10, iou_threshold=0.3)
    gate.update([("fire", BOX)])
    gate.update([("fire", grown(BOX, 0.8))])
    events = gate.update([("fire", grown(BOX, 0.6))])
    assert len(events) == 1
    assert events[0].escalate is False


# -- empty / no-detection ticks --------------------------------------------------


def test_empty_detections_never_crash_and_confirm_nothing():
    gate = TemporalGate(k=3, n=10, iou_threshold=0.3)
    for _ in range(5):
        assert gate.update([]) == []


def test_candidates_do_not_leak_forever_after_fully_aging_out():
    gate = TemporalGate(k=3, n=3, iou_threshold=0.3)
    gate.update([("fire", BOX)])
    for _ in range(3):
        gate.update([])  # ages fully out of the n=3 window - should be dropped internally
    assert gate._candidates == []
