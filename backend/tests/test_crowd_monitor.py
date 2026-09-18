"""Tests for CrowdMonitor's DBSCAN clustering and per-zone hysteresis (Expansion Plan
Phase G; PLATFORM_EXPANSION_PLAN.md sections 4.2/5)."""

from __future__ import annotations

import numpy as np

from perimeter.boundary.crowd import CrowdMonitor, dbscan_cluster_sizes


def cluster(center, n, spread=5.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.array(center, dtype=np.float32) + rng.normal(scale=spread, size=(n, 2)).astype(
        np.float32
    )


# -- DBSCAN primitive --------------------------------------------------------------


def test_a_tight_group_forms_one_cluster():
    points = cluster((100, 100), 6, spread=5.0)
    sizes = dbscan_cluster_sizes(points, eps=30.0, min_samples=2)
    assert sizes == [6]


def test_spread_out_points_form_no_cluster():
    points = np.array([[0, 0], [500, 0], [0, 500], [500, 500]], dtype=np.float32)
    assert dbscan_cluster_sizes(points, eps=30.0, min_samples=2) == []


def test_two_separate_huddles_are_two_clusters():
    points = np.concatenate(
        [cluster((0, 0), 4, spread=3.0, seed=1), cluster((1000, 1000), 3, spread=3.0, seed=2)]
    )
    sizes = sorted(dbscan_cluster_sizes(points, eps=20.0, min_samples=2))
    assert sizes == [3, 4]


def test_empty_input_yields_no_clusters():
    assert dbscan_cluster_sizes(np.zeros((0, 2), np.float32), eps=30.0, min_samples=2) == []


def test_min_samples_of_one_treats_every_point_as_its_own_cluster():
    points = np.array([[0, 0], [500, 0]], dtype=np.float32)
    assert sorted(dbscan_cluster_sizes(points, eps=30.0, min_samples=1)) == [1, 1]


# -- per-zone hysteresis -------------------------------------------------------------


def test_no_event_below_threshold():
    monitor = CrowdMonitor(eps_px=30.0, min_samples=2)
    points = cluster((0, 0), 2, spread=2.0)  # a pair, but threshold is 5
    assert monitor.update("z1", points, threshold=5, min_frames=1) is None


def test_fires_once_threshold_and_min_frames_are_met():
    monitor = CrowdMonitor(eps_px=30.0, min_samples=2)
    points = cluster((0, 0), 6, spread=3.0)

    assert monitor.update("z1", points, threshold=5, min_frames=3) is None  # age 1
    assert monitor.update("z1", points, threshold=5, min_frames=3) is None  # age 2
    event = monitor.update("z1", points, threshold=5, min_frames=3)  # age 3

    assert event is not None
    assert event.zone_id == "z1"
    assert event.cluster_size == 6


def test_does_not_refire_while_the_crowd_persists():
    monitor = CrowdMonitor(eps_px=30.0, min_samples=2)
    points = cluster((0, 0), 6, spread=3.0)
    for _ in range(3):
        monitor.update("z1", points, threshold=5, min_frames=3)

    # Same crowd, still there - must not fire again every subsequent frame.
    assert monitor.update("z1", points, threshold=5, min_frames=3) is None
    assert monitor.update("z1", points, threshold=5, min_frames=3) is None


def test_refires_after_the_crowd_disperses_and_reforms():
    monitor = CrowdMonitor(eps_px=30.0, min_samples=2)
    crowd = cluster((0, 0), 6, spread=3.0)
    empty = np.zeros((0, 2), np.float32)

    for _ in range(3):
        monitor.update("z1", crowd, threshold=5, min_frames=3)
    for _ in range(3):  # disperses, and stays dispersed for min_frames
        monitor.update("z1", empty, threshold=5, min_frames=3)

    for _ in range(2):
        assert monitor.update("z1", crowd, threshold=5, min_frames=3) is None
    event = monitor.update("z1", crowd, threshold=5, min_frames=3)
    assert event is not None


def test_a_single_dropped_detection_frame_does_not_refire_the_same_crowd():
    monitor = CrowdMonitor(eps_px=30.0, min_samples=2)
    crowd = cluster((0, 0), 6, spread=3.0)
    flicker = crowd[:3]  # detector momentarily loses half the group

    events = []
    for frame in [crowd] * 3 + [flicker] + [crowd] * 5 + [flicker, flicker] + [crowd] * 5:
        events.append(monitor.update("z1", frame, threshold=5, min_frames=3))

    assert sum(e is not None for e in events) == 1


def test_zones_are_tracked_independently():
    monitor = CrowdMonitor(eps_px=30.0, min_samples=2)
    crowd = cluster((0, 0), 6, spread=3.0)
    sparse = cluster((0, 0), 2, spread=2.0)

    for _ in range(3):
        monitor.update("z1", crowd, threshold=5, min_frames=3)
        assert monitor.update("z2", sparse, threshold=5, min_frames=3) is None
