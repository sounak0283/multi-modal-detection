"""Tests for classify_state's threshold logic and PPEMonitor's hysteresis (Expansion
Plan Phase H). Pure logic - no ONNX model involved, mirrors CrowdMonitor's (Phase G)
testing style."""

from __future__ import annotations

from perimeter.detect.ppe import PPE_ITEMS, PPEMonitor, classify_state


def test_ppe_items_matches_the_plan_verbatim():
    assert PPE_ITEMS == ("helmet", "vest", "gloves", "shoes", "glasses")


# -- classify_state -----------------------------------------------------------------


def test_high_probability_is_present():
    assert classify_state(0.9) == "present"


def test_low_probability_is_absent():
    assert classify_state(0.1) == "absent"


def test_the_gap_between_is_indeterminate():
    assert classify_state(0.5) == "indeterminate"


def test_boundary_values_are_inclusive():
    assert classify_state(0.65) == "present"
    assert classify_state(0.35) == "absent"


def test_custom_thresholds():
    assert classify_state(0.55, present=0.5, absent=0.2) == "present"


# -- PPEMonitor hysteresis -----------------------------------------------------------


def test_no_violation_on_a_single_absent_observation():
    monitor = PPEMonitor()
    assert monitor.update("z1", 1, "helmet", "absent", min_frames=3) is False


def test_fires_once_min_frames_of_absence_are_met():
    monitor = PPEMonitor()
    assert monitor.update("z1", 1, "helmet", "absent", min_frames=3) is False
    assert monitor.update("z1", 1, "helmet", "absent", min_frames=3) is False
    assert monitor.update("z1", 1, "helmet", "absent", min_frames=3) is True


def test_present_resets_progress_immediately():
    monitor = PPEMonitor()
    monitor.update("z1", 1, "helmet", "absent", min_frames=3)
    monitor.update("z1", 1, "helmet", "absent", min_frames=3)
    monitor.update("z1", 1, "helmet", "present", min_frames=3)
    assert monitor.update("z1", 1, "helmet", "absent", min_frames=3) is False  # age back to 1


def test_indeterminate_neither_advances_nor_resets():
    monitor = PPEMonitor()
    monitor.update("z1", 1, "helmet", "absent", min_frames=3)
    monitor.update("z1", 1, "helmet", "absent", min_frames=3)
    assert monitor.update("z1", 1, "helmet", "indeterminate", min_frames=3) is False
    # Still only 2 confirmed "absent" observations - one more should now confirm.
    assert monitor.update("z1", 1, "helmet", "absent", min_frames=3) is True


def test_does_not_refire_while_still_absent():
    monitor = PPEMonitor()
    for _ in range(3):
        monitor.update("z1", 1, "helmet", "absent", min_frames=3)
    assert monitor.update("z1", 1, "helmet", "absent", min_frames=3) is False


def test_refires_after_present_and_absent_again():
    monitor = PPEMonitor()
    for _ in range(3):
        monitor.update("z1", 1, "helmet", "absent", min_frames=3)
    monitor.update("z1", 1, "helmet", "present", min_frames=3)
    for _ in range(2):
        assert monitor.update("z1", 1, "helmet", "absent", min_frames=3) is False
    assert monitor.update("z1", 1, "helmet", "absent", min_frames=3) is True


def test_items_and_tracks_and_zones_are_independent():
    monitor = PPEMonitor()
    monitor.update("z1", 1, "helmet", "absent", min_frames=1)
    assert monitor.update("z1", 1, "vest", "absent", min_frames=1) is True
    assert monitor.update("z1", 2, "helmet", "absent", min_frames=1) is True
    assert monitor.update("z2", 1, "helmet", "absent", min_frames=1) is True


def test_evict_stale_removes_entries_not_seen_recently():
    monitor = PPEMonitor()
    monitor.update("z1", 1, "helmet", "absent", min_frames=1)  # last_seen = 1
    for _ in range(5):
        monitor.update("z2", 2, "vest", "present", min_frames=1)  # last_seen up to 6

    monitor.evict_stale(max_age_calls=3)  # cutoff = 6 - 3 = 3

    assert ("z1", 1, "helmet") not in monitor._states
    assert ("z2", 2, "vest") in monitor._states


def test_evict_stale_keeps_recently_seen_entries():
    monitor = PPEMonitor()
    monitor.update("z1", 1, "helmet", "absent", min_frames=1)
    monitor.evict_stale(max_age_calls=100)
    assert ("z1", 1, "helmet") in monitor._states
