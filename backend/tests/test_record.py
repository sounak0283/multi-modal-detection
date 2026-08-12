"""Tests for the site recorder's pure helpers.

The capture loop itself needs a camera, so it is exercised manually against a video
file (see README). What is tested here is the logic that can silently misbehave for
weeks on an unattended box: source parsing and disk pruning.
"""

from __future__ import annotations

import os
import time

import pytest

from fsbd.record import RecorderConfig, parse_source, prune_to_budget


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0),
        ("1", 1),
        ("rtsp://user:pw@10.0.0.5:554/stream1", "rtsp://user:pw@10.0.0.5:554/stream1"),
        ("clip.mp4", "clip.mp4"),
        ("C:/footage/clip.mp4", "C:/footage/clip.mp4"),
    ],
)
def test_parse_source(raw, expected):
    assert parse_source(raw) == expected


def _write(path, size_bytes, mtime):
    path.write_bytes(b"\0" * size_bytes)
    os.utime(path, (mtime, mtime))


def test_prune_removes_oldest_first(tmp_path):
    now = time.time()
    # 3 MiB total across three files, oldest first.
    _write(tmp_path / "old.mp4", 1024 * 1024, now - 300)
    _write(tmp_path / "mid.mp4", 1024 * 1024, now - 200)
    _write(tmp_path / "new.mp4", 1024 * 1024, now - 100)

    # Budget of 2 MiB must evict exactly the oldest file.
    removed = prune_to_budget(tmp_path, budget_gb=2 / 1024)

    assert removed == 1
    assert not (tmp_path / "old.mp4").exists()
    assert (tmp_path / "mid.mp4").exists()
    assert (tmp_path / "new.mp4").exists()


def test_prune_noop_when_under_budget(tmp_path):
    _write(tmp_path / "a.mp4", 1024, time.time())
    assert prune_to_budget(tmp_path, budget_gb=1.0) == 0
    assert (tmp_path / "a.mp4").exists()


def test_prune_recurses_into_day_directories(tmp_path):
    """Segments live under recordings/<date>/, so pruning must not be top-level only."""
    now = time.time()
    day = tmp_path / "2026-08-11"
    day.mkdir()
    _write(day / "seg_090000.mp4", 1024 * 1024, now - 300)
    _write(day / "seg_091500.mp4", 1024 * 1024, now - 100)

    removed = prune_to_budget(tmp_path, budget_gb=1 / 1024)

    assert removed == 1
    assert not (day / "seg_090000.mp4").exists()


def test_config_defaults_are_conservative():
    """Defaults must be safe for an unattended box at a customer site."""
    cfg = RecorderConfig(source=0, outdir=None)  # type: ignore[arg-type]
    assert cfg.fps <= 5, "recording rate should stay low; purpose is negative mining"
    assert cfg.disk_budget_gb > 0, "an unbounded recorder can fill a customer's disk"
    assert cfg.still_interval_s == 0, "stills should be opt-in"
