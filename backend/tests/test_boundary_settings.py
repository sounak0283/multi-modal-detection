"""Tests for BoundarySettings parsing from app.yaml, in particular
`track_activation_threshold` - the tracker's bar for STARTING a track, separate from
the detector's own `person_conf` cutoff (see settings.py's docstring for the measured
backlit-scene evidence behind the 2026-09-22 default change)."""

from __future__ import annotations

import os

import pytest

from perimeter.settings import load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("PERIMETER_"):
            monkeypatch.delenv(key, raising=False)


def write_yaml(tmp_path, text):
    path = tmp_path / "app.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_the_block_is_absent(tmp_path):
    path = write_yaml(tmp_path, "storage:\n  retention_days: 30\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.boundary.default_min_frames == 4
    assert settings.boundary.person_conf == 0.30
    assert settings.boundary.track_activation_threshold == 0.40


def test_reads_a_custom_boundary_block(tmp_path):
    path = write_yaml(
        tmp_path,
        "boundary:\n  default_min_frames: 6\n  person_conf: 0.25\n"
        "  track_activation_threshold: 0.55\n",
    )
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.boundary.default_min_frames == 6
    assert settings.boundary.person_conf == 0.25
    assert settings.boundary.track_activation_threshold == 0.55


def test_a_partial_block_falls_back_to_defaults_for_the_rest(tmp_path):
    path = write_yaml(tmp_path, "boundary:\n  track_activation_threshold: 0.35\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.boundary.track_activation_threshold == 0.35
    assert settings.boundary.default_min_frames == 4  # untouched default
    assert settings.boundary.person_conf == 0.30  # untouched default
