"""Tests for CrowdSettings parsing from app.yaml (Expansion Plan Phase G)."""

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


def test_defaults_match_the_committed_app_yaml_when_the_block_is_absent(tmp_path):
    path = write_yaml(tmp_path, "storage:\n  retention_days: 30\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.crowd.eps_px == 80.0
    assert settings.crowd.min_samples == 2


def test_reads_a_custom_crowd_block(tmp_path):
    path = write_yaml(tmp_path, "crowd:\n  eps_px: 40.0\n  min_samples: 3\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.crowd.eps_px == 40.0
    assert settings.crowd.min_samples == 3


def test_a_partial_block_falls_back_to_defaults_for_the_rest(tmp_path):
    path = write_yaml(tmp_path, "crowd:\n  eps_px: 100.0\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.crowd.eps_px == 100.0
    assert settings.crowd.min_samples == 2  # untouched default
