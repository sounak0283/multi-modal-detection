"""Tests for FireSmokeSettings parsing from app.yaml (Expansion Plan Phase E).

The `firesmoke:` block in config/app.yaml has existed since before this phase but was
never read by settings.py - these tests are what prove the wiring actually works now.
"""

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

    assert settings.firesmoke.conf_fire == 0.45
    assert settings.firesmoke.conf_smoke == 0.35
    assert settings.firesmoke.gate_k == 6
    assert settings.firesmoke.gate_n == 10
    assert settings.firesmoke.gate_iou == 0.3


def test_reads_a_custom_firesmoke_block(tmp_path):
    path = write_yaml(
        tmp_path,
        "firesmoke:\n"
        "  conf_fire: 0.6\n"
        "  conf_smoke: 0.4\n"
        "  gate_k: 4\n"
        "  gate_n: 8\n"
        "  gate_iou: 0.25\n",
    )
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.firesmoke.conf_fire == 0.6
    assert settings.firesmoke.conf_smoke == 0.4
    assert settings.firesmoke.gate_k == 4
    assert settings.firesmoke.gate_n == 8
    assert settings.firesmoke.gate_iou == 0.25


def test_a_partial_block_falls_back_to_defaults_for_the_rest(tmp_path):
    path = write_yaml(tmp_path, "firesmoke:\n  conf_fire: 0.9\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.firesmoke.conf_fire == 0.9
    assert settings.firesmoke.conf_smoke == 0.35  # untouched default
    assert settings.firesmoke.gate_k == 6


def test_alert_gap_defaults_to_30_seconds(tmp_path):
    path = write_yaml(tmp_path, "storage:\n  retention_days: 30\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")
    assert settings.firesmoke.alert_gap_seconds == 30.0


def test_alert_gap_zero_and_negative_mean_the_k_of_n_gate(tmp_path):
    for raw, expected in (("0", 0.0), ("-5", 0.0), ("45", 45.0)):
        path = write_yaml(tmp_path, f"firesmoke:\n  alert_gap_seconds: {raw}\n")
        settings = load_settings(config_path=path, env_file=tmp_path / ".env")
        assert settings.firesmoke.alert_gap_seconds == expected
