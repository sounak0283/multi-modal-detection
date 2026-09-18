"""Tests for InferenceSettings.ppe_every_n parsing from app.yaml (Expansion Plan
Phase H)."""

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


def test_default_matches_the_committed_app_yaml_when_absent(tmp_path):
    path = write_yaml(tmp_path, "storage:\n  retention_days: 30\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.inference.ppe_every_n == 3


def test_reads_a_custom_value(tmp_path):
    path = write_yaml(tmp_path, "inference:\n  ppe_every_n: 5\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.inference.ppe_every_n == 5
