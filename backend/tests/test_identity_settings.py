"""Tests for IdentitySettings parsing from app.yaml (Expansion Plan Phase F).

The `identity:` block in config/app.yaml has existed since before this phase (the
scaffold reserved for the face layer) but was never read by settings.py - mirrors
test_firesmoke_settings.py's proof-of-wiring shape.
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

    assert settings.identity.enabled is False
    assert settings.identity.min_person_box_px == 120
    assert settings.identity.cosine_threshold == 0.363


def test_reads_a_custom_identity_block(tmp_path):
    path = write_yaml(
        tmp_path,
        "identity:\n"
        "  enabled: true\n"
        "  min_person_box_px: 150\n"
        "  cosine_threshold: 0.4\n",
    )
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.identity.enabled is True
    assert settings.identity.min_person_box_px == 150
    assert settings.identity.cosine_threshold == 0.4


def test_a_partial_block_falls_back_to_defaults_for_the_rest(tmp_path):
    path = write_yaml(tmp_path, "identity:\n  enabled: true\n")
    settings = load_settings(config_path=path, env_file=tmp_path / ".env")

    assert settings.identity.enabled is True
    assert settings.identity.min_person_box_px == 120  # untouched default
    assert settings.identity.cosine_threshold == 0.363
