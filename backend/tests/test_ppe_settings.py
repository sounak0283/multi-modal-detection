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


# -- ppe.backend: the switch between the helmet detector and the old classifier ----------


def test_helmet_detector_is_the_default_backend(tmp_path):
    path = write_yaml(tmp_path, "storage:\n  retention_days: 30\n")
    ppe = load_settings(config_path=path, env_file=tmp_path / ".env").ppe

    assert ppe.backend == "helmet_detector"
    assert ppe.model_path == "models/ppe/ppe_final_C_v2_320.onnx"
    assert (ppe.head_threshold, ppe.helmet_threshold) == (0.45, 0.55)
    assert ppe.min_person_height_px == 96


def test_classifier_backend_reverts_to_the_old_model_path(tmp_path):
    path = write_yaml(tmp_path, "ppe:\n  backend: classifier\n")
    ppe = load_settings(config_path=path, env_file=tmp_path / ".env").ppe

    assert ppe.backend == "classifier"
    assert ppe.model_path == "models/ppe/model.onnx"


def test_custom_thresholds_and_paths(tmp_path):
    path = write_yaml(
        tmp_path,
        "ppe:\n  helmet_detector_model: elsewhere/h.onnx\n  head_threshold: 0.5\n"
        "  helmet_threshold: 0.6\n  min_person_height_px: 120\n",
    )
    ppe = load_settings(config_path=path, env_file=tmp_path / ".env").ppe

    assert ppe.model_path == "elsewhere/h.onnx"
    assert (ppe.head_threshold, ppe.helmet_threshold, ppe.min_person_height_px) == (0.5, 0.6, 120)


def test_an_unknown_backend_is_rejected_at_load(tmp_path):
    path = write_yaml(tmp_path, "ppe:\n  backend: vest_magic\n")
    with pytest.raises(ValueError, match="ppe.backend"):
        load_settings(config_path=path, env_file=tmp_path / ".env")
