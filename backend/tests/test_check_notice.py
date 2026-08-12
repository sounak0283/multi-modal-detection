"""Tests for the asset licence gate.

This gate is a commercial control, not a convenience: it is what stops an unrecorded
model or dataset reaching a customer build. A gate that silently passes everything is
worse than no gate, so its failure paths are tested explicitly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
GATE = BACKEND_ROOT / "tools" / "check_notice.py"


def run_gate(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), "--root", str(root)],
        capture_output=True,
        text=True,
    )


def make_repo(tmp_path: Path, notice: str) -> Path:
    (tmp_path / "NOTICE.md").write_text(notice, encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


def write_model(root: Path, name: str, manifest: dict | None) -> Path:
    model_dir = root / "models" / name
    model_dir.mkdir(parents=True)
    if manifest is not None:
        (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return model_dir


VALID_MANIFEST = {"name": "yunet", "licence": "MIT", "source": "https://example.invalid/yunet"}


def test_passes_on_empty_repo(tmp_path):
    make_repo(tmp_path, "# NOTICE\n")
    assert run_gate(tmp_path).returncode == 0


def test_fails_on_unlisted_dataset(tmp_path):
    root = make_repo(tmp_path, "# NOTICE\n")
    (root / "data" / "mystery_dataset").mkdir()

    result = run_gate(root)

    assert result.returncode == 1
    assert "mystery_dataset" in result.stderr


def test_passes_when_dataset_is_listed(tmp_path):
    root = make_repo(tmp_path, "# NOTICE\n\n| dfire | CC0 | ... |\n")
    (root / "data" / "dfire").mkdir()

    assert run_gate(root).returncode == 0


def test_fails_when_model_has_no_manifest(tmp_path):
    root = make_repo(tmp_path, "# NOTICE\n\nyunet\n")
    write_model(root, "yunet", manifest=None)

    result = run_gate(root)

    assert result.returncode == 1
    assert "manifest.json" in result.stderr


def test_fails_on_manifest_missing_required_key(tmp_path):
    root = make_repo(tmp_path, "# NOTICE\n\nyunet\n")
    write_model(root, "yunet", {"name": "yunet", "licence": "MIT"})  # no source

    result = run_gate(root)

    assert result.returncode == 1
    assert "source" in result.stderr


def test_inhouse_weights_need_training_provenance(tmp_path):
    """Weights we trained ourselves carry the heavier burden - dataset traceability."""
    root = make_repo(tmp_path, "# NOTICE\n\nfiresmoke\n")
    write_model(root, "firesmoke", {"name": "firesmoke", "licence": "ours", "source": "internal"})

    result = run_gate(root)

    assert result.returncode == 1
    assert "datasets" in result.stderr
    assert "training_commit" in result.stderr


def test_inhouse_weights_pass_with_full_provenance(tmp_path):
    root = make_repo(tmp_path, "# NOTICE\n\nfiresmoke\n")
    write_model(
        root,
        "firesmoke",
        {
            "name": "firesmoke",
            "licence": "ours",
            "source": "internal",
            "datasets": [{"name": "dfire", "sha256": "abc123"}],
            "training_commit": "deadbeef",
            "trained_at": "2026-08-11T10:00:00Z",
        },
    )

    assert run_gate(root).returncode == 0


def test_fails_on_malformed_manifest(tmp_path):
    root = make_repo(tmp_path, "# NOTICE\n\nyunet\n")
    model_dir = write_model(root, "yunet", VALID_MANIFEST)
    (model_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    result = run_gate(root)

    assert result.returncode == 1
    assert "not valid JSON" in result.stderr


def test_missing_notice_file_is_invocation_error(tmp_path):
    (tmp_path / "models").mkdir()
    result = run_gate(tmp_path)
    assert result.returncode == 2
