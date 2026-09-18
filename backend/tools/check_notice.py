#!/usr/bin/env python3
"""Asset licence gate.

`pip-licenses` reads Python package metadata only. It cannot see an .onnx file, a
dataset directory, or a vendored source file - which is exactly where this project's
licensing risk lives (NOTICE.md section 6.1). This gate covers that blind spot.

Two checks:

1. Every immediate subdirectory of models/ and data/ is mentioned in NOTICE.md.
   An unlisted asset is a build failure, not a warning.
2. Every models/<name>/ carries a manifest.json recording provenance, so the question
   "which shipped weights are affected?" is answerable if a licence later proves
   different from what was recorded.

Exit code 0 = pass, 1 = fail, 2 = bad invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keys every models/<name>/manifest.json must carry.
REQUIRED_MANIFEST_KEYS = ("name", "licence", "source")

# Additional keys required when the manifest declares the weights as our own,
# so a dataset licence problem can be traced to specific shipped checkpoints.
REQUIRED_INHOUSE_KEYS = ("datasets", "training_commit", "trained_at")

SKIP_NAMES = {".gitkeep", ".gitignore", "README.md", "__pycache__"}


def iter_assets(root: Path) -> list[Path]:
    """Immediate subdirectories and loose files under `root`, excluding housekeeping."""
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir() if p.name not in SKIP_NAMES and not p.name.startswith(".")
    )


def check_listed(assets: list[Path], notice_text: str) -> list[str]:
    """Every asset directory name must appear somewhere in NOTICE.md."""
    failures = []
    for asset in assets:
        if asset.name not in notice_text:
            failures.append(
                f"{asset.as_posix()} has no entry in NOTICE.md. "
                f"Add a row recording its licence and source before it can ship."
            )
    return failures


def check_manifest(model_dir: Path) -> list[str]:
    """Every models/<name>/ must carry a provenance manifest."""
    if not model_dir.is_dir():
        return []

    # A directory holding only housekeeping (e.g. a README describing weights that are not
    # trained yet) ships nothing, so there is no provenance to record.
    if not iter_assets(model_dir):
        return []

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return [
            f"{model_dir.as_posix()} has no manifest.json. "
            f"Provenance is required for every shipped model (NOTICE.md section 1)."
        ]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{manifest_path.as_posix()} is not valid JSON: {exc}"]

    if not isinstance(manifest, dict):
        return [f"{manifest_path.as_posix()} must contain a JSON object."]

    failures = [
        f"{manifest_path.as_posix()} is missing required key '{key}'."
        for key in REQUIRED_MANIFEST_KEYS
        if key not in manifest
    ]

    # In-house weights carry the heavier provenance burden: they are the ones whose
    # licence status depends on training data rather than on an upstream declaration.
    if str(manifest.get("licence", "")).strip().lower() in {"ours", "in-house", "proprietary"}:
        failures.extend(
            f"{manifest_path.as_posix()} declares in-house weights but is missing '{key}'."
            for key in REQUIRED_INHOUSE_KEYS
            if key not in manifest
        )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["models", "data"],
        help="Directories whose contents must be recorded in NOTICE.md.",
    )
    parser.add_argument("--notice", default="NOTICE.md", help="Path to NOTICE.md.")
    parser.add_argument(
        "--root", default=".", help="Repository root (paths are resolved against this)."
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    notice_path = root / args.notice
    if not notice_path.is_file():
        print(f"FAIL: {args.notice} not found at {notice_path}", file=sys.stderr)
        return 2

    notice_text = notice_path.read_text(encoding="utf-8")
    failures: list[str] = []
    checked = 0

    for asset_root_name in args.assets:
        asset_root = root / asset_root_name
        assets = iter_assets(asset_root)
        checked += len(assets)
        failures.extend(check_listed(assets, notice_text))

        if asset_root.name == "models":
            for model_dir in assets:
                failures.extend(check_manifest(model_dir))

    if failures:
        print(f"Asset licence gate FAILED ({len(failures)} problem(s)):\n", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nNothing enters models/ or data/ without a NOTICE.md row recorded at "
            "download time. See NOTICE.md section 6.1.",
            file=sys.stderr,
        )
        return 1

    print(f"Asset licence gate passed ({checked} asset(s) checked against {args.notice}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
