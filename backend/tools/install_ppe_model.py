#!/usr/bin/env python3
"""Install the PPE helmet model from demo_package.zip into models/ppe/.

Model weights are git-ignored (NOTICE.md section 1), so a fresh checkout needs this step.
The file is verified against the SHA-256 recorded in models/ppe/manifest.json before it is
written, so a different or corrupted model can never silently take its place.

    python tools/install_ppe_model.py --zip "C:/path/to/demo_package.zip"
    python tools/install_ppe_model.py --package-dir ../third_party/demo_package

Exit code 0 = installed (or already present and verified), 1 = verification failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_ROOT / "models" / "ppe"
MEMBER = "models/ppe_final_C_v2_320.onnx"


def expected_sha256(filename: str) -> str:
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["file"] == filename:
            return entry["sha256"]
    raise SystemExit(f"{filename} is not listed in {MODEL_DIR / 'manifest.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="path to demo_package.zip")
    source.add_argument("--package-dir", type=Path, help="an already-extracted demo_package/")
    args = parser.parse_args(argv)

    filename = Path(MEMBER).name
    if args.zip:
        with zipfile.ZipFile(args.zip) as archive:
            data = archive.read(MEMBER)
    else:
        data = (args.package_dir / MEMBER).read_bytes()

    want = expected_sha256(filename)
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        print(f"FAIL: {filename} sha256 {got} does not match manifest {want}", file=sys.stderr)
        return 1

    target = MODEL_DIR / filename
    target.write_bytes(data)
    print(f"installed {target} ({len(data)} bytes, sha256 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
