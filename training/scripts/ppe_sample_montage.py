#!/usr/bin/env python3
"""Build one montage image per unified PPE class, cropping around each ground-truth
box (with a margin) and drawing it, so a human can eyeball label correctness quickly -
this is diagnostic tooling only, not used by training or the validation harness.

Usage:
    python ppe_sample_montage.py --manifest /path/to/domain_manifest.jsonl \
        --dataset-root /path/to/ppe_merged --out-dir /path/to/montages --n 20
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ppe_taxonomy import UNIFIED_CLASSES

THUMB = 180
COLS = 5
MARGIN_FRAC = 0.25  # extra context around the box, as a fraction of box size


def crop_with_box(img: np.ndarray, cx: float, cy: float, bw: float, bh: float) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
    x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
    mx, my = bw * w * MARGIN_FRAC, bh * h * MARGIN_FRAC
    cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    cx2, cy2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    crop = img[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return np.full((THUMB, THUMB, 3), 64, np.uint8)
    # box coords relative to the crop
    bx1, by1 = int(x1 - cx1), int(y1 - cy1)
    bx2, by2 = int(x2 - cx1), int(y2 - cy1)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    return cv2.resize(crop, (THUMB, THUMB))


def build_montage(thumbs: list[np.ndarray], label: str) -> np.ndarray:
    rows = (len(thumbs) + COLS - 1) // COLS
    grid = np.full((rows * THUMB + 30, COLS * THUMB, 3), 30, np.uint8)
    cv2.putText(grid, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, COLS)
        y0 = 30 + r * THUMB
        grid[y0:y0 + THUMB, c * THUMB:(c + 1) * THUMB] = thumb
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_class: dict[str, list[tuple[Path, float, float, float, float]]] = defaultdict(list)
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        img_path = Path(entry["image"])
        # find the matching label file: <root>/labels/<split>/<stem>.txt
        # image path is <root>/images/<split>/<file> - derive the split from that.
        parts = img_path.parts
        split = parts[-2]
        label_path = args.dataset_root / "labels" / split / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        lines = label_path.read_text().splitlines()
        for line2 in lines:
            p = line2.split()
            if len(p) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            by_class[UNIFIED_CLASSES[cls_id]].append((img_path, cx, cy, bw, bh))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cls_name in UNIFIED_CLASSES:
        instances = by_class.get(cls_name, [])
        if not instances:
            print(f"{cls_name}: NO DATA, skipping montage")
            continue
        rng.shuffle(instances)
        sample = instances[: args.n]
        thumbs = []
        for img_path, cx, cy, bw, bh in sample:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            thumbs.append(crop_with_box(img, cx, cy, bw, bh))
        montage = build_montage(thumbs, f"{cls_name}  (n={len(thumbs)} of {len(instances)} total instances)")
        out_path = args.out_dir / f"{cls_name}.jpg"
        cv2.imwrite(str(out_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{cls_name}: wrote {out_path} ({len(thumbs)} samples)")


if __name__ == "__main__":
    main()
