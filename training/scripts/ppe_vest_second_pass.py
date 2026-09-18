#!/usr/bin/env python3
"""Second-pass spot-check on the 417 vest_present instances still flagged after
removing the image/Image/img naming families. Two things:

1. Source-family clustering check on JUST this remainder (same method as before)
   to catch a second concentrated sub-source, since every "looks clean" estimate
   this session undershot on the first pass.
2. A visual montage of a random sample for manual review of genuine-mislabel vs
   false-positive-flag (shadowed/occluded/unusual-colour legitimate vest).

Usage:
    python ppe_vest_second_pass.py --dataset A:/fsbd_training/ppe_merged \\
        --manifest A:/fsbd_training/ppe_merged/domain_manifest.jsonl \\
        --out-dir A:/fsbd_training/ppe_review_montages --n 75
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ppe_taxonomy import UNIFIED_CLASSES

HIVIS_HUE_RANGES = [(10, 45), (45, 85)]
SAT_MIN, VAL_MIN = 70, 80
THRESH = 0.12
BAD_FAMILIES_ALREADY_REMOVED = {"image", "Image", "img"}

ROBOFLOW_SUFFIX_RE = re.compile(r"\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)
TRAILING_FRAME_IDX_RE = re.compile(r"[-_]\d+$")
LEADING_ALPHA_RE = re.compile(r"^[A-Za-z]+")


def original_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_jpg$", "", stem, flags=re.IGNORECASE)
    return ROBOFLOW_SUFFIX_RE.sub("", stem)


def fine_group(stem: str) -> str:
    return TRAILING_FRAME_IDX_RE.sub("", stem)


def coarse_group(stem: str) -> str:
    m = LEADING_ALPHA_RE.match(stem)
    return m.group(0) if m else stem


def hivis_fraction(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for lo, hi in HIVIS_HUE_RANGES:
        mask |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi) & (hsv[..., 1] >= SAT_MIN) & (hsv[..., 2] >= VAL_MIN)
    return float(mask.mean())


def crop_box(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
    x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
    return img[y1:y2, x1:x2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_class: dict[str, list[dict]] = defaultdict(list)
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        img_path = Path(entry["image"])
        split = img_path.parts[-2]
        label_path = args.dataset / "labels" / split / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        for line2 in label_path.read_text().splitlines():
            p = line2.split()
            if len(p) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            by_class[UNIFIED_CLASSES[cls_id]].append({"image": img_path, "box": (cx, cy, bw, bh)})

    instances = by_class.get("vest_present", [])
    flagged = []
    fine_total, fine_flagged = defaultdict(int), defaultdict(int)
    coarse_total, coarse_flagged = defaultdict(int), defaultdict(int)
    n_checked = 0

    for inst in instances:
        stem = original_stem(inst["image"].name)
        cg = coarse_group(stem)
        if cg in BAD_FAMILIES_ALREADY_REMOVED:
            continue  # already excluded from the "remaining" pool
        img = cv2.imread(str(inst["image"]))
        if img is None:
            continue
        n_checked += 1
        frac = hivis_fraction(crop_box(img, inst["box"]))
        fg = fine_group(stem)
        fine_total[fg] += 1
        coarse_total[cg] += 1
        if frac < THRESH:
            fine_flagged[fg] += 1
            coarse_flagged[cg] += 1
            flagged.append({"image": inst["image"], "box": inst["box"], "frac": frac, "fine": fg, "coarse": cg})

    print(f"checked {n_checked} (bad families already excluded), flagged {len(flagged)} "
          f"({len(flagged)/n_checked*100:.2f}%)\n")

    print("--- second-pass FINE grouping (>=3 instances) ---")
    fine_rows = sorted(
        ((g, fine_total[g], fine_flagged.get(g, 0)) for g in fine_total if fine_total[g] >= 3 and fine_flagged.get(g, 0) > 0),
        key=lambda r: -r[2],
    )
    for g, tot, fl in fine_rows[:15]:
        print(f"  {g!r:30s} total={tot:4d} flagged={fl:4d} rate={fl/tot*100:5.1f}%")
    top5 = sum(r[2] for r in fine_rows[:5])
    print(f"  top-5 fine groups: {top5}/{len(flagged)} = {top5/len(flagged)*100:.1f}% of remaining flagged\n")

    print("--- second-pass COARSE grouping (>=10 instances) ---")
    coarse_rows = sorted(
        ((g, coarse_total[g], coarse_flagged.get(g, 0)) for g in coarse_total if coarse_total[g] >= 10 and coarse_flagged.get(g, 0) > 0),
        key=lambda r: -r[2],
    )
    for g, tot, fl in coarse_rows[:15]:
        print(f"  {g!r:30s} total={tot:4d} flagged={fl:4d} rate={fl/tot*100:5.1f}%")
    top5c = sum(r[2] for r in coarse_rows[:5])
    print(f"  top-5 coarse groups: {top5c}/{len(flagged)} = {top5c/len(flagged)*100:.1f}% of remaining flagged\n")

    # ---- montage of a random sample for visual review ----
    rng = random.Random(args.seed)
    sample = rng.sample(flagged, min(args.n, len(flagged)))
    THUMB, COLS = 160, 8
    thumbs = []
    for item in sample:
        img = cv2.imread(str(item["image"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy, bw, bh = item["box"]
        mx, my = bw * w * 0.25, bh * h * 0.25
        x1, y1 = max(0, int((cx - bw / 2) * w - mx)), max(0, int((cy - bh / 2) * h - my))
        x2, y2 = min(w, int((cx + bw / 2) * w + mx)), min(h, int((cy + bh / 2) * h + my))
        crop = img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        bx1, by1 = int((cx - bw / 2) * w - x1), int((cy - bh / 2) * h - y1)
        bx2, by2 = int((cx + bw / 2) * w - x1), int((cy + bh / 2) * h - y1)
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(crop, f"{item['frac']:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        thumbs.append(cv2.resize(crop, (THUMB, THUMB)))

    rows_n = (len(thumbs) + COLS - 1) // COLS
    grid = np.full((rows_n * THUMB + 30, COLS * THUMB, 3), 30, np.uint8)
    cv2.putText(grid, f"vest_present 2nd-pass flagged spot-check (n={len(thumbs)} of {len(flagged)} remaining-flagged, "
                       f"hivis-fraction shown per crop)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, COLS)
        y0 = 30 + r * THUMB
        grid[y0:y0 + THUMB, c * THUMB:(c + 1) * THUMB] = thumb

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "vest_present_2ndpass_flagged.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote montage: {out_path} ({len(thumbs)} samples)")


if __name__ == "__main__":
    main()
