#!/usr/bin/env python3
"""Two-signal vest_present mislabel check, replacing the colour-only heuristic that
overcounted false positives (blue/cyan/pink/black vests the original hue range
couldn't see - confirmed by the user's own visual review of the spot-check montage).

Signal 1 (colour): widened hi-vis hue range - orange/yellow, green, AND now
    blue/cyan (~90-130) and pink/magenta (~150-170). Real safety vests come in more
    colours than orange/yellow/green; the check should not conflate "not
    orange-yellow-green" with "not a vest".
Signal 2 (stripe, colour-independent): every real safety vest has retroreflective
    stripes regardless of base colour (including a black vest with a reflective
    stripe found in the sample). Detected via Canny edges + probabilistic Hough
    transform, looking for multiple near-parallel line segments in the box region.

A crop is only flagged as a likely mislabel if it fails BOTH signals - passing
either one (a colour match OR a visible stripe pattern) is enough to call it a
plausible vest.

Usage:
    python ppe_vest_two_signal_check.py --dataset A:/fsbd_training/ppe_merged \\
        --manifest A:/fsbd_training/ppe_merged/domain_manifest.jsonl \\
        --out-dir A:/fsbd_training/ppe_review_montages
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

# Widened again: orange/yellow, green, AND blue/cyan, pink/magenta.
HIVIS_HUE_RANGES = [(10, 45), (45, 85), (90, 130), (150, 170)]
SAT_MIN, VAL_MIN = 60, 70  # relaxed further - blue/pink hi-vis can read less saturated
COLOUR_THRESH = 0.12

BAD_FAMILIES_ALREADY_REMOVED = {"image", "Image", "img"}
ROBOFLOW_SUFFIX_RE = re.compile(r"\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)
LEADING_ALPHA_RE = re.compile(r"^[A-Za-z]+")


def coarse_group(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_jpg$", "", stem, flags=re.IGNORECASE)
    stem = ROBOFLOW_SUFFIX_RE.sub("", stem)
    m = LEADING_ALPHA_RE.match(stem)
    return m.group(0) if m else stem


def colour_signal(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for lo, hi in HIVIS_HUE_RANGES:
        mask |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi) & (hsv[..., 1] >= SAT_MIN) & (hsv[..., 2] >= VAL_MIN)
    return float(mask.mean())


def stripe_signal(crop: np.ndarray, min_lines: int = 2, angle_tol_deg: float = 12.0) -> tuple[bool, int]:
    """Detect a set of >=min_lines roughly-parallel high-contrast line segments -
    the geometric signature of retroreflective vest striping, independent of colour.
    Returns (has_stripes, largest_parallel_group_size)."""
    if crop.size == 0:
        return False, 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 60, 160)
    h, w = crop.shape[:2]
    min_len = max(10, int(min(h, w) * 0.25))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=min_len, maxLineGap=6)
    if lines is None or len(lines) < min_lines:
        return False, 0 if lines is None else len(lines)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        angles.append(angle)
    angles = np.array(angles)

    # bucket by angle, find the largest group of mutually-near-parallel lines
    best_group = 0
    for a in angles:
        group = np.sum(np.abs(((angles - a + 90) % 180) - 90) <= angle_tol_deg)
        best_group = max(best_group, int(group))
    return best_group >= min_lines, best_group


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
    ap.add_argument("--n-montage", type=int, default=60)
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
    n_checked = 0
    n_colour_only_fail = 0
    n_both_fail = 0
    both_failed = []

    for inst in instances:
        fam = coarse_group(inst["image"].name)
        if fam in BAD_FAMILIES_ALREADY_REMOVED:
            continue
        img = cv2.imread(str(inst["image"]))
        if img is None:
            continue
        n_checked += 1
        crop = crop_box(img, inst["box"])
        colour_frac = colour_signal(crop)
        colour_pass = colour_frac >= COLOUR_THRESH
        stripe_pass, stripe_group = stripe_signal(crop)

        if not colour_pass:
            n_colour_only_fail += 1
        if not colour_pass and not stripe_pass:
            n_both_fail += 1
            both_failed.append({
                "image": inst["image"], "box": inst["box"],
                "colour_frac": colour_frac, "stripe_group": stripe_group,
            })

    print(f"checked {n_checked} vest_present instances (bad families already excluded)\n")
    print(f"colour-signal-only fail rate: {n_colour_only_fail}/{n_checked} = {n_colour_only_fail/n_checked*100:.2f}%")
    print(f"BOTH signals fail (two-signal mislabel rate): {n_both_fail}/{n_checked} = {n_both_fail/n_checked*100:.2f}%")
    print(f"  -> colour check alone overcounted by {n_colour_only_fail - n_both_fail} instances "
          f"({(n_colour_only_fail - n_both_fail)/n_checked*100:.2f} percentage points)\n")

    # fresh montage of what's still flagged under the two-signal check
    rng = random.Random(0)
    sample = rng.sample(both_failed, min(args.n_montage, len(both_failed)))
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
        cv2.putText(crop, f"c{item['colour_frac']:.2f} s{item['stripe_group']}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        thumbs.append(cv2.resize(crop, (THUMB, THUMB)))

    rows_n = (len(thumbs) + COLS - 1) // COLS
    grid = np.full((rows_n * THUMB + 30, COLS * THUMB, 3), 30, np.uint8)
    cv2.putText(grid, f"vest_present two-signal-fail spot-check (n={len(thumbs)} of {len(both_failed)}, "
                       f"c=colour-fraction s=parallel-line-group-size)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, COLS)
        y0 = 30 + r * THUMB
        grid[y0:y0 + THUMB, c * THUMB:(c + 1) * THUMB] = thumb

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "vest_present_twosignal_flagged.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote montage: {out_path} ({len(thumbs)} samples of {len(both_failed)} still flagged)")

    out = {
        "n_checked": n_checked,
        "colour_only_fail": n_colour_only_fail,
        "both_fail": n_both_fail,
        "corrected_rate": n_both_fail / n_checked if n_checked else 0,
    }
    (args.dataset / "vest_two_signal_report.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.dataset / 'vest_two_signal_report.json'}")


if __name__ == "__main__":
    main()
