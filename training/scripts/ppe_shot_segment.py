#!/usr/bin/env python3
"""Video-shot segmentation for Construction-PPE's sequentially-numbered frame
exports, built after both global DCT-hash clustering (hamming<=6 through 26) and
global HSV-histogram clustering (correl 0.80-0.90) failed to isolate the
recurring rooftop gloves_present burst without either under- or wildly
over-merging unrelated construction photos (confirmed: hamming<=18 collapsed
603/705 images into one cluster; histogram threshold 0.80 collapsed 601/705).
Both failed because they compare ALL pairs globally, and many unrelated outdoor
construction photos share similar sky/concrete/hi-vis-orange palettes.

This instead only compares NEIGHBOURS in numeric-filename order (image1082,
image1083, ...) - since Construction-PPE's images are literal sequential video
frame exports (confirmed: image1082-image1109's correlation to a hand-picked
reference frame drops off smoothly and matches the visually-identified rooftop
burst), a shot boundary is wherever consecutive-by-ID similarity drops, not
wherever any two images anywhere in the dataset happen to look similar. This
avoids the over-merging failure mode entirely, at the cost of only working for
sequentially-exported sources (fine here - this is exactly what Construction-PPE
is).

Usage: imported by ppe_curate_gloves_glasses.py; run standalone for inspection:
    python ppe_shot_segment.py --dataset A:/fsbd_training/curated_gloves_glasses/construction_ppe/extracted --class-id 1
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def hist_feature(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def numeric_key(path: Path) -> tuple:
    m = re.search(r"(\d+)", path.stem)
    return (0, int(m.group(1))) if m else (1, path.stem)


def segment_shots(paths: list[Path], corr_thresh: float = 0.85) -> list[list[Path]]:
    """Sort by numeric filename id, walk sequentially, start a new shot whenever
    consecutive-frame similarity drops below corr_thresh."""
    ordered = sorted(paths, key=numeric_key)
    shots: list[list[Path]] = []
    prev_feat = None
    for p in ordered:
        img = cv2.imread(str(p))
        if img is None:
            continue
        feat = hist_feature(img)
        if prev_feat is not None and cv2.compareHist(prev_feat, feat, cv2.HISTCMP_CORREL) >= corr_thresh:
            shots[-1].append(p)
        else:
            shots.append([p])
        prev_feat = feat
    return shots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--class-id", type=int, required=True)
    ap.add_argument("--corr-thresh", type=float, default=0.85)
    args = ap.parse_args()

    paths = []
    for split in ("train", "val", "test"):
        img_dir = args.dataset / "images" / split
        lbl_dir = args.dataset / "labels" / split
        if not img_dir.is_dir():
            continue
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            ids = set(int(l.split()[0]) for l in lbl_path.read_text().splitlines() if l.split())
            if args.class_id in ids:
                paths.append(img_path)

    shots = segment_shots(sorted(set(paths)), args.corr_thresh)
    sizes = sorted((len(s) for s in shots), reverse=True)
    print(f"{len(paths)} images -> {len(shots)} shots (corr_thresh={args.corr_thresh})")
    print("top 10 shot sizes:", sizes[:10])


if __name__ == "__main__":
    main()
