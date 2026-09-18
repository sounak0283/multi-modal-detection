#!/usr/bin/env python3
"""Curate a small, multi-source, domain-filtered replacement for gloves_present/
absent and glasses_present/absent (2026-09-17 decision). Two rejected sources
(agts/handglove-detection - disposable/nitrile office gloves; database-sjrvw/
safety-goggles - generic Google-Images eyeglasses stock photos) are NOT included
here at all - domain relevance was checked before this script was written, not by
this script.

Shot-diversity-biased sampling: every present-class source showed real
same-shoot near-duplicate concentration when checked with DCT-hash clustering
(ppe_cppe_variety_check.py) - Construction-PPE's gloves/goggles ~75% cluster
ratio (bursts up to 20 frames), construction-images/gloves-7zhos far worse at
12.6% (bursts up to 35 frames). A first attempt sampled 1 image per DCT-hash
cluster (hamming<=6) and reported 100% post-sampling cluster ratio - but the
resulting montage showed a single rooftop scene still dominating ~20/60 images,
because hamming<=6 splits one continuous shoot into many small sub-clusters
(pose/zoom drift exceeds the threshold within seconds) without ever fully
merging them, so "1 per cluster" still pulled many frames from the same shoot.
Neither loosening the hash threshold (tested up to hamming<=26: collapses to 1
giant cluster) nor global HSV-histogram correlation (tested 0.80-0.90: collapses
600+/705 into one cluster, since unrelated outdoor construction photos share
similar sky/concrete/hi-vis-orange palettes) could isolate shots without
over-merging unrelated photos.

Fixed with ppe_shot_segment.py: since these are literal sequential video-frame
exports (confirmed via numeric filename ordering), only comparing NEIGHBOURS in
filename-numeric order and breaking the "shot" wherever consecutive-frame
similarity drops (corr < 0.85) segments the data correctly without the
all-pairs over-merging failure mode. Sampling AT MOST 1 image per shot.

Absent-class sources (Construction-PPE's no_gloves/no_goggle) were confirmed
STRUCTURALLY diverse already (100% cluster ratio, every image its own cluster) -
still run through the same one-per-cluster logic for consistency, but it will not
change their composition since every image is already a singleton cluster.

Sources included (see NOTICE.md for licence rows):
  - Ultralytics Construction-PPE (AGPL-3.0): gloves=1, no_gloves=9, goggles=4, no_goggle=8
  - construction-images/gloves-7zhos (CC BY 4.0): glove=0 (present only)
  - siabar/ppe-plsuk (CC BY 4.0): Glass=2 (present only, n=6, no absent class)

Usage:
    python ppe_curate_gloves_glasses.py --out A:/fsbd_training/curated_gloves_glasses/final \\
        --per-source-cap 30
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ppe_shot_segment import segment_shots, hist_feature

ROOT = Path("A:/fsbd_training/curated_gloves_glasses")

# One recurring rooftop shoot (orange helmet, red vest, black pants, brown boots,
# white tile floor) turned out to account for 203/705 (29%) of Construction-PPE's
# gloves_present and 191/480 (40%) of goggles_present - confirmed by correlation
# to these two reference frames, both still visibly dominating even after
# shot-segmentation (video panning meant the shoot fragmented into many small
# "shots" under corr_thresh=0.85, none individually flagged as a burst, but
# collectively still ~1/3-1/2 of the source pool). Capped explicitly below
# rather than left to general-purpose clustering, which this dataset defeated
# twice (DCT-hash: hamming<=26 collapses to 1 cluster; global HSV histogram:
# 0.80 threshold collapses 601/705 into 1 cluster; shot-segmentation: correctly
# avoids over-merging but doesn't catch a panning video as one shoot).
ROOFTOP_REF_STEMS = ("image1104", "image855")
ROOFTOP_CORR_THRESH = 0.75
ROOFTOP_MAX_KEEP = 2
CPPE = ROOT / "construction_ppe" / "extracted"

SOURCES = [
    {
        "name": "ultralytics_construction_ppe",
        "license": "AGPL-3.0",
        "img_label_pairs": [
            (CPPE / "images" / s, CPPE / "labels" / s) for s in ("train", "val", "test")
        ],
        "class_map": {"gloves_present": 1, "gloves_absent": 9, "glasses_present": 4, "glasses_absent": 8},
    },
    {
        "name": "construction_images_gloves",
        "license": "CC BY 4.0",
        "img_label_pairs": [
            (ROOT / "construction_images_gloves" / "extracted" / s / "images",
             ROOT / "construction_images_gloves" / "extracted" / s / "labels")
            for s in ("train", "valid", "test")
        ],
        "class_map": {"gloves_present": 0},
    },
    {
        "name": "siabar_ppe_plsuk",
        "license": "CC BY 4.0",
        "img_label_pairs": [
            (ROOT / "siabar_ppe" / "extracted" / s / "images", ROOT / "siabar_ppe" / "extracted" / s / "labels")
            for s in ("train", "valid", "test")
        ],
        "class_map": {"glasses_present": 2},
    },
]

TARGET_CLASSES = ["gloves_present", "gloves_absent", "glasses_present", "glasses_absent"]


def filter_known_bursts(paths: list[Path], all_class_paths: list[Path], rng: random.Random) -> list[Path]:
    """Cap the recurring rooftop shoot (see ROOFTOP_* constants) to ROOFTOP_MAX_KEEP
    images, using correlation to two reference frames rather than general clustering
    (which this dataset defeated twice - see module docstring). Only meaningful for
    Construction-PPE; a no-op (returns paths unchanged) if the reference stems aren't
    present in this pool at all."""
    refs = []
    for stem in ROOFTOP_REF_STEMS:
        match = next((p for p in all_class_paths if p.stem == stem), None)
        if match is not None:
            img = cv2.imread(str(match))
            if img is not None:
                refs.append(hist_feature(img))
    if not refs:
        return paths

    flagged, clean = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        feat = hist_feature(img)
        if max(cv2.compareHist(r, feat, cv2.HISTCMP_CORREL) for r in refs) > ROOFTOP_CORR_THRESH:
            flagged.append(p)
        else:
            clean.append(p)
    kept_flagged = rng.sample(flagged, min(ROOFTOP_MAX_KEEP, len(flagged)))
    if flagged:
        print(f"    (rooftop-shoot filter: {len(flagged)} flagged, keeping {len(kept_flagged)})")
    return clean + kept_flagged


def collect_source_instances(source: dict) -> dict[str, list[Path]]:
    out = defaultdict(list)
    for img_dir, lbl_dir in source["img_label_pairs"]:
        if not img_dir.is_dir():
            continue
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            present_ids = set()
            for line in lbl_path.read_text().splitlines():
                p = line.split()
                if p:
                    present_ids.add(int(p[0]))
            for cname, cid in source["class_map"].items():
                if cid in present_ids:
                    out[cname].append(img_path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-source-cap", type=int, default=30,
                     help="max images sampled per source per class, at most 1 per near-dup cluster")
    args = ap.parse_args()

    rng = random.Random(0)
    manifest = []
    per_class_per_source_count = defaultdict(lambda: defaultdict(int))

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for cname in TARGET_CLASSES:
        (args.out / cname).mkdir(exist_ok=True)

    for source in SOURCES:
        by_class = collect_source_instances(source)
        for cname, paths in by_class.items():
            paths = sorted(set(paths))
            if source["name"] == "ultralytics_construction_ppe" and cname.endswith("_present"):
                paths = sorted(filter_known_bursts(paths, paths, rng))
            shots = segment_shots(paths, corr_thresh=0.85)
            n_clusters = len(shots)

            # one representative per shot (random within shot), shuffled shot order,
            # capped at per-source-cap - this is what prevents a 20/28/35-frame burst
            # from contributing more than 1 image no matter how large the burst is.
            reps = [rng.choice(shot) for shot in shots]
            rng.shuffle(reps)
            sample = reps[: args.per_source_cap]

            for p in sample:
                dest_name = f"{source['name']}__{p.name}"
                dest = args.out / cname / dest_name
                shutil.copy2(p, dest)
                manifest.append({"class": cname, "source": source["name"], "license": source["license"],
                                  "orig_path": str(p), "dest": str(dest)})
            per_class_per_source_count[cname][source["name"]] = len(sample)
            print(f"{cname:16s} <- {source['name']:32s}: {len(paths):5d} instances -> "
                  f"{n_clusters:5d} shots -> sampled {len(sample):3d} (1/shot, capped)")

    (args.out / "curation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} curated images to {args.out}")

    print("\n=== Per-class totals ===")
    for cname in TARGET_CLASSES:
        total = sum(per_class_per_source_count[cname].values())
        print(f"  {cname}: {total} images from {len(per_class_per_source_count[cname])} source(s)")


if __name__ == "__main__":
    main()
