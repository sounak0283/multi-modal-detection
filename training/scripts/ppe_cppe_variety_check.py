#!/usr/bin/env python3
"""Within-dataset scene/shoot variety check for Ultralytics Construction-PPE's
no_gloves and no_goggle classes (the candidate sole origin for gloves_absent/
glasses_absent in the 2026-09-17 gloves/glasses re-sourcing), run because the
"goggles" (present) class turned out to be concentrated in a handful of
same-video frame sequences (same person, same rooftop, consecutive frames) -
the same single-shoot-concentration pattern PP02 had for gloves, just smaller
scale. Requested check: does no_gloves/no_goggle show the same concentration,
or genuine variety across people/scenes/lighting?

Method: DCT-based perceptual hash (ppe_step1_refine.py's dct_phash, hamming<=2
clustering) - a cluster of near-identical hashes is consecutive video frames or
near-duplicate shots of the same person/scene; a low cluster-to-image ratio means
a few shoots dominate, near 1:1 means genuine per-image variety. Also reports raw
image dimensions (constant across most images since Construction-PPE is 640x640
letterboxed, so dimension isn't a useful diversity signal here - noted for
completeness only) and brightness/blur spread as a coarse "different lighting/
scene" proxy, same stats already used in ppe_step1_data_safety.py's confound_check.

Also runs the identical check on gloves/goggles (present) for direct comparison -
this is what lets us say concretely "no_gloves is/isn't like the goggles problem"
rather than asserting it.

Usage:
    python ppe_cppe_variety_check.py --dataset A:/fsbd_training/curated_gloves_glasses/construction_ppe/extracted
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASS_IDS = {"gloves": 1, "no_gloves": 9, "goggles": 4, "no_goggle": 8}


def dct_phash(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    return (dct_low > med).flatten()


def near_dup_cluster_count(hashes: list[np.ndarray], max_hamming: int) -> tuple[int, list[list[int]]]:
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.count_nonzero(hashes[i] != hashes[j]) <= max_hamming:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return len(clusters), sorted(clusters.values(), key=len, reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()

    by_class: dict[str, list[Path]] = defaultdict(list)
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
            ids = set()
            for line in lbl_path.read_text().splitlines():
                p = line.split()
                if p:
                    ids.add(int(p[0]))
            for cname, cid in CLASS_IDS.items():
                if cid in ids:
                    by_class[cname].append(img_path)

    results = {}
    for cname in ("gloves", "no_gloves", "goggles", "no_goggle"):
        paths = sorted(set(by_class.get(cname, [])))
        n = len(paths)
        hashes, brightness, blur = [], [], []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hashes.append(dct_phash(gray))
            brightness.append(float(gray.mean()))
            blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        n_clusters, clusters = near_dup_cluster_count(hashes, max_hamming=6)
        top5_sizes = [len(c) for c in clusters[:5]]
        results[cname] = {
            "n_images": n,
            "n_clusters_hamming6": n_clusters,
            "cluster_to_image_ratio": n_clusters / n if n else 0,
            "top5_cluster_sizes": top5_sizes,
            "brightness_mean": float(np.mean(brightness)) if brightness else 0,
            "brightness_std": float(np.std(brightness)) if brightness else 0,
            "blur_mean": float(np.mean(blur)) if blur else 0,
            "blur_std": float(np.std(blur)) if blur else 0,
        }
        print(f"{cname:12s}: {n:4d} images -> {n_clusters:4d} distinct-scene clusters "
              f"(ratio {results[cname]['cluster_to_image_ratio']*100:.1f}%), "
              f"top-5 cluster sizes {top5_sizes}, "
              f"brightness std={results[cname]['brightness_std']:.1f}, blur std={results[cname]['blur_std']:.1f}")

    out_path = args.dataset.parent / "cppe_variety_check.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
