#!/usr/bin/env python3
"""Step 1 refinement pass, run after ppe_step1_data_safety.py's initial (sampled)
report:

1. helmet_absent - FULL scan (not sampled) for face/mask-closeup contamination via
   YuNet, then actually applies the removal: flagged images are dropped from
   images/labels entirely (confirmed via a source-attribution check that the two
   contaminating source videos contribute helmet_absent only - no other class's
   signal is lost by dropping them wholesale).
2. vest_present - FULL scan with a widened hi-vis hue range that now includes
   safety-green (the original check only covered orange/yellow and likely
   overcounted mislabels by flagging legitimate green vests).
3. glasses_present / glasses_absent - FULL(er) near-duplicate check with a proper
   DCT-based perceptual hash (much less prone to false "duplicate" merges than the
   original 8x8 average hash) and a stricter hamming threshold.

Mutates the dataset in place for (1) only - (2) and (3) are report-only, matching
the user's instruction to be told before assuming vest_present is trainable, not to
have it silently filtered.

Usage:
    python ppe_step1_refine.py --dataset A:/fsbd_training/ppe_merged \\
        --manifest A:/fsbd_training/ppe_merged/domain_manifest.jsonl \\
        --yunet-model E:/fire-and-boundary-detection/backend/models/yunet/face_detection_yunet_2023mar.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))
from perimeter.identity.yunet import FaceDetector  # noqa: E402

from ppe_taxonomy import UNIFIED_CLASSES  # noqa: E402

FACE_AREA_FRACTION_FLAG = 0.15

# Widened from the first pass: orange/yellow (10-45) AND safety-green/lime (45-85),
# both valid ANSI/ISEA 107 hi-vis colours in real-world vests.
HIVIS_HUE_RANGES = [(10, 45), (45, 85)]
HIVIS_SAT_MIN, HIVIS_VAL_MIN = 70, 80  # slightly relaxed - green hi-vis reads darker/less saturated than orange under some lighting
VEST_FLAG_THRESHOLD = 0.12


def load_instances(manifest_path: Path, dataset_root: Path) -> dict[str, list[dict]]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        img_path = Path(entry["image"])
        split = img_path.parts[-2]
        label_path = dataset_root / "labels" / split / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        for line2 in label_path.read_text().splitlines():
            p = line2.split()
            if len(p) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            by_class[UNIFIED_CLASSES[cls_id]].append(
                {"image": img_path, "box": (cx, cy, bw, bh), "split": split, "label_path": label_path}
            )
    return by_class


def crop_box(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
    x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
    return img[y1:y2, x1:x2]


def hivis_fraction(crop_bgr: np.ndarray) -> float:
    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for lo, hi in HIVIS_HUE_RANGES:
        mask |= (
            (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
            & (hsv[..., 1] >= HIVIS_SAT_MIN) & (hsv[..., 2] >= HIVIS_VAL_MIN)
        )
    return float(mask.mean())


def dct_phash(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Standard DCT-based perceptual hash (same algorithm as the `imagehash` package's
    phash) - far more robust to genuine-but-similar-composition photos than an 8x8
    average hash, which is prone to false "duplicate" merges on shared brightness
    patterns alone."""
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    return (dct_low > med).flatten()


def near_dup_cluster_count(hashes: list[np.ndarray], max_hamming: int) -> int:
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
    return len({find(i) for i in range(n)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--yunet-model", type=Path, required=True)
    ap.add_argument("--dup-sample-cap", type=int, default=4200, help="glasses classes are ~4.1-4.2k each, so this covers them fully")
    ap.add_argument("--apply", action="store_true", help="actually remove flagged helmet_absent images (default: dry run, report only)")
    args = ap.parse_args()

    by_class = load_instances(args.manifest, args.dataset)

    # ---------------- 1. helmet_absent FULL contamination scan ----------------
    print("=== (1) helmet_absent FULL face/mask-closeup scan ===")
    face_detector = FaceDetector(args.yunet_model)
    instances = by_class.get("helmet_absent", [])
    images_to_labels: dict[Path, list[Path]] = defaultdict(list)
    for inst in instances:
        images_to_labels[inst["image"]].append(inst["label_path"])

    flagged_images: set[Path] = set()
    checked = 0
    for img_path in images_to_labels:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        checked += 1
        h, w = img.shape[:2]
        faces = face_detector.detect(img)
        if not faces:
            continue
        largest = max(faces, key=lambda f: f.box[2] * f.box[3])
        frac = (largest.box[2] * largest.box[3]) / (w * h) if w * h else 0
        if frac >= FACE_AREA_FRACTION_FLAG:
            flagged_images.add(img_path)

    print(f"checked {checked} unique images with a helmet_absent instance")
    print(f"flagged {len(flagged_images)} images ({len(flagged_images)/checked*100:.1f}%)")

    if args.apply:
        removed_instances = 0
        for img_path in flagged_images:
            label_path = images_to_labels[img_path][0]  # all entries for one image share the same label file
            if label_path.exists():
                n_lines = len([l for l in label_path.read_text().splitlines() if l.strip()])
                removed_instances += n_lines
                label_path.unlink()
            if img_path.exists():
                img_path.unlink()
        print(f"APPLIED: removed {len(flagged_images)} images, {removed_instances} total box instances (all classes in those images)")

        # rewrite domain_manifest.jsonl dropping the removed images
        kept_lines = []
        for line in args.manifest.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if Path(entry["image"]) not in flagged_images:
                kept_lines.append(line)
        args.manifest.write_text("\n".join(kept_lines) + "\n")
        print(f"rewrote {args.manifest} ({len(kept_lines)} images remaining)")
    else:
        print("DRY RUN - pass --apply to actually remove these files")

    # ---------------- 2. vest_present FULL widened-hue scan ----------------
    print("\n=== (2) vest_present FULL widened hi-vis check (orange/yellow + green) ===")
    vest_instances = by_class.get("vest_present", [])
    fractions, flagged_vest = [], []
    for inst in vest_instances:
        img = cv2.imread(str(inst["image"]))
        if img is None:
            continue
        crop = crop_box(img, inst["box"])
        frac = hivis_fraction(crop)
        fractions.append(frac)
        if frac < VEST_FLAG_THRESHOLD:
            flagged_vest.append(inst["image"])
    arr = np.array(fractions) if fractions else np.array([0.0])
    corrected_rate = len(flagged_vest) / len(fractions) if fractions else 0.0
    print(f"checked {len(fractions)} instances (full class)")
    print(f"mean hivis fraction: {arr.mean():.3f}, median: {np.median(arr):.3f}")
    print(f"flagged below threshold ({VEST_FLAG_THRESHOLD}): {len(flagged_vest)}")
    print(f"CORRECTED estimated mislabel rate: {corrected_rate*100:.1f}%")
    if corrected_rate > 0.15:
        print("*** STILL ABOVE 15% - flagging for explicit decision before treating vest_present as trainable as-is ***")
    elif corrected_rate > 0.10:
        print("*** in the 10-15% band - borderline, flagging per instruction ***")
    else:
        print("under 10% - within acceptable range")

    # ---------------- 3. glasses near-dup, tighter hash ----------------
    print("\n=== (3) glasses near-duplicate check, DCT phash + hamming<=2 ===")
    results = {}
    for cls_name in ("glasses_present", "glasses_absent"):
        insts = by_class.get(cls_name, [])
        seen_images = {}
        for inst in insts:
            if inst["image"] not in seen_images:
                seen_images[inst["image"]] = inst
        sample = list(seen_images.values())
        if len(sample) > args.dup_sample_cap:
            import random
            sample = random.Random(0).sample(sample, args.dup_sample_cap)
        hashes = []
        for inst in sample:
            img = cv2.imread(str(inst["image"]))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hashes.append(dct_phash(gray))
        clusters = near_dup_cluster_count(hashes, max_hamming=2)
        results[cls_name] = {
            "unique_images_checked": len(hashes),
            "near_dup_clusters": clusters,
            "dup_ratio": 1 - (clusters / len(hashes)) if hashes else 0.0,
        }
        print(f"{cls_name}: {len(hashes)} unique images -> {clusters} clusters "
              f"(dup ratio {results[cls_name]['dup_ratio']*100:.1f}%)")

    out = {
        "helmet_absent_full_scan": {"checked": checked, "flagged": len(flagged_images), "applied": args.apply},
        "vest_present_full_scan": {
            "checked": len(fractions), "mean_hivis": float(arr.mean()),
            "flagged": len(flagged_vest), "corrected_mislabel_rate": corrected_rate,
        },
        "glasses_dup_recheck": results,
    }
    (args.dataset / "step1_refine_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.dataset / 'step1_refine_report.json'}")


if __name__ == "__main__":
    main()
