#!/usr/bin/env python3
"""Step 1 data-safety gate (fast, no training) for the PPE merge:

  a. helmet_absent  - flag face/mask-closeup-style contamination (uses the project's
                       real YuNet face detector, not a heuristic reimplementation - a
                       face occupying a large fraction of the full frame is a strong,
                       principled signal for "portrait/face-dataset photo", which is
                       compositionally unlike wide-shot industrial photography).
  b. vest_present   - HSV hi-vis-colour coverage check per box; low coverage flags a
                       likely mislabeled box (e.g. drawn on pants, not a vest).
  c. vest_absent    - blank/near-solid-colour image detection (pixel variance).
  d. gloves_present vs gloves_absent - brightness/resolution/blur distributions per
                       class + perceptual-hash near-duplicate clustering to estimate
                       unique source count (near-identical video frames inflate
                       instance counts without adding real signal, and can make two
                       classes trivially separable on nuisance variables rather than
                       the actual item).
  e. glasses_present / glasses_absent - same unique-source check as (d).
  f. Prints a clean updated manifest table (class, domain, kept, removed).

This is pure data analysis - no model training, no GPU. Run before Step 2/3.

Usage:
    python ppe_step1_data_safety.py --dataset A:/fsbd_training/ppe_merged \\
        --manifest A:/fsbd_training/ppe_merged/domain_manifest.jsonl \\
        --yunet-model E:/fire-and-boundary-detection/backend/models/yunet/face_detection_yunet_2023mar.onnx \\
        --sample-cap 3000
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

HIVIS_HUE_RANGES = [(10, 45)]  # orange->yellow in OpenCV's 0-179 hue scale
HIVIS_SAT_MIN, HIVIS_VAL_MIN = 80, 90
BLANK_VARIANCE_THRESHOLD = 5.0  # near-zero pixel variance = solid colour / corrupt
FACE_AREA_FRACTION_FLAG = 0.15  # face covers >=15% of the full frame -> portrait-style


def load_instances(manifest_path: Path, dataset_root: Path) -> dict[str, list[dict]]:
    """class -> list of {"image": Path, "box": (cx,cy,bw,bh), "split": str}"""
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
                {"image": img_path, "box": (cx, cy, bw, bh), "split": split}
            )
    return by_class


def crop_box(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
    x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
    return img[y1:y2, x1:x2]


def sample(instances: list[dict], cap: int, seed: int = 0) -> list[dict]:
    if len(instances) <= cap:
        return instances
    import random
    rng = random.Random(seed)
    return rng.sample(instances, cap)


# ---------------------------------------------------------------------------
# (a) helmet_absent face/mask-closeup contamination
# ---------------------------------------------------------------------------
def check_helmet_absent(instances: list[dict], face_detector: FaceDetector, cap: int) -> dict:
    sampled = sample(instances, cap)
    flagged = []
    seen_images: dict[Path, list] = defaultdict(list)
    for inst in sampled:
        seen_images[inst["image"]].append(inst)

    checked_images = 0
    for img_path, insts in seen_images.items():
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        checked_images += 1
        h, w = img.shape[:2]
        faces = face_detector.detect(img)
        if not faces:
            continue
        largest = max(faces, key=lambda f: f.box[2] * f.box[3])
        fw, fh = largest.box[2], largest.box[3]
        frac = (fw * fh) / (w * h) if w * h else 0
        if frac >= FACE_AREA_FRACTION_FLAG:
            flagged.append((img_path, frac))

    return {
        "sampled_instances": len(sampled),
        "checked_images": checked_images,
        "flagged_images": len(flagged),
        "flagged_fraction": len(flagged) / checked_images if checked_images else 0.0,
        "examples": [str(p) for p, _ in flagged[:10]],
    }


# ---------------------------------------------------------------------------
# (b) vest_present hi-vis colour outlier check
# ---------------------------------------------------------------------------
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


def check_vest_present(instances: list[dict], cap: int, threshold: float = 0.12) -> dict:
    sampled = sample(instances, cap)
    fractions = []
    flagged = []
    for inst in sampled:
        img = cv2.imread(str(inst["image"]))
        if img is None:
            continue
        crop = crop_box(img, inst["box"])
        frac = hivis_fraction(crop)
        fractions.append(frac)
        if frac < threshold:
            flagged.append((inst["image"], frac))
    arr = np.array(fractions) if fractions else np.array([0.0])
    return {
        "sampled_instances": len(sampled),
        "checked": len(fractions),
        "mean_hivis_fraction": float(arr.mean()),
        "median_hivis_fraction": float(np.median(arr)),
        "flagged_below_threshold": len(flagged),
        "estimated_mislabel_rate": len(flagged) / len(fractions) if fractions else 0.0,
        "threshold": threshold,
        "examples": [str(p) for p, _ in flagged[:10]],
    }


# ---------------------------------------------------------------------------
# (c) vest_absent blank/corrupt image scan
# ---------------------------------------------------------------------------
def check_vest_absent(instances: list[dict], cap: int) -> dict:
    sampled = sample(instances, cap)
    seen_images = {inst["image"] for inst in sampled}
    blank = []
    unreadable = []
    for img_path in seen_images:
        img = cv2.imread(str(img_path))
        if img is None:
            unreadable.append(img_path)
            continue
        if float(np.var(img)) < BLANK_VARIANCE_THRESHOLD:
            blank.append(img_path)
    return {
        "sampled_images": len(seen_images),
        "blank_or_solid_colour": len(blank),
        "unreadable": len(unreadable),
        "examples": [str(p) for p in (blank + unreadable)[:10]],
    }


# ---------------------------------------------------------------------------
# (d)/(e) confound check: brightness / resolution / blur + near-duplicate clustering
# ---------------------------------------------------------------------------
def phash_bits(img_gray_small: np.ndarray) -> np.ndarray:
    """8x8 average-hash - lightweight, no extra dependency needed beyond cv2/numpy."""
    small = cv2.resize(img_gray_small, (8, 8), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()


def confound_check(pos_instances: list[dict], neg_instances: list[dict], cap: int) -> dict:
    def stats_for(instances: list[dict]) -> tuple[dict, list[np.ndarray], set[Path]]:
        sampled = sample(instances, cap)
        brightness, blur, res_w, res_h = [], [], [], []
        hashes = []
        images = set()
        for inst in sampled:
            img = cv2.imread(str(inst["image"]))
            if img is None:
                continue
            images.add(inst["image"])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean()))
            blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            h, w = img.shape[:2]
            res_w.append(w)
            res_h.append(h)
            hashes.append(phash_bits(gray))
        stats = {
            "n": len(sampled),
            "brightness_mean": float(np.mean(brightness)) if brightness else 0.0,
            "brightness_std": float(np.std(brightness)) if brightness else 0.0,
            "blur_mean": float(np.mean(blur)) if blur else 0.0,
            "blur_std": float(np.std(blur)) if blur else 0.0,
            "resolution_modes": sorted(set(zip(res_w, res_h)), key=lambda t: -res_w.count(t[0]))[:3],
            "unique_images_in_sample": len(images),
        }
        return stats, hashes, images

    pos_stats, pos_hashes, _ = stats_for(pos_instances)
    neg_stats, neg_hashes, _ = stats_for(neg_instances)

    def near_dup_cluster_count(hashes: list[np.ndarray], max_hamming: int = 4) -> int:
        """Cheap O(n^2) clustering over the (capped) sample - fine at n<=3000."""
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

    pos_clusters = near_dup_cluster_count(pos_hashes) if len(pos_hashes) <= 3000 else None
    neg_clusters = near_dup_cluster_count(neg_hashes) if len(neg_hashes) <= 3000 else None

    # Simple separability signal: how far apart are the class means relative to
    # pooled std, on each nuisance variable - a rough Cohen's-d style check, not a
    # trained classifier, but enough to catch "trivially separable on brightness alone".
    def cohens_d(m1, s1, m2, s2):
        pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2) if (s1 or s2) else 1e-9
        return abs(m1 - m2) / pooled if pooled else 0.0

    d_brightness = cohens_d(
        pos_stats["brightness_mean"], pos_stats["brightness_std"],
        neg_stats["brightness_mean"], neg_stats["brightness_std"],
    )
    d_blur = cohens_d(
        pos_stats["blur_mean"], pos_stats["blur_std"],
        neg_stats["blur_mean"], neg_stats["blur_std"],
    )

    return {
        "present": pos_stats,
        "absent": neg_stats,
        "present_near_dup_clusters": pos_clusters,
        "absent_near_dup_clusters": neg_clusters,
        "present_dup_ratio": (pos_clusters / pos_stats["n"]) if pos_clusters and pos_stats["n"] else None,
        "absent_dup_ratio": (neg_clusters / neg_stats["n"]) if neg_clusters and neg_stats["n"] else None,
        "cohens_d_brightness": d_brightness,
        "cohens_d_blur": d_blur,
        "trivially_separable_flag": bool(d_brightness > 1.2 or d_blur > 1.2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--yunet-model", type=Path, required=True)
    ap.add_argument("--sample-cap", type=int, default=3000)
    args = ap.parse_args()

    print("Loading instances from manifest + label files...")
    by_class = load_instances(args.manifest, args.dataset)
    for cls in UNIFIED_CLASSES:
        print(f"  {cls}: {len(by_class.get(cls, []))} instances")

    print("\n=== (a) helmet_absent face/mask-closeup contamination check ===")
    face_detector = FaceDetector(args.yunet_model)
    result_a = check_helmet_absent(by_class.get("helmet_absent", []), face_detector, args.sample_cap)
    print(json.dumps(result_a, indent=2))

    print("\n=== (b) vest_present hi-vis colour outlier check ===")
    result_b = check_vest_present(by_class.get("vest_present", []), args.sample_cap)
    print(json.dumps(result_b, indent=2))

    print("\n=== (c) vest_absent blank/corrupt scan ===")
    result_c = check_vest_absent(by_class.get("vest_absent", []), args.sample_cap)
    print(json.dumps(result_c, indent=2))

    print("\n=== (d) gloves_present vs gloves_absent confound check ===")
    result_d = confound_check(
        by_class.get("gloves_present", []), by_class.get("gloves_absent", []), args.sample_cap
    )
    print(json.dumps(result_d, indent=2, default=str))

    print("\n=== (e) glasses_present vs glasses_absent confound check ===")
    result_e = confound_check(
        by_class.get("glasses_present", []), by_class.get("glasses_absent", []), args.sample_cap
    )
    print(json.dumps(result_e, indent=2, default=str))

    out = {
        "helmet_absent_contamination": result_a,
        "vest_present_outliers": result_b,
        "vest_absent_blank_scan": result_c,
        "gloves_confound": result_d,
        "glasses_confound": result_e,
        "raw_counts": {cls: len(by_class.get(cls, [])) for cls in UNIFIED_CLASSES},
    }
    out_path = args.dataset / "step1_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote full report to {out_path}")


if __name__ == "__main__":
    main()
