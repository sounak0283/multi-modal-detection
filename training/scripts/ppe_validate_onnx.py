#!/usr/bin/env python3
"""Validation harness for a PPE object-detector ONNX model: per-class precision/
recall on held-out data, reported SEPARATELY per domain (cctv_real, cctv_real_blurred,
posed_web, posed_web_proxy_augmented) rather than as one aggregate number - training
loss convergence is deliberately NOT what this script reports. A model can converge
perfectly on posed/web photos and still fail on CCTV; the only way to know is to look
at the CCTV-real slice on its own.

Reuses the real runtime decode path (backend/src/perimeter/detect/yolox_onnx.py's
YoloxOnnx) rather than reimplementing YOLOX's anchor-grid decode - the same reasoning
dfire_to_coco.py and fire_nano.py already applied: the eventual deployment code and
this harness must agree on what the model's raw output means, or a discrepancy here
would be worse than no harness at all.

Matching: standard greedy IoU matching per class per image (IoU >= --iou-thresh,
default 0.5) between predicted and ground-truth boxes; unmatched predictions are false
positives, unmatched ground truth are false negatives.

Usage:
    python ppe_validate_onnx.py --model /path/to/ppe.onnx \\
        --dataset /path/to/ppe_merged --split test \\
        --manifest /path/to/ppe_merged/domain_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ppe_taxonomy import UNIFIED_CLASSES

# backend's real runtime decode logic, not a reimplementation - see module docstring.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))
from perimeter.detect.yolox_onnx import YoloxOnnx  # noqa: E402


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
        out.append((cls, np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])))
    return out


def match_and_score(
    preds: list[tuple[int, float, np.ndarray]],
    gts: list[tuple[int, np.ndarray]],
    iou_thresh: float,
    counts: dict[str, dict[str, int]],
    domain: str,
) -> None:
    """Mutates `counts[class_name][domain + '_tp'/'_fp'/'_fn']` in place."""
    matched_gt = set()
    preds_sorted = sorted(preds, key=lambda p: -p[1])
    for cls, _score, pbox in preds_sorted:
        cls_name = UNIFIED_CLASSES[cls]
        best_iou, best_j = 0.0, -1
        for j, (gcls, gbox) in enumerate(gts):
            if gcls != cls or j in matched_gt:
                continue
            i = iou(pbox, gbox)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            counts[cls_name][f"{domain}_tp"] += 1
        else:
            counts[cls_name][f"{domain}_fp"] += 1
    for j, (gcls, _gbox) in enumerate(gts):
        if j not in matched_gt:
            counts[UNIFIED_CLASSES[gcls]][f"{domain}_fn"] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="output of ppe_dataset_prep.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", type=Path, required=True, help="domain_manifest.jsonl")
    ap.add_argument("--conf-thresh", type=float, default=0.25)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--to-rgb", action="store_true", help="pass through to YoloxOnnx if this model was exported with colour conversion")
    args = ap.parse_args()

    domain_by_image: dict[str, str] = {}
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        domains = {inst["domain"] for inst in entry["instances"]}
        # one dominant domain per image for reporting purposes; an image mixing
        # blurred and non-blurred SFCHD instances is rare and reported under
        # whichever domain appears first - acceptable for a diagnostic harness.
        domain_by_image[Path(entry["image"]).name] = next(iter(domains), "unknown")

    model = YoloxOnnx(args.model, conf_threshold=args.conf_thresh, nms_threshold=0.5, to_rgb=args.to_rgb)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    domains_seen: set[str] = set()

    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    n_images = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        domain = domain_by_image.get(img_path.name, "unknown")
        domains_seen.add(domain)

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        dets = model.detect(frame)
        preds = list(zip(dets.class_ids.tolist(), dets.scores.tolist(), dets.xyxy, strict=False))
        gts = load_ground_truth(lbl_dir / (img_path.stem + ".txt"), w, h)

        match_and_score(preds, gts, args.iou_thresh, counts, domain)
        n_images += 1

    print(f"Evaluated {n_images} images across domains: {sorted(domains_seen)}\n")
    header = f"{'class':<18}{'domain':<28}{'precision':>10}{'recall':>10}{'tp':>6}{'fp':>6}{'fn':>6}"
    print(header)
    print("-" * len(header))
    for cls in UNIFIED_CLASSES:
        for domain in sorted(domains_seen):
            tp = counts[cls].get(f"{domain}_tp", 0)
            fp = counts[cls].get(f"{domain}_fp", 0)
            fn = counts[cls].get(f"{domain}_fn", 0)
            if tp + fp + fn == 0:
                continue
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            print(f"{cls:<18}{domain:<28}{precision:>10.3f}{recall:>10.3f}{tp:>6}{fp:>6}{fn:>6}")


if __name__ == "__main__":
    main()
