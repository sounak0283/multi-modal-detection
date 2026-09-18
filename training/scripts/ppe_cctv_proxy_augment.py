#!/usr/bin/env python3
"""Synthetic-CCTV proxy augmentation for the posed/web PPE images (shlokraval), so
the gloves/glasses classes - which have zero real-CCTV coverage - see *something*
resembling a ceiling-mounted wide-area camera during training, rather than only
ever seeing eye-level, high-resolution, well-lit photos.

This is a togglable PREPROCESSING PASS, not a permanent transform: it reads a
prepared YOLO split (as produced by ppe_dataset_prep.py) and writes a SEPARATE
output directory with degraded copies of the posed_web images plus boxes carried
through unchanged (the four operations below are all box-preserving except the
perspective warp, which does move box coordinates - handled explicitly).
cctv_real images are left untouched by design (they are already real CCTV footage;
degrading them further would be actively wrong) - the script skips any image whose
domain_manifest.jsonl entry isn't tagged posed_web/posed_web_proxy_augmented.

Four operations approximate a ceiling/high-wall mount looking down at a wide area:

1. **Downscale + upscale** (simulates low resolution relative to scene size - a
   person at typical CCTV range is a small fraction of the frame, so small PPE
   items like gloves/goggles are effectively low-res even on a 1080p/4K camera).
2. **Gaussian blur** (motion blur + lens softness at range).
3. **JPEG re-compression at a low quality factor** (most CCTV/NVR pipelines
   compress hard to save storage - this is a real, measurable degradation source,
   not a cosmetic one).
4. **Steep top-down perspective warp** (a ceiling mount does not see PPE items
   face-on the way a posed photo does - a helmet viewed from above looks very
   different from a helmet viewed at eye level, and this is the one operation
   that meaningfully moves box geometry, not just pixel quality).

Compare with/without by pointing training at either the original prepared split or
this script's output - nothing here is baked into ppe_dataset_prep.py itself.

Usage:
    python ppe_cctv_proxy_augment.py --src /path/to/ppe_merged --dst /path/to/ppe_merged_cctv_proxy
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

DOWNSCALE_FACTOR_RANGE = (0.25, 0.45)  # simulate a person occupying 25-45% of native res
BLUR_KERNEL_RANGE = (3, 7)  # odd kernel sizes only
JPEG_QUALITY_RANGE = (25, 45)
PERSPECTIVE_STRENGTH_RANGE = (0.08, 0.18)  # fraction of image size the top corners pull inward


def degrade_image(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]

    factor = rng.uniform(*DOWNSCALE_FACTOR_RANGE)
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    k = rng.choice(range(BLUR_KERNEL_RANGE[0], BLUR_KERNEL_RANGE[1] + 1, 2))
    img = cv2.GaussianBlur(img, (k, k), 0)

    quality = rng.randint(*JPEG_QUALITY_RANGE)
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


def perspective_warp(img: np.ndarray, boxes_xyxy: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Steep top-down warp: pull the top two corners inward to simulate looking
    down from a high mount. boxes_xyxy is (N, 4) in pixel coords; returns the
    warped image and the corresponding warped boxes (axis-aligned bounding box of
    the warped corners - an approximation, since a true perspective warp turns a
    rectangle into a quadrilateral, but YOLO/COCO boxes must stay axis-aligned)."""
    h, w = img.shape[:2]
    strength = rng.uniform(*PERSPECTIVE_STRENGTH_RANGE)
    dx = w * strength

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, 0], [w - dx, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderValue=(114, 114, 114))

    if boxes_xyxy.size == 0:
        return warped, boxes_xyxy

    warped_boxes = np.zeros_like(boxes_xyxy)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
        warped_boxes[i] = [
            warped_corners[:, 0].min(), warped_corners[:, 1].min(),
            warped_corners[:, 0].max(), warped_corners[:, 1].max(),
        ]
    return warped, warped_boxes


def yolo_to_xyxy(line: str, w: int, h: int) -> tuple[int, float, float, float, float]:
    cls, cx, cy, bw, bh = line.split()
    cls, cx, cy, bw, bh = int(cls), float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return cls, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def xyxy_to_yolo(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="output of ppe_dataset_prep.py")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--no-perspective", action="store_true", help="skip the box-moving warp step, keep only pixel-quality degradation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    domain_by_image = {}
    manifest_path = args.src / "domain_manifest.jsonl"
    for line in manifest_path.read_text().splitlines():
        entry = json.loads(line)
        domain_by_image[Path(entry["image"]).name] = entry

    n_converted, n_skipped = 0, 0
    for split_dir in (args.src / "images").iterdir():
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        out_img_dir = args.dst / "images" / split
        out_lbl_dir = args.dst / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(split_dir.iterdir()):
            entry = domain_by_image.get(img_path.name)
            is_posed = entry is not None and entry.get("source") != "sfchd"
            if not is_posed:
                n_skipped += 1
                continue  # never degrade real CCTV footage

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            label_path = args.src / "labels" / split / (img_path.stem + ".txt")
            lines = label_path.read_text().splitlines() if label_path.exists() else []
            parsed = [yolo_to_xyxy(l, w, h) for l in lines if l.strip()]
            classes = np.array([p[0] for p in parsed], dtype=int)
            boxes = np.array([p[1:] for p in parsed], dtype=np.float32) if parsed else np.zeros((0, 4), np.float32)

            img = degrade_image(img, rng)
            if not args.no_perspective:
                img, boxes = perspective_warp(img, boxes, rng)

            cv2.imwrite(str(out_img_dir / img_path.name), img)
            out_lines = [
                xyxy_to_yolo(int(c), *b, w, h) for c, b in zip(classes, boxes, strict=False)
            ]
            (out_lbl_dir / (img_path.stem + ".txt")).write_text("\n".join(out_lines))
            n_converted += 1

    print(f"proxy-augmented {n_converted} posed_web images, skipped {n_skipped} non-posed images (left untouched, not copied)")


if __name__ == "__main__":
    main()
