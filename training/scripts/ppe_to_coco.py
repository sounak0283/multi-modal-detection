#!/usr/bin/env python3
"""Convert ppe_merged's YOLO-format split (ppe_dataset_prep.py's output, already
cleaned by ppe_step1_refine.py's helmet_absent removal) to COCO JSON for YOLOX
training. Mirrors dfire_to_coco.py's shape exactly.

Class mapping: ppe_taxonomy.UNIFIED_CLASSES defines 8 YOLO class ids (0-7), but
glasses_present/glasses_absent (ids 6, 7) are EXCLUDED here - the 2026-09-16 data
review found 65-72% near-duplication in both glasses classes via DCT perceptual
hash and excluded them from this training run (see
training/scripts/_artifact/ppe_review.html#glasses_present). Any box labeled
glasses_present/glasses_absent is dropped during conversion; source images that
contain ONLY glasses boxes are also dropped (nothing left to annotate). Remaining
6 classes keep UNIFIED_CLASSES' order, remapped to COCO category ids 1-6.

shoes_present/shoes_absent have zero data in any source dataset (ppe_taxonomy.py's
UNCOVERED_ITEMS) and were never in UNIFIED_CLASSES to begin with - nothing to drop.

Usage:
    python ppe_to_coco.py --src A:/fsbd_training/ppe_merged --dst A:/fsbd_training/ppe_coco
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from ppe_taxonomy import UNIFIED_CLASSES

EXCLUDED_CLASSES = {"glasses_present", "glasses_absent"}
KEPT_CLASSES = [c for c in UNIFIED_CLASSES if c not in EXCLUDED_CLASSES]
# YOLO class id (0-7, ppe_merged's labels) -> COCO category id (1-based, kept classes only).
CLASS_ID_MAP = {
    UNIFIED_CLASSES.index(name): KEPT_CLASSES.index(name) + 1
    for name in KEPT_CLASSES
}
CATEGORIES = [{"id": i + 1, "name": name} for i, name in enumerate(KEPT_CLASSES)]


def pairs_for(img_dir: Path, lbl_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    return [(p, lbl_dir / (p.stem + ".txt")) for p in imgs]


def build_coco(image_label_pairs, out_json: Path, images_out: Path):
    images_out.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    kept, skipped_no_boxes, skipped_unreadable = 0, 0, 0

    for img_path, lbl_path in image_label_pairs:
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                if cls_id not in CLASS_ID_MAP:
                    continue  # glasses_* - excluded
                cx, cy, bw, bh = map(float, parts[1:])
                boxes.append((CLASS_ID_MAP[cls_id], cx, cy, bw, bh))
        if not boxes:
            skipped_no_boxes += 1
            continue

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            skipped_unreadable += 1
            continue

        dest = images_out / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)

        img_id = len(coco["images"]) + 1
        coco["images"].append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        for cat_id, cx, cy, bw, bh in boxes:
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            box_w, box_h = bw * w, bh * h
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cat_id,
                "bbox": [x, y, box_w, box_h], "area": box_w * box_h, "iscrowd": 0,
            })
            ann_id += 1
        kept += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    print(f"{out_json.name}: {kept} images kept, {skipped_no_boxes} dropped (glasses-only/no boxes), "
          f"{skipped_unreadable} unreadable, {len(coco['annotations'])} boxes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    print(f"kept classes ({len(KEPT_CLASSES)}): {KEPT_CLASSES}")
    print(f"excluded classes: {sorted(EXCLUDED_CLASSES)}\n")

    for split, out_name in (("train", "train2017"), ("val", "val2017"), ("test", "test2017")):
        img_dir = args.src / "images" / split
        lbl_dir = args.src / "labels" / split
        if not img_dir.is_dir():
            print(f"skipping split '{split}' - {img_dir} does not exist")
            continue
        pairs = pairs_for(img_dir, lbl_dir)
        build_coco(pairs, args.dst / "annotations" / f"{split}.json", args.dst / out_name)


if __name__ == "__main__":
    main()
