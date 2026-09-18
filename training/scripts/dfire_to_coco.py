#!/usr/bin/env python3
"""Convert the D-Fire YOLO-format dataset to COCO JSON for YOLOX training.

Per PLAN.md Section 5.3: 2 classes {0: fire, 1: smoke}. Handles the common D-Fire
release layout (train/images, train/labels, test/images, test/labels — this Kaggle
mirror ships no separate val split, so val is carved out of train) as well as a
flat images/ + labels/ layout, auto-detected.

Usage:
    python dfire_to_coco.py --src /a/fsbd_training/dfire_extracted --dst /a/fsbd_training/dfire_coco
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

CATEGORIES = [{"id": 1, "name": "fire"}, {"id": 2, "name": "smoke"}]
# IMPORTANT: this Kaggle mirror's data.yaml declares `names: ['smoke', 'fire']` -
# i.e. YOLO class 0 = smoke, class 1 = fire in the source labels. That is the REVERSE
# of PLAN.md section 5.3's project convention (0: fire, 1: smoke), which
# src/perimeter/detect/firesmoke.py's FIRE_CLASS_ID=0/SMOKE_CLASS_ID=1 also assumes.
# Map source class 0 (smoke) -> COCO category 2 (smoke), source class 1 (fire) ->
# COCO category 1 (fire) so the trained model's output order matches the runtime.
# Verify against the actual data.yaml before reusing this script on a different mirror.
CLASS_ID_MAP = {0: 2, 1: 1}


def find_split_dirs(src: Path):
    """Return {"train": (images_dir, labels_dir), "test": (...)} for whatever exists."""
    splits = {}
    for name in ("train", "test", "val", "valid"):
        img_dir = src / name / "images"
        lbl_dir = src / name / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            splits[name] = (img_dir, lbl_dir)
    if not splits:
        # flat layout: src/images, src/labels
        img_dir, lbl_dir = src / "images", src / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            splits["all"] = (img_dir, lbl_dir)
    if not splits:
        raise SystemExit(f"Could not find an images/+labels/ layout under {src}")
    return splits


def build_coco(image_label_pairs, out_json: Path, images_out: Path):
    images_out.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    kept, skipped = 0, 0

    for img_id, (img_path, lbl_path) in enumerate(image_label_pairs, start=1):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            skipped += 1
            continue

        dest = images_out / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)

        coco["images"].append({
            "id": img_id, "file_name": img_path.name, "width": w, "height": h,
        })

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = parts
                cls = CLASS_ID_MAP.get(int(cls))
                if cls is None:
                    continue
                cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                box_w = bw * w
                box_h = bh * h
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cls,
                    "bbox": [x, y, box_w, box_h], "area": box_w * box_h,
                    "iscrowd": 0,
                })
                ann_id += 1
        kept += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    print(f"{out_json.name}: {kept} images kept, {skipped} skipped, {len(coco['annotations'])} boxes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.1,
                     help="fraction of train carved into val if no val split exists")
    args = ap.parse_args()

    splits = find_split_dirs(args.src)
    print("found splits:", {k: str(v[0]) for k, v in splits.items()})

    def pairs_for(img_dir: Path, lbl_dir: Path):
        exts = {".jpg", ".jpeg", ".png"}
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
        return [(p, lbl_dir / (p.stem + ".txt")) for p in imgs]

    train_pairs, val_pairs, test_pairs = [], [], []

    if "train" in splits:
        train_pairs = pairs_for(*splits["train"])
    if "val" in splits:
        val_pairs = pairs_for(*splits["val"])
    elif "valid" in splits:
        val_pairs = pairs_for(*splits["valid"])
    if "test" in splits:
        test_pairs = pairs_for(*splits["test"])
    if "all" in splits:
        train_pairs = pairs_for(*splits["all"])

    if not val_pairs and train_pairs:
        n_val = max(1, int(len(train_pairs) * args.val_fraction))
        val_pairs = train_pairs[:n_val]
        train_pairs = train_pairs[n_val:]

    print(f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    build_coco(train_pairs, args.dst / "annotations" / "train.json", args.dst / "train2017")
    build_coco(val_pairs, args.dst / "annotations" / "val.json", args.dst / "val2017")
    if test_pairs:
        build_coco(test_pairs, args.dst / "annotations" / "test.json", args.dst / "test2017")


if __name__ == "__main__":
    main()
