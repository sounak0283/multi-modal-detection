#!/usr/bin/env python3
"""Merge SFCHD (CCTV-real) + shlokraval/ppe-dataset-yolov8 (posed/web, Apache-2.0/
CC BY 4.0) into one YOLO-format dataset under `ppe_taxonomy.UNIFIED_CLASSES`.

Does NOT download anything - takes already-downloaded, already-extracted source
directories as arguments. Downloading SFCHD is blocked on the pending licence
confirmation (NOTICE.md) and is a large file either way; downloading shlokraval's
~2.5GB archive is a decision for whoever runs this, not something this script does on
its own.

Per source:

- **shlokraval/ppe-dataset-yolov8**: confirmed layout (peeked via `kaggle datasets
  files` + a single-file `data.yaml` download, no bulk pull) - standard Roboflow
  YOLOv8 export: `{split}/images/*.jpg` + `{split}/labels/*.txt`, splits named
  train/valid/test, class ids 0..13 indexing `data.yaml`'s `names` list.
- **SFCHD**: layout NOT verified - the repo README doesn't state an annotation format
  and the archive hasn't been downloaded (large file + pending licence). This script
  supports both a YOLO-txt layout and a COCO-JSON layout, auto-detected, as a
  reasonable hedge - re-check against the real archive once downloaded and adjust
  `load_sfchd_yolo`/`load_sfchd_coco` if neither matches.

Output layout (mirrors the shlokraval/Roboflow convention so it drops into the same
YOLOX COCO-conversion path `dfire_to_coco.py` already established for fire/smoke):

    <out>/images/{train,val,test}/*.jpg
    <out>/labels/{train,val,test}/*.txt        (unified class ids)
    <out>/domain_manifest.jsonl                (per-image domain provenance)
    <out>/data.yaml                             (unified class list)

Usage:
    python ppe_dataset_prep.py --shlokraval-dir /path/to/ppe-dataset-yolov8 \\
        [--sfchd-dir /path/to/SFCHD] --out-dir /path/to/ppe_merged
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ppe_taxonomy import SFCHD_CLASS_MAP, SHLOKRAVAL_CLASS_MAP, UNIFIED_CLASSES, DomainTag

# Confirmed via `python -m kaggle datasets download -d shlokraval/ppe-dataset-yolov8
# -f data.yaml` on 2026-09-16 - do not hand-edit without re-checking the real file,
# order is load-bearing (label files reference these by integer index).
SHLOKRAVAL_SOURCE_NAMES = [
    "Fall-Detected", "Gloves", "Goggles", "Hardhat", "Ladder", "Mask",
    "NO-Gloves", "NO-Goggles", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest",
]
SHLOKRAVAL_SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def convert_split_yolo(
    src_images: Path,
    src_labels: Path,
    source_names: list[str],
    class_map: dict[str, tuple[str, DomainTag]],
    out_images: Path,
    out_labels: Path,
    manifest_lines: list[dict],
) -> tuple[int, int]:
    """Copy one YOLO-format split, remapping class ids. Returns (images, boxes)."""
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    n_images = 0
    n_boxes = 0

    for img_path in sorted(src_images.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = src_labels / (img_path.stem + ".txt")
        kept_lines: list[str] = []
        domain_instances: list[dict] = []

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                src_id, cx, cy, w, h = parts
                src_name = source_names[int(src_id)]
                mapped = class_map.get(src_name)
                if mapped is None:
                    continue  # out-of-scope source class (Person, Ladder, Mask, ...)
                unified_name, domain = mapped
                unified_id = UNIFIED_CLASSES.index(unified_name)
                kept_lines.append(f"{unified_id} {cx} {cy} {w} {h}")
                domain_instances.append({"class": unified_name, "domain": domain.value})
                n_boxes += 1

        dest_img = out_images / img_path.name
        shutil.copy2(img_path, dest_img)
        (out_labels / (img_path.stem + ".txt")).write_text("\n".join(kept_lines))
        manifest_lines.append({
            "image": str(dest_img),
            "source": "shlokraval",
            "instances": domain_instances,
        })
        n_images += 1

    return n_images, n_boxes


def load_sfchd_yolo(src_root: Path) -> bool:
    """Heuristic: does this SFCHD extraction look like a YOLO-txt export?"""
    return (src_root / "labels").is_dir() or any(src_root.glob("*/labels"))


def load_sfchd_coco(src_root: Path) -> bool:
    return any(src_root.glob("**/*.json"))


def convert_sfchd(src_root: Path, out_root: Path, manifest_lines: list[dict]) -> tuple[int, int]:
    """SFCHD layout is unverified pending download - see module docstring. Raises
    NotImplementedError with a clear message rather than silently producing wrong
    output if the real archive doesn't match either guess."""
    if load_sfchd_yolo(src_root):
        raise NotImplementedError(
            "SFCHD looks YOLO-formatted but the exact split/class-index layout has "
            "not been verified against the real archive - inspect a few label files "
            "and source_names ordering by hand, then adapt convert_split_yolo's "
            "callers below (SFCHD_CLASS_MAP is keyed by class NAME, not index, so "
            "only the source_names list ordering needs to be supplied correctly)."
        )
    if load_sfchd_coco(src_root):
        raise NotImplementedError(
            "SFCHD looks COCO-JSON formatted - adapt the COCO-reading logic already "
            "written for D-Fire in dfire_to_coco.py (same shape: images[]/"
            "annotations[]/categories[]), remapping category names through "
            "SFCHD_CLASS_MAP instead of the fire/smoke CLASS_ID_MAP."
        )
    raise NotImplementedError(
        f"Could not detect SFCHD's annotation format under {src_root} - no labels/ "
        "directory and no .json files found. Inspect the extracted archive by hand."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shlokraval-dir", type=Path, default=None)
    ap.add_argument("--sfchd-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if not args.shlokraval_dir and not args.sfchd_dir:
        raise SystemExit("provide at least one of --shlokraval-dir / --sfchd-dir")

    manifest_lines: list[dict] = []
    totals = {"images": 0, "boxes": 0}

    if args.shlokraval_dir:
        for unified_split, source_split in SHLOKRAVAL_SPLIT_DIRS.items():
            src_images = args.shlokraval_dir / source_split / "images"
            src_labels = args.shlokraval_dir / source_split / "labels"
            if not src_images.is_dir():
                print(f"skip shlokraval split {source_split!r}: {src_images} not found")
                continue
            n_img, n_box = convert_split_yolo(
                src_images, src_labels, SHLOKRAVAL_SOURCE_NAMES, SHLOKRAVAL_CLASS_MAP,
                args.out_dir / "images" / unified_split,
                args.out_dir / "labels" / unified_split,
                manifest_lines,
            )
            print(f"shlokraval/{source_split} -> {unified_split}: {n_img} images, {n_box} boxes")
            totals["images"] += n_img
            totals["boxes"] += n_box

    if args.sfchd_dir:
        n_img, n_box = convert_sfchd(args.sfchd_dir, args.out_dir, manifest_lines)
        totals["images"] += n_img
        totals["boxes"] += n_box

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "domain_manifest.jsonl").open("w") as f:
        for line in manifest_lines:
            f.write(json.dumps(line) + "\n")

    data_yaml = args.out_dir / "data.yaml"
    data_yaml.write_text(
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(UNIFIED_CLASSES)}\nnames: {list(UNIFIED_CLASSES)}\n"
    )

    print(f"\nTotal: {totals['images']} images, {totals['boxes']} boxes")
    print(f"Wrote {args.out_dir / 'domain_manifest.jsonl'} and {data_yaml}")


if __name__ == "__main__":
    main()
