#!/usr/bin/env python3
"""Wrong-object spot-check for helmet and gloves classes, extending
ppe_person_overlap_check.py's finding beyond vest_present. That script found a
confirmed wrong-object mislabel (a shoe held in someone's hand, boxed and labeled
vest_present) via "no overlapping person detection nearby" as a class-agnostic
signal, then flagged the SAME signal is noisy for vest_present specifically (most
flags there are genuine vests the whole-body detector just couldn't find a person
for - tight crop, motion blur, unusual pose - not mislabels).

This script does NOT re-derive that caveat's conclusion for helmet/gloves - it
builds the visual evidence (montages of the actual flagged crops) so that can be
judged directly, the same way the vest montage let the user spot blue/cyan/pink/
black vests being wrongly flagged by the color check. No conclusion is asserted
here about mislabel rate - only the material to review it.

Reuses backend/models/yolox_person the same way ppe_person_overlap_check.py does.

Usage:
    python ppe_wrong_object_montage.py --dataset A:/fsbd_training/ppe_merged \\
        --manifest A:/fsbd_training/ppe_merged/domain_manifest.jsonl \\
        --person-model E:/multi-modal-detection/backend/models/yolox_person/yolox_nano.onnx \\
        --out-dir A:/fsbd_training/ppe_review_montages
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))
from perimeter.detect.person import PersonDetector  # noqa: E402

from ppe_taxonomy import UNIFIED_CLASSES  # noqa: E402

MARGIN_FRAC = 0.15
DEFAULT_TARGET_CLASSES = ("helmet_present", "helmet_absent", "gloves_present", "gloves_absent")


def use_gpu_session(detector: PersonDetector) -> None:
    inner = detector.model
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    inner.session = ort.InferenceSession(
        str(inner.model_path), options,
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )
    print("person detector providers:", inner.session.get_providers())


def person_boxes_expanded(detector: PersonDetector, img: np.ndarray) -> np.ndarray:
    dets = detector.detect(img)
    if len(dets) == 0:
        return np.zeros((0, 4), np.float32)
    h, w = img.shape[:2]
    boxes = dets.xyxy.copy()
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] = np.clip(boxes[:, 0] - bw * MARGIN_FRAC, 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1] - bh * MARGIN_FRAC, 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2] + bw * MARGIN_FRAC, 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3] + bh * MARGIN_FRAC, 0, h)
    return boxes


def center_in_any_box(cx: float, cy: float, boxes: np.ndarray) -> bool:
    if len(boxes) == 0:
        return False
    return bool(np.any((boxes[:, 0] <= cx) & (cx <= boxes[:, 2]) & (boxes[:, 1] <= cy) & (cy <= boxes[:, 3])))


def write_montage(cls_name: str, examples: list, total_flagged: int, out_dir: Path, n_montage: int) -> None:
    rng = random.Random(0)
    sample = rng.sample(examples, min(n_montage, len(examples)))
    THUMB, COLS = 160, 8
    thumbs = []
    for item in sample:
        img = cv2.imread(str(item["image"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy, bw, bh = item["box"]
        mx, my = bw * w * 0.3, bh * h * 0.3
        x1, y1 = max(0, int((cx - bw / 2) * w - mx)), max(0, int((cy - bh / 2) * h - my))
        x2, y2 = min(w, int((cx + bw / 2) * w + mx)), min(h, int((cy + bh / 2) * h + my))
        crop = img[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        bx1, by1 = int((cx - bw / 2) * w - x1), int((cy - bh / 2) * h - y1)
        bx2, by2 = int((cx + bw / 2) * w - x1), int((cy + bh / 2) * h - y1)
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        thumbs.append(cv2.resize(crop, (THUMB, THUMB)))
    rows_n = (len(thumbs) + COLS - 1) // COLS
    grid = np.full((rows_n * THUMB + 30, COLS * THUMB, 3), 30, np.uint8)
    cv2.putText(grid, f"{cls_name}: boxes with NO overlapping person detection (n={len(thumbs)} of {total_flagged})",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, COLS)
        y0 = 30 + r * THUMB
        grid[y0:y0 + THUMB, c * THUMB:(c + 1) * THUMB] = thumb
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cls_name}_no_person_overlap.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote montage: {out_path} ({len(thumbs)} of {total_flagged} flagged)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--person-model", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-montage", type=int, default=40)
    ap.add_argument("--classes", nargs="+", default=list(DEFAULT_TARGET_CLASSES),
                     help="which classes to check (default: helmet + gloves, all present/absent)")
    args = ap.parse_args()
    target_classes = args.classes

    by_class: dict[str, list[dict]] = defaultdict(list)
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        img_path = Path(entry["image"])
        split = img_path.parts[-2]
        label_path = args.dataset / "labels" / split / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        for line2 in label_path.read_text().splitlines():
            p = line2.split()
            if len(p) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            by_class[UNIFIED_CLASSES[cls_id]].append({"image": img_path, "box": (cx, cy, bw, bh)})

    detector = PersonDetector(args.person_model, conf_threshold=0.25)
    use_gpu_session(detector)

    results = {}
    for cls_name in target_classes:
        instances = by_class.get(cls_name, [])
        if not instances:
            continue
        person_cache: dict[Path, tuple[np.ndarray, int, int]] = {}
        n_checked, n_no_person = 0, 0
        no_person_examples = []
        for i, inst in enumerate(instances):
            img_path = inst["image"]
            if img_path not in person_cache:
                img = cv2.imread(str(img_path))
                if img is None:
                    person_cache[img_path] = (np.zeros((0, 4), np.float32), 0, 0)
                else:
                    h, w = img.shape[:2]
                    person_cache[img_path] = (person_boxes_expanded(detector, img), w, h)
            boxes, w, h = person_cache[img_path]
            if w == 0:
                continue
            cx, cy, bw, bh = inst["box"]
            px, py = cx * w, cy * h
            n_checked += 1
            if not center_in_any_box(px, py, boxes):
                n_no_person += 1
                if len(no_person_examples) < 300:
                    no_person_examples.append(inst)
            if (i + 1) % 5000 == 0:
                print(f"  {cls_name}: {i+1}/{len(instances)} instances, "
                      f"{n_no_person}/{n_checked} no-overlap so far", flush=True)
        results[cls_name] = {
            "checked": n_checked, "no_person_overlap": n_no_person,
            "rate": n_no_person / n_checked if n_checked else 0,
        }
        print(f"{cls_name}: {n_no_person}/{n_checked} = {results[cls_name]['rate']*100:.2f}% have NO overlapping person box")
        if no_person_examples:
            write_montage(cls_name, no_person_examples, n_no_person, args.out_dir, args.n_montage)

    (args.dataset / "wrong_object_montage_report.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.dataset / 'wrong_object_montage_report.json'}")


if __name__ == "__main__":
    main()
