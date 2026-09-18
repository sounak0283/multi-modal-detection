#!/usr/bin/env python3
"""Source-attribution check for vest_present's flagged (low hi-vis-fraction) boxes -
same method that isolated helmet_absent's two contaminating source videos, applied
here to see whether the 25.7% mislabel rate is concentrated in a few sub-sources
(safe to isolate and drop) or diffuse across the whole class (not fixable by
dropping a sub-source; a real decision point for the user).

Source grouping: Roboflow-exported filenames encode the original upload name before
the "_jpg.rf.<hash>" suffix (e.g. "IMG_3099_mp4-18_jpg.rf.0858c27d...jpg" -> original
stem "IMG_3099_mp4-18"). Two grouping levels are reported since a single project can
mix naming conventions:
  - fine: the original stem with a trailing frame index stripped (groups sequential
    video-frame exports like "IMG_3099_mp4-18" / "IMG_3099_mp4-19" together)
  - coarse: just the leading alphabetic naming-convention prefix (e.g. "pos", "image",
    "img", "Image") - a cheap proxy for "which sub-dataset within the Roboflow
    'combined' project this came from", since different contributors/sources tend to
    use different naming conventions.

Usage:
    python ppe_vest_source_attribution.py --dataset A:/fsbd_training/ppe_merged \\
        --manifest A:/fsbd_training/ppe_merged/domain_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ppe_taxonomy import UNIFIED_CLASSES

HIVIS_HUE_RANGES = [(10, 45), (45, 85)]
HIVIS_SAT_MIN, HIVIS_VAL_MIN = 70, 80
VEST_FLAG_THRESHOLD = 0.12

ROBOFLOW_SUFFIX_RE = re.compile(r"\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)
TRAILING_FRAME_IDX_RE = re.compile(r"[-_]\d+$")
LEADING_ALPHA_RE = re.compile(r"^[A-Za-z]+")


def original_stem(filename: str) -> str:
    stem = Path(filename).stem  # drops .jpg
    stem = re.sub(r"_jpg$", "", stem, flags=re.IGNORECASE)
    stem = ROBOFLOW_SUFFIX_RE.sub("", stem)
    return stem


def fine_group(stem: str) -> str:
    return TRAILING_FRAME_IDX_RE.sub("", stem)


def coarse_group(stem: str) -> str:
    m = LEADING_ALPHA_RE.match(stem)
    return m.group(0) if m else stem


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


def crop_box(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
    x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
    return img[y1:y2, x1:x2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

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
            by_class[UNIFIED_CLASSES[cls_id]].append(
                {"image": img_path, "box": (cx, cy, bw, bh)}
            )

    instances = by_class.get("vest_present", [])
    print(f"scanning {len(instances)} vest_present instances...")

    fine_total, fine_flagged = defaultdict(int), defaultdict(int)
    coarse_total, coarse_flagged = defaultdict(int), defaultdict(int)
    all_flagged_images = []
    n_checked = 0

    for inst in instances:
        img = cv2.imread(str(inst["image"]))
        if img is None:
            continue
        n_checked += 1
        crop = crop_box(img, inst["box"])
        frac = hivis_fraction(crop)
        is_flagged = frac < VEST_FLAG_THRESHOLD

        stem = original_stem(inst["image"].name)
        fg, cg = fine_group(stem), coarse_group(stem)
        fine_total[fg] += 1
        coarse_total[cg] += 1
        if is_flagged:
            fine_flagged[fg] += 1
            coarse_flagged[cg] += 1
            all_flagged_images.append(inst["image"].name)

    total_flagged = sum(fine_flagged.values())
    print(f"checked {n_checked}, flagged {total_flagged} ({total_flagged/n_checked*100:.1f}%)\n")

    def report(level_name, total_d, flagged_d, min_group_size=5):
        print(f"--- {level_name} grouping: top contaminated groups (>= {min_group_size} instances, sorted by flagged count) ---")
        rows = []
        for g, tot in total_d.items():
            fl = flagged_d.get(g, 0)
            if tot >= min_group_size and fl > 0:
                rows.append((g, tot, fl, fl / tot))
        rows.sort(key=lambda r: -r[2])
        top10 = rows[:15]
        for g, tot, fl, rate in top10:
            flag = "  <-- high concentration" if rate > 0.5 else ""
            print(f"  {g!r:30s} total={tot:5d}  flagged={fl:5d}  rate={rate*100:5.1f}%{flag}")
        top_n_flagged = sum(r[2] for r in rows[:5])
        print(f"  top-5 groups account for {top_n_flagged}/{total_flagged} = {top_n_flagged/total_flagged*100:.1f}% of all flagged instances")
        print(f"  total distinct groups with >=1 flagged instance: {sum(1 for g in flagged_d if flagged_d[g] > 0)}")
        print()
        return rows

    fine_rows = report("FINE (per source video/image-set)", fine_total, fine_flagged)
    coarse_rows = report("COARSE (per naming-convention family)", coarse_total, coarse_flagged, min_group_size=20)

    out = {
        "n_checked": n_checked,
        "total_flagged": total_flagged,
        "flagged_rate": total_flagged / n_checked if n_checked else 0,
        "fine_groups": [{"group": g, "total": t, "flagged": f, "rate": f / t} for g, t, f, _ in fine_rows],
        "coarse_groups": [{"group": g, "total": t, "flagged": f, "rate": f / t} for g, t, f, _ in coarse_rows],
        "all_flagged_filenames": all_flagged_images,
    }
    out_path = args.dataset / "vest_source_attribution.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
