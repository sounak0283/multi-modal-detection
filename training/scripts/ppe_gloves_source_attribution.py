#!/usr/bin/env python3
"""Source-attribution + confound-isolation check for gloves_present/gloves_absent,
built after ppe_wrong_object_montage.py's gloves montages showed the "no
overlapping person" flagged subset dominated by on-screen timestamp overlays from
a single continuous CCTV feed (2021-02 through 2021-03-11) - the same
single-source-concentration pattern that caused helmet_absent's confirmed two-video
contamination and inflated vest_present's original flagged rate.

Filenames encode this: box instances prefixed "PP02<frame-id>" (e.g.
"PP02img458_jpg.rf.<hash>.jpg") are one continuous source-video export; every other
naming family in this merge (nglov###, fglov###, goggle######, img###, frame####,
NVR, PPE) is a one-off ID with no shared sequential prefix, i.e. not a video frame
sequence. "PP02" is the one real "distinct source" candidate to check.

Three things this script answers, in order:
  1. How many instances (not unique images - box count, matching how the model
     actually trains) does PP02 contribute to gloves_present and gloves_absent,
     out of the FULL class (not just the earlier no-person-overlap subset)?
  2. Does PP02's share differ between present/absent - i.e. is the split uneven
     enough to itself explain (or partially explain) Step 1's blur confound
     (cohens_d_blur=0.647)?
  3. Re-run ppe_step1_data_safety.py's confound_check() three ways: full class,
     PP02 excluded, PP02-only - to see whether the confound is driven entirely by
     this one source, or persists independent of it.

Usage:
    python ppe_gloves_source_attribution.py --dataset A:/fsbd_training/ppe_merged \\
        --manifest A:/fsbd_training/ppe_merged/domain_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from ppe_step1_data_safety import confound_check, load_instances

SOURCE_TOKEN_RE = re.compile(r"^([A-Za-z]+\d+)")


def source_group(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_jpg$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\.rf\.[0-9a-f]{16,}$", "", stem, flags=re.IGNORECASE)
    m = SOURCE_TOKEN_RE.match(stem)
    return m.group(1) if m else stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=3000, help="sample cap for confound_check, matches Step 1's own cap")
    args = ap.parse_args()

    by_class = load_instances(args.manifest, args.dataset)
    present = by_class.get("gloves_present", [])
    absent = by_class.get("gloves_absent", [])

    print("=== 1. Instance-level source breakdown (full classes, not just flagged subset) ===\n")
    group_counts: dict[str, dict[str, int]] = {"gloves_present": defaultdict(int), "gloves_absent": defaultdict(int)}
    for cname, instances in (("gloves_present", present), ("gloves_absent", absent)):
        for inst in instances:
            group_counts[cname][source_group(inst["image"].name)] += 1

    pp02_present = group_counts["gloves_present"].get("PP02", 0)
    pp02_absent = group_counts["gloves_absent"].get("PP02", 0)
    pp02_present_pct = pp02_present / len(present) * 100 if present else 0
    pp02_absent_pct = pp02_absent / len(absent) * 100 if absent else 0

    print(f"gloves_present: {len(present)} instances total, PP02 = {pp02_present} ({pp02_present_pct:.1f}%)")
    print(f"gloves_absent:  {len(absent)} instances total, PP02 = {pp02_absent} ({pp02_absent_pct:.1f}%)")
    print(f"\ntop non-PP02 groups by instance count:")
    for cname in ("gloves_present", "gloves_absent"):
        rest = sorted(((g, c) for g, c in group_counts[cname].items() if g != "PP02"), key=lambda x: -x[1])
        print(f"  {cname}: {rest[:10]}")

    print(f"\n=== 2. Uneven split ===")
    gap = pp02_absent_pct - pp02_present_pct
    print(f"PP02 share of gloves_present: {pp02_present_pct:.1f}%")
    print(f"PP02 share of gloves_absent:  {pp02_absent_pct:.1f}%")
    print(f"gap: {gap:+.1f} percentage points {'(absent-skewed)' if gap > 0 else '(present-skewed)' if gap < 0 else '(balanced)'}")

    def filt(instances, want_pp02: bool):
        return [i for i in instances if (source_group(i["image"].name) == "PP02") == want_pp02]

    present_excl, absent_excl = filt(present, False), filt(absent, False)
    present_only, absent_only = filt(present, True), filt(absent, True)

    print(f"\n=== 3. Blur confound, re-scoped ===\n")
    print("--- (a) FULL class (Step 1's original scope, cap={}) ---".format(args.cap))
    full = confound_check(present, absent, args.cap)
    print(f"  cohens_d_blur = {full['cohens_d_blur']:.3f}  (present blur_mean={full['present']['blur_mean']:.1f} std={full['present']['blur_std']:.1f}, "
          f"absent blur_mean={full['absent']['blur_mean']:.1f} std={full['absent']['blur_std']:.1f})")

    print(f"\n--- (b) PP02 EXCLUDED (n_present={len(present_excl)}, n_absent={len(absent_excl)}) ---")
    excl = confound_check(present_excl, absent_excl, args.cap)
    print(f"  cohens_d_blur = {excl['cohens_d_blur']:.3f}  (present blur_mean={excl['present']['blur_mean']:.1f} std={excl['present']['blur_std']:.1f}, "
          f"absent blur_mean={excl['absent']['blur_mean']:.1f} std={excl['absent']['blur_std']:.1f})")

    print(f"\n--- (c) PP02 ONLY (n_present={len(present_only)}, n_absent={len(absent_only)}) ---")
    only = confound_check(present_only, absent_only, args.cap)
    print(f"  cohens_d_blur = {only['cohens_d_blur']:.3f}  (present blur_mean={only['present']['blur_mean']:.1f} std={only['present']['blur_std']:.1f}, "
          f"absent blur_mean={only['absent']['blur_mean']:.1f} std={only['absent']['blur_std']:.1f})")

    out = {
        "pp02_share": {
            "gloves_present_count": pp02_present, "gloves_present_pct": pp02_present_pct,
            "gloves_absent_count": pp02_absent, "gloves_absent_pct": pp02_absent_pct,
            "gap_pp": gap,
        },
        "source_group_counts": {k: dict(v) for k, v in group_counts.items()},
        "confound_full": full,
        "confound_pp02_excluded": excl,
        "confound_pp02_only": only,
    }
    out_path = args.dataset / "gloves_source_attribution.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
