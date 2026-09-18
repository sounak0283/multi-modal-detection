#!/usr/bin/env python3
"""Per-class image/instance counts from ppe_dataset_prep.py's domain_manifest.jsonl,
broken down by domain so the CCTV-real vs posed/proxy-augmented split is visible at a
glance rather than buried in a single aggregate number.

Usage:
    python ppe_dataset_manifest.py --manifest /path/to/ppe_merged/domain_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ppe_taxonomy import UNCOVERED_ITEMS, UNIFIED_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    # counts[class][domain] = (images_containing_it, instances)
    images_with_class: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    instance_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        image = entry["image"]
        for inst in entry["instances"]:
            cls, domain = inst["class"], inst["domain"]
            images_with_class[cls][domain].add(image)
            instance_counts[cls][domain] += 1

    print(f"{'class':<18} {'domain':<28} {'images':>8} {'instances':>10}")
    print("-" * 68)
    for cls in UNIFIED_CLASSES:
        domains = images_with_class.get(cls, {})
        if not domains:
            print(f"{cls:<18} {'NO DATA':<28} {0:>8} {0:>10}")
            continue
        for domain in sorted(domains):
            n_img = len(domains[domain])
            n_inst = instance_counts[cls][domain]
            print(f"{cls:<18} {domain:<28} {n_img:>8} {n_inst:>10}")

    print()
    for item in UNCOVERED_ITEMS:
        print(f"NOT TRAINED (no data source found): {item}")


if __name__ == "__main__":
    main()
