"""Unified PPE class taxonomy for the object-detector training pipeline.

Architecture decision (2026-09-16): every available PPE dataset (SFCHD, the
Apache-2.0/CC BY 4.0 replacements for the disqualified SH17) annotates PPE items as
bounding boxes across the whole frame, not per-person-crop labels. That is a natural
fit for a YOLOX-style multi-class object detector - the same architecture and
YoloxOnnx wrapper backend/models/firesmoke uses - NOT the whole-person-crop multi-label
classifier `backend/src/perimeter/detect/ppe.py:PPEClassifier` currently implements.
This taxonomy module targets the object-detector shape. `ppe.py`'s classifier and its
PPEMonitor hysteresis logic are a separate, later runtime-integration decision - not
touched by this data-prep work.

Both source families already model "item present" and "item absent" as two distinct,
independently-annotated visual patterns (a helmeted head looks different from a bare
one) rather than treating absence as "no detection" - this taxonomy keeps that design
rather than collapsing to a single positive-only class per item.

Per-class data source, spelled out because it matters for how much to trust the
trained detector on each one:

    | Class            | Source(s)                          | Domain                        |
    |------------------|-------------------------------------|--------------------------------|
    | helmet_present   | SFCHD "Safety Helmet"; shlokraval "Hardhat" | cctv_real + posed_web  |
    | helmet_absent    | SFCHD "Head"; shlokraval "NO-Hardhat"       | cctv_real + posed_web  |
    | vest_present     | SFCHD "Safety Clothing" (UNVERIFIED); shlokraval "Safety Vest" | cctv_real (unverified) + posed_web |
    | vest_absent      | SFCHD "Other Clothing" (UNVERIFIED); shlokraval "NO-Safety Vest" | cctv_real (unverified) + posed_web |
    | gloves_present   | shlokraval "Gloves"                | posed_web only                |
    | gloves_absent    | shlokraval "NO-Gloves"              | posed_web only                |
    | glasses_present  | shlokraval "Goggles"                | posed_web only                |
    | glasses_absent   | shlokraval "NO-Goggles"              | posed_web only                |
    | shoes_present    | NONE FOUND                          | no permissively-licensed source located as of 2026-09-16 |
    | shoes_absent     | NONE FOUND                          | ditto                          |

Shoes/boots has no data at all. Do not silently drop it from PPE_ITEMS
(backend/src/perimeter/detect/ppe.py) - surface it as an explicit known gap in any
model card / manifest for whatever ships, per the same "don't overstate provenance"
principle NOTICE.md already enforces for datasets.

SFCHD's "Safety Clothing" / "Other Clothing" -> vest mapping is UNVERIFIED. "Safety
clothing" in the source annotation guidelines may be broader than a hi-vis vest (could
include full coveralls, jackets, etc.) - this needs actual sample-image inspection
before being trusted, which requires downloading the SFCHD archive (blocked pending
both the size and the licence-confirmation email, per NOTICE.md's pending-confirmation
row). Every function below still emits the mapping so the pipeline is ready to run once
that's unblocked, but `MAPPING_CONFIDENCE["vest"]["sfchd"]` is deliberately marked
"unverified" so nothing downstream can accidentally treat it as confirmed.

SFCHD's "Blurred Clothing" / "Blurred Head" classes are a real CCTV-domain phenomenon
(motion blur, distance blur) that posed/web photos cannot teach at all. Rather than
merge them into the base present/absent classes (which would teach the model that
blur looks like a normal detection), they are kept as their own domain tag
(`blurred_cctv`) on the derived instance - see `DomainTag` below - so a training/
validation split can hold them out separately and the eventual runtime should treat a
detection under this tag as a cue to abstain (report "indeterminate"), the same
"abstention must never be read as a violation" principle PLATFORM_EXPANSION_PLAN.md
section 5 already established for the identity layer's `no_face` state.
"""

from __future__ import annotations

from enum import Enum


class DomainTag(str, Enum):
    CCTV_REAL = "cctv_real"
    CCTV_REAL_BLURRED = "cctv_real_blurred"
    POSED_WEB = "posed_web"
    POSED_WEB_PROXY_AUGMENTED = "posed_web_proxy_augmented"


# Final, ordered training taxonomy. Index position is the YOLO/COCO class id emitted
# by the dataset-prep scripts and expected by the training exp config - do not reorder
# without regenerating the prepared dataset.
UNIFIED_CLASSES: tuple[str, ...] = (
    "helmet_present",
    "helmet_absent",
    "vest_present",
    "vest_absent",
    "gloves_present",
    "gloves_absent",
    "glasses_present",
    "glasses_absent",
)

# PPE items with NO annotated data in any dataset evaluated so far. Not present in
# UNIFIED_CLASSES at all - kept here so downstream tooling can generate an explicit
# "not trained: no data" line in the manifest instead of silently omitting it.
UNCOVERED_ITEMS: tuple[str, ...] = ("shoes",)

MAPPING_CONFIDENCE: dict[str, dict[str, str]] = {
    "helmet": {"sfchd": "confirmed", "shlokraval": "confirmed"},
    "vest": {"sfchd": "unverified", "shlokraval": "confirmed"},
    "gloves": {"shlokraval": "confirmed"},
    "glasses": {"shlokraval": "confirmed"},
}

# Source dataset class name -> (unified class name, domain tag).
# SFCHD label names per its README (github.com/lijfrank/SFCHD-SCALE):
#   Person, Safety Helmet, Safety Clothing, Other Clothing, Head, Blurred Clothing,
#   Blurred Head. "Person" and bare "Head"/"Blurred Head" that aren't specifically a
#   helmet-absence signal are handled separately - see NOTES below.
SFCHD_CLASS_MAP: dict[str, tuple[str, DomainTag]] = {
    "Safety Helmet": ("helmet_present", DomainTag.CCTV_REAL),
    "Head": ("helmet_absent", DomainTag.CCTV_REAL),
    "Blurred Head": ("helmet_absent", DomainTag.CCTV_REAL_BLURRED),
    "Safety Clothing": ("vest_present", DomainTag.CCTV_REAL),  # UNVERIFIED, see module docstring
    "Other Clothing": ("vest_absent", DomainTag.CCTV_REAL),  # UNVERIFIED, see module docstring
    "Blurred Clothing": ("vest_absent", DomainTag.CCTV_REAL_BLURRED),  # UNVERIFIED
    # "Person" is intentionally NOT mapped - the pipeline's own YOLOX person detector
    # already produces person boxes; re-detecting "person" here would duplicate that
    # role and dilute training signal on the PPE items this detector actually exists
    # for. Drop "Person" instances during conversion.
}

# shlokraval/ppe-dataset-yolov8 (Kaggle mirror of Roboflow Universe's
# roboflow-universe-projects/personal-protective-equipment-combined-model, v4).
# Kaggle's own licence field says Apache-2.0; the embedded roboflow.license metadata
# in that dataset's own data.yaml says CC BY 4.0 - record the more conservative
# (attribution-requiring) one in NOTICE.md since it's from the original project, not
# the re-uploader.
SHLOKRAVAL_CLASS_MAP: dict[str, tuple[str, DomainTag]] = {
    "Hardhat": ("helmet_present", DomainTag.POSED_WEB),
    "NO-Hardhat": ("helmet_absent", DomainTag.POSED_WEB),
    "Safety Vest": ("vest_present", DomainTag.POSED_WEB),
    "NO-Safety Vest": ("vest_absent", DomainTag.POSED_WEB),
    "Gloves": ("gloves_present", DomainTag.POSED_WEB),
    "NO-Gloves": ("gloves_absent", DomainTag.POSED_WEB),
    "Goggles": ("glasses_present", DomainTag.POSED_WEB),
    "NO-Goggles": ("glasses_absent", DomainTag.POSED_WEB),
    # Fall-Detected, Ladder, Mask, NO-Mask, Person, Safety Cone are out of scope for
    # this project's PPE_ITEMS set - dropped during conversion, not mapped.
}


def class_index(name: str) -> int:
    return UNIFIED_CLASSES.index(name)
