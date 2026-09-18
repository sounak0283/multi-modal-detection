"""Identity resolution: the single entry point `Pipeline` calls (Expansion Plan Phase F;
PLAN.md section 7).

Invoked only on a confirmed boundary ENTRY - never on every frame. The cost (two extra
model inferences) is unjustifiable on CPU at frame rate, and the privacy exposure of
running face recognition continuously is unnecessary when the product only needs to know
who crossed a boundary.

Three outcomes, matching PLAN.md section 7 exactly:

* `known`         - a confident match against an enrolled person.
* `unknown_face`  - a usable face was found but matched no one.
* `no_face`       - no usable face (box too small, or none detected in the crop).

`no_face` must never be rendered as "intruder" - see `alerts/messages.py:subject_for`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perimeter.identity.gallery import FaceGallery
from perimeter.identity.sface import FaceEmbedder
from perimeter.identity.yunet import FaceBox, FaceDetector

# The top ~35% of a person's bounding box is where a face sits under normal camera
# geometry (PLAN.md section 7 step 2). Cropped from the full-resolution frame, never
# from the letterboxed inference input.
FACE_CROP_TOP_FRACTION = 0.35


@dataclass(frozen=True)
class IdentityResult:
    status: str  # "known" | "unknown_face" | "no_face"
    person_id: str | None = None
    name: str | None = None


NO_FACE = IdentityResult(status="no_face")

# A face whose box ends within this many pixels of the crop's bottom edge is treated as
# cut off by the crop.
FACE_CUT_OFF_MARGIN_PX = 4


def _cut_off_at_bottom(face: FaceBox, crop_height: int) -> bool:
    _x, y, _w, h = (float(v) for v in face.raw[:4])
    return y + h >= crop_height - FACE_CUT_OFF_MARGIN_PX


class IdentityResolver:
    """Shared across every camera's `Pipeline` - the models and gallery are stateless per
    call and expensive to load, so one instance serves the whole process rather than one
    per camera (unlike the per-camera fire/smoke detector)."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        gallery: FaceGallery,
        min_person_box_px: int = 120,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.gallery = gallery
        self.min_person_box_px = min_person_box_px

    def resolve(
        self, frame_bgr: np.ndarray, person_box_px: tuple[float, float, float, float]
    ) -> IdentityResult:
        x1, y1, x2, y2 = person_box_px
        box_height = y2 - y1
        if box_height < self.min_person_box_px:
            # Step 1: below this, a face would be under ~40px and YuNet returns nothing
            # useful - skip the inference entirely rather than run it for nothing.
            return NO_FACE

        frame_height, frame_width = frame_bgr.shape[:2]
        crop_bottom = y1 + box_height * FACE_CROP_TOP_FRACTION
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(frame_width, int(x2)), min(frame_height, int(crop_bottom))
        if cx2 <= cx1 or cy2 <= cy1:
            return NO_FACE

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        face = self.detector.largest(crop)
        if face is None or _cut_off_at_bottom(face, crop.shape[0]):
            # The top-fraction rule assumes the whole body is in frame. Close to the
            # camera (a desk webcam, someone right under a ceiling camera) the box is
            # head-and-shoulders only and the crop slices through the face. YuNet still
            # finds the top half of it, but aligning and embedding half a face scores
            # like a stranger (0.03 against the person's own enrolment in one real
            # case) - so retry on the full person box.
            cy2 = min(frame_height, int(y2))
            if cy2 <= cy1:
                return NO_FACE
            crop = frame_bgr[cy1:cy2, cx1:cx2]
            face = self.detector.largest(crop)
        if face is None:
            return NO_FACE

        embedding = self.embedder.embed(crop, face)
        match = self.gallery.match(embedding)
        if match is None:
            return IdentityResult(status="unknown_face")
        person_id, name = match
        return IdentityResult(status="known", person_id=person_id, name=name)
