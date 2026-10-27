"""Looking closely at the people in a snapshot or frames (Phase J, "E2").

The chat model (Haiku) cannot judge protective equipment in a wide CCTV picture: measured
2026-10-27 on a workshop clip, it called a blue hard hat "a blue balloon", put helmets
lying on a bench onto people, and gave one worker's helmet to the welder. Sonnet 4.6 got
all 7 test crops right. So for every look at an image:

1. people are found with the system's own person detector and cropped at full size;
2. the whole scene goes out with each person boxed and numbered - and the person the alert
   flagged (its `bbox`, PPE alerts) boxed in red - plus a close-up of each person;
3. `ai.vision_model` writes one line per person (clothing, helmet ON the head yes / no /
   cannot tell, vest), and answers the question.

The chat gets that text, not the pictures, so it cannot re-judge them itself. People the
detector misses (e.g. a welder hidden in the glare) are still in the boxed scene, and the
model is told to list anyone visible but unboxed too.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perimeter.ai.bedrock import converse_text
from perimeter.settings import AISettings

log = logging.getLogger("perimeter.ai.inspect")

MAX_PEOPLE = 8  # close-ups per image; the boxed scene still shows everyone
SCENE_WIDTH = 1568  # the most of a picture the model takes in at once
CROP_SIZE = 512
DETECT_CONF = 0.35
FLAG_IOU = 0.3

PROMPT = """You are checking CCTV images for a site-safety system.
{about}Each image shows the whole scene with every detected person boxed and numbered
(yellow){flagged}, followed by a close-up of each numbered person.

For EACH numbered person, one line: where they are, clothing (shirt colour), helmet ON
their head - yes / no / cannot tell - and vest - yes / no / cannot tell.
Rules: a helmet lying nearby or held in a hand does not count; a cloth, cap, turban or
goggles is not a hard hat; if the head is hidden, in glare, blurred or too small, say
"cannot tell" - never guess. Judge each person only from their own close-up and box.
Then list anyone else clearly visible but not boxed (e.g. a welder hidden by sparks) the
same way. Finally answer the question in one or two sentences.

Question: {question}"""


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _jpeg(image: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode an image")
    return buf.tobytes()


class PeopleInspector:
    def __init__(self, ai: AISettings, client: Any, root: Path, detector: Any = None) -> None:
        self.ai = ai
        self._client = client  # a LazyClient
        self._model_path = root / ai.vision_person_model
        self._detector = detector
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.ai.vision_model)

    def _people(self, image: np.ndarray) -> list[tuple[float, float, float, float]]:
        with self._lock:
            if self._detector is None:
                from perimeter.detect.person import PersonDetector

                self._detector = PersonDetector(self._model_path, conf_threshold=DETECT_CONF)
        boxes = [tuple(float(v) for v in b) for b in self._detector.detect(image).xyxy]
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return sorted(boxes[:MAX_PEOPLE], key=lambda b: b[0])  # biggest, then left to right

    def inspect(
        self,
        images: list[tuple[str, bytes]],
        question: str,
        flagged_bbox: list[float] | None = None,
        about: str = "",
    ) -> dict[str, Any]:
        """`images` are (label, jpeg). `flagged_bbox` is the alert's own normalised box,
        drawn on every image. Returns {"report": text, "people": [...per image counts]}."""
        content: list[dict[str, Any]] = []
        counts = []
        for label, data in images:
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            people = self._people(image)
            flag_px = None
            if flagged_bbox:
                fx1, fy1, fx2, fy2 = flagged_bbox
                flag_px = (fx1 * w, fy1 * h, fx2 * w, fy2 * h)
            scene = image.copy()
            thick = max(2, w // 400)
            flagged_number = None
            crops: list[dict[str, Any]] = []
            for number, (x1, y1, x2, y2) in enumerate(people, start=1):
                if flag_px and _iou((x1, y1, x2, y2), flag_px) >= FLAG_IOU:
                    flagged_number = number
                cv2.rectangle(scene, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), thick)
                cv2.putText(scene, str(number), (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, thick * 0.6, (0, 255, 255), thick)
                bw, bh = x2 - x1, y2 - y1
                crop = image[int(max(0, y1 - 0.15 * bh)):int(min(h, y2 + 0.05 * bh)),
                             int(max(0, x1 - 0.25 * bw)):int(min(w, x2 + 0.25 * bw))]
                if crop.size:
                    scale = CROP_SIZE / max(crop.shape[:2])
                    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    crops += [{"text": f"{label} - close-up of person {number}:"},
                              {"image": {"format": "jpeg", "source": {"bytes": _jpeg(crop)}}}]
            if flag_px:
                cv2.rectangle(scene, (int(flag_px[0]), int(flag_px[1])),
                              (int(flag_px[2]), int(flag_px[3])), (0, 0, 255), thick + 2)
            if w > SCENE_WIDTH:
                scene = cv2.resize(scene, (SCENE_WIDTH, int(SCENE_WIDTH * h / w)),
                                   interpolation=cv2.INTER_AREA)
            note = ""
            if flag_px:
                note = (" The alert flagged the person in the RED box"
                        + (f" (person {flagged_number})." if flagged_number else "."))
            content += [{"text": f"{label} - whole scene, {len(people)} people boxed.{note}"},
                        {"image": {"format": "jpeg", "source": {"bytes": _jpeg(scene, 85)}}},
                        *crops]
            counts.append({"image": label, "people_detected": len(people),
                           "flagged_person": flagged_number})
        if not content:
            raise ValueError("no readable image")
        prompt = PROMPT.format(
            about=f"{about}\n" if about else "",
            flagged=", and the person the alert flagged in RED" if flagged_bbox else "",
            question=question or "Who is and is not wearing a helmet?",
        )
        content.append({"text": prompt})
        settings = replace(self.ai, chat_model=self.ai.vision_model)
        report = converse_text(self._client.get(), settings, content, max_tokens=900)
        return {"report": report.strip(), "images": counts}
