"""YOLOX ONNX inference wrapper (PLAN.md sections 3 and 4).

Shared by the person detector and, once trained, the fire/smoke detector - they are the
same architecture with different weights and class maps.

Preprocessing contract
----------------------
    letterbox to the model's input size with pad value 114,
    ratio = min(H_in/h, W_in/w), image placed top-left, NO normalisation
    (raw 0-255 float32), NCHW layout.

Getting any part of that wrong yields plausible-looking garbage rather than an error, so
it is pinned here and recorded per-model in each manifest.json.

**Colour order is per-model and must not be guessed.** Upstream YOLOX's own `preproc`
does no colour conversion, so the official Megvii exports expect **BGR**. OpenCV Zoo
re-exported the same architecture and its demo applies `cvtColor(BGR2RGB)` first, so the
zoo weights expect **RGB**. Measured on 2026-08-11 against a pedestrian clip: feeding the
official yolox_nano BGR found 8 people, RGB found 6 with lower scores. Wrong colour order
degrades quietly rather than failing, which is exactly why `to_rgb` is an explicit
constructor argument with no default that suits both.

The model output is UNDECODED: [1, 8400, 85] for a 640x640 input, where
8400 = 80^2 + 40^2 + 20^2 over strides 8/16/32, and 85 = 4 box + 1 objectness + 80 classes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

log = logging.getLogger("perimeter.detect.yolox")

YOLOX_STRIDES = (8, 16, 32)
PAD_VALUE = 114


@dataclass(frozen=True)
class Detections:
    """Axis-aligned detections in ORIGINAL image pixel coordinates.

    Array-of-struct rather than struct-of-array would be tidier to read but slower to
    hand to the tracker, which wants columns. This layout maps directly onto
    supervision.Detections in Phase 2.
    """

    xyxy: np.ndarray  # (N, 4) float32
    scores: np.ndarray  # (N,)  float32
    class_ids: np.ndarray  # (N,)  int32

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @classmethod
    def empty(cls) -> Detections:
        return cls(
            xyxy=np.zeros((0, 4), np.float32),
            scores=np.zeros((0,), np.float32),
            class_ids=np.zeros((0,), np.int32),
        )

    def filter_classes(self, keep: set[int]) -> Detections:
        if len(self) == 0:
            return self
        mask = np.isin(self.class_ids, list(keep))
        return Detections(self.xyxy[mask], self.scores[mask], self.class_ids[mask])


def default_intra_op_threads() -> int:
    """Physical cores minus one, per PLAN.md section 4.

    os.cpu_count() reports logical CPUs. On an SMT machine that is double the physical
    count, and oversubscribing ORT is the classic cause of "it got slower when I
    optimised it". Halving is a heuristic - override explicitly via config on hardware
    where it guesses wrong.
    """
    logical = os.cpu_count() or 2
    physical = max(1, logical // 2)
    return max(1, physical - 1)


def configure_opencv_threads() -> None:
    """Stop OpenCV and onnxruntime from each claiming every core (PLAN.md section 4).

    Call once at application start, not from library code - it mutates global state.
    """
    cv2.setNumThreads(1)


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize preserving aspect ratio, pad to `size` with 114, image at top-left.

    Returns the padded image and the scale factor needed to map boxes back.
    """
    in_h, in_w = size
    src_h, src_w = image.shape[:2]
    ratio = min(in_h / src_h, in_w / src_w)
    new_w, new_h = int(src_w * ratio), int(src_h * ratio)

    padded = np.full((in_h, in_w, 3), PAD_VALUE, dtype=np.uint8)
    if new_w > 0 and new_h > 0:
        padded[:new_h, :new_w] = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
    return padded, ratio


def build_grids(input_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Anchor grid and per-anchor stride for YOLOX's undecoded output.

    Precomputed once per model: it depends only on input size, and rebuilding it per
    frame would cost more than the decode itself.
    """
    grids, strides = [], []
    in_h, in_w = input_size
    for stride in YOLOX_STRIDES:
        h, w = in_h // stride, in_w // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        strides.append(np.full((1, grid.shape[1], 1), stride))
    return (
        np.concatenate(grids, axis=1).astype(np.float32),
        np.concatenate(strides, axis=1).astype(np.float32),
    )


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS on xyxy boxes. Pure numpy: deterministic and testable.

    Preferred over cv2.dnn.NMSBoxesBatched so behaviour does not vary with the OpenCV
    build, and so the suppression logic can be unit-tested directly.
    """
    if boxes.shape[0] == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[best], x1[rest])
        yy1 = np.maximum(y1[best], y1[rest])
        xx2 = np.minimum(x2[best], x2[rest])
        yy2 = np.minimum(y2[best], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[best] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)

        order = rest[iou <= iou_threshold]
    return keep


def class_aware_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> list[int]:
    """NMS applied per class, so a person standing in front of a car is not suppressed."""
    keep: list[int] = []
    for cls in np.unique(class_ids):
        idx = np.flatnonzero(class_ids == cls)
        keep.extend(idx[i] for i in nms(boxes[idx], scores[idx], iou_threshold))
    return sorted(keep)


class YoloxOnnx:
    """ONNX Runtime session wrapper for a YOLOX detector."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.5,
        intra_op_threads: int | None = None,
        to_rgb: bool | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                f"See README for the download step and NOTICE.md for provenance."
            )

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.to_rgb = self._resolve_colour_order(self.model_path, to_rgb)

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads or default_intra_op_threads()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Ops run sequentially; parallelism comes from intra-op threading. Inter-op
        # parallelism on a single-branch CNN buys nothing and competes for the same cores.
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"]
        )

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        if not (isinstance(shape[2], int) and isinstance(shape[3], int)):
            raise ValueError(
                f"{self.model_path.name} has a dynamic input shape {shape}. "
                f"This wrapper assumes a fixed input size."
            )
        self.input_size: tuple[int, int] = (int(shape[2]), int(shape[3]))
        self.output_name = self.session.get_outputs()[0].name

        self._grids, self._strides = build_grids(self.input_size)
        self.threads = options.intra_op_num_threads

        log.info(
            "loaded %s input=%s threads=%d",
            self.model_path.name,
            self.input_size,
            self.threads,
        )

    @staticmethod
    def _resolve_colour_order(model_path: Path, explicit: bool | None) -> bool:
        """Colour order comes from the model's manifest, never from a guess.

        Making the manifest load-bearing for correctness - not just for licence
        provenance - means a model dropped in without one fails loudly at load time
        rather than quietly detecting 25% fewer people.
        """
        if explicit is not None:
            return explicit

        manifest_path = model_path.parent / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path} is not valid JSON: {exc}") from exc

            for entry in manifest.get("files", []):
                if entry.get("file") == model_path.name and "to_rgb" in entry:
                    return bool(entry["to_rgb"])
            if "to_rgb" in manifest:
                return bool(manifest["to_rgb"])

        raise ValueError(
            f"Colour order for {model_path.name} is unknown. Official Megvii YOLOX "
            f"exports expect BGR; OpenCV Zoo re-exports expect RGB. Record 'to_rgb' in "
            f"{manifest_path.name} or pass to_rgb= explicitly. Guessing costs ~25% recall."
        )

    # -- pipeline stages, split so the benchmark can time them individually ----

    def preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if self.to_rgb else frame_bgr
        padded, ratio = letterbox(image, self.input_size)
        blob = padded.transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(blob, dtype=np.float32), ratio

    def forward(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: blob})[0]

    def postprocess(self, raw: np.ndarray, ratio: float) -> Detections:
        predictions = raw[0].astype(np.float32, copy=True)

        # Decode: centres are grid-relative, extents are log-space. Both in input-image
        # pixels after multiplying by the anchor's stride.
        centres = (predictions[:, :2] + self._grids[0]) * self._strides[0]
        extents = np.exp(predictions[:, 2:4]) * self._strides[0]

        scores_all = predictions[:, 4:5] * predictions[:, 5:]
        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids]

        keep_mask = scores >= self.conf_threshold
        if not keep_mask.any():
            return Detections.empty()

        centres, extents = centres[keep_mask], extents[keep_mask]
        scores, class_ids = scores[keep_mask], class_ids[keep_mask]

        half = extents / 2.0
        boxes = np.concatenate([centres - half, centres + half], axis=1)
        boxes /= ratio  # undo letterbox scaling, back to original image pixels

        keep = class_aware_nms(boxes, scores, class_ids, self.nms_threshold)
        if not keep:
            return Detections.empty()

        return Detections(
            xyxy=boxes[keep].astype(np.float32),
            scores=scores[keep].astype(np.float32),
            class_ids=class_ids[keep].astype(np.int32),
        )

    def detect(self, frame_bgr: np.ndarray) -> Detections:
        blob, ratio = self.preprocess(frame_bgr)
        raw = self.forward(blob)
        detections = self.postprocess(raw, ratio)
        return self._clip_to_frame(detections, frame_bgr.shape[:2])

    @staticmethod
    def _clip_to_frame(detections: Detections, hw: tuple[int, int]) -> Detections:
        """Clamp boxes to the frame.

        Matters beyond tidiness: the boundary engine takes the foot point from y2, and
        a box extending past the bottom edge would place a person's feet outside the
        image and outside every zone.
        """
        if len(detections) == 0:
            return detections
        height, width = hw
        xyxy = detections.xyxy.copy()
        np.clip(xyxy[:, [0, 2]], 0, width - 1, out=xyxy[:, [0, 2]])
        np.clip(xyxy[:, [1, 3]], 0, height - 1, out=xyxy[:, [1, 3]])
        return Detections(xyxy, detections.scores, detections.class_ids)
