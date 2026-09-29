"""PPE helmet demo - CPU only (onnxruntime, opencv-python, numpy).

Pipeline (same as the product's stage-2 design):
  frame -> person detector (YOLOX-nano, 416) -> each person >= 96 px tall is cropped with
  padding -> PPE detector (YOLOX-nano, 320) finds head / helmet in the crop -> boxes mapped
  back to the frame -> IoU tracker -> "NO HELMET" only after 3 of the last 5 frames.

  python demo.py --source 0                      # webcam 0
  python demo.py --source video.mp4 --save out.mp4
  python demo.py --source rtsp://user:pass@cam/stream

Keys: q / Esc quits.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
STRIDES = (8, 16, 32)

# Person crop - identical to the training crops (perimeter/detect/ppe_crop.py).
PAD_X, PAD_TOP, PAD_BOTTOM = 0.10, 0.12, 0.04
MIN_PERSON_HEIGHT_PX = 96

GREEN, RED, GREY, WHITE, YELLOW = (0, 200, 0), (0, 0, 255), (160, 160, 160), (255, 255, 255), (0, 220, 255)


class Yolox:
    """Minimal YOLOX ONNX runner: letterbox (pad 114, top-left), BGR, raw head output."""

    def __init__(self, path: Path, threads: int):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name, (self.h, self.w) = inp.name, inp.shape[2:4]
        grids, strides = [], []
        for s in STRIDES:
            xv, yv = np.meshgrid(np.arange(self.w // s), np.arange(self.h // s))
            g = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(g)
            strides.append(np.full((g.shape[0], 1), s))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def __call__(self, img: np.ndarray, conf: float, nms_iou: float = 0.45):
        """-> (boxes xyxy in img pixels, scores, class ids)."""
        r = min(self.h / img.shape[0], self.w / img.shape[1])
        nw, nh = int(img.shape[1] * r), int(img.shape[0] * r)
        pad = np.full((self.h, self.w, 3), 114, np.uint8)
        if nw > 0 and nh > 0:
            pad[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(pad.transpose(2, 0, 1)[None], dtype=np.float32)
        p = self.sess.run(None, {self.name: blob})[0][0]
        xy = (p[:, :2] + self.grid) * self.stride
        wh = np.exp(p[:, 2:4]) * self.stride
        cls_scores = p[:, 4:5] * p[:, 5:]
        cls = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(cls)), cls]
        m = score >= conf
        if not m.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        boxes = np.concatenate([xy[m] - wh[m] / 2, xy[m] + wh[m] / 2], 1) / r
        score, cls = score[m], cls[m]
        keep = []
        for c in np.unique(cls):
            idx = np.flatnonzero(cls == c)
            keep += [idx[i] for i in nms(boxes[idx], score[idx], nms_iou)]
        keep = sorted(keep)
        return boxes[keep], score[keep], cls[keep]


def nms(b: np.ndarray, s: np.ndarray, thr: float) -> list[int]:
    order, keep = s.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        ix = np.maximum(0, np.minimum(b[i, 2], b[rest, 2]) - np.maximum(b[i, 0], b[rest, 0]))
        iy = np.maximum(0, np.minimum(b[i, 3], b[rest, 3]) - np.maximum(b[i, 1], b[rest, 1]))
        inter = ix * iy
        union = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]) + (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1]) - inter
        order = rest[inter / np.maximum(union, 1e-9) <= thr]
    return keep


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class Track:
    def __init__(self, tid, box, is_head):
        self.id, self.box, self.missed = tid, box, 0
        self.hist = deque([is_head], maxlen=5)
        self.flagged = False


class Tracker:
    """Greedy IoU tracker over head/helmet boxes. NO HELMET = bare head in >= 3 of last 5 frames."""

    def __init__(self, iou_thr=0.3, max_missed=5):
        self.tracks: list[Track] = []
        self.next_id, self.iou_thr, self.max_missed = 1, iou_thr, max_missed
        self.violation_ids: set[int] = set()

    def update(self, dets):  # dets: list of (box, is_head)
        pairs = sorted(((iou(t.box, d[0]), ti, di) for ti, t in enumerate(self.tracks) for di, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for v, ti, di in pairs:
            if v < self.iou_thr or ti in used_t or di in used_d:
                continue
            t = self.tracks[ti]
            t.box, t.missed = dets[di][0], 0
            t.hist.append(dets[di][1])
            used_t.add(ti)
            used_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(self.next_id, d[0], d[1]))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for t in self.tracks:
            t.flagged = sum(t.hist) >= 3
            if t.flagged:
                self.violation_ids.add(t.id)
        return [t for t in self.tracks if t.missed == 0]


def detect_ppe(frame, person, ppe, cfg):
    fh, fw = frame.shape[:2]
    boxes, scores, cls = person(frame, cfg["person_conf"])
    out = []  # (box in frame, score, class_name)
    for (x1, y1, x2, y2), c in zip(boxes, cls):
        if c != 0 or (y2 - y1) < MIN_PERSON_HEIGHT_PX:  # COCO class 0 = person
            continue
        w, h = x2 - x1, y2 - y1
        rx1, ry1 = int(max(0, x1 - PAD_X * w)), int(max(0, y1 - PAD_TOP * h))
        rx2, ry2 = int(min(fw, x2 + PAD_X * w)), int(min(fh, y2 + PAD_BOTTOM * h))
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue
        b, s, k = ppe(crop, min(cfg["thresholds"].values()))
        for bb, ss, kk in zip(b, s, k):
            name = cfg["classes"][int(kk)]
            if ss >= cfg["thresholds"][name]:
                out.append((bb + np.array([rx1, ry1, rx1, ry1], np.float32), float(ss), name))
    # neighbours' crops overlap: merge duplicates across crops, per class
    merged = []
    for name in set(o[2] for o in out):
        items = [o for o in out if o[2] == name]
        b = np.array([o[0] for o in items])
        s = np.array([o[1] for o in items])
        merged += [items[i] for i in nms(b, s, 0.5)]
    # a head covered by a helmet box is the same head: keep the stronger label
    final = []
    for o in merged:
        dup = [p for p in merged if p is not o and p[2] != o[2] and iou(p[0], o[0]) > 0.5]
        if all(o[1] >= p[1] for p in dup):
            final.append(o)
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PPE helmet demo (CPU).")
    ap.add_argument("--source", required=True, help="webcam index, video file, or RTSP URL")
    ap.add_argument("--save", help="optional output mp4 path")
    ap.add_argument("--no-show", action="store_true", help="do not open a window (headless)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    person = Yolox(HERE / cfg["person_model"], args.threads)
    ppe = Yolox(HERE / cfg["ppe_model"], args.threads)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")
    writer = None
    tracker = Tracker()
    fps, n = 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        live = tracker.update([(d[0], d[2] == "head") for d in detect_ppe(frame, person, ppe, cfg)])
        dt = time.perf_counter() - t0
        fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
        for t in live:
            x1, y1, x2, y2 = (int(v) for v in t.box)
            if t.flagged:
                colour, label = RED, "NO HELMET"
            elif not t.hist[-1]:
                colour, label = GREEN, "HELMET"
            else:
                colour, label = GREY, "checking"
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        now_violations = sum(t.flagged for t in live)
        cv2.rectangle(frame, (0, 0), (360, 54), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS {fps:5.1f}  (CPU)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        cv2.putText(frame, f"NO HELMET now: {now_violations}  total: {len(tracker.violation_ids)}", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED if now_violations else WHITE, 2)
        if args.save:
            if writer is None:
                fps_in = cap.get(cv2.CAP_PROP_FPS) or 10
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("PPE demo (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    cap.release()
    if writer:
        writer.release()
    if not args.no_show:  # headless OpenCV builds have no window functions at all
        cv2.destroyAllWindows()
    print(f"frames {n}, final FPS {fps:.1f}, NO HELMET tracks {len(tracker.violation_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
