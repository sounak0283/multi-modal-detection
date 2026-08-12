#!/usr/bin/env python3
"""Phase 2 verification: do track IDs survive occlusion in the real space?

PLAN.md section 9, Phase 2. This is not a demo - it is the measurement that validates
the boundary state machine's design. Section 6.3 initialises new tracks as "born already
settled" precisely because ByteTrack reassigns IDs after occlusion, and every reassigned
ID inside a zone would otherwise fire a phantom ENTRY. How often that happens in the
customer's actual space decides whether the hysteresis and initialisation are sufficient.

Run it against footage from the real camera, not a test clip.

Usage:
    python tools/track_preview.py --source clip.mp4 --out annotated.mp4
    python tools/track_preview.py --source rtsp://... --max-frames 900
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402

from fsbd.detect.person import PersonDetector  # noqa: E402
from fsbd.detect.yolox_onnx import configure_opencv_threads  # noqa: E402
from fsbd.track.tracker import PersonTracker  # noqa: E402

# Distinct, colour-blind-safe-ish palette cycled by track ID.
PALETTE = [
    (255, 128, 0), (0, 200, 255), (60, 220, 60), (200, 100, 255),
    (255, 80, 80), (255, 220, 0), (0, 160, 255), (180, 180, 180),
]


def colour_for(track_id: int) -> tuple[int, int, int]:
    return PALETTE[int(track_id) % len(PALETTE)]


def annotate(frame, tracked, detection_frame: bool):
    canvas = frame.copy()
    for box, track_id, score in zip(
        tracked.xyxy.astype(int), tracked.track_ids, tracked.scores, strict=True
    ):
        x1, y1, x2, y2 = box
        colour = colour_for(track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

        label = f"#{track_id} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            canvas, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

        # The point the boundary engine will actually test (PLAN.md section 6.1).
        cv2.circle(canvas, ((x1 + x2) // 2, y2), 4, (255, 255, 255), -1)
        cv2.circle(canvas, ((x1 + x2) // 2, y2), 4, colour, 1)

    tag = "DETECT" if detection_frame else "skip"
    cv2.putText(canvas, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def report(
    lifetimes: dict[int, int],
    concurrency: list[int],
    detection_frames: int,
    detection_fps: float,
) -> int:
    if not lifetimes:
        print("\nNo tracks were formed. Check that the source contains people.")
        return 1

    lengths = sorted(lifetimes.values())
    max_concurrent = max(concurrency) if concurrency else 0
    mean_concurrent = statistics.mean(concurrency) if concurrency else 0.0
    short_threshold = max(2, round(detection_fps))  # roughly one second of tracking
    fragments = sum(1 for n in lengths if n < short_threshold)

    print()
    print("=" * 72)
    print("PHASE 2 - track stability")
    print("=" * 72)
    print(f"detection frames processed : {detection_frames}")
    print(f"detection cadence          : {detection_fps:.1f} Hz")
    print()
    print(f"unique track IDs created   : {len(lifetimes)}")
    print(f"max concurrent tracks      : {max_concurrent}")
    print(f"mean concurrent tracks     : {mean_concurrent:.1f}")
    print()
    print(f"track length (detection frames): median {statistics.median(lengths)}, "
          f"min {lengths[0]}, max {lengths[-1]}")
    print(f"fragments (< {short_threshold} frames ~1s) : {fragments} of {len(lifetimes)} "
          f"({100 * fragments / len(lifetimes):.0f}%)")

    # Churn proxy: with perfect tracking, unique IDs ~ the number of distinct people
    # that ever appeared, which is bounded below by peak concurrency. Many IDs against
    # low concurrency means the tracker is re-labelling the same people.
    churn = len(lifetimes) / max_concurrent if max_concurrent else float("inf")
    print(f"churn ratio (IDs / peak concurrent) : {churn:.1f}")

    print()
    print("-" * 72)
    if churn <= 3.0 and fragments / len(lifetimes) <= 0.35:
        print("STABLE - section 6.3's born-settled initialisation is sufficient here.")
        return 0
    print(
        "CHURNY - IDs are being reassigned often. Before building the boundary\n"
        "engine on this, consider:\n"
        "  - raising lost_track_seconds (occlusion tolerance)\n"
        "  - raising min_consecutive_frames (fewer spurious short tracks)\n"
        "  - raising the person detector's confidence threshold\n"
        "Section 6.3 already suppresses phantom ENTRYs from ID churn, but high churn\n"
        "also means genuine crossings get missed when a track is reborn mid-zone."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="models/yolox_person/yolox_nano.onnx", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Write an annotated video.")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument(
        "--every-n", type=int, default=2, help="Person-detection cadence (PLAN.md section 4)."
    )
    parser.add_argument("--decode-fps", type=float, default=15.0)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--lost-track-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    configure_opencv_threads()

    source: str | int = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"cannot open source: {args.source}", file=sys.stderr)
        return 2

    detection_fps = args.decode_fps / args.every_n
    detector = PersonDetector(args.model, conf_threshold=args.conf)
    tracker = PersonTracker(
        detection_fps=detection_fps, lost_track_seconds=args.lost_track_seconds
    )

    writer: cv2.VideoWriter | None = None
    lifetimes: dict[int, int] = defaultdict(int)
    concurrency: list[int] = []
    tracked = None
    seq = 0
    detection_frames = 0

    while seq < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        seq += 1

        # The cadence from PLAN.md section 4. On skip frames the tracker is NOT called -
        # it is not merely skipped for speed, calling it would corrupt track state.
        is_detection_frame = seq % args.every_n == 0
        if is_detection_frame:
            detection_frames += 1
            tracked = tracker.update(detector.detect(frame))
            for track_id in tracked.track_ids:
                lifetimes[int(track_id)] += 1
            concurrency.append(len(tracked))

        if args.out is not None and tracked is not None:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.decode_fps, (w, h)
                )
            writer.write(annotate(frame, tracked, is_detection_frame))

        if seq % 100 == 0:
            print(f"  {seq} frames, {len(lifetimes)} IDs so far", file=sys.stderr)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nwrote {args.out}")

    return report(dict(lifetimes), concurrency, detection_frames, detection_fps)


if __name__ == "__main__":
    raise SystemExit(main())
