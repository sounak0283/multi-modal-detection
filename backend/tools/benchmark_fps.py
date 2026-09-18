#!/usr/bin/env python3
"""Phase 1 gate: measure real inference throughput on the target machine.

PLAN.md section 9.1 - "Measure real fps at the end of Phase 1, before building anything
else." If the person detector cannot sustain the cadence in section 4, the whole
threading model needs rebalancing, and discovering that now is cheap.

PLAN.md section 9.2 - quantisation is an EXPERIMENT, not a mandate. The commonly quoted
2-3x INT8 speedup assumes AVX512-VNNI. This tool exists to replace that assumption with
a number from this CPU.

Usage:
    python tools/benchmark_fps.py --models models/yolox_person/*.onnx
    python tools/benchmark_fps.py --models a.onnx b.onnx --source clip.mp4 --runs 50
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402

from perimeter.detect.yolox_onnx import YoloxOnnx, configure_opencv_threads  # noqa: E402

# From PLAN.md section 4: decode at 12-15 fps, person inference every 2nd frame.
# That is the throughput the pipeline design assumes.
REQUIRED_PERSON_FPS = 8.0


@dataclass
class Result:
    name: str
    size_mb: float
    input_size: tuple[int, int]
    threads: int
    pre_ms: list[float]
    fwd_ms: list[float]
    post_ms: list[float]
    detections: int

    @property
    def total_ms(self) -> list[float]:
        return [p + f + q for p, f, q in zip(self.pre_ms, self.fwd_ms, self.post_ms, strict=True)]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.total_ms)

    @property
    def p95_ms(self) -> float:
        ordered = sorted(self.total_ms)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    @property
    def fps(self) -> float:
        return 1000.0 / self.median_ms if self.median_ms else 0.0


def load_frames(source: str | None, count: int) -> list[np.ndarray]:
    """Real frames if given, otherwise synthetic noise.

    Content barely affects convolution cost, but it does affect postprocess: an empty
    scene skips NMS entirely and flatters the numbers. Noise produces a realistic
    number of candidate boxes, so synthetic is a fair-to-pessimistic default.
    """
    if source is None:
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(count)]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"--source not found: {path}")

    image = cv2.imread(str(path))
    if image is not None:
        return [image] * count

    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"could not read any frames from {path}")
    # Loop the clip if it is shorter than the requested run count.
    return [frames[i % len(frames)] for i in range(count)]


def benchmark(
    model_path: Path, frames: list[np.ndarray], warmup: int, threads: int | None
) -> Result:
    model = YoloxOnnx(model_path, intra_op_threads=threads)

    # ORT lazily allocates arenas and picks kernels on the first passes; timing those
    # would report startup cost as steady-state cost.
    for frame in frames[:warmup]:
        model.detect(frame)

    pre_ms, fwd_ms, post_ms = [], [], []
    detections = 0

    for frame in frames:
        t0 = time.perf_counter()
        blob, ratio = model.preprocess(frame)
        t1 = time.perf_counter()
        raw = model.forward(blob)
        t2 = time.perf_counter()
        result = model.postprocess(raw, ratio)
        t3 = time.perf_counter()

        pre_ms.append((t1 - t0) * 1000)
        fwd_ms.append((t2 - t1) * 1000)
        post_ms.append((t3 - t2) * 1000)
        detections += len(result)

    return Result(
        name=model_path.name,
        size_mb=model_path.stat().st_size / 1024**2,
        input_size=model.input_size,
        threads=model.threads,
        pre_ms=pre_ms,
        fwd_ms=fwd_ms,
        post_ms=post_ms,
        detections=detections,
    )


def report(results: list[Result], runs: int) -> int:
    print()
    print("=" * 78)
    print("PHASE 1 GATE - person detector throughput")
    print("=" * 78)
    print(f"machine   : {platform.processor() or platform.machine()}")
    print(f"python    : {platform.python_version()}   opencv: {cv2.__version__}")
    print(f"runs      : {runs} timed passes per model")
    print()

    header = f"{'model':<44}{'input':>10}{'median':>9}{'p95':>8}{'fps':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        size = f"{r.input_size[0]}x{r.input_size[1]}"
        print(f"{r.name:<44}{size:>10}{r.median_ms:>8.1f}m{r.p95_ms:>7.1f}{r.fps:>7.1f}")

    print()
    print("stage breakdown (median ms)")
    print("-" * len(header))
    for r in results:
        pre = statistics.median(r.pre_ms)
        fwd = statistics.median(r.fwd_ms)
        post = statistics.median(r.post_ms)
        print(f"{r.name:<44}{'pre':>7} {pre:5.1f}{'  fwd':>7} {fwd:6.1f}{'  post':>7} {post:5.1f}")

    # Quantisation verdict - the reason PLAN.md section 9.2 stopped mandating INT8.
    fp32 = next((r for r in results if "int8" not in r.name.lower()), None)
    int8 = next((r for r in results if "int8" in r.name.lower()), None)
    if fp32 and int8:
        speedup = fp32.median_ms / int8.median_ms if int8.median_ms else 0.0
        print()
        print(f"INT8 speedup vs FP32: {speedup:.2f}x")
        if speedup < 1.5:
            print(
                "  -> Below the 2-3x commonly quoted. Consistent with a CPU lacking\n"
                "     AVX512-VNNI. Do not assume INT8 is the answer on this hardware."
            )

    best = max(results, key=lambda r: r.fps)
    print()
    print("-" * len(header))
    print(f"VERDICT (best: {best.name} at {best.fps:.1f} fps, need >= {REQUIRED_PERSON_FPS:.0f})")
    if best.fps >= REQUIRED_PERSON_FPS:
        print("  PASS - the section 4 cadence table holds at this input size.")
        return 0
    print(
        "  FAIL - rebalance before building further (PLAN.md section 9.1):\n"
        "     - drop decode fps, or\n"
        "     - widen the person frame-skip beyond n%2, or\n"
        "     - re-export at a smaller input size (YOLOX-nano @416)."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--source", default=None, help="Image or video. Default: synthetic noise.")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)

    configure_opencv_threads()
    frames = load_frames(args.source, args.runs + args.warmup)

    results = []
    for model_path in args.models:
        if not model_path.is_file():
            print(f"skipping missing model: {model_path}", file=sys.stderr)
            continue
        print(f"benchmarking {model_path.name} ...", file=sys.stderr)
        results.append(benchmark(model_path, frames[: args.runs], args.warmup, args.threads))

    if not results:
        print("No models benchmarked.", file=sys.stderr)
        return 2
    return report(results, args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
