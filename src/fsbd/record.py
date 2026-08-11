"""Continuous site recorder for negative collection (PLAN.md section 5.2, Phase 0).

Why this exists before any detector does
----------------------------------------
The fire/smoke model's false-alarm rate is decided by site-specific negatives -
welding sparks, forklift beacons, machinery steam, sunlight through skylights. Those
depend on the customer's operations happening, not on developer time. Collecting them
is a calendar dependency we do not control, so recording starts on day one and the
footage is mined later.

Design notes
------------
* Records at a LOW frame rate (default 4 fps) and re-encodes rather than stream-copying.
  The purpose is harvesting still frames for a training set, not producing evidence
  video, so 4 fps is ample and keeps weeks of footage affordable in both CPU and disk.
* Segments are fixed-duration files, so pruning is a whole-file delete and a corrupted
  tail costs one segment rather than the entire recording.
* Disk is a hard budget with oldest-first pruning. An unattended recorder that fills a
  customer's disk is a support incident.
* Reconnects with exponential backoff. Cameras drop; the recorder must not.

Recordings contain customer premises footage. They are git-ignored and must be handled
under the retention policy in PLAN.md section 10.3.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from fsbd.settings import parse_source, redact  # re-exported: shared with the app config

log = logging.getLogger("fsbd.record")

# Tried in order; the first fourcc the platform can actually open wins.
CODEC_CANDIDATES: tuple[tuple[str, str], ...] = (("mp4v", ".mp4"), ("XVID", ".avi"))

RECONNECT_BACKOFF_START_S = 2.0
RECONNECT_BACKOFF_MAX_S = 60.0
READ_FAILURE_GRACE = 15  # consecutive failed reads before declaring the feed lost


@dataclass(frozen=True)
class RecorderConfig:
    source: str | int
    outdir: Path
    fps: float = 4.0
    segment_minutes: float = 15.0
    disk_budget_gb: float = 50.0
    width: int | None = None
    height: int | None = None
    still_interval_s: float = 0.0  # 0 disables JPEG stills
    jpeg_quality: int = 92


def open_capture(source: str | int) -> cv2.VideoCapture:
    # CAP_FFMPEG is the backend that handles RTSP; let OpenCV choose for device indices.
    cap = (
        cv2.VideoCapture(source)
        if isinstance(source, int)
        else cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    )
    if cap.isOpened():
        # Keep the driver-side buffer minimal. We sample by wall clock, so a deep buffer
        # only adds latency and hands us stale frames after any hiccup.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def open_writer(path_stem: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    """Open a VideoWriter, falling back through codec candidates.

    OpenCV's VideoWriter fails by returning a non-opened object rather than raising, and
    which fourccs work varies by platform and build - so probe rather than assume.
    """
    for fourcc_name, suffix in CODEC_CANDIDATES:
        path = path_stem.with_suffix(suffix)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            log.debug("segment codec=%s path=%s", fourcc_name, path.name)
            return writer, path
        writer.release()
        path.unlink(missing_ok=True)
    raise RuntimeError(
        f"No usable video codec. Tried {[c for c, _ in CODEC_CANDIDATES]}. "
        f"Install a build of OpenCV with FFmpeg support."
    )


def directory_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def prune_to_budget(outdir: Path, budget_gb: float) -> int:
    """Delete oldest segments until the directory fits the budget. Returns files removed."""
    budget_bytes = int(budget_gb * 1024**3)
    files = sorted(
        (f for f in outdir.rglob("*") if f.is_file()),
        key=lambda f: f.stat().st_mtime,
    )
    total = sum(f.stat().st_size for f in files)
    removed = 0

    for f in files:
        if total <= budget_bytes:
            break
        size = f.stat().st_size
        try:
            f.unlink()
        except OSError as exc:
            log.warning("could not prune %s: %s", f.name, exc)
            continue
        total -= size
        removed += 1

    if removed:
        log.info("pruned %d old file(s) to stay under %.1f GB", removed, budget_gb)
    return removed


class Recorder:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self._stop = False
        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._segment_path: Path | None = None
        self._segment_started: float = 0.0
        self._last_write: float = 0.0
        self._last_still: float = 0.0
        self._frames_in_segment = 0

    def request_stop(self, *_: object) -> None:
        log.info("stop requested, finishing current segment")
        self._stop = True

    # -- segment lifecycle -------------------------------------------------

    def _start_segment(self, size: tuple[int, int]) -> None:
        self._close_segment()
        day_dir = self.cfg.outdir / datetime.now().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        stem = day_dir / datetime.now().strftime("seg_%H%M%S")
        self._writer, self._segment_path = open_writer(stem, self.cfg.fps, size)
        self._segment_started = time.monotonic()
        self._frames_in_segment = 0

    def _close_segment(self) -> None:
        if self._writer is not None:
            self._writer.release()
            log.info(
                "segment closed: %s (%d frames)",
                self._segment_path.name if self._segment_path else "?",
                self._frames_in_segment,
            )
        self._writer = None
        self._segment_path = None

    def _segment_expired(self) -> bool:
        return time.monotonic() - self._segment_started >= self.cfg.segment_minutes * 60

    # -- main loop ---------------------------------------------------------

    def run(self) -> int:
        self.cfg.outdir.mkdir(parents=True, exist_ok=True)
        backoff = RECONNECT_BACKOFF_START_S
        write_interval = 1.0 / self.cfg.fps

        while not self._stop:
            self._cap = open_capture(self.cfg.source)
            if not self._cap.isOpened():
                log.warning("cannot open source, retrying in %.0fs", backoff)
                self._sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)
                continue

            if self.cfg.width and self.cfg.height:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)

            log.info("connected to %s", redact(self.cfg.source))
            backoff = RECONNECT_BACKOFF_START_S
            consecutive_failures = 0

            while not self._stop:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= READ_FAILURE_GRACE:
                        log.warning("feed lost after %d failed reads", consecutive_failures)
                        break
                    time.sleep(0.1)
                    continue
                consecutive_failures = 0

                now = time.monotonic()
                if now - self._last_write < write_interval:
                    continue  # sample by wall clock, not by source frame rate
                self._last_write = now

                height, width = frame.shape[:2]
                if self._writer is None or self._segment_expired():
                    self._start_segment((width, height))
                    prune_to_budget(self.cfg.outdir, self.cfg.disk_budget_gb)

                assert self._writer is not None
                self._writer.write(frame)
                self._frames_in_segment += 1

                still_due = now - self._last_still >= self.cfg.still_interval_s
                if self.cfg.still_interval_s and still_due:
                    self._save_still(frame)
                    self._last_still = now

            self._close_segment()
            self._cap.release()
            self._cap = None

            if not self._stop:
                log.info("reconnecting in %.0fs", backoff)
                self._sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)

        self._close_segment()
        log.info("recorder stopped")
        return 0

    def _save_still(self, frame) -> None:
        stills_dir = self.cfg.outdir / "stills" / datetime.now().strftime("%Y-%m-%d")
        stills_dir.mkdir(parents=True, exist_ok=True)
        path = stills_dir / datetime.now().strftime("still_%H%M%S_%f.jpg")
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep so Ctrl-C during backoff does not hang for a minute."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self._stop:
            time.sleep(0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous low-fps site recorder for training-negative collection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="RTSP URL, video file, or local camera index (e.g. 0).",
    )
    parser.add_argument("--outdir", type=Path, default=Path("recordings"))
    parser.add_argument("--fps", type=float, default=4.0, help="Recording rate, not source rate.")
    parser.add_argument("--segment-minutes", type=float, default=15.0)
    parser.add_argument("--disk-budget-gb", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--still-interval",
        type=float,
        default=0.0,
        dest="still_interval_s",
        help="Also save a JPEG every N seconds. 0 disables.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = RecorderConfig(
        source=parse_source(args.source),
        outdir=args.outdir,
        fps=args.fps,
        segment_minutes=args.segment_minutes,
        disk_budget_gb=args.disk_budget_gb,
        width=args.width,
        height=args.height,
        still_interval_s=args.still_interval_s,
    )

    recorder = Recorder(cfg)
    signal.signal(signal.SIGINT, recorder.request_stop)
    signal.signal(signal.SIGTERM, recorder.request_stop)

    log.info(
        "recording %s at %.1f fps, %.0f min segments, %.0f GB budget -> %s",
        redact(cfg.source),
        cfg.fps,
        cfg.segment_minutes,
        cfg.disk_budget_gb,
        cfg.outdir,
    )
    try:
        return recorder.run()
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
