"""JPEG pre-roll ring buffer (Expansion Plan Phase C).

Feeds `evidence.writer.EvidenceWriter`: on every alert, the writer reads whatever is
currently in a camera's buffer and muxes it into a clip. Storing JPEG bytes rather than
raw frames is deliberate (PLAN.md section 4's ring-buffer sizing note) - 15 s of 720p
JPEG is ~10-15 MB; the same window of raw frames would be closer to 900 MB.
"""

from __future__ import annotations

import threading
from collections import deque


class JpegRingBuffer:
    """Thread-safe, time-bounded buffer of `(ts, jpeg_bytes)`.

    Bounded by age (`seconds`), not by count - the frame rate is not fixed (a stalled
    decode thread should shrink the buffer's frame count, not silently discard older
    seconds and admit newer ones out of a fixed-size ring).
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._frames: deque[tuple[float, bytes]] = deque()
        self._lock = threading.Lock()

    def append(self, ts: float, jpeg: bytes) -> None:
        with self._lock:
            self._frames.append((ts, jpeg))
            cutoff = ts - self.seconds
            while self._frames and self._frames[0][0] < cutoff:
                self._frames.popleft()

    def snapshot(self) -> list[tuple[float, bytes]]:
        """A copy of everything currently buffered, oldest first."""
        with self._lock:
            return list(self._frames)

    def closest_to(self, ts: float) -> bytes | None:
        """The JPEG whose timestamp is nearest `ts` - used to pick the alert snapshot."""
        frames = self.snapshot()
        if not frames:
            return None
        return min(frames, key=lambda item: abs(item[0] - ts))[1]

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
