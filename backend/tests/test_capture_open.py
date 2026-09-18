"""open_capture must bound FFmpeg's open/read timeouts.

OpenCV opens FFmpeg streams under a process-wide lock with a 30 s default interrupt, so
an unbounded open on one unreachable camera stalls every other camera's (re)connect and
the Test-connection probe behind it.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from perimeter.capture import source as source_module
from perimeter.capture.latest_slot import LatestSlot


class RecordingCapture:
    calls: list[tuple] = []

    def __init__(self, *args):
        RecordingCapture.calls.append(args)

    def isOpened(self):  # noqa: N802 - mirrors cv2's API
        return False


def test_network_sources_get_a_bounded_open_and_read_timeout(monkeypatch):
    RecordingCapture.calls = []
    monkeypatch.setattr(source_module.cv2, "VideoCapture", RecordingCapture)

    source_module.open_capture("rtsp://cam/stream", open_timeout_ms=5000, read_timeout_ms=9000)

    _url, backend, params = RecordingCapture.calls[0]
    assert backend == cv2.CAP_FFMPEG
    settings = dict(zip(params[::2], params[1::2], strict=True))
    assert settings[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC] == 5000
    assert settings[cv2.CAP_PROP_READ_TIMEOUT_MSEC] == 9000


def test_device_index_uses_the_native_backend(monkeypatch):
    RecordingCapture.calls = []
    monkeypatch.setattr(source_module.cv2, "VideoCapture", RecordingCapture)

    source_module.open_capture(0)

    assert RecordingCapture.calls[0] == (0,)


def test_capture_thread_reads_time_out_at_the_feed_lost_threshold(monkeypatch):
    RecordingCapture.calls = []
    monkeypatch.setattr(source_module.cv2, "VideoCapture", RecordingCapture)
    thread = source_module.CaptureThread("rtsp://cam/stream", slot=None, feed_lost_after_s=10.0)

    thread._open()

    params = RecordingCapture.calls[0][2]
    settings = dict(zip(params[::2], params[1::2], strict=True))
    assert settings[cv2.CAP_PROP_READ_TIMEOUT_MSEC] == 10_000


class InstantCapture:
    """Delivers frames as fast as they are read, like a local file or a fast camera."""

    def __init__(self, native_fps=0.0):
        self.native_fps = native_fps
        self.reads = 0

    def isOpened(self):  # noqa: N802
        return True

    def set(self, *_):
        return True

    def get(self, prop):
        return self.native_fps if prop == cv2.CAP_PROP_FPS else 640.0

    def read(self):
        self.reads += 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        pass


def _run_for(thread, seconds):
    thread.start()
    time.sleep(seconds)
    thread.stop()


def test_frames_are_published_at_decode_fps_not_at_source_rate(monkeypatch):
    fake = InstantCapture()
    monkeypatch.setattr(source_module, "open_capture", lambda *a, **k: fake)
    slot = LatestSlot()
    thread = source_module.CaptureThread("rtsp://cam/stream", slot, decode_fps=10)

    _run_for(thread, 1.0)

    assert fake.reads > 200  # a live source is still drained as fast as it delivers
    assert 7 <= thread.frames_decoded <= 13


def test_a_video_file_plays_back_at_its_native_rate(monkeypatch):
    fake = InstantCapture(native_fps=25.0)
    monkeypatch.setattr(source_module, "open_capture", lambda *a, **k: fake)
    thread = source_module.CaptureThread("clip.mp4", LatestSlot(), decode_fps=100)

    _run_for(thread, 1.0)

    assert 20 <= fake.reads <= 30
