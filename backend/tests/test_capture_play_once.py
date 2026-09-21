"""A file source can play once and stop (used by the dashboard's video-test page), instead of
restarting from the top the way the live-camera reconnect loop does."""

from __future__ import annotations

import time

import cv2
import numpy as np

from perimeter.capture.latest_slot import LatestSlot
from perimeter.capture.source import CaptureThread


def make_video(path, frames=8, fps=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()


def run_until(cond, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_a_non_looping_file_stops_at_the_end(tmp_path):
    video = tmp_path / "clip.avi"
    make_video(video)
    capture = CaptureThread(str(video), LatestSlot(), decode_fps=30, loop=False)
    capture.start()
    assert run_until(lambda: capture.finished)
    frames = capture.frames_decoded
    time.sleep(0.5)
    capture.stop()

    assert capture.frames_decoded == frames  # did not restart from the top
    assert frames >= 1


def test_a_looping_file_keeps_going(tmp_path):
    video = tmp_path / "clip.avi"
    make_video(video)
    capture = CaptureThread(str(video), LatestSlot(), decode_fps=30, loop=True)
    capture.start()
    assert run_until(lambda: capture.frames_decoded >= 10, timeout=10.0)  # more than 8 = restarted
    capture.stop()
    assert not capture.finished
