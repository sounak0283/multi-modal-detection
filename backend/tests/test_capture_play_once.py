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
        writer.write(np.full((48, 64, 3), (i * 20) % 256, np.uint8))
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


# -- non-realtime mode (the Video test page) --------------------------------------------


def test_non_realtime_finishes_much_faster_than_the_videos_own_length(tmp_path):
    """The whole point: a low, deliberately-slow native fps would take multiple real
    seconds to play out at realtime pacing - non-realtime must not wait for any of it."""
    video = tmp_path / "clip.avi"
    make_video(video, frames=40, fps=8)  # 5s of native playback at realtime pacing

    capture = CaptureThread(str(video), LatestSlot(), decode_fps=8, loop=False, realtime=False)
    started = time.time()
    capture.start()
    assert run_until(lambda: capture.finished, timeout=3.0)
    elapsed = time.time() - started
    capture.stop()

    assert elapsed < 2.0  # well under the 5s the video's own length would otherwise force


def test_non_realtime_still_samples_at_the_configured_decode_fps(tmp_path):
    """Not pacing reads to real time must not mean publishing every frame either - the
    proportion kept has to still match decode_fps against the file's OWN timeline, or
    downstream hysteresis (tracker occlusion tolerance, zone entry frame-counts) would be
    silently recalibrated against a file that happens to hand back frames faster."""
    video = tmp_path / "clip.avi"
    make_video(video, frames=40, fps=20)  # native: 20fps, decode_fps below asks for 1-in-4

    capture = CaptureThread(str(video), LatestSlot(), decode_fps=5, loop=False, realtime=False)
    capture.start()
    assert run_until(lambda: capture.finished, timeout=3.0)
    capture.stop()

    assert capture.frames_decoded == 10  # 40 read frames / (20fps -> 5fps = every 4th) = 10


def test_non_realtime_keeps_the_right_rate_even_when_it_does_not_divide_evenly(tmp_path):
    """Regression: an early version used `round(file_fps / decode_fps)` as a fixed
    integer stride - for a ratio like 20/12 that rounds to 2, silently sampling at an
    effective 10fps instead of the requested 12. The fractional accumulator here must
    track the true average rate instead."""
    video = tmp_path / "clip.avi"
    make_video(video, frames=100, fps=20)  # native 20fps, asking for 12fps (ratio 1.667)

    capture = CaptureThread(str(video), LatestSlot(), decode_fps=12, loop=False, realtime=False)
    capture.start()
    assert run_until(lambda: capture.finished, timeout=3.0)
    capture.stop()

    # Exact accumulator result for 100 reads at step=20/12: 60 keeps. A fixed stride of
    # 2 (round(20/12)) would instead give 50 - the bug this test exists to catch.
    assert capture.frames_decoded == 60
