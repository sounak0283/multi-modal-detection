"""Tests for Pipeline's crowd-formation wiring (Expansion Plan Phase G).

Constructing a real `Pipeline` needs the real person-detection ONNX weights, same
`pytest.mark.skipif` pattern `test_firesmoke_pipeline.py`/`test_identity_pipeline.py`
use. `_process_crowd` is exercised directly with a hand-built `TrackedDetections`,
bypassing the full capture/detect/track path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perimeter.boundary.zones import ZoneStore, zone_from_dict
from perimeter.cameras.models import camera_from_dict
from perimeter.pipeline import Pipeline
from perimeter.settings import Settings
from perimeter.track.tracker import TrackedDetections

PERSON_MODEL = Path("models/yolox_person/yolox_nano.onnx")

pytestmark = pytest.mark.skipif(
    not PERSON_MODEL.is_file(),
    reason="yolox_nano.onnx not present - see README for the download step",
)

SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]  # covers the whole frame


class RecordingAlertBus:
    def __init__(self):
        self.published = []

    def publish(self, event):
        self.published.append(event)


def make_store():
    import mongomock

    db = mongomock.MongoClient()["perimeter_test"]
    return ZoneStore(db["zones"], refresh_interval=0)


def make_camera(enabled_modules):
    return camera_from_dict(
        {"id": "cam_01", "source_type": "webcam", "source": 0, "enabled_modules": enabled_modules}
    )


def make_pipeline(enabled_modules=("boundary", "crowd")):
    alerts = RecordingAlertBus()
    pipeline = Pipeline(
        Settings(), make_camera(list(enabled_modules)), make_store(), str(PERSON_MODEL),
        alerts=alerts,
    )
    return pipeline, alerts


def make_tracked(foot_points_px: list[tuple[float, float]]) -> TrackedDetections:
    """A person box per foot point: 20x50px, bottom-centre at the given point."""
    boxes = np.array(
        [[x - 10, y - 50, x + 10, y] for x, y in foot_points_px], dtype=np.float32
    )
    n = len(foot_points_px)
    return TrackedDetections(
        xyxy=boxes,
        scores=np.ones(n, np.float32),
        class_ids=np.zeros(n, np.int32),
        track_ids=np.arange(n, dtype=np.int32),
    )


CLUSTER = [(100, 100), (110, 105), (95, 110), (105, 95), (100, 90)]  # 5 tight points


def test_crowd_monitor_is_none_when_the_module_is_not_enabled():
    pipeline, _alerts = make_pipeline(enabled_modules=["boundary"])
    assert pipeline.crowd_monitor is None


def test_crowd_monitor_is_built_when_the_module_is_enabled():
    pipeline, _alerts = make_pipeline()
    assert pipeline.crowd_monitor is not None


def test_no_alert_without_a_crowd_threshold_on_the_zone():
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [zone_from_dict({"id": "z1", "type": "polygon", "points": SQUARE})], "cam_01"
    )

    tracked = make_tracked(CLUSTER)
    masked = np.zeros(len(CLUSTER), dtype=bool)
    for _ in range(5):
        pipeline._process_crowd(tracked, masked, 640, 480, ts=0.0)

    assert alerts.published == []


def test_fires_once_threshold_and_hysteresis_are_met():
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [
            zone_from_dict(
                {
                    "id": "z1", "type": "polygon", "points": SQUARE,
                    "crowd_threshold": 4, "crowd_min_frames": 3,
                }
            )
        ],
        "cam_01",
    )

    tracked = make_tracked(CLUSTER)
    masked = np.zeros(len(CLUSTER), dtype=bool)
    for _ in range(3):
        pipeline._process_crowd(tracked, masked, 640, 480, ts=0.0)

    assert len(alerts.published) == 1
    payload = alerts.published[0]
    assert payload["kind"] == "crowd"
    assert payload["zone_id"] == "z1"
    assert "5" in payload["message"]  # cluster size


def test_excluded_people_do_not_count_toward_a_crowd():
    """Exclusion-masked points are stripped before clustering - a queue line marked as
    an exclusion zone should never itself be read as a crowd."""
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [
            zone_from_dict(
                {
                    "id": "z1", "type": "polygon", "points": SQUARE,
                    "crowd_threshold": 4, "crowd_min_frames": 1,
                }
            )
        ],
        "cam_01",
    )

    tracked = make_tracked(CLUSTER)
    masked = np.ones(len(CLUSTER), dtype=bool)  # everyone suppressed
    pipeline._process_crowd(tracked, masked, 640, 480, ts=0.0)

    assert alerts.published == []


def test_crowd_zones_only_apply_to_polygon_type():
    pipeline, alerts = make_pipeline()
    pipeline.store.save(
        [
            zone_from_dict(
                {
                    "id": "z1", "type": "tripwire", "points": [[0.0, 0.5], [1.0, 0.5]],
                    "applies_to": [], "crowd_threshold": 2, "crowd_min_frames": 1,
                }
            )
        ],
        "cam_01",
    )

    tracked = make_tracked(CLUSTER)
    masked = np.zeros(len(CLUSTER), dtype=bool)
    pipeline._process_crowd(tracked, masked, 640, 480, ts=0.0)

    assert alerts.published == []
