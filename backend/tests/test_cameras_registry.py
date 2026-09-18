"""Tests for the camera domain model, registry and PipelineManager reconciliation
(Expansion Plan Phase A.2).

`PipelineManager` tests substitute a `FakePipeline` for the real `Pipeline` (monkeypatch
on `perimeter.cameras.manager.Pipeline`) rather than constructing real ones - a real `Pipeline`
opens camera hardware and loads an ONNX model in its constructor, neither of which
belongs in a reconciliation-logic unit test. This mirrors the `FakeDatabase` pattern
already used in test_alerts.py.
"""

from __future__ import annotations

import mongomock
import pytest

from perimeter.boundary.zones import ZoneStore
from perimeter.cameras import manager as manager_module
from perimeter.cameras.models import MODULE_BOUNDARY, Camera, SourceType, camera_from_dict
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import Settings

# -- Camera model -----------------------------------------------------------


def test_minimal_webcam_camera_gets_sensible_defaults():
    camera = camera_from_dict({"id": "cam_01"})

    assert camera.source_type is SourceType.WEBCAM
    assert camera.enabled is True
    assert camera.enabled_modules == frozenset()


def test_no_module_runs_unless_selected_including_boundary():
    camera = camera_from_dict({"id": "cam_01", "enabled_modules": []})
    assert camera.enabled_modules == frozenset()


def test_fire_smoke_runs_without_boundary():
    camera = camera_from_dict({"id": "cam_01", "enabled_modules": ["fire_smoke"]})
    assert camera.enabled_modules == frozenset({"fire_smoke"})


def test_modules_that_need_person_tracking_enable_boundary():
    camera = camera_from_dict({"id": "cam_01", "enabled_modules": ["crowd", "identity"]})
    assert MODULE_BOUNDARY in camera.enabled_modules


def test_unknown_module_is_rejected():
    with pytest.raises(ValueError, match="unknown module"):
        camera_from_dict({"id": "cam_01", "enabled_modules": ["teleportation"]})


def test_rtsp_source_without_a_url_is_rejected():
    with pytest.raises(ValueError, match="rtsp_url"):
        camera_from_dict({"id": "cam_01", "source_type": "rtsp"})


def test_missing_id_is_rejected():
    with pytest.raises(ValueError, match="missing 'id'"):
        camera_from_dict({})


def test_resolution_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="width"):
        camera_from_dict({"id": "cam_01", "width": 4})


def test_public_dict_redacts_the_rtsp_url_but_reports_it_is_set():
    camera = camera_from_dict(
        {"id": "cam_01", "source_type": "rtsp", "rtsp_url": "rtsp://admin:secret@10.0.0.5/s1"}
    )
    public = camera.to_public_dict()

    assert "secret" not in str(public)
    assert public["rtsp_url_set"] is True
    assert "rtsp_url" not in public


def test_to_dict_round_trips_through_camera_from_dict():
    original = camera_from_dict(
        {"id": "cam_01", "name": "Gate", "enabled_modules": ["crowd", "fire_smoke"]}
    )
    restored = camera_from_dict(original.to_dict())
    assert restored == original


# -- CameraRegistry -----------------------------------------------------------


@pytest.fixture
def registry() -> CameraRegistry:
    return CameraRegistry(mongomock.MongoClient()["perimeter_test"]["cameras"])


def test_create_and_get(registry):
    registry.create(camera_from_dict({"id": "cam_01", "name": "Gate"}))
    assert registry.get("cam_01").name == "Gate"


def test_get_missing_camera_returns_none(registry):
    assert registry.get("does-not-exist") is None


def test_create_duplicate_id_is_rejected(registry):
    registry.create(camera_from_dict({"id": "cam_01"}))
    with pytest.raises(ValueError, match="already exists"):
        registry.create(camera_from_dict({"id": "cam_01"}))


def test_update_replaces_the_stored_camera(registry):
    registry.create(camera_from_dict({"id": "cam_01", "name": "Old"}))
    registry.update("cam_01", camera_from_dict({"id": "cam_01", "name": "New"}))
    assert registry.get("cam_01").name == "New"


def test_update_missing_camera_raises_key_error(registry):
    with pytest.raises(KeyError):
        registry.update("does-not-exist", camera_from_dict({"id": "does-not-exist"}))


def test_delete_removes_the_camera(registry):
    registry.create(camera_from_dict({"id": "cam_01"}))
    registry.delete("cam_01")
    assert registry.get("cam_01") is None


def test_delete_missing_camera_raises_key_error(registry):
    with pytest.raises(KeyError):
        registry.delete("does-not-exist")


def test_list_returns_every_camera(registry):
    registry.create(camera_from_dict({"id": "cam_01"}))
    registry.create(camera_from_dict({"id": "cam_02"}))
    assert {c.id for c in registry.list()} == {"cam_01", "cam_02"}


def test_version_bumps_on_every_write(registry):
    before = registry.version
    registry.create(camera_from_dict({"id": "cam_01"}))
    assert registry.version > before


# -- PipelineManager reconciliation ------------------------------------------


class FakePipeline:
    """Records lifecycle calls instead of touching real camera hardware or a model."""

    instances: list[FakePipeline] = []

    def __init__(
        self,
        settings,
        camera: Camera,
        store,
        model_path,
        alerts=None,
        firesmoke_model_path=None,
        identity_resolver=None,
        ppe_model_path=None,
    ):
        self.camera = camera
        self.started = False
        self.stopped = False
        self.reconfigured_with: Camera | None = None
        FakePipeline.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True

    def reconfigure(self, camera: Camera) -> None:
        self.camera = camera
        self.reconfigured_with = camera


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch):
    FakePipeline.instances = []
    monkeypatch.setattr(manager_module, "Pipeline", FakePipeline)


@pytest.fixture
def manager_registry_store():
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"])
    manager = manager_module.PipelineManager(Settings(), registry, store, "unused-model.onnx")
    return manager, registry


def test_start_launches_a_pipeline_for_each_enabled_camera(manager_registry_store):
    manager, registry = manager_registry_store
    registry.create(camera_from_dict({"id": "cam_01"}))
    registry.create(camera_from_dict({"id": "cam_02", "enabled": False}))

    manager.start()

    assert set(manager.all()) == {"cam_01"}
    assert manager.get("cam_01").started is True


def test_notify_camera_changed_starts_a_newly_created_camera(manager_registry_store):
    manager, registry = manager_registry_store
    manager.start()

    registry.create(camera_from_dict({"id": "cam_02"}))
    manager.notify_camera_changed("cam_02")

    assert manager.get("cam_02") is not None
    assert manager.get("cam_02").started is True


def test_notify_camera_changed_is_a_noop_before_start(manager_registry_store):
    """A camera created through the API before the manager is running (e.g.
    --no-pipeline) must not open real hardware."""
    manager, registry = manager_registry_store
    registry.create(camera_from_dict({"id": "cam_01"}))

    manager.notify_camera_changed("cam_01")

    assert manager.all() == {}


def test_deleting_a_camera_stops_its_pipeline(manager_registry_store):
    manager, registry = manager_registry_store
    registry.create(camera_from_dict({"id": "cam_01"}))
    manager.start()
    pipeline = manager.get("cam_01")

    registry.delete("cam_01")
    manager.notify_camera_changed("cam_01")

    assert manager.all() == {}
    assert pipeline.stopped is True


def test_disabling_a_camera_stops_its_pipeline(manager_registry_store):
    manager, registry = manager_registry_store
    registry.create(camera_from_dict({"id": "cam_01"}))
    manager.start()
    pipeline = manager.get("cam_01")

    registry.update("cam_01", camera_from_dict({"id": "cam_01", "enabled": False}))
    manager.notify_camera_changed("cam_01")

    assert manager.all() == {}
    assert pipeline.stopped is True


def test_changing_a_reconnect_field_reconfigures_in_place(manager_registry_store):
    manager, registry = manager_registry_store
    registry.create(camera_from_dict({"id": "cam_01", "width": 1280}))
    manager.start()
    pipeline = manager.get("cam_01")

    updated = camera_from_dict({"id": "cam_01", "width": 640})
    registry.update("cam_01", updated)
    manager.notify_camera_changed("cam_01")

    assert manager.get("cam_01") is pipeline, "the same Pipeline instance survives"
    assert pipeline.reconfigured_with == updated


def test_changing_enabled_modules_reconfigures_in_place(manager_registry_store):
    """Expansion Plan Phase E: toggling fire_smoke on the Cameras page for an
    already-running camera must not be a silent no-op until a full restart."""
    manager, registry = manager_registry_store
    registry.create(camera_from_dict({"id": "cam_01", "enabled_modules": []}))
    manager.start()
    pipeline = manager.get("cam_01")

    updated = camera_from_dict({"id": "cam_01", "enabled_modules": ["fire_smoke"]})
    registry.update("cam_01", updated)
    manager.notify_camera_changed("cam_01")

    assert pipeline.reconfigured_with == updated


def test_changing_a_non_reconnect_field_does_not_reconfigure(manager_registry_store):
    """Only source/resolution/fps changes need a capture-thread restart; something like
    a name change should not disturb a running pipeline."""
    manager, registry = manager_registry_store
    registry.create(camera_from_dict({"id": "cam_01", "name": "Old name"}))
    manager.start()
    pipeline = manager.get("cam_01")

    registry.update("cam_01", camera_from_dict({"id": "cam_01", "name": "New name"}))
    manager.notify_camera_changed("cam_01")

    assert pipeline.reconfigured_with is None


def test_stop_stops_every_running_pipeline(manager_registry_store):
    manager, registry = manager_registry_store
    registry.create(camera_from_dict({"id": "cam_01"}))
    manager.start()
    pipeline = manager.get("cam_01")

    manager.stop()

    assert pipeline.stopped is True
    assert manager.all() == {}
