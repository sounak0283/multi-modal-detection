"""Camera registry - MongoDB-backed (Expansion Plan, Phase A.2).

Replaces `main.py` constructing a single `Pipeline` from `.env`. One document per camera
in the `cameras` collection; `PipelineManager` (manager.py) polls `list()` and reconciles
running pipelines against it, so adding, editing or removing a camera from the dashboard
takes effect without a process restart.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pymongo.collection import Collection

from perimeter.cameras.models import Camera, camera_from_dict

log = logging.getLogger("perimeter.cameras.registry")

COLLECTION = "cameras"


class CameraRegistry:
    """Thread-safe CRUD over the `cameras` collection.

    A monotonic `version` counter (mirroring `boundary.zones.ZoneStore`'s pattern) lets
    `PipelineManager` cheaply notice "something changed" without diffing every field of
    every camera on each poll tick.
    """

    def __init__(self, collection: Collection) -> None:
        self._collection = collection
        self._lock = threading.Lock()
        self._version = 0
        self._collection.create_index("id", unique=True)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def _bump(self) -> None:
        with self._lock:
            self._version += 1

    def list(self) -> list[Camera]:
        return [camera_from_dict(doc) for doc in self._collection.find({}, {"_id": 0})]

    def get(self, camera_id: str) -> Camera | None:
        doc = self._collection.find_one({"id": camera_id}, {"_id": 0})
        return camera_from_dict(doc) if doc else None

    def create(self, camera: Camera) -> Camera:
        if self._collection.find_one({"id": camera.id}, {"_id": 1}):
            raise ValueError(f"camera {camera.id!r} already exists")
        self._collection.insert_one(camera.to_dict())
        self._bump()
        log.info("camera created: %s", camera)
        return camera

    def update(self, camera_id: str, camera: Camera) -> Camera:
        if camera.id != camera_id:
            raise ValueError("camera id cannot be changed")
        result = self._collection.replace_one({"id": camera_id}, camera.to_dict())
        if result.matched_count == 0:
            raise KeyError(f"camera {camera_id!r} not found")
        self._bump()
        log.info("camera updated: %s", camera)
        return camera

    def delete(self, camera_id: str) -> None:
        result = self._collection.delete_one({"id": camera_id})
        if result.deleted_count == 0:
            raise KeyError(f"camera {camera_id!r} not found")
        self._bump()
        log.info("camera deleted: %s", camera_id)

    def upsert_from_dict(self, data: dict[str, Any]) -> Camera:
        """Used only by `main.py`'s dev-mode bootstrap (see module docstring there)."""
        camera = camera_from_dict(data)
        self._collection.replace_one({"id": camera.id}, camera.to_dict(), upsert=True)
        self._bump()
        return camera
