"""Dashboard-managed application settings (currently: the video-test section on/off).

`config/app.yaml` holds the engineering defaults; anything an admin should be able to flip
from the dashboard without editing a file lives here, in one MongoDB document, the same
"one small document, reloaded on a timer, previous good value kept if the read fails"
shape as `alerts/config_store.py`. A value never saved falls back to the app.yaml default.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

log = logging.getLogger("perimeter.appsettings")

DOC_ID = "global"
REFRESH_INTERVAL_S = 2.0


@dataclass(frozen=True)
class AppSettings:
    video_test_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"video_test_enabled": self.video_test_enabled}


def app_settings_from_dict(data: dict[str, Any], default: AppSettings) -> AppSettings:
    if not isinstance(data, dict):
        raise ValueError("app settings must be a mapping")
    value = data.get("video_test_enabled", default.video_test_enabled)
    if not isinstance(value, bool):
        raise ValueError("app settings: video_test_enabled must be true or false")
    return AppSettings(video_test_enabled=value)


class AppSettingsStore:
    def __init__(
        self,
        collection: Collection | None,
        default: AppSettings | None = None,
        refresh_interval: float = REFRESH_INTERVAL_S,
    ) -> None:
        """`collection=None` keeps settings in memory only (tests, no-database runs)."""
        self._collection = collection
        self._default = default or AppSettings()
        self._refresh_interval = refresh_interval
        self._lock = threading.Lock()
        self._settings = self._default
        self._last_poll = 0.0
        self.reload(force=True)

    def get(self) -> AppSettings:
        self.reload()
        with self._lock:
            return self._settings

    def reload(self, force: bool = False) -> None:
        if self._collection is None:
            return
        now = time.monotonic()
        if not force and now - self._last_poll < self._refresh_interval:
            return
        self._last_poll = now
        try:
            doc = self._collection.find_one({"id": DOC_ID})
        except PyMongoError as exc:
            log.warning("could not reload app settings, keeping the previous value: %s", exc)
            return
        if doc is None:
            return
        doc = dict(doc)
        doc.pop("_id", None)
        doc.pop("id", None)
        try:
            settings = app_settings_from_dict(doc, self._default)
        except ValueError as exc:
            log.error("app settings document rejected, keeping the previous value: %s", exc)
            return
        with self._lock:
            self._settings = settings

    def save(self, settings: AppSettings) -> None:
        if self._collection is not None:
            self._collection.replace_one(
                {"id": DOC_ID}, settings.to_dict() | {"id": DOC_ID}, upsert=True
            )
        with self._lock:
            self._settings = settings
        log.info("app settings saved: %s", settings.to_dict())
