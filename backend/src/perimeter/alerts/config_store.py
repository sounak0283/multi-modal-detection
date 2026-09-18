"""MongoDB-backed alert configuration (Expansion Plan Phase D).

Structurally copied from `boundary/zones.py:ZoneStore` - one document instead of
per-camera documents (rules are edited together as a small set in one dashboard page,
the same "hold the whole desired state, PUT it back" pattern `Boundaries.jsx` already
uses via `ZoneStore.save`), same throttled-reload-for-hot-editing behaviour, same
reject-a-bad-document-and-keep-the-previous-good-one safety property - a bad edit must
never silently turn off alerting.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from perimeter.alerts.cooldown import AlertConfig, alert_config_from_dict

log = logging.getLogger("perimeter.alerts.config_store")

DOC_ID = "global"
REFRESH_INTERVAL_S = 2.0


class AlertConfigStore:
    def __init__(
        self, collection: Collection, refresh_interval: float = REFRESH_INTERVAL_S
    ) -> None:
        self._collection = collection
        self._refresh_interval = refresh_interval
        self._lock = threading.Lock()
        self._config = AlertConfig()
        self._last_error: str | None = None
        self._version = 0
        self._last_poll = 0.0
        self.reload(force=True)

    # -- reading -----------------------------------------------------------

    def get(self) -> AlertConfig:
        with self._lock:
            return self._config

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def reload(self, force: bool = False) -> bool:
        """Re-read from MongoDB, throttled to at most once per `refresh_interval`.

        A document that fails validation is rejected and the previous good config is
        kept - a bad edit must never silently disarm alerting.
        """
        now = time.monotonic()
        if not force and now - self._last_poll < self._refresh_interval:
            return False
        self._last_poll = now

        try:
            doc = self._collection.find_one({"id": DOC_ID})
        except PyMongoError as exc:
            # Alert routing needs no database to send an email. Keep the last good
            # config so a critical alert raised during an outage is still delivered.
            with self._lock:
                self._last_error = f"alert config storage unavailable: {exc.__class__.__name__}"
            log.warning("could not reload alert config, keeping previous config: %s", exc)
            return False
        if doc is None:
            return False  # nothing saved yet - keep the in-memory default (empty) config

        doc = dict(doc)
        doc.pop("_id", None)
        doc.pop("id", None)
        try:
            config = alert_config_from_dict(doc)
        except ValueError as exc:
            with self._lock:
                self._last_error = str(exc)
            log.error("alert_config document rejected, keeping previous config: %s", exc)
            return False

        with self._lock:
            self._config = config
            self._last_error = None
            self._version += 1
        log.info("alert config loaded: %d sink rule(s)", len(config.sinks))
        return True

    # -- writing -----------------------------------------------------------

    def save(self, config: AlertConfig) -> None:
        """Replace the whole document. Never writes something we cannot read back."""
        doc: dict[str, Any] = config.to_dict()
        alert_config_from_dict(doc)  # round-trip check before it ever reaches Mongo

        self._collection.replace_one({"id": DOC_ID}, doc | {"id": DOC_ID}, upsert=True)

        with self._lock:
            self._config = config
            self._last_error = None
            self._version += 1
        log.info("alert config saved: %d sink rule(s)", len(config.sinks))
