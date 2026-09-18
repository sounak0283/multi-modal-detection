"""Tests for AlertConfigStore (Expansion Plan Phase D)."""

from __future__ import annotations

import mongomock

from perimeter.alerts.config_store import AlertConfigStore
from perimeter.alerts.cooldown import AlertConfig, sink_rule_from_dict
from perimeter.boundary.zones import Severity


def make_store(refresh_interval: float = 0) -> AlertConfigStore:
    collection = mongomock.MongoClient()["perimeter_test"]["alert_config"]
    return AlertConfigStore(collection, refresh_interval=refresh_interval)


# -- defaults / empty state -------------------------------------------------------


def test_starts_empty_with_sensible_defaults():
    store = make_store()
    config = store.get()
    assert config.sinks == ()
    assert config.sound_enabled is True
    assert config.sound_min_severity == Severity.HIGH


# -- save / reload roundtrip -------------------------------------------------------


def test_save_then_get_returns_the_saved_config():
    store = make_store()
    rule = sink_rule_from_dict({"type": "smtp", "to": ["ops@example.com"], "min_severity": "high"})
    config = AlertConfig(sinks=(rule,), sound_enabled=False, sound_min_severity=Severity.CRITICAL)

    store.save(config)

    fetched = store.get()
    assert fetched.sound_enabled is False
    assert fetched.sound_min_severity == Severity.CRITICAL
    assert len(fetched.sinks) == 1
    assert fetched.sinks[0].to == ("ops@example.com",)


def test_save_persists_to_the_collection_a_fresh_store_can_read(monkeypatch):
    collection = mongomock.MongoClient()["perimeter_test"]["alert_config"]
    store_a = AlertConfigStore(collection, refresh_interval=0)
    rule = sink_rule_from_dict({"type": "smtp", "to": ["ops@example.com"]})
    store_a.save(AlertConfig(sinks=(rule,)))

    store_b = AlertConfigStore(collection, refresh_interval=0)  # fresh in-memory cache

    assert len(store_b.get().sinks) == 1
    assert store_b.get().sinks[0].to == ("ops@example.com",)


def test_reload_picks_up_a_change_made_directly_in_the_collection():
    collection = mongomock.MongoClient()["perimeter_test"]["alert_config"]
    store = AlertConfigStore(collection, refresh_interval=0)
    rule = sink_rule_from_dict({"type": "smtp", "to": ["a@b.com"]})
    store.save(AlertConfig(sinks=(rule,)))

    # A second store instance, standing in for "the dashboard's admin PUT".
    store2 = AlertConfigStore(collection, refresh_interval=0)
    store2.save(AlertConfig(sinks=(), sound_enabled=False))

    assert store.reload(force=True) is True
    assert store.get().sinks == ()
    assert store.get().sound_enabled is False


# -- reject-bad-document-keep-previous ------------------------------------------


def test_a_malformed_document_is_rejected_and_previous_config_is_kept():
    collection = mongomock.MongoClient()["perimeter_test"]["alert_config"]
    store = AlertConfigStore(collection, refresh_interval=0)
    good_rule = sink_rule_from_dict({"type": "smtp", "to": ["a@b.com"]})
    store.save(AlertConfig(sinks=(good_rule,)))

    # Corrupt the document directly, bypassing save()'s own validation.
    collection.update_one({"id": "global"}, {"$set": {"sinks": [{"type": "smtp", "to": []}]}})

    reloaded = store.reload(force=True)

    assert reloaded is False
    assert len(store.get().sinks) == 1  # previous good config, not the corrupted one
    assert store.last_error is not None


def test_reload_throttles_when_not_forced():
    store = make_store(refresh_interval=999)
    collection = store._collection
    rule = sink_rule_from_dict({"type": "smtp", "to": ["a@b.com"]})
    collection.replace_one(
        {"id": "global"}, AlertConfig(sinks=(rule,)).to_dict() | {"id": "global"}, upsert=True
    )

    assert store.reload(force=False) is False  # still within the (huge) refresh interval
    assert store.get().sinks == ()  # unchanged


# -- version -----------------------------------------------------------------------


def test_version_increments_on_save():
    store = make_store()
    before = store.version
    store.save(AlertConfig())
    assert store.version == before + 1
