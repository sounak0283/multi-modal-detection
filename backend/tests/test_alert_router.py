"""Tests for AlertRouter (Expansion Plan Phase D)."""

from __future__ import annotations

import mongomock

from perimeter.alerts.config_store import AlertConfigStore
from perimeter.alerts.cooldown import AlertConfig, CooldownGate, sink_rule_from_dict
from perimeter.alerts.router import AlertRouter


class FakeEmailSink:
    def __init__(self):
        self.calls = []

    def on_alert(self, payload, to):
        self.calls.append((payload, to))


def make_store(config: AlertConfig | None = None) -> AlertConfigStore:
    collection = mongomock.MongoClient()["perimeter_test"]["alert_config"]
    store = AlertConfigStore(collection, refresh_interval=0)
    if config is not None:
        store.save(config)
    return store


def event(severity="high") -> dict:
    return {
        "id": "evt1", "camera_id": "cam_01", "kind": "boundary",
        "zone_id": "z1", "severity": severity,
    }


def test_dispatch_forwards_to_email_sink_for_a_matching_enabled_rule():
    rule = sink_rule_from_dict(
        {"type": "smtp", "to": ["ops@example.com"], "min_severity": "medium"}
    )
    store = make_store(AlertConfig(sinks=(rule,)))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch(event())

    assert len(sink.calls) == 1
    payload, to = sink.calls[0]
    assert payload["id"] == "evt1"
    assert to == ("ops@example.com",)


def test_dispatch_skips_a_disabled_rule():
    rule = sink_rule_from_dict(
        {"type": "smtp", "to": ["ops@example.com"], "min_severity": "low", "enabled": False}
    )
    store = make_store(AlertConfig(sinks=(rule,)))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch(event())

    assert sink.calls == []


def test_dispatch_skips_a_rule_below_its_severity_floor():
    rule = sink_rule_from_dict(
        {"type": "smtp", "to": ["ops@example.com"], "min_severity": "critical"}
    )
    store = make_store(AlertConfig(sinks=(rule,)))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch(event(severity="medium"))

    assert sink.calls == []


def test_dispatch_with_no_email_sink_configured_does_not_crash():
    rule = sink_rule_from_dict({"type": "smtp", "to": ["ops@example.com"], "min_severity": "low"})
    store = make_store(AlertConfig(sinks=(rule,)))
    router = AlertRouter(store, CooldownGate(), None)

    router.dispatch(event())  # must not raise


def test_dispatch_reflects_a_live_dashboard_edit_without_restart():
    """The store's own reload() throttle is what makes an edit visible within
    refresh_interval - dispatch() must actually call it, not just read a cached config
    grabbed once at construction time."""
    store = make_store(AlertConfig(sinks=()))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch(event())
    assert sink.calls == []

    new_rule = sink_rule_from_dict(
        {"type": "smtp", "to": ["ops@example.com"], "min_severity": "low"}
    )
    store.save(AlertConfig(sinks=(new_rule,)))

    router.dispatch(event())
    assert len(sink.calls) == 1


def test_dispatch_applies_the_cooldown_gate():
    rule = sink_rule_from_dict(
        {"type": "smtp", "to": ["ops@example.com"], "min_severity": "low", "cooldown_seconds": 60}
    )
    store = make_store(AlertConfig(sinks=(rule,)))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch(event())
    router.dispatch(event())

    assert len(sink.calls) == 1


def test_dispatch_skips_a_rule_whose_kinds_do_not_include_the_event():
    rule = sink_rule_from_dict(
        {"type": "smtp", "to": ["fire@example.com"], "min_severity": "low", "kinds": ["fire", "smoke"]}
    )
    store = make_store(AlertConfig(sinks=(rule,)))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch(event())  # kind="boundary", not in the rule's kinds

    assert sink.calls == []


def test_dispatch_forwards_a_rule_whose_kinds_include_the_event():
    rule = sink_rule_from_dict(
        {"type": "smtp", "to": ["boundary@example.com"], "min_severity": "low", "kinds": ["boundary"]}
    )
    store = make_store(AlertConfig(sinks=(rule,)))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch(event())  # kind="boundary"

    assert len(sink.calls) == 1


def test_empty_kinds_matches_every_alert_kind():
    """Backward compatible default: a rule with no 'kinds' set (or an old rule saved
    before this field existed) still matches everything."""
    rule = sink_rule_from_dict({"type": "smtp", "to": ["all@example.com"], "min_severity": "low"})
    store = make_store(AlertConfig(sinks=(rule,)))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch(event())
    router.dispatch({**event(), "kind": "fire"})

    assert len(sink.calls) == 2


def test_kinds_and_recipients_can_differ_per_rule():
    fire_rule = sink_rule_from_dict(
        {"type": "smtp", "to": ["safety@example.com"], "min_severity": "low", "kinds": ["fire", "smoke"]}
    )
    boundary_rule = sink_rule_from_dict(
        {"type": "smtp", "to": ["security@example.com"], "min_severity": "low", "kinds": ["boundary", "crowd"]}
    )
    store = make_store(AlertConfig(sinks=(fire_rule, boundary_rule)))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch({**event(), "kind": "fire"})

    assert len(sink.calls) == 1
    payload, to = sink.calls[0]
    assert to == ("safety@example.com",)


def test_multiple_rules_can_each_independently_match():
    rule_a = sink_rule_from_dict({"type": "smtp", "to": ["a@example.com"], "min_severity": "low"})
    rule_b = sink_rule_from_dict({"type": "smtp", "to": ["b@example.com"], "min_severity": "low"})
    store = make_store(AlertConfig(sinks=(rule_a, rule_b)))
    sink = FakeEmailSink()
    router = AlertRouter(store, CooldownGate(), sink)

    router.dispatch(event())

    assert len(sink.calls) == 2
    recipients = {to for _payload, to in sink.calls}
    assert recipients == {("a@example.com",), ("b@example.com",)}
