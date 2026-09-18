"""Tests for alert routing rules and cooldown gating (Expansion Plan Phase D)."""

from __future__ import annotations

import time

import pytest

from perimeter.alerts.cooldown import (
    AlertRule,
    CooldownGate,
    alert_config_from_dict,
    sink_rule_from_dict,
)
from perimeter.boundary.zones import Severity


def rule(**overrides) -> AlertRule:
    defaults = {
        "id": "r1", "sink_type": "smtp", "min_severity": Severity.LOW, "cooldown_seconds": 0,
    }
    defaults.update(overrides)
    return AlertRule(**defaults)


def event(camera_id="cam_01", kind="boundary", zone_id="loading_bay", severity="medium") -> dict:
    return {
        "id": "evt1", "camera_id": camera_id, "kind": kind,
        "zone_id": zone_id, "severity": severity,
    }


# -- sink_rule_from_dict -----------------------------------------------------


def test_parses_a_valid_smtp_rule():
    parsed = sink_rule_from_dict(
        {
            "id": "abc123", "type": "smtp", "min_severity": "high",
            "cooldown_seconds": 300, "to": ["ops@example.com"], "enabled": False,
        }
    )
    assert parsed.id == "abc123"
    assert parsed.sink_type == "smtp"
    assert parsed.min_severity == Severity.HIGH
    assert parsed.cooldown_seconds == 300
    assert parsed.to == ("ops@example.com",)
    assert parsed.enabled is False


def test_defaults_min_severity_cooldown_and_enabled():
    parsed = sink_rule_from_dict({"type": "smtp", "to": ["ops@example.com"]})
    assert parsed.min_severity == Severity.MEDIUM
    assert parsed.cooldown_seconds == 0
    assert parsed.enabled is True


def test_generates_an_id_when_omitted():
    """A new rule created from the dashboard doesn't have to invent one."""
    a = sink_rule_from_dict({"type": "smtp", "to": ["x@y.com"]})
    b = sink_rule_from_dict({"type": "smtp", "to": ["x@y.com"]})
    assert a.id and b.id
    assert a.id != b.id


def test_rejects_an_unsupported_sink_type():
    with pytest.raises(ValueError, match="unsupported type"):
        sink_rule_from_dict({"type": "telegram", "to": ["x"]})


def test_rejects_smtp_without_recipients():
    with pytest.raises(ValueError, match="requires a non-empty 'to'"):
        sink_rule_from_dict({"type": "smtp", "to": []})


def test_rejects_an_invalid_min_severity():
    with pytest.raises(ValueError, match="invalid min_severity"):
        sink_rule_from_dict({"type": "smtp", "min_severity": "extreme", "to": ["x@y.com"]})


def test_blank_recipients_are_dropped():
    parsed = sink_rule_from_dict({"type": "smtp", "to": ["  ", "ops@example.com", ""]})
    assert parsed.to == ("ops@example.com",)


# -- alert_config_from_dict -------------------------------------------------------


def test_parses_a_full_config():
    config = alert_config_from_dict(
        {
            "sinks": [{"type": "smtp", "to": ["a@b.com"]}],
            "sound_enabled": False,
            "sound_min_severity": "critical",
        }
    )
    assert len(config.sinks) == 1
    assert config.sound_enabled is False
    assert config.sound_min_severity == Severity.CRITICAL


def test_defaults_to_sound_on_and_high_with_no_sinks():
    config = alert_config_from_dict({})
    assert config.sinks == ()
    assert config.sound_enabled is True
    assert config.sound_min_severity == Severity.HIGH


def test_rejects_duplicate_rule_ids():
    with pytest.raises(ValueError, match="duplicate rule id"):
        alert_config_from_dict(
            {
                "sinks": [
                    {"id": "same", "type": "smtp", "to": ["a@b.com"]},
                    {"id": "same", "type": "smtp", "to": ["c@d.com"]},
                ]
            }
        )


def test_rejects_the_whole_document_on_one_bad_rule():
    with pytest.raises(ValueError, match="requires a non-empty 'to'"):
        alert_config_from_dict({"sinks": [{"type": "smtp", "to": []}]})


def test_round_trips_through_to_dict():
    original = alert_config_from_dict(
        {"sinks": [{"type": "smtp", "to": ["a@b.com"], "min_severity": "high"}]}
    )
    round_tripped = alert_config_from_dict(original.to_dict())
    assert round_tripped == original


# -- CooldownGate: severity floor ---------------------------------------------


def test_below_floor_is_rejected():
    gate = CooldownGate()
    assert gate.allow("r1", rule(min_severity=Severity.HIGH), event(severity="medium")) is False


def test_at_or_above_floor_is_allowed():
    gate = CooldownGate()
    r = rule(min_severity=Severity.MEDIUM)
    assert gate.allow("r1", r, event(severity="medium")) is True
    assert gate.allow("r1", r, event(severity="high")) is True
    assert gate.allow("r1", r, event(severity="critical")) is True


def test_unparseable_severity_falls_back_to_medium():
    gate = CooldownGate()
    assert gate.allow("r1", rule(min_severity=Severity.MEDIUM), event(severity="not-real")) is True
    assert gate.allow("r1", rule(min_severity=Severity.HIGH), event(severity=None)) is False


# -- CooldownGate: cooldown ------------------------------------------------------


def test_cooldown_zero_never_throttles():
    gate = CooldownGate()
    r = rule(cooldown_seconds=0)
    for _ in range(5):
        assert gate.allow("r1", r, event()) is True


def test_second_alert_within_the_window_is_suppressed():
    gate = CooldownGate()
    r = rule(cooldown_seconds=60)
    assert gate.allow("r1", r, event()) is True
    assert gate.allow("r1", r, event()) is False


def test_cooldown_expires_after_the_window():
    gate = CooldownGate()
    r = rule(cooldown_seconds=0.2)
    assert gate.allow("r1", r, event()) is True
    time.sleep(0.25)
    assert gate.allow("r1", r, event()) is True


def test_cooldown_key_is_independent_per_camera():
    gate = CooldownGate()
    r = rule(cooldown_seconds=60)
    assert gate.allow("r1", r, event(camera_id="cam_01")) is True
    assert gate.allow("r1", r, event(camera_id="cam_02")) is True


def test_cooldown_key_is_independent_per_zone():
    gate = CooldownGate()
    r = rule(cooldown_seconds=60)
    assert gate.allow("r1", r, event(zone_id="loading_bay")) is True
    assert gate.allow("r1", r, event(zone_id="main_gate")) is True


def test_cooldown_key_is_independent_per_rule():
    """Two rules (e.g. different recipients) must not share cooldown state."""
    gate = CooldownGate()
    r = rule(cooldown_seconds=60)
    assert gate.allow("r1", r, event()) is True
    assert gate.allow("r2", r, event()) is True


def test_entry_and_exit_of_the_same_zone_share_one_cooldown():
    """Deliberate: subtype is NOT part of the cooldown key, so a busy door's entry/exit
    churn is throttled as one routine stream."""
    gate = CooldownGate()
    r = rule(cooldown_seconds=60)
    entry = event()
    entry["subtype"] = "entry"
    exit_ = event()
    exit_["subtype"] = "exit"
    assert gate.allow("r1", r, entry) is True
    assert gate.allow("r1", r, exit_) is False


@pytest.mark.parametrize("cooldown", [-30, "soon", 10 * 365 * 24 * 3600])
def test_out_of_range_or_non_numeric_cooldown_is_rejected(cooldown):
    with pytest.raises(ValueError, match="cooldown_seconds"):
        sink_rule_from_dict(
            {"type": "smtp", "to": ["ops@example.com"], "cooldown_seconds": cooldown}
        )


@pytest.mark.parametrize(
    "to", [["not an email"], ["ops@example.com", "typo@nodot"], "ops@example.com"]
)
def test_invalid_recipients_are_rejected(to):
    with pytest.raises(ValueError):
        sink_rule_from_dict({"type": "smtp", "to": to})
