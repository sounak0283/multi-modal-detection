"""Alert routing rules and cooldown gating (Expansion Plan Phase D).

`AlertBus.sinks` (bus.py) has always been a flat list of callables with no filtering -
every sink saw every alert. This module is what turns one dashboard-managed rule
(`alerts/config_store.py`'s `AlertConfig.sinks`) into an actual gate: a severity floor
plus a per-key cooldown. Sink-agnostic on purpose - Telegram/MQTT/webhook sinks land here
the same way email does, without re-implementing the gating logic.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

from perimeter.boundary.zones import Severity

_EMAIL_PATTERN = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")
MAX_COOLDOWN_SECONDS = 7 * 24 * 60 * 60

# Declared order, low to high - payload severities are compared against a rule's floor
# using this ordering, not string comparison.
_SEVERITY_ORDER = (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)
_SEVERITY_RANK = {severity: rank for rank, severity in enumerate(_SEVERITY_ORDER)}

SUPPORTED_SINK_TYPES = frozenset({"smtp"})


def _new_rule_id() -> str:
    return secrets.token_hex(6)


@dataclass(frozen=True)
class AlertRule:
    """One rule in the dashboard-managed `alert_config` document (Expansion Plan Phase D).

    `kinds` filters which alert kinds this rule applies to ('boundary', 'crowd', 'ppe',
    'access', 'fire', 'smoke', ...) - empty means every kind, which keeps every rule
    written before this field existed behaving exactly as it did (matches everything),
    same "absence means unrestricted" convention `min_severity`'s own default already
    uses.
    """

    id: str
    sink_type: str
    min_severity: Severity = Severity.MEDIUM
    cooldown_seconds: float = 0.0
    to: tuple[str, ...] = ()
    enabled: bool = True
    kinds: tuple[str, ...] = ()

    def matches_kind(self, kind: Any) -> bool:
        return not self.kinds or kind in self.kinds

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.sink_type,
            "min_severity": self.min_severity.value,
            "cooldown_seconds": self.cooldown_seconds,
            "to": list(self.to),
            "enabled": self.enabled,
            "kinds": list(self.kinds),
        }


def sink_rule_from_dict(data: dict[str, Any]) -> AlertRule:
    """Parse and validate one sink rule. Raises ValueError with an actionable message,
    matching `cameras/models.py:camera_from_dict` and `boundary/zones.py:zone_from_dict`'s
    contract. Generates an `id` if the incoming dict omits one, so creating a new rule
    from the dashboard doesn't require the frontend to invent one."""
    if not isinstance(data, dict):
        raise ValueError("alert sink rule must be a mapping")

    sink_type = str(data.get("type", "")).strip().lower()
    if sink_type not in SUPPORTED_SINK_TYPES:
        valid = ", ".join(sorted(SUPPORTED_SINK_TYPES))
        raise ValueError(f"alert sink: unsupported type {data.get('type')!r} (expected: {valid})")

    raw_min_severity = str(data.get("min_severity", Severity.MEDIUM.value)).strip().lower()
    try:
        min_severity = Severity(raw_min_severity)
    except ValueError:
        valid = ", ".join(s.value for s in Severity)
        raise ValueError(
            f"alert sink: invalid min_severity {data.get('min_severity')!r} (expected: {valid})"
        ) from None

    try:
        cooldown_seconds = float(data.get("cooldown_seconds", 0) or 0)
    except (TypeError, ValueError):
        raise ValueError(
            f"alert sink: cooldown_seconds {data.get('cooldown_seconds')!r} is not a number"
        ) from None
    if not 0 <= cooldown_seconds <= MAX_COOLDOWN_SECONDS:
        raise ValueError(
            f"alert sink: cooldown_seconds must be between 0 and {MAX_COOLDOWN_SECONDS}"
        )

    raw_to = data.get("to", [])
    if not isinstance(raw_to, list):
        raise ValueError("alert sink: 'to' must be a list of email addresses")
    to = tuple(str(addr).strip() for addr in raw_to if str(addr).strip())
    if sink_type == "smtp" and not to:
        raise ValueError("alert sink: type 'smtp' requires a non-empty 'to' list")
    # A typo here does not fail loudly anywhere else: the SMTP server accepts the
    # message and the alert silently goes nowhere.
    invalid = [addr for addr in to if not _EMAIL_PATTERN.match(addr)]
    if invalid:
        raise ValueError(f"alert sink: invalid recipient address(es) {invalid}")

    rule_id = str(data.get("id", "")).strip() or _new_rule_id()

    raw_kinds = data.get("kinds", [])
    if not isinstance(raw_kinds, list):
        raise ValueError("alert sink: 'kinds' must be a list of alert kind names")
    kinds = tuple(str(k).strip().lower() for k in raw_kinds if str(k).strip())

    return AlertRule(
        id=rule_id,
        sink_type=sink_type,
        min_severity=min_severity,
        cooldown_seconds=cooldown_seconds,
        to=to,
        enabled=bool(data.get("enabled", True)),
        kinds=kinds,
    )


@dataclass(frozen=True)
class AlertConfig:
    """The single `alert_config` document (Expansion Plan Phase D)."""

    sinks: tuple[AlertRule, ...] = ()
    sound_enabled: bool = True
    sound_min_severity: Severity = Severity.HIGH

    def to_dict(self) -> dict[str, Any]:
        return {
            "sinks": [rule.to_dict() for rule in self.sinks],
            "sound_enabled": self.sound_enabled,
            "sound_min_severity": self.sound_min_severity.value,
        }


def alert_config_from_dict(data: dict[str, Any]) -> AlertConfig:
    """Parse and validate the whole `alert_config` document. Rejects the whole save on
    any one bad rule - never write something we cannot read back, same rule
    `boundary/zones.py:ZoneStore.save` already follows."""
    if not isinstance(data, dict):
        raise ValueError("alert config must be a mapping")

    raw_sinks = data.get("sinks", [])
    if not isinstance(raw_sinks, list):
        raise ValueError("alert config: 'sinks' must be a list")
    sinks = tuple(sink_rule_from_dict(entry) for entry in raw_sinks)

    ids = [rule.id for rule in sinks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"alert config: duplicate rule id(s) {sorted(duplicates)}")

    raw_sound_severity = str(data.get("sound_min_severity", Severity.HIGH.value)).strip().lower()
    try:
        sound_min_severity = Severity(raw_sound_severity)
    except ValueError:
        valid = ", ".join(s.value for s in Severity)
        raise ValueError(
            f"alert config: invalid sound_min_severity {data.get('sound_min_severity')!r} "
            f"(expected: {valid})"
        ) from None

    return AlertConfig(
        sinks=sinks,
        sound_enabled=bool(data.get("sound_enabled", True)),
        sound_min_severity=sound_min_severity,
    )


class CooldownGate:
    """Thread-safe: severity floor plus a per-(rule, camera, kind, zone) cooldown.

    Deliberately keyed WITHOUT `subtype` (entry/exit) - a busy door's constant
    entry/exit churn is exactly the "routine boundary crossing" traffic the plan says a
    cooldown should throttle as one stream, not reset on every direction change.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_sent: dict[tuple[str, str | None, str | None, str | None], float] = {}

    def allow(self, rule_id: str, rule: AlertRule, payload: dict[str, Any]) -> bool:
        payload_severity = _parse_severity(payload.get("severity"))
        if _SEVERITY_RANK[payload_severity] < _SEVERITY_RANK[rule.min_severity]:
            return False

        if rule.cooldown_seconds <= 0:
            return True  # explicit bypass (PLATFORM_EXPANSION_PLAN.md §9)

        key = (rule_id, payload.get("camera_id"), payload.get("kind"), payload.get("zone_id"))
        now = time.monotonic()
        with self._lock:
            last = self._last_sent.get(key)
            if last is not None and now - last < rule.cooldown_seconds:
                return False
            self._last_sent[key] = now
        return True


def _parse_severity(value: Any) -> Severity:
    try:
        return Severity(str(value).strip().lower())
    except ValueError:
        return Severity.MEDIUM  # same fallback zone_from_dict uses for a missing/bad severity
