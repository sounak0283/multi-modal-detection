"""AlertRouter (Expansion Plan Phase D).

The one callable actually registered on `AlertBus.sinks`. `AlertBus.sinks` is never
mutated after startup - a dashboard edit to the rule list takes effect by `dispatch()`
reading live state from `AlertConfigStore` on every call (cheap: `reload()` is throttled
internally, the same "call it liberally" pattern `BoundaryEngine`/the API routes already
use against `ZoneStore`), not by rebuilding the sinks list.
"""

from __future__ import annotations

import logging
from typing import Any

from perimeter.alerts.config_store import AlertConfigStore
from perimeter.alerts.cooldown import CooldownGate
from perimeter.alerts.sinks.smtp import EmailSink

log = logging.getLogger("perimeter.alerts.router")


class AlertRouter:
    def __init__(
        self, config_store: AlertConfigStore, gate: CooldownGate, email_sink: EmailSink | None
    ) -> None:
        self.config_store = config_store
        self.gate = gate
        self.email_sink = email_sink

    def dispatch(
        self,
        payload: dict[str, Any],
        followup: bool = False,
        only_rules: set[str] | None = None,
        scope: str = "",
    ) -> set[str]:
        """Route one alert; returns the ids of the rules it was sent to.

        `followup=True` re-sends an already-alerted event once its video is ready (see
        evidence/writer.py): cooldown is skipped, everything else still filters, and
        `only_rules` restricts it to the rules whose first email actually went out - so an
        alert the cooldown suppressed does not leak out later as a video email."""
        self.config_store.reload()
        config = self.config_store.get()
        sent: set[str] = set()

        for rule in config.sinks:
            if not rule.enabled or rule.sink_type != "smtp":
                continue
            if only_rules is not None and rule.id not in only_rules:
                continue
            if not rule.matches_kind(payload.get("kind")):
                continue
            allowed = self.gate.allow(
                rule.id, rule, payload, enforce_cooldown=not followup, scope=scope
            )
            if not allowed:
                continue
            if self.email_sink is None:
                log.warning(
                    "alert rule %s wants to send email but SMTP is not configured "
                    "(PERIMETER_SMTP_HOST) - skipping",
                    rule.id,
                )
                continue
            self.email_sink.on_alert(payload, rule.to, followup=followup)
            sent.add(rule.id)
        return sent
