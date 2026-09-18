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

    def dispatch(self, payload: dict[str, Any]) -> None:
        self.config_store.reload()
        config = self.config_store.get()

        for rule in config.sinks:
            if not rule.enabled or rule.sink_type != "smtp":
                continue
            if not rule.matches_kind(payload.get("kind")):
                continue
            if not self.gate.allow(rule.id, rule, payload):
                continue
            if self.email_sink is None:
                log.warning(
                    "alert rule %s wants to send email but SMTP is not configured "
                    "(PERIMETER_SMTP_HOST) - skipping",
                    rule.id,
                )
                continue
            self.email_sink.on_alert(payload, rule.to)
