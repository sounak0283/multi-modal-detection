"""Human-readable times for people (alert emails).

Stored times stay UTC. What a person reads is the server's local time with the UTC offset
spelled out - e.g. "30 Sep 2026, 11:48:30 (UTC+05:30)" - so an email is right for the site
it comes from and still unambiguous if the server itself runs in UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime


def local_time(value: float | datetime, seconds: bool = True) -> str:
    moment = (
        datetime.fromtimestamp(value, tz=UTC) if isinstance(value, int | float) else value
    )
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    local = moment.astimezone()  # this machine's timezone
    offset = local.strftime("%z")  # +0530
    clock = "%H:%M:%S" if seconds else "%H:%M"
    return f"{local.strftime(f'%d %b %Y, {clock}')} (UTC{offset[:3]}:{offset[3:]})"
