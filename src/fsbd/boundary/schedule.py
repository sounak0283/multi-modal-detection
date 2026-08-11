"""Per-zone active hours (PLAN.md section 6.4).

Close to as important as the exclusion mask. A warehouse zone without a schedule fires
four hundred times during the working day and gets switched off in week one, at which
point the product protects nothing.

Overnight windows are the normal case, not the edge case - "18:00 to 06:00" is what a
site actually asks for - so the wrap is handled explicitly rather than left to whoever
writes the config to discover.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
ALL_DAYS = frozenset(DAY_NAMES)


def parse_time(value: str) -> time:
    """Accepts 'HH:MM' or 'HH:MM:SS'."""
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid time {value!r}, expected HH:MM")
    try:
        numbers = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"invalid time {value!r}, expected HH:MM") from exc
    hour, minute = numbers[0], numbers[1]
    second = numbers[2] if len(numbers) == 3 else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"time {value!r} out of range")
    return time(hour, minute, second)


@dataclass(frozen=True)
class Schedule:
    """An always-on schedule is the default: a zone with no schedule is always active."""

    days: frozenset[str] = ALL_DAYS
    start: time | None = None
    end: time | None = None

    @property
    def always_on(self) -> bool:
        return self.start is None or self.end is None

    @property
    def wraps_midnight(self) -> bool:
        return not self.always_on and self.start > self.end  # type: ignore[operator]

    def is_active(self, when: datetime) -> bool:
        today = DAY_NAMES[when.weekday()]
        now = when.time()

        if self.always_on:
            return today in self.days

        if not self.wraps_midnight:
            return today in self.days and self.start <= now < self.end  # type: ignore[operator]

        # Overnight window. The evening half belongs to today; the morning half belongs
        # to the window that *started* yesterday, so it is yesterday's day name that
        # must be in the set. Getting this backwards makes a Fri-night-to-Sat-morning
        # schedule silently stop at midnight - exactly when it is needed.
        if now >= self.start:  # type: ignore[operator]
            return today in self.days
        if now < self.end:  # type: ignore[operator]
            yesterday = DAY_NAMES[(when.weekday() - 1) % 7]
            return yesterday in self.days
        return False

    def to_dict(self) -> dict[str, Any] | None:
        if self.always_on and self.days == ALL_DAYS:
            return None
        payload: dict[str, Any] = {"days": [d for d in DAY_NAMES if d in self.days]}
        if not self.always_on:
            payload["from"] = self.start.strftime("%H:%M")  # type: ignore[union-attr]
            payload["to"] = self.end.strftime("%H:%M")  # type: ignore[union-attr]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Schedule:
        if not data:
            return cls()

        raw_days = data.get("days")
        if raw_days is None:
            days = ALL_DAYS
        else:
            days = frozenset(str(d).strip().lower()[:3] for d in raw_days)
            unknown = days - ALL_DAYS
            if unknown:
                raise ValueError(f"unknown day(s): {sorted(unknown)}")

        raw_start, raw_end = data.get("from"), data.get("to")
        if (raw_start is None) != (raw_end is None):
            raise ValueError("schedule needs both 'from' and 'to', or neither")

        start = parse_time(str(raw_start)) if raw_start is not None else None
        end = parse_time(str(raw_end)) if raw_end is not None else None
        if start is not None and start == end:
            raise ValueError("schedule 'from' and 'to' are identical; use no schedule instead")

        return cls(days=days, start=start, end=end)

    def describe(self) -> str:
        if self.always_on and self.days == ALL_DAYS:
            return "always"
        day_part = "every day" if self.days == ALL_DAYS else ",".join(
            d for d in DAY_NAMES if d in self.days
        )
        if self.always_on:
            return day_part
        window = f"{self.start:%H:%M}-{self.end:%H:%M}"  # type: ignore[str-format]
        return f"{day_part} {window}" + (" (overnight)" if self.wraps_midnight else "")
