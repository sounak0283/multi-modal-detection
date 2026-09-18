"""Tests for per-zone active hours (PLAN.md section 6.4).

The overnight wrap is the case that matters. "18:00 to 06:00" is what a site actually
asks for, and getting the day attribution wrong makes a Friday-night schedule silently
stop at midnight - exactly when it is needed.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from perimeter.boundary.schedule import ALL_DAYS, Schedule, parse_time

# 2026-08-10 is a Monday.
MON = datetime(2026, 8, 10)
SAT = datetime(2026, 8, 15)
SUN = datetime(2026, 8, 16)


def at(day: datetime, hour: int, minute: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute)


# -- parsing --------------------------------------------------------------


def test_parse_time_hh_mm():
    assert parse_time("18:30").hour == 18


def test_parse_time_with_seconds():
    assert parse_time("06:00:30").second == 30


@pytest.mark.parametrize("bad", ["25:00", "12:99", "noon", "12", ""])
def test_parse_time_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        parse_time(bad)


# -- always on ------------------------------------------------------------


def test_no_schedule_is_always_active():
    schedule = Schedule.from_dict(None)
    assert schedule.always_on
    assert schedule.is_active(at(MON, 3))
    assert schedule.is_active(at(SUN, 14))


def test_days_only_schedule_filters_by_day():
    schedule = Schedule.from_dict({"days": ["sat", "sun"]})
    assert not schedule.is_active(at(MON, 12))
    assert schedule.is_active(at(SAT, 12))


# -- ordinary daytime window ---------------------------------------------


def test_daytime_window():
    schedule = Schedule.from_dict({"from": "09:00", "to": "17:00"})
    assert not schedule.is_active(at(MON, 8, 59))
    assert schedule.is_active(at(MON, 9))
    assert schedule.is_active(at(MON, 16, 59))
    assert not schedule.is_active(at(MON, 17)), "end is exclusive"


def test_daytime_window_respects_days():
    schedule = Schedule.from_dict({"days": ["mon"], "from": "09:00", "to": "17:00"})
    assert schedule.is_active(at(MON, 12))
    assert not schedule.is_active(at(SAT, 12))


# -- overnight wrap -------------------------------------------------------


def test_overnight_window_is_detected():
    assert Schedule.from_dict({"from": "18:00", "to": "06:00"}).wraps_midnight


def test_overnight_window_covers_both_halves():
    schedule = Schedule.from_dict({"from": "18:00", "to": "06:00"})
    assert schedule.is_active(at(MON, 23))
    assert schedule.is_active(at(MON, 2))
    assert not schedule.is_active(at(MON, 12))


def test_overnight_morning_half_belongs_to_the_previous_day():
    """A Fri 18:00-06:00 window must still be active at 02:00 on Saturday morning."""
    schedule = Schedule.from_dict({"days": ["fri"], "from": "18:00", "to": "06:00"})
    friday = datetime(2026, 8, 14)
    saturday = datetime(2026, 8, 15)

    assert schedule.is_active(at(friday, 20)), "Friday evening"
    assert schedule.is_active(at(saturday, 2)), "spills into Saturday morning"
    assert not schedule.is_active(at(saturday, 20)), "Saturday evening is not scheduled"


def test_overnight_does_not_leak_into_an_unscheduled_morning():
    schedule = Schedule.from_dict({"days": ["mon"], "from": "18:00", "to": "06:00"})
    tuesday = datetime(2026, 8, 11)
    wednesday = datetime(2026, 8, 12)

    assert schedule.is_active(at(tuesday, 2)), "Monday night spills into Tuesday"
    assert not schedule.is_active(at(wednesday, 2)), "Tuesday night was never scheduled"


# -- errors ---------------------------------------------------------------


def test_from_without_to_is_rejected():
    with pytest.raises(ValueError, match="both"):
        Schedule.from_dict({"from": "18:00"})


def test_identical_from_and_to_is_rejected():
    with pytest.raises(ValueError, match="identical"):
        Schedule.from_dict({"from": "18:00", "to": "18:00"})


def test_unknown_day_is_rejected():
    with pytest.raises(ValueError, match="unknown day"):
        Schedule.from_dict({"days": ["mon", "funday"]})


def test_day_names_are_normalised():
    schedule = Schedule.from_dict({"days": ["Monday", "TUE"]})
    assert schedule.days == frozenset({"mon", "tue"})


# -- round trip -----------------------------------------------------------


def test_round_trip_through_dict():
    original = Schedule.from_dict({"days": ["mon", "fri"], "from": "18:00", "to": "06:00"})
    restored = Schedule.from_dict(original.to_dict())
    assert restored == original


def test_always_on_serialises_to_none():
    assert Schedule(days=ALL_DAYS).to_dict() is None


def test_describe_flags_overnight():
    schedule = Schedule.from_dict({"from": "18:00", "to": "06:00"})
    assert "overnight" in schedule.describe()
