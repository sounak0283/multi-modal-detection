"""Tests for alert wording and the alert bus.

The bus carries a policy that matters more than its code: detection outranks
persistence. A database outage must never stop the pipeline, and must never silently
swallow an alert either.
"""

from __future__ import annotations

import time

from perimeter.alerts.bus import AlertBus
from perimeter.alerts.messages import (
    boundary_message,
    firesmoke_message,
    health_message,
    subject_for,
)
from perimeter.boundary.zones import EventKind


def event(message: str = "Someone entered Bay") -> dict:
    return {"camera_id": "cam_01", "kind": "boundary", "subtype": "entry", "message": message}


# -- wording --------------------------------------------------------------


def test_entry_and_exit_wording():
    assert boundary_message(EventKind.ENTRY, "Loading Bay") == "Someone entered Loading Bay"
    assert boundary_message(EventKind.EXIT, "Loading Bay") == "Someone left Loading Bay"


def test_message_without_a_zone_name_still_reads():
    assert boundary_message(EventKind.ENTRY, "") == "Someone entered"


def test_named_person_replaces_someone():
    """The extension point for the face layer: only the subject changes."""
    message = boundary_message(EventKind.ENTRY, "Gate", "known", "Sounak Bera")
    assert message == "Sounak Bera entered Gate"


def test_a_face_with_no_match_is_unrecognised_not_an_intruder():
    assert subject_for("unknown_face") == "An unrecognised person"


def test_no_usable_face_reads_the_same_as_no_identity_layer():
    """Outdoors at a gate this is the common case. Rendering abstention as an
    accusation would be worse than saying nothing (PLAN.md section 7)."""
    assert subject_for("no_face") == subject_for(None) == "Someone"


def test_known_status_without_a_name_falls_back_safely():
    assert subject_for("known", None) == "Someone"


def test_health_and_firesmoke_wording():
    assert "stopped responding" in health_message("feed_lost", "cam_01")
    assert firesmoke_message("fire", "Server Room") == "Fire detected in Server Room"
    assert firesmoke_message("smoke") == "Smoke detected"


# -- coordinate coercion ---------------------------------------------------


def test_coordinates_survive_storage_even_as_numpy_scalars():
    """REGRESSION (originally against the Postgres/json.dumps path): foot_point came
    from a numpy float32 array, and np.float32 / int stays np.float32. A naive Mongo
    insert of a numpy scalar raises `bson.errors.InvalidDocument`, so EVERY insert would
    fail the same way the old `json.dumps` did, while detection carried on looking
    perfectly healthy."""
    import numpy as np

    from perimeter.store.db import _coerce_coords

    assert _coerce_coords((np.float32(0.5), np.float32(0.25))) == [0.5, 0.25]
    assert _coerce_coords([np.float64(1.0), 2.0]) == [1.0, 2.0]
    assert _coerce_coords(None) is None


# -- bus ------------------------------------------------------------------


class FakeDatabase:
    def __init__(self, fail_times: int = 0) -> None:
        self.rows: list[dict] = []
        self.fail_times = fail_times
        self.attempts = 0

    def insert_event(self, payload: dict) -> int:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError("connection refused")
        self.rows.append(payload)
        return len(self.rows)


def drain(bus: AlertBus, expected: int, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bus.stats.persisted >= expected:
            return
        time.sleep(0.02)


def test_events_are_persisted():
    database = FakeDatabase()
    bus = AlertBus(database=database)
    bus.start()
    try:
        bus.publish(event())
        drain(bus, 1)
    finally:
        bus.stop()

    assert len(database.rows) == 1
    assert database.rows[0]["message"] == "Someone entered Bay"


def test_publish_never_blocks_the_caller():
    """Called from the inference thread; blocking here would stall detection."""
    bus = AlertBus(database=FakeDatabase())
    started = time.perf_counter()
    for _ in range(500):
        bus.publish(event())
    assert time.perf_counter() - started < 1.0


def test_a_database_outage_does_not_lose_the_alert():
    """Fails twice, then succeeds. The alert must still land."""
    database = FakeDatabase(fail_times=2)
    bus = AlertBus(database=database)
    bus.start()
    try:
        bus.publish(event())
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and not database.rows:
            time.sleep(0.05)
    finally:
        bus.stop()

    assert database.rows, "the alert should have been persisted on retry"
    assert bus.stats.failed >= 1


def test_sinks_still_fire_when_the_database_is_down():
    """A delivered warning with no database row beats silence."""
    seen: list[dict] = []
    bus = AlertBus(database=FakeDatabase(fail_times=99), sinks=[seen.append])
    bus.start()
    try:
        bus.publish(event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not seen:
            time.sleep(0.02)
    finally:
        bus.stop()

    assert seen, "notification must not depend on persistence succeeding"


def test_one_broken_sink_does_not_stop_the_others():
    def explode(_: dict) -> None:
        raise RuntimeError("sink is broken")

    seen: list[dict] = []
    bus = AlertBus(database=None, sinks=[explode, seen.append])
    bus.start()
    try:
        bus.publish(event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not seen:
            time.sleep(0.02)
    finally:
        bus.stop()

    assert seen


def test_bus_works_with_no_database_configured():
    seen: list[dict] = []
    bus = AlertBus(database=None, sinks=[seen.append])
    bus.start()
    try:
        bus.publish(event())
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not seen:
            time.sleep(0.02)
    finally:
        bus.stop()

    assert len(seen) == 1


def test_queue_overflow_drops_the_oldest_not_the_newest():
    """If the backlog is being shed, the most recent state of the site is what matters."""
    from perimeter.alerts import bus as bus_module

    bus = AlertBus(database=None)
    for i in range(bus_module.QUEUE_MAXSIZE + 25):
        bus.publish(event(f"alert {i}"))

    assert bus.stats.dropped >= 25
    newest = list(bus._queue.queue)[-1]
    assert newest["message"] == f"alert {bus_module.QUEUE_MAXSIZE + 24}"


def test_stop_flushes_queued_alerts():
    database = FakeDatabase()
    bus = AlertBus(database=database)
    for i in range(5):
        bus.publish(event(f"alert {i}"))
    bus.start()
    bus.stop()

    assert len(database.rows) == 5
