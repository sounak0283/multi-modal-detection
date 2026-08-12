"""Tests for alert wording, the alert bus, and DSN handling.

The bus carries a policy that matters more than its code: detection outranks
persistence. A database outage must never stop the pipeline, and must never silently
swallow an alert either.
"""

from __future__ import annotations

import time

import pytest

from fsbd.alerts.bus import AlertBus
from fsbd.alerts.messages import boundary_message, firesmoke_message, health_message, subject_for
from fsbd.boundary.zones import EventKind
from fsbd.store.db import parse_dsn


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


# -- DSN ------------------------------------------------------------------


def test_dsn_parses_all_parts():
    dsn = parse_dsn("postgresql://user:pw@db.local:5433/fsbd")
    assert (dsn.user, dsn.host, dsn.port, dsn.database) == ("user", "db.local", 5433, "fsbd")


def test_dsn_defaults_the_port():
    assert parse_dsn("postgresql://u:p@localhost/fsbd").port == 5432


def test_dsn_percent_decodes_the_password():
    """Database passwords routinely contain characters that must be escaped in a URL,
    and a silently wrong password is a confusing failure to debug."""
    assert parse_dsn("postgresql://u:p%40ss%3Aword@h/db").password == "p@ss:word"


def test_dsn_safe_string_hides_the_password():
    safe = parse_dsn("postgresql://user:hunter2@h:5432/fsbd").safe
    assert "hunter2" not in safe
    assert "user@h:5432/fsbd" in safe


@pytest.mark.parametrize(
    "url",
    ["mysql://u:p@h/db", "postgresql://u:p@h/", "not a url at all"],
)
def test_bad_dsn_is_rejected(url):
    with pytest.raises(ValueError):
        parse_dsn(url)


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
    from fsbd.alerts import bus as bus_module

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
