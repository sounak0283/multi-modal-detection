"""Tests for EmailSink (Expansion Plan Phase D).

Every test injects a fake `send` instead of the real SMTP one - deliberately, following
Phase C's evidence-writer lesson: a real network call inside a fast unit test suite is
both slow and a source of cross-test flakiness that has nothing to do with the logic
being tested.
"""

from __future__ import annotations

import threading
import time

from perimeter.alerts.sinks.smtp import EmailSink


class FakeSender:
    def __init__(self, fail: bool = False):
        self.sent = []
        self.fail = fail
        self.lock = threading.Lock()

    def __call__(self, message):
        if self.fail:
            raise ConnectionRefusedError("smtp relay unreachable")
        with self.lock:
            self.sent.append(message)


def make_sink(send=None) -> EmailSink:
    return EmailSink(
        host="smtp.example.com", port=587, user="bot@example.com", password="hunter2",
        from_addr="", use_tls=True, send=send or FakeSender(),
    )


def event(**overrides) -> dict:
    payload = {
        "id": "evt1", "camera_id": "cam_01", "kind": "boundary", "subtype": "entry",
        "zone_id": "loading_bay", "zone_name": "Loading Bay", "severity": "high",
        "message": "Someone entered Loading Bay", "ts": 1750000000.0,
    }
    payload.update(overrides)
    return payload


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# -- content --------------------------------------------------------------------


def test_from_addr_falls_back_to_user_when_unset():
    sink = make_sink()
    assert sink.from_addr == "bot@example.com"


def test_sent_message_has_the_expected_subject_and_body():
    sender = FakeSender()
    sink = make_sink(send=sender)
    sink.start()
    try:
        sink.on_alert(event(), to=("ops@example.com",))
        assert wait_for(lambda: sender.sent)
    finally:
        sink.stop()

    message = sender.sent[0]
    assert message["Subject"] == "[HIGH] boundary — cam_01"
    assert message["From"] == "bot@example.com"
    assert message["To"] == "ops@example.com"
    body = message.get_content()
    assert "Someone entered Loading Bay" in body
    assert "Loading Bay" in body
    assert "cam_01" in body
    assert "HIGH" in body


def test_multiple_recipients_are_comma_joined():
    sender = FakeSender()
    sink = make_sink(send=sender)
    sink.start()
    try:
        sink.on_alert(event(), to=("a@example.com", "b@example.com"))
        assert wait_for(lambda: sender.sent)
    finally:
        sink.stop()

    assert sender.sent[0]["To"] == "a@example.com, b@example.com"


def test_missing_optional_fields_do_not_crash_message_building():
    sender = FakeSender()
    sink = make_sink(send=sender)
    sink.start()
    try:
        sink.on_alert({"id": "evt2"}, to=("ops@example.com",))
        assert wait_for(lambda: sender.sent)
    finally:
        sink.stop()

    assert "(no message)" in sender.sent[0].get_content()


# -- failure handling -------------------------------------------------------------


def test_a_failed_send_is_logged_and_does_not_crash_the_thread():
    sender = FakeSender(fail=True)
    sink = make_sink(send=sender)
    sink.start()
    try:
        sink.on_alert(event(), to=("ops@example.com",))
        time.sleep(0.3)  # give the worker a chance to hit (and survive) the failure

        # The thread must still be alive and able to process a next message.
        sender.fail = False
        sink.on_alert(event(), to=("ops@example.com",))
        assert wait_for(lambda: sender.sent)
    finally:
        sink.stop()

    assert len(sender.sent) == 1


# -- lifecycle --------------------------------------------------------------------


def test_start_is_idempotent():
    sink = make_sink()
    sink.start()
    thread = sink._thread
    sink.start()
    assert sink._thread is thread
    sink.stop()


def test_stop_joins_the_thread():
    sink = make_sink()
    sink.start()
    sink.stop(timeout=2.0)
    assert sink._thread is None
