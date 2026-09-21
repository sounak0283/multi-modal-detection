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
    threads = list(sink._threads)
    sink.start()
    assert sink._threads == threads and len(threads) == sink._workers
    sink.stop()


def test_stop_joins_the_thread():
    sink = make_sink()
    sink.start()
    sink.stop(timeout=2.0)
    assert sink._threads == []


def _drain(sink, expected, timeout=3.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline and len(expected) == 0:
        time.sleep(0.02)


def test_reports_success_so_delivered_can_be_recorded():
    results = []
    sink = EmailSink("h", 25, "u", "p", "from@x.com", send=lambda m: None,
                     on_result=lambda eid, err: results.append((eid, err)))
    sink.start()
    sink.on_alert({"id": "evt1", "severity": "high", "kind": "boundary"}, ("to@x.com",))
    _drain(sink, results)
    sink.stop()

    assert results == [("evt1", None)]


def test_reports_the_error_when_a_send_fails():
    results = []

    def boom(_m):
        raise OSError("smtp down")

    sink = EmailSink("h", 25, "u", "p", "from@x.com", send=boom,
                     on_result=lambda eid, err: results.append((eid, err)))
    sink.start()
    sink.on_alert({"id": "evt2", "severity": "high", "kind": "boundary"}, ("to@x.com",))
    _drain(sink, results)
    sink.stop()

    assert results and results[0][0] == "evt2" and "smtp down" in results[0][1]


def test_several_emails_are_sent_at_once_not_one_after_another():
    """A slow SMTP session (5-20s on Gmail) must not hold up the email behind it - the one
    carrying the video link was arriving a minute late queued behind no-video alerts."""
    active, peak = [0], [0]
    lock = threading.Lock()

    def slow_send(_message):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.3)
        with lock:
            active[0] -= 1

    sink = EmailSink("h", 25, "u", "p", "from@x.com", send=slow_send, workers=3)
    sink.start()
    started = time.time()
    for i in range(3):
        sink.on_alert({"id": f"e{i}", "severity": "high", "kind": "boundary"}, ("to@x.com",))
    deadline = time.time() + 3
    while time.time() < deadline and active[0] + peak[0] < 4 and time.time() - started < 1.2:
        time.sleep(0.02)
    time.sleep(0.5)
    sink.stop()

    assert peak[0] == 3  # all three in flight together
    assert time.time() - started < 1.5  # ~0.3s of work, not 0.9s serial
