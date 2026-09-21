"""Which alerts get a video clip, and how urgent alerts are handled (evidence/policy.py and
the two-worker EvidenceWriter)."""

from __future__ import annotations

import threading
import time

import pytest
from test_s3_evidence import (
    FakeDb,
    FakeS3,
    Notifier,
    fake_mux,
    ring,
    s3_store,
    wait_for,
)

from perimeter.evidence.policy import ClipPolicy, policy_from_settings
from perimeter.evidence.writer import EvidenceWriter


def alert(kind="boundary", subtype="entry", identity_status=None, event_id="e1"):
    return {"id": event_id, "camera_id": "cam_01", "ts": 101.0, "kind": kind,
            "subtype": subtype, "identity_status": identity_status, "severity": "high"}


# -- the rules ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "record"),
    [
        (alert(identity_status="known"), False),  # registered person walks in: not recorded
        (alert(identity_status="unknown_face"), True),  # stranger: recorded
        (alert(identity_status="no_face"), True),  # face unreadable: cannot vouch for them
        (alert(identity_status=None), True),  # identity module off: nobody is 'registered'
        (alert(subtype="exit", identity_status="unknown_face"), False),  # exits off by default
        (alert(kind="fire", subtype=None), True),
        (alert(kind="smoke", subtype=None), True),
        (alert(kind="crowd", subtype=None), True),
        (alert(kind="access", subtype="unauthorized"), True),
    ],
)
def test_default_rules(payload, record):
    assert ClipPolicy().decide(payload).record is record


def test_fire_smoke_crowd_and_access_are_urgent_but_boundary_is_not():
    policy = ClipPolicy()
    for kind in ("fire", "smoke", "crowd", "access"):
        assert policy.decide(alert(kind=kind, subtype=None)).urgent
    assert not policy.decide(alert(identity_status="unknown_face")).urgent


def test_boundary_entry_all_records_even_a_registered_person():
    policy = policy_from_settings({"boundary_entry": "all"}, None)
    assert policy.decide(alert(identity_status="known")).record


def test_boundary_exit_can_be_turned_on():
    policy = policy_from_settings({"boundary_exit": "unregistered"}, None)
    assert policy.decide(alert(subtype="exit", identity_status="unknown_face")).record
    assert not policy.decide(alert(subtype="exit", identity_status="known")).record


def test_any_kind_can_be_switched_off():
    policy = policy_from_settings({"crowd": "off"}, None)
    assert not policy.decide(alert(kind="crowd", subtype=None)).record


def test_yaml_off_and_on_arrive_as_booleans_and_are_accepted():
    """YAML 1.1 turns an unquoted off/on into False/True - people will write it that way."""
    policy = policy_from_settings({"boundary_exit": False, "crowd": False, "ppe": True}, None)
    assert not policy.decide(alert(subtype="exit")).record
    assert not policy.decide(alert(kind="crowd", subtype=None)).record
    assert policy.decide(alert(kind="ppe", subtype=None)).record


def test_a_typo_fails_loudly_instead_of_silently_recording_nothing():
    with pytest.raises(ValueError, match="unknown trigger"):
        policy_from_settings({"boundry_entry": "all"}, None)
    with pytest.raises(ValueError, match="not valid"):
        policy_from_settings({"boundary_entry": "sometimes"}, None)
    with pytest.raises(ValueError, match="not valid"):
        policy_from_settings({"fire": "unregistered"}, None)


def test_urgent_kinds_are_configurable():
    policy = policy_from_settings(None, ["fire"])
    assert policy.decide(alert(kind="fire", subtype=None)).urgent
    assert not policy.decide(alert(kind="crowd", subtype=None)).urgent


# -- the writer ---------------------------------------------------------------------------


def make_writer(buffers=None, mux=fake_mux, **kw):
    notifier = Notifier()
    writer = EvidenceWriter(
        database=FakeDb(),
        store=s3_store(FakeS3()),
        ring_buffer_lookup=lambda cam: (
            buffers if buffers is not None else {"cam_01": ring()}
        ).get(cam),
        post_roll_seconds=0.0,
        mux=mux,
        notify=notifier,
        policy=ClipPolicy(),
        urgent_post_roll_seconds=0.0,
        retry_backoff_s=(0.0,),
        **kw,
    )
    return writer, notifier


def test_registered_person_gets_the_alert_email_but_no_recording():
    writer, notifier = make_writer()
    writer.start()
    writer.on_alert(alert(identity_status="known"))
    assert wait_for(lambda: len(notifier.calls) == 1)
    writer.stop()

    assert "clip_url" not in notifier.calls[0]["payload"]
    assert writer.database.last("clip_status") == "skipped"
    assert writer.store._client.uploads == []  # nothing was cut or uploaded


def test_unregistered_person_is_recorded_and_the_email_carries_the_video():
    writer, notifier = make_writer()
    writer.start()
    writer.on_alert(alert(identity_status="unknown_face"))
    assert wait_for(lambda: len(notifier.calls) == 1)
    writer.stop()

    assert notifier.calls[0]["payload"]["clip_url"]
    assert notifier.calls[0]["followup"] is False
    assert writer.database.last("clip_status") == "saved"


def test_an_exit_is_not_recorded():
    writer, notifier = make_writer()
    writer.start()
    writer.on_alert(alert(subtype="exit", identity_status="unknown_face"))
    assert wait_for(lambda: len(notifier.calls) == 1)
    writer.stop()
    assert writer.database.last("clip_status") == "skipped"


def test_fire_emails_immediately_then_sends_the_video_as_a_followup():
    writer, notifier = make_writer()
    writer.start()
    writer.on_alert(alert(kind="fire", subtype=None))
    assert wait_for(lambda: len(notifier.calls) == 2)
    writer.stop()

    first, second = notifier.calls
    assert first["followup"] is False and "clip_url" not in first["payload"]
    assert second["followup"] is True and second["payload"]["clip_url"]
    assert second["only"] == {"r1"}


def test_crowd_is_urgent_too():
    writer, notifier = make_writer()
    writer.start()
    writer.on_alert(alert(kind="crowd", subtype=None))
    assert wait_for(lambda: len(notifier.calls) == 2)
    writer.stop()
    assert notifier.calls[1]["followup"] is True


def test_fire_is_not_held_up_by_a_slow_clip_already_in_progress():
    """The reason for the second worker: a normal alert's clip is mid-mux/upload when a fire
    alert arrives - the fire email must still go out at once."""
    release = threading.Event()

    def slow_mux(frames, event_id, tmp_dir):
        if event_id == "slow":
            release.wait(10)
        return fake_mux(frames, event_id, tmp_dir)

    writer, notifier = make_writer(mux=slow_mux)
    writer.start()
    writer.on_alert(alert(identity_status="unknown_face", event_id="slow"))
    time.sleep(0.4)  # the normal worker is now stuck inside slow_mux
    writer.on_alert(alert(kind="fire", subtype=None, event_id="fire1"))

    assert wait_for(lambda: any(c["payload"]["id"] == "fire1" for c in notifier.calls), 3.0)
    assert not any(c["payload"]["id"] == "slow" for c in notifier.calls)  # still stuck
    release.set()
    writer.stop()


def test_urgent_alert_does_not_wait_for_a_fuller_buffer():
    from test_s3_evidence import jpeg

    from perimeter.capture.ring_buffer import JpegRingBuffer

    buf = JpegRingBuffer(seconds=60)
    for i in range(4):
        buf.append(100.0 + i * 0.5, jpeg())  # 1.5s buffered, minimum below is 100s
    writer, notifier = make_writer(buffers={"cam_01": buf}, min_clip_seconds=100.0)

    started = time.time()
    writer.start()
    writer.on_alert(alert(kind="fire", subtype=None))
    assert wait_for(lambda: len(notifier.calls) == 2, 5.0)
    writer.stop()
    assert time.time() - started < 3.0  # a normal alert would have waited the full extra wait


def test_no_policy_means_every_alert_is_recorded_the_old_way():
    notifier = Notifier()
    writer = EvidenceWriter(
        database=FakeDb(), store=s3_store(FakeS3()),
        ring_buffer_lookup=lambda cam: ring(), post_roll_seconds=0.0, mux=fake_mux, notify=notifier,
    )
    writer.start()
    writer.on_alert(alert(identity_status="known"))
    assert wait_for(lambda: len(notifier.calls) == 1)
    writer.stop()
    assert notifier.calls[0]["payload"]["clip_url"]


# -- a no-video email must not use up the slot of the email that has the video ---------------


def test_a_skipped_alert_does_not_starve_the_video_email_that_follows_it():
    """Regression, seen live: a registered person's exit (no clip) emailed at once and started
    the 60s cooldown; the unregistered entry one second later - the one carrying the video
    link - then hit that cooldown and its email was never sent."""
    import mongomock

    from perimeter.alerts.config_store import AlertConfigStore
    from perimeter.alerts.cooldown import AlertConfig, CooldownGate, sink_rule_from_dict
    from perimeter.alerts.router import AlertRouter

    store = AlertConfigStore(mongomock.MongoClient()["t"]["alert_config"], refresh_interval=0)
    store.save(AlertConfig(sinks=(sink_rule_from_dict(
        {"id": "r1", "type": "smtp", "to": ["a@b.com"], "cooldown_seconds": 60}),)))

    class Sink:
        def __init__(self):
            self.sent = []

        def on_alert(self, payload, to, followup=False):
            self.sent.append((payload["id"], payload.get("clip_url")))

    sink = Sink()
    router = AlertRouter(store, CooldownGate(), sink)
    writer = EvidenceWriter(
        database=FakeDb(), store=s3_store(FakeS3()),
        ring_buffer_lookup=lambda cam: ring(), post_roll_seconds=0.0, mux=fake_mux,
        notify=router.dispatch, policy=ClipPolicy(), urgent_post_roll_seconds=0.0,
    )
    writer.start()
    same_zone = {"zone_id": "z1"}
    exit_known = alert(subtype="exit", identity_status="known", event_id="exit_known")
    writer.on_alert({**exit_known, **same_zone})
    assert wait_for(lambda: len(sink.sent) == 1)
    stranger = alert(identity_status="unknown_face", event_id="entry_stranger")
    writer.on_alert({**stranger, **same_zone})
    assert wait_for(lambda: len(sink.sent) == 2, 5.0)
    writer.stop()

    assert sink.sent[0] == ("exit_known", None)
    assert sink.sent[1][0] == "entry_stranger" and sink.sent[1][1]  # sent, and with the video


def test_two_video_alerts_in_a_row_are_still_throttled():
    """The cooldown itself is unchanged for alerts that compete for the same slot."""
    from perimeter.alerts.cooldown import AlertRule, CooldownGate

    gate = CooldownGate()
    rule = AlertRule(id="r1", sink_type="smtp", cooldown_seconds=60, to=("a@b.com",))
    payload = {"camera_id": "c", "kind": "boundary", "zone_id": "z", "severity": "high"}
    assert gate.allow("r1", rule, payload)
    assert not gate.allow("r1", rule, payload)
    assert gate.allow("r1", rule, payload, scope="novideo")  # separate clock
