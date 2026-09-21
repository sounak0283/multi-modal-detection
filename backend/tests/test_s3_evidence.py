"""S3 evidence storage, the video link in alert emails, and clip-save retries.

A fake S3 client is injected everywhere - no AWS account or network is touched. The writer
is driven with real threads and zero backoff so the retry/follow-up flow is exercised for
real, not mocked out.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import mongomock
import numpy as np
import pytest
from conftest import authed_client, seed_admin

from perimeter.alerts.cooldown import AlertConfig, CooldownGate, sink_rule_from_dict
from perimeter.alerts.router import AlertRouter
from perimeter.alerts.sinks.smtp import _build_message
from perimeter.api.app import create_app
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.capture.ring_buffer import JpegRingBuffer
from perimeter.evidence.store import LocalEvidenceStore, S3EvidenceStore, build_evidence_store
from perimeter.evidence.writer import EvidenceWriter
from perimeter.settings import Settings


class FakeS3:
    """Records calls; `fail_uploads` makes the next N uploads raise, like a flaky network."""

    def __init__(self, fail_uploads: int = 0):
        self.fail_uploads = fail_uploads
        self.uploads: list[dict] = []
        self.deleted: list[str] = []

    def upload_file(self, filename, bucket, key, ExtraArgs=None):  # noqa: N803
        if self.fail_uploads > 0:
            self.fail_uploads -= 1
            raise OSError("network down")
        assert Path(filename).is_file(), "upload must read the still-present temp file"
        self.uploads.append({"bucket": bucket, "key": key, "extra": ExtraArgs, "kind": "file"})

    def put_object(self, Bucket, Key, Body, **extra):  # noqa: N803
        self.uploads.append({"bucket": Bucket, "key": Key, "extra": extra, "kind": "bytes"})

    def generate_presigned_url(self, op, Params, ExpiresIn):  # noqa: N803
        return f"https://signed.example/{Params['Bucket']}/{Params['Key']}?exp={ExpiresIn}"

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.deleted.append(Key)


def s3_store(client=None, prefix="") -> S3EvidenceStore:
    return S3EvidenceStore("evid", "snapshots", "clips", prefix=prefix, client=client or FakeS3())


# -- the store ---------------------------------------------------------------------


def test_clip_upload_is_private_encrypted_and_typed(tmp_path):
    fake = FakeS3()
    clip = tmp_path / "e1.mp4"
    clip.write_bytes(b"video")

    key = s3_store(fake).save_clip("cam_01", "e1", clip)

    assert key.startswith("clips/cam_01/") and key.endswith("/e1.mp4")
    up = fake.uploads[0]
    assert up["bucket"] == "evid" and up["key"] == key
    assert up["extra"]["ServerSideEncryption"] == "AES256"
    assert up["extra"]["ContentType"] == "video/mp4"
    assert "ACL" not in up["extra"]  # never public


def test_snapshot_upload_uses_jpeg_content_type():
    fake = FakeS3()
    key = s3_store(fake).save_snapshot("cam_01", "e1", b"\xff\xd8jpeg")
    assert key.endswith("/e1.jpg")
    assert fake.uploads[0]["extra"]["ContentType"] == "image/jpeg"


def test_prefix_applies_to_the_object_but_not_the_stored_key():
    fake = FakeS3()
    store = s3_store(fake, prefix="prod/site1")
    snap = store.save_snapshot("cam_01", "e1", b"x")
    assert fake.uploads[0]["key"] == f"prod/site1/{snap}"
    assert store.uri(snap) == f"s3://evid/prod/site1/{snap}"


def test_presigned_url_and_uri():
    store = s3_store()
    assert "exp=3600" in store.presigned_url("clips/a.mp4", 3600)
    assert store.uri("clips/a.mp4") == "s3://evid/clips/a.mp4"


def test_delete_removes_the_object():
    fake = FakeS3()
    s3_store(fake).delete("clips/a.mp4")
    assert fake.deleted == ["clips/a.mp4"]


def s3_settings(bucket):
    from dataclasses import replace

    base = Settings()
    return replace(base, storage=replace(base.storage, backend="s3", s3_bucket=bucket))


def test_factory_falls_back_to_local_without_a_bucket(tmp_path):
    settings = s3_settings(bucket="")
    assert isinstance(build_evidence_store(settings, tmp_path), LocalEvidenceStore)


def test_factory_builds_s3_when_configured(tmp_path):
    settings = s3_settings(bucket="evid")
    store = build_evidence_store(settings, tmp_path, client=FakeS3())
    assert isinstance(store, S3EvidenceStore) and store.bucket == "evid"


# -- the writer: link in the email, retries, follow-up ------------------------------


class FakeDb:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def update_evidence_paths(self, event_id, **fields):
        with self.lock:
            self.calls.append({"event_id": event_id, **fields})

    def last(self, field):
        with self.lock:
            vals = [c[field] for c in self.calls if c.get(field) is not None]
        return vals[-1] if vals else None


class Notifier:
    def __init__(self, sends_to=("r1",)):
        self.sends_to = set(sends_to)
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, payload, followup=False, only_rules=None, scope=""):
        with self.lock:
            self.calls.append({"payload": payload, "followup": followup, "only": only_rules})
        return self.sends_to


def jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((40, 60, 3), np.uint8))
    return buf.tobytes()


def ring(n=6) -> JpegRingBuffer:
    buf = JpegRingBuffer(seconds=60)
    for i in range(n):
        buf.append(100.0 + i * 0.5, jpeg())
    return buf


def fake_mux(frames, event_id, tmp_dir):
    if len(frames) < 2:
        return None
    path = tmp_dir / f"{event_id}.mp4"
    path.write_bytes(b"clip")
    return path


def make_writer(fake_s3, notifier, buffers, mux=fake_mux, attempts=4):
    return EvidenceWriter(
        database=FakeDb(),
        store=s3_store(fake_s3),
        ring_buffer_lookup=lambda cam: buffers.get(cam),
        post_roll_seconds=0.0,
        mux=mux,
        notify=notifier,
        link_expiry_hours=2,
        upload_attempts=attempts,
        retry_backoff_s=(0.0,),
    )


def wait_for(cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


ALERT = {"id": "evt1", "camera_id": "cam_01", "ts": 101.0, "kind": "boundary", "severity": "high"}


def test_first_email_carries_the_video_link_and_mongo_gets_the_uri():
    fake, notifier = FakeS3(), Notifier()
    writer = make_writer(fake, notifier, {"cam_01": ring()})
    writer.start()
    writer.on_alert(dict(ALERT))
    assert wait_for(lambda: len(notifier.calls) == 1)
    writer.stop()

    call = notifier.calls[0]
    assert call["followup"] is False
    assert call["payload"]["clip_url"].startswith("https://signed.example/evid/clips/cam_01/")
    assert "exp=7200" in call["payload"]["clip_url"]
    assert call["payload"]["clip_link_expires_at"].endswith("UTC")
    db = writer.database
    assert db.last("clip_status") == "saved"
    assert db.last("clip_uri").startswith("s3://evid/clips/cam_01/")
    assert db.last("evidence_backend") == "s3"
    # the link the email carried is recorded for audit, with its expiry, and matches the email
    assert db.last("clip_link_url") == call["payload"]["clip_url"]
    expires = db.last("clip_link_expires_at")
    remaining = (expires - datetime.now(UTC)).total_seconds()
    assert expires.tzinfo is not None and 7100 < remaining <= 7200
    # ...but the permanent reference fields never contain a signed URL
    assert "signed.example" not in db.last("clip_uri") + db.last("clip_path")


def test_failed_upload_sends_the_email_without_a_link_then_a_followup_with_one():
    fake, notifier = FakeS3(fail_uploads=2), Notifier(sends_to=("r1",))
    writer = make_writer(fake, notifier, {"cam_01": ring()})
    writer.start()
    writer.on_alert(dict(ALERT))
    assert wait_for(lambda: len(notifier.calls) == 2)
    writer.stop()

    first, second = notifier.calls
    assert first["followup"] is False and "clip_url" not in first["payload"]
    assert second["followup"] is True and second["payload"]["clip_url"]
    assert second["only"] == {"r1"}  # only rules whose first email actually went out
    assert writer.database.last("clip_status") == "saved"
    assert writer.database.last("clip_attempts") == 3


def test_gives_up_after_the_configured_attempts_and_marks_failed():
    fake, notifier = FakeS3(fail_uploads=99), Notifier()
    writer = make_writer(fake, notifier, {"cam_01": ring()}, attempts=3)
    writer.start()
    writer.on_alert(dict(ALERT))
    assert wait_for(lambda: writer.database.last("clip_status") == "failed")
    time.sleep(0.3)
    writer.stop()

    assert len(notifier.calls) == 1  # the email always went out; no follow-up without a video
    assert writer.database.last("clip_attempts") == 3
    assert [u for u in fake.uploads if u["kind"] == "file"] == []  # no clip ever landed


def test_a_retry_after_the_ring_buffer_has_moved_on_still_uploads():
    """The 15s ring buffer overwrites long before a slow retry fires, so a retry must use
    frames it kept - not re-read the buffer. Here the mux fails once, the buffer is then
    emptied, and the retry still produces the clip."""
    fake, notifier = FakeS3(), Notifier()
    buf = ring()
    seen = {"calls": 0}

    def flaky_mux(frames, event_id, tmp_dir):
        seen["calls"] += 1
        if seen["calls"] == 1:
            raise RuntimeError("codec hiccup")
        assert len(frames) == 6  # the frames captured at alert time, not the emptied buffer
        return fake_mux(frames, event_id, tmp_dir)

    writer = make_writer(fake, notifier, {"cam_01": buf}, mux=flaky_mux)
    writer.start()
    writer.on_alert(dict(ALERT))
    assert wait_for(lambda: seen["calls"] >= 1)
    buf._frames.clear()
    assert wait_for(lambda: len(notifier.calls) == 2)
    writer.stop()

    assert notifier.calls[1]["followup"] is True
    assert notifier.calls[1]["payload"]["clip_url"]


def test_no_ring_buffer_still_sends_the_email_once():
    notifier = Notifier()
    writer = make_writer(FakeS3(), notifier, {})
    writer.start()
    writer.on_alert(dict(ALERT))
    assert wait_for(lambda: len(notifier.calls) == 1)
    writer.stop()
    assert "clip_url" not in notifier.calls[0]["payload"]


def test_a_local_store_never_adds_a_link(tmp_path):
    notifier = Notifier()
    writer = EvidenceWriter(
        database=FakeDb(),
        store=LocalEvidenceStore(tmp_path / "ev", "snapshots", "clips"),
        ring_buffer_lookup=lambda cam: ring(),
        post_roll_seconds=0.0,
        mux=fake_mux,
        notify=notifier,
    )
    writer.start()
    writer.on_alert(dict(ALERT))
    assert wait_for(lambda: len(notifier.calls) == 1)
    writer.stop()
    assert "clip_url" not in notifier.calls[0]["payload"]


# -- router follow-up ---------------------------------------------------------------


class RecordingSink:
    def __init__(self):
        self.calls = []

    def on_alert(self, payload, to, followup=False):
        self.calls.append((payload["id"], to, followup))


def make_router(*rules):
    coll = mongomock.MongoClient()["t"]["alert_config"]
    from perimeter.alerts.config_store import AlertConfigStore

    store = AlertConfigStore(coll, refresh_interval=0)
    store.save(AlertConfig(sinks=tuple(sink_rule_from_dict(r) for r in rules)))
    sink = RecordingSink()
    return AlertRouter(store, CooldownGate(), sink), sink


EVENT = {"id": "e1", "camera_id": "c", "kind": "boundary", "zone_id": "z", "severity": "high"}


def test_followup_bypasses_the_cooldown_that_the_first_email_started():
    router, sink = make_router(
        {"id": "r1", "type": "smtp", "to": ["a@b.com"], "cooldown_seconds": 300}
    )
    sent = router.dispatch(dict(EVENT))
    assert sent == {"r1"}
    router.dispatch(dict(EVENT), followup=True, only_rules=sent)

    assert [c[2] for c in sink.calls] == [False, True]


def test_followup_skips_rules_whose_first_email_was_suppressed():
    router, sink = make_router(
        {"id": "r1", "type": "smtp", "to": ["a@b.com"], "cooldown_seconds": 300},
        {"id": "r2", "type": "smtp", "to": ["c@d.com"], "min_severity": "critical"},
    )
    sent = router.dispatch(dict(EVENT))  # r2 filtered out by severity
    assert sent == {"r1"}
    sink.calls.clear()
    router.dispatch(dict(EVENT), followup=True, only_rules=sent)

    assert [c[1] for c in sink.calls] == [("a@b.com",)]  # r2 never leaks a video email


def test_followup_still_respects_the_severity_floor():
    router, sink = make_router(
        {"id": "r1", "type": "smtp", "to": ["a@b.com"], "min_severity": "critical"}
    )
    router.dispatch(dict(EVENT), followup=True, only_rules={"r1"})
    assert sink.calls == []


# -- the email ------------------------------------------------------------------------


def test_email_body_includes_the_link_and_expiry():
    msg = _build_message(
        {**EVENT, "ts": 1.0, "message": "m",
         "clip_url": "https://signed.example/x", "clip_link_expires_at": "2026-09-24 10:00 UTC"},
        "from@x.com", ("to@x.com",),
    )
    body = msg.get_content()
    assert "https://signed.example/x" in body and "2026-09-24 10:00 UTC" in body
    assert not msg["Subject"].startswith("[VIDEO READY]")


def test_followup_subject_is_marked():
    msg = _build_message({**EVENT, "ts": 1.0, "message": "m", "clip_url": "https://s/x"},
                         "from@x.com", ("to@x.com",), followup=True)
    assert msg["Subject"].startswith("[VIDEO READY] ")


def test_email_without_a_video_is_unchanged():
    msg = _build_message({**EVENT, "ts": 1.0, "message": "m"}, "f@x.com", ("t@x.com",))
    body = msg.get_content()
    assert "Video" not in body


# -- the API --------------------------------------------------------------------------


EVENT_ID = "68f0000000000000000000bb"


class OnlyGetEvent:
    """Just what the evidence routes call, like test_evidence_api.py's FakeDatabase."""

    def get_event(self, event_id):
        if event_id != EVENT_ID:
            return None
        return {"id": EVENT_ID, "clip_path": "clips/cam_01/2026-09-24/e.mp4",
                "snapshot_path": "snapshots/cam_01/2026-09-24/e.jpg"}


@pytest.fixture
def s3_client():
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    zone_store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    settings = Settings()
    manager = PipelineManager(settings, registry, zone_store, "unused.onnx")
    app = create_app(settings, zone_store, registry, manager, users,
                     database=OnlyGetEvent(), evidence_store=s3_store())
    return authed_client(app), EVENT_ID


def test_api_redirects_to_a_short_lived_presigned_url(s3_client):
    client, event_id = s3_client
    resp = client.get(f"/api/events/{event_id}/clip", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"].startswith("https://signed.example/evid/clips/cam_01/")
    assert "exp=300" in resp.headers["location"]


def test_api_still_requires_login_for_s3_evidence(s3_client):
    client, event_id = s3_client
    client.cookies.clear()
    assert client.get(f"/api/events/{event_id}/clip", follow_redirects=False).status_code == 401


def test_client_is_built_with_the_configured_region_and_regional_addressing(monkeypatch):
    """Regression, found against a real bucket: without region_name the presigned links
    were signed for a generic endpoint and every video link failed SignatureDoesNotMatch."""
    import boto3

    seen = {}
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: seen.update(kw) or object())

    S3EvidenceStore("evid", "snapshots", "clips", region="ap-south-1")

    assert seen["region_name"] == "ap-south-1"
    assert seen["config"].s3["addressing_style"] == "virtual"
    assert seen["config"].signature_version == "s3v4"


# -- minimum clip length ---------------------------------------------------------------


def _writer_with_span_recorder(buf, min_clip_seconds, max_extra_wait_s):
    spans = []

    def recording_mux(frames, event_id, tmp_dir):
        spans.append(frames[-1][0] - frames[0][0])
        return fake_mux(frames, event_id, tmp_dir)

    writer = EvidenceWriter(
        database=FakeDb(),
        store=s3_store(),
        ring_buffer_lookup=lambda cam: buf,
        post_roll_seconds=0.0,
        mux=recording_mux,
        notify=Notifier(),
        min_clip_seconds=min_clip_seconds,
        max_extra_wait_s=max_extra_wait_s,
    )
    return writer, spans


def test_clip_waits_for_the_buffer_to_reach_the_minimum_length():
    """An alert right after a start/reconnect finds a half-empty buffer; the clip must wait
    for more footage instead of being saved short."""
    buf = JpegRingBuffer(seconds=60)
    for i in range(4):  # only 1.5s buffered
        buf.append(100.0 + i * 0.5, jpeg())
    writer, spans = _writer_with_span_recorder(buf, min_clip_seconds=3.0, max_extra_wait_s=10.0)

    def keep_recording():
        ts = 101.5
        for _ in range(12):
            time.sleep(0.15)
            ts += 0.5
            buf.append(ts, jpeg())

    threading.Thread(target=keep_recording, daemon=True).start()
    writer.start()
    writer.on_alert(dict(ALERT))
    assert wait_for(lambda: spans, timeout=8.0)
    writer.stop()

    assert spans[0] >= 3.0


def test_clip_is_saved_short_rather_than_waiting_forever():
    buf = JpegRingBuffer(seconds=60)
    for i in range(4):
        buf.append(100.0 + i * 0.5, jpeg())
    writer, spans = _writer_with_span_recorder(buf, min_clip_seconds=100.0, max_extra_wait_s=0.6)

    started = time.time()
    writer.start()
    writer.on_alert(dict(ALERT))
    assert wait_for(lambda: spans, timeout=5.0)
    writer.stop()

    assert 0.5 <= time.time() - started < 3.0  # held back briefly, then gave up
    assert spans[0] < 100.0  # what existed, not padded


def test_a_full_buffer_is_not_delayed_at_all():
    buf = ring(6)  # 2.5s buffered, min is 2.0 -> already long enough
    writer, spans = _writer_with_span_recorder(buf, min_clip_seconds=2.0, max_extra_wait_s=10.0)
    started = time.time()
    writer.start()
    writer.on_alert(dict(ALERT))
    assert wait_for(lambda: spans)
    writer.stop()
    assert time.time() - started < 2.0


def test_the_ring_buffer_is_never_smaller_than_the_minimum_clip(tmp_path):
    from perimeter.settings import load_settings

    cfg = tmp_path / "app.yaml"
    cfg.write_text("storage:\n  ring_buffer_seconds: 8\n  min_clip_seconds: 15\n")
    storage = load_settings(config_path=cfg, env_file=tmp_path / ".env").storage
    assert storage.ring_buffer_seconds >= storage.min_clip_seconds == 15.0


# -- browser-playable clips ---------------------------------------------------------------


class _FakeVideoWriter:
    def __init__(self, path):
        self.path, self.frames = path, 0

    def write(self, image):
        assert image.shape[0] % 2 == 0 and image.shape[1] % 2 == 0, "H.264 needs even dimensions"
        self.frames += 1

    def release(self):
        self.path.write_bytes(b"video")


def _stub_codecs(monkeypatch, tags_by_fourcc, unopenable=()):
    """Stand in for OpenCV: which fourcc was asked for decides the tag 'found' in the file."""
    import perimeter.evidence.writer as wm

    asked = []

    def fake_open_writer(stem, fps, size, candidates):
        fourcc, suffix = candidates[0]
        asked.append(fourcc)
        if fourcc in unopenable:
            raise RuntimeError(f"cannot open {fourcc}")
        path = stem.with_suffix(suffix)
        return _FakeVideoWriter(path), path

    monkeypatch.setattr(wm, "open_writer", fake_open_writer)
    monkeypatch.setattr(wm, "_video_tag", lambda path: tags_by_fourcc[asked[-1]])
    return wm, asked


def _frames(n=4, size=(60, 80)):
    ok, buf = cv2.imencode(".jpg", np.zeros((*size, 3), np.uint8))
    return [(100.0 + i * 0.5, buf.tobytes()) for i in range(n)]


def test_clip_uses_h264_so_the_dashboard_can_play_it(monkeypatch, tmp_path):
    wm, asked = _stub_codecs(monkeypatch, {"avc1": "h264", "mp4v": "FMP4"})
    assert wm.mux_clip(_frames(), "e1", tmp_path).suffix == ".mp4"
    assert asked == ["avc1"]  # first choice worked, nothing else tried


def test_falls_back_to_mp4v_when_the_h264_encoder_only_pretends(monkeypatch, tmp_path, caplog):
    """OpenCV reports a writer as opened even when the encoder is missing and something else
    was substituted - so the tag actually in the file is what counts."""
    wm, asked = _stub_codecs(monkeypatch, {"avc1": "FMP4", "mp4v": "FMP4"})
    with caplog.at_level("WARNING"):
        assert wm.mux_clip(_frames(), "e1", tmp_path) is not None
    assert asked == ["avc1", "mp4v"]
    assert "will NOT play in a browser" in caplog.text  # never silent about an unplayable clip


def test_falls_back_when_h264_cannot_even_be_opened(monkeypatch, tmp_path):
    wm, asked = _stub_codecs(monkeypatch, {"mp4v": "FMP4"}, unopenable=("avc1",))
    assert wm.mux_clip(_frames(), "e1", tmp_path) is not None
    assert asked == ["avc1", "mp4v"]


def test_odd_frame_sizes_are_trimmed_for_h264(monkeypatch, tmp_path):
    wm, _asked = _stub_codecs(monkeypatch, {"avc1": "h264"})
    assert wm.mux_clip(_frames(size=(359, 641)), "e1", tmp_path) is not None  # writer asserts even
