"""Phase J steps 2 + 2b: the background analysis worker, frame RAG and the /api/ai routes -
against an in-memory MongoDB and a fake Bedrock (no AWS, no cost)."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
from datetime import UTC, datetime, timedelta

import cv2
import mongomock
import numpy as np
import pytest
from conftest import authed_client, seed_admin

from perimeter.ai.bedrock import AIUnavailableError
from perimeter.ai.frames import (
    FrameIndex,
    FrameRAG,
    _parse_captions,
    extract_keyframes,
)
from perimeter.ai.services import AIServices
from perimeter.ai.video import VideoAnalysis
from perimeter.ai.worker import MAX_ATTEMPTS, RETRY_AFTER_S, AnalysisError, AnalysisWorker
from perimeter.api.app import create_app
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import AISettings, Settings, load_settings

DIMS = 256


def ai(**overrides):
    return AISettings(**{"enabled": True, "api_key": "k", "embed_dims": DIMS,
                         "frames_per_clip": 3, **overrides})


def make_clip(frames=30, fps=10.0, size=(640, 480)) -> bytes:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for i in range(frames):
            writer.write(np.full((size[1], size[0], 3), (i * 8) % 255, dtype=np.uint8))
        writer.release()
        return path.read_bytes()


CLIP = make_clip()


def word_vector(text: str) -> list[float]:
    """A stand-in embedding: hashed bag of words, so shared words mean nearby vectors."""
    vector = [0.0] * DIMS
    for word in re.findall(r"[a-z]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeBedrock:
    def __init__(self, captions=None, answer="Two people, no helmets [1]."):
        self.captions = captions or {}
        self.answer = answer
        self.embedded: list[str] = []
        self.conversations: list[dict] = []
        self.fail_embed = False

    def invoke_model(self, modelId, body, contentType, accept):  # noqa: N803 - boto3's names
        request = json.loads(body)
        if self.fail_embed:
            raise RuntimeError("bedrock is down")
        assert request["dimensions"] == DIMS
        self.embedded.append(request["inputText"])
        payload = {"embedding": word_vector(request["inputText"])}
        return {"body": io.BytesIO(json.dumps(payload).encode())}

    def converse(self, modelId, messages, inferenceConfig, system=None):  # noqa: N803
        self.conversations.append({"messages": messages, "system": system})
        content = messages[0]["content"]
        prompt = content[-1]["text"]
        if prompt.startswith("Question:"):
            text = self.answer
        else:
            images = sum(1 for part in content if "image" in part)
            kind = re.search(r"raised a '(\w+)'", prompt).group(1)
            text = json.dumps(self.captions.get(kind, [f"{kind} frame"] * images)[:images])
        return {"output": {"message": {"content": [{"text": text}]}}}


class FakeStore:
    def __init__(self, data=CLIP):
        self.data = data

    def read(self, key):
        if self.data is None:
            raise FileNotFoundError(key)
        return self.data

    def uri(self, key):
        return None


class FakeAnalyzer:
    def __init__(self, error=None):
        self.error = error
        self.seen: list[str] = []

    def analyze(self, event):
        self.seen.append(event["id"])
        if self.error is not None:
            raise self.error
        return VideoAnalysis(
            summary=f"Summary of {event['kind']}", people_count=2, activities=["walking"],
            verdict="likely_real", findings="details", model="pegasus", input_mode="inline",
            seconds=1.0, raw={"message": "{}"},
        )


def now():
    return datetime.now(UTC)


def add_event(db, kind="ppe", clip=True, test_video=False, ts=None, camera_id="cam_01"):
    doc = {
        "kind": kind, "camera_id": camera_id, "zone_name": "Gate",
        "message": ("[TEST VIDEO] " if test_video else "") + f"{kind} alert",
        "ts": ts or now(),
        "clip_status": "saved" if clip else "pending",
        "clip_path": f"clips/{camera_id}/x.mp4" if clip else None,
    }
    return str(db["events"].insert_one(doc).inserted_id)


def make_worker(db=None, analyzer=None, bedrock=None, settings=None, rag=True):
    db = db if db is not None else mongomock.MongoClient()["perimeter_test"]
    settings = settings or ai()
    bedrock = bedrock or FakeBedrock()
    frame_rag = FrameRAG(settings, FakeStore(), FrameIndex(), bedrock) if rag else None
    worker = AnalysisWorker(settings, db, analyzer or FakeAnalyzer(), frame_rag)
    worker.ensure_indexes()
    return worker, db


# -- settings --------------------------------------------------------------------------------


def test_settings_reject_an_embedding_size_titan_does_not_offer(tmp_path):
    cfg = tmp_path / "app.yaml"
    cfg.write_text("ai:\n  embed_dims: 384\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ai.embed_dims"):
        load_settings(config_path=cfg, env_file=tmp_path / ".env")


def test_the_shipped_config_loads_with_frame_rag_on():
    s = load_settings().ai
    assert s.frame_rag and s.embed_dims == 512 and s.frames_per_clip == 6


# -- reading frames ----------------------------------------------------------------------------


def test_keyframes_are_spread_over_the_clip_and_shrunk():
    frames = extract_keyframes(CLIP, count=3, height=240)
    offsets = [offset for offset, _jpeg in frames]
    assert len(frames) == 3 and offsets == sorted(offsets)
    assert offsets[0] < 1.0 < offsets[-1] <= 3.0  # a 3 s clip
    image = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[0] == 240


def test_garbled_captions_are_padded_rather_than_misaligned():
    assert _parse_captions('Sure! ["a", "b"]', 3) == ["a", "b", ""]
    assert _parse_captions("no json here", 2) == ["", ""]


# -- the index -----------------------------------------------------------------------------------


def event(kind="ppe", event_id="e1", **extra):
    return {"id": event_id, "kind": kind, "camera_id": "cam_01", "zone_name": "Gate",
            "ts": now(), "clip_path": "clips/cam_01/x.mp4", **extra}


def test_captions_are_embedded_with_their_alert_type_and_the_summary_is_kept():
    bedrock = FakeBedrock(captions={"fire": ["bright light", "bright light", "bright light"]})
    rag = FrameRAG(ai(), FakeStore(), FrameIndex(), bedrock)
    stored = rag.index_event(event("fire"), summary="A fire burns and grows.")
    assert stored == 4
    assert bedrock.embedded[0] == "fire alert, zone Gate: bright light"
    hits = rag.search("fire burns")
    assert hits[0].whole_clip and hits[0].caption == "A fire burns and grows."
    assert hits[0].to_dict()["whole_clip"] is True and "clip_path" not in hits[0].to_dict()


def test_search_finds_the_right_clip_and_respects_filters():
    bedrock = FakeBedrock(captions={
        "ppe": ["worker without helmet"] * 3, "boundary": ["two men sitting"] * 3})
    rag = FrameRAG(ai(), FakeStore(), FrameIndex(), bedrock)
    rag.index_event(event("ppe", "e1"))
    rag.index_event(event("boundary", "e2", camera_id="cam_02"))
    assert rag.search("helmet")[0].event_id == "e1"
    assert rag.search("men sitting")[0].event_id == "e2"
    assert {h.event_id for h in rag.search("helmet", camera_id="cam_02")} == {"e2"}
    assert {h.event_id for h in rag.search("helmet", kind="boundary")} == {"e2"}
    assert {h.event_id for h in rag.search("helmet", event_id="e2")} == {"e2"}
    assert rag.search("helmet", since=(now() + timedelta(hours=1)).timestamp()) == []


def test_a_clip_is_embedded_once_and_only_its_summary_is_refreshed():
    bedrock = FakeBedrock()
    rag = FrameRAG(ai(), FakeStore(), FrameIndex(), bedrock)
    rag.index_event(event(), summary="first")
    calls = (len(bedrock.embedded), len(bedrock.conversations))
    assert calls == (4, 1)  # 3 frames + summary, one caption call

    rag.index_event(event(), summary="first")  # e.g. a retry: nothing to do
    assert (len(bedrock.embedded), len(bedrock.conversations)) == calls

    rag.index_event(event(), summary="second")  # re-analysed: only the summary changes
    assert (len(bedrock.embedded), len(bedrock.conversations)) == (5, 1)
    assert rag.index.count() == 4 and rag.index.summary_caption("e1") == "second"


def test_forcing_re_embeds_the_frames_without_duplicating_them():
    bedrock = FakeBedrock()
    rag = FrameRAG(ai(), FakeStore(), FrameIndex(), bedrock)
    rag.index_event(event(), summary="first")
    rag.index_event(event(), summary="first", force=True)
    assert len(bedrock.conversations) == 2 and rag.index.count() == 4


def test_frames_stay_embedded_across_a_restart(tmp_path):
    bedrock = FakeBedrock()
    FrameRAG(ai(), FakeStore(), FrameIndex(tmp_path), bedrock).index_event(event(), "s")
    before = len(bedrock.embedded)
    restarted = FrameRAG(ai(), FakeStore(), FrameIndex(tmp_path), bedrock)
    restarted.index_event(event(), "s")
    assert len(bedrock.embedded) == before and restarted.index.count() == 4


def test_analyse_again_does_not_re_embed_the_frames():
    bedrock = FakeBedrock()
    worker, db = make_worker(bedrock=bedrock)
    event_id = add_event(db, "fire")
    worker.run_once()
    captions, entries = len(bedrock.conversations), worker.rag.index.count()
    for _ in range(3):
        worker.request(event_id)
        worker.run_once()
    assert len(bedrock.conversations) == captions  # no new caption calls
    assert worker.rag.index.count() == entries  # nothing piled up
    assert db["video_analyses"].find_one({"_id": event_id})["frames_count"] == 4


def test_old_entries_are_purged():
    rag = FrameRAG(ai(), FakeStore(), FrameIndex(), FakeBedrock())
    rag.index_event(event(ts=now() - timedelta(days=40)))
    rag.index_event(event(event_id="e2"))
    rag.index.purge_before((now() - timedelta(days=30)).timestamp())
    assert not rag.index.has_event("e1") and rag.index.has_event("e2")


def test_two_in_memory_indexes_do_not_share_entries():
    first, second = FrameIndex(), FrameIndex()
    first.replace_event(event(), [(0.5, "x", word_vector("x"))])
    assert first.count() == 1 and second.count() == 0


# -- answers ---------------------------------------------------------------------------------------


def test_an_answer_looks_at_the_matching_frames_and_cites_them():
    bedrock = FakeBedrock(captions={"ppe": ["worker without helmet"] * 3})
    rag = FrameRAG(ai(), FakeStore(), FrameIndex(), bedrock)
    rag.index_event(event(), summary="Two workers walk past.")
    result = rag.answer("who has no helmet?")
    assert result["answer"] == "Two people, no helmets [1]."
    content = bedrock.conversations[-1]["messages"][0]["content"]
    assert sum(1 for part in content if "image" in part) == 3  # frames, not the summary
    assert any("video summary of the whole clip" in p.get("text", "") for p in content)
    assert {s["event_id"] for s in result["sources"]} == {"e1"}


def test_an_answer_still_comes_from_captions_when_the_clip_cannot_be_read():
    bedrock = FakeBedrock()
    store = FakeStore()
    rag = FrameRAG(ai(), store, FrameIndex(), bedrock)
    rag.index_event(event())
    store.data = None  # clip gone from storage since it was indexed
    assert rag.answer("ppe frame")["answer"]
    content = bedrock.conversations[-1]["messages"][0]["content"]
    assert not any("image" in part for part in content)


def test_no_matching_footage_means_no_model_call():
    bedrock = FakeBedrock()
    result = FrameRAG(ai(), FakeStore(), FrameIndex(), bedrock).answer("anything?")
    assert result["sources"] == [] and bedrock.conversations == []


# -- the worker: what gets analysed automatically ------------------------------------------------


def test_a_new_clip_of_an_auto_type_is_analysed_and_indexed():
    worker, db = make_worker()
    event_id = add_event(db, "ppe")
    assert worker.run_once() == 1
    doc = db["video_analyses"].find_one({"_id": event_id})
    assert doc["status"] == "done" and doc["trigger"] == "auto" and doc["attempts"] == 1
    assert doc["analysis"]["verdict"] == "likely_real"  # stored...
    assert doc["frames_status"] == "done" and doc["frames_count"] == 4  # 3 frames + summary
    assert worker.rag.index.has_event(event_id)


def test_other_alerts_are_left_alone():
    worker, db = make_worker()
    add_event(db, "boundary")  # not an auto type
    add_event(db, "ppe", clip=False)  # clip not saved yet
    add_event(db, "ppe", test_video=True)  # Video test footage
    add_event(db, "ppe", ts=now() - timedelta(hours=2))  # from before the worker started
    assert worker.run_once() == 0
    assert db["video_analyses"].count_documents({}) == 0


def test_test_video_clips_are_analysed_when_skip_test_video_is_off():
    worker, db = make_worker(settings=ai(skip_test_video=False))
    add_event(db, "ppe", test_video=True)
    assert worker.run_once() == 1


def test_a_clip_is_analysed_once():
    analyzer = FakeAnalyzer()
    worker, db = make_worker(analyzer=analyzer)
    add_event(db, "fire")
    worker.run_once()
    worker.run_once()
    assert len(analyzer.seen) == 1


def test_the_daily_cap_stops_auto_analysis():
    worker, db = make_worker(settings=ai(daily_cap=2))
    for _ in range(3):
        add_event(db, "fire")
    assert worker.run_once() == 2
    assert worker.run_once() == 0


def test_frame_rag_can_be_switched_off():
    worker, db = make_worker(rag=False)
    event_id = add_event(db, "fire")
    worker.run_once()
    doc = db["video_analyses"].find_one({"_id": event_id})
    assert doc["status"] == "done" and "frames_status" not in doc


# -- the worker: manual requests ---------------------------------------------------------------


def test_a_manual_request_works_for_any_alert_with_a_clip_and_hides_the_verdict():
    worker, db = make_worker()
    event_id = add_event(db, "boundary", test_video=True)
    queued = worker.request(event_id)
    assert queued["status"] == "queued" and queued["trigger"] == "manual"
    assert worker.run_once() == 1
    shown = worker.status(event_id)
    assert shown["status"] == "done" and shown["analysis"]["summary"] == "Summary of boundary"
    assert "verdict" not in shown["analysis"] and "raw" not in shown
    assert db["video_analyses"].find_one({"_id": event_id})["analysis"]["verdict"]


def test_a_manual_request_re_runs_a_finished_analysis():
    analyzer = FakeAnalyzer()
    worker, db = make_worker(analyzer=analyzer)
    event_id = add_event(db, "fire")
    worker.run_once()
    worker.request(event_id)
    worker.run_once()
    assert analyzer.seen == [event_id, event_id]


def test_manual_requests_are_not_capped():
    worker, db = make_worker(settings=ai(daily_cap=0))
    worker.request(add_event(db, "fire"))
    assert worker.run_once() == 1


def test_a_manual_request_needs_a_known_alert_with_a_clip():
    worker, db = make_worker()
    with pytest.raises(AnalysisError, match="not found"):
        worker.request("0123456789abcdef01234567")
    with pytest.raises(AnalysisError, match="not found"):
        worker.request("not-an-id")
    with pytest.raises(AnalysisError, match="no saved clip"):
        worker.request(add_event(db, "fire", clip=False))


# -- the worker: failures ------------------------------------------------------------------------


def age(db, event_id, seconds):
    db["video_analyses"].update_one(
        {"_id": event_id}, {"$set": {"updated_at": now() - timedelta(seconds=seconds)}})


def test_a_failed_analysis_is_retried_later_up_to_the_limit():
    analyzer = FakeAnalyzer(error=RuntimeError("throttled"))
    worker, db = make_worker(analyzer=analyzer)
    event_id = add_event(db, "fire")
    worker.run_once()
    worker.run_once()  # too soon to retry
    assert len(analyzer.seen) == 1
    for _ in range(MAX_ATTEMPTS + 1):
        age(db, event_id, RETRY_AFTER_S + 1)
        worker.run_once()
    doc = db["video_analyses"].find_one({"_id": event_id})
    assert len(analyzer.seen) == MAX_ATTEMPTS and doc["status"] == "failed"
    assert doc["error"] == "throttled"


def test_an_error_no_retry_can_fix_is_not_retried():
    analyzer = FakeAnalyzer(error=AIUnavailableError("BEDROCK_API_KEY is not set"))
    worker, db = make_worker(analyzer=analyzer)
    event_id = add_event(db, "fire")
    worker.run_once()
    age(db, event_id, RETRY_AFTER_S + 1)
    worker.run_once()
    assert len(analyzer.seen) == 1


def test_a_frame_indexing_failure_does_not_fail_the_analysis():
    bedrock = FakeBedrock()
    bedrock.fail_embed = True
    worker, db = make_worker(bedrock=bedrock)
    event_id = add_event(db, "fire")
    worker.run_once()
    doc = db["video_analyses"].find_one({"_id": event_id})
    assert doc["status"] == "done" and doc["frames_status"] == "failed"
    assert "bedrock is down" in doc["frames_error"]


def test_a_job_cut_off_by_a_restart_is_picked_up_again():
    analyzer = FakeAnalyzer()
    worker, db = make_worker(analyzer=analyzer)
    event_id = add_event(db, "fire")
    worker._claim(worker._event(event_id))
    db["video_analyses"].update_one({"_id": event_id}, {"$set": {"status": "running"}})

    restarted = AnalysisWorker(ai(), db, analyzer, None)
    restarted.recover_interrupted()  # what start() does first
    restarted.run_once()
    assert db["video_analyses"].find_one({"_id": event_id})["status"] == "done"


def test_a_retry_for_an_alert_that_was_deleted_stops_retrying():
    analyzer = FakeAnalyzer(error=RuntimeError("throttled"))
    worker, db = make_worker(analyzer=analyzer)
    event_id = add_event(db, "fire")
    worker.run_once()
    db["events"].delete_many({})
    age(db, event_id, RETRY_AFTER_S + 1)
    worker.run_once()
    age(db, event_id, RETRY_AFTER_S + 1)
    assert worker._retry_ids() == []
    assert len(analyzer.seen) == 1


def test_retention_removes_old_analyses_and_their_frames():
    worker, db = make_worker()
    worker.retention_days = 30
    old, new = add_event(db, "fire"), add_event(db, "fire")
    worker.run_once()
    db["video_analyses"].update_one(
        {"_id": old}, {"$set": {"event_ts": now() - timedelta(days=31)}})
    worker.rag.index.replace_event(
        {**worker._event(old), "ts": now() - timedelta(days=31)}, [(0.5, "x", word_vector("x"))])
    worker._last_sweep = float("-inf")
    worker._sweep_retention()
    assert db["video_analyses"].find_one({"_id": old}) is None
    assert db["video_analyses"].find_one({"_id": new}) is not None
    assert not worker.rag.index.has_event(old) and worker.rag.index.has_event(new)


def test_the_worker_thread_analyses_in_the_background_and_stops_cleanly():
    worker, db = make_worker(settings=ai(poll_seconds=1.0))
    event_id = add_event(db, "fire")
    worker.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        doc = db["video_analyses"].find_one({"_id": event_id})
        if doc and doc["status"] == "done":
            break
        time.sleep(0.05)
    worker.stop(timeout=5)
    assert doc["status"] == "done" and worker._thread is None


# -- routes ----------------------------------------------------------------------------------


def build_app(tmp_path, worker):
    db = mongomock.MongoClient()["perimeter_test"]
    registry = CameraRegistry(db["cameras"])
    store = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, store, "unused-model-path.onnx")
    services = None if worker is None else AIServices(ai=worker.ai, db=db, worker=worker)
    return create_app(settings, store, registry, manager, users, ai_services=services)


def test_routes_answer_503_while_ai_is_off(tmp_path):
    client = authed_client(build_app(tmp_path, None))
    assert client.get("/api/ai/status").json() == {"available": False}
    assert client.post("/api/ai/events/abc/analyze").status_code == 503
    assert client.get("/api/ai/search", params={"q": "fire"}).status_code == 503


def test_analyse_then_read_the_result_over_the_api(tmp_path):
    worker, db = make_worker()
    event_id = add_event(db, "boundary")
    client = authed_client(build_app(tmp_path, worker))
    assert client.get(f"/api/ai/events/{event_id}/analysis").status_code == 404
    response = client.post(f"/api/ai/events/{event_id}/analyze")
    assert response.status_code == 202 and response.json()["status"] == "queued"
    worker.run_once()
    body = client.get(f"/api/ai/events/{event_id}/analysis").json()
    assert body["status"] == "done" and "verdict" not in body["analysis"]
    assert client.post("/api/ai/events/nope/analyze").status_code == 404


def test_search_and_ask_over_the_api(tmp_path):
    bedrock = FakeBedrock(captions={"ppe": ["worker without helmet"] * 3})
    worker, db = make_worker(bedrock=bedrock)
    add_event(db, "ppe")
    worker.run_once()
    client = authed_client(build_app(tmp_path, worker))
    assert client.get("/api/ai/status").json()["indexed_entries"] == 4
    results = client.get("/api/ai/search", params={"q": "helmet", "kind": "ppe"}).json()
    assert results["results"] and results["results"][0]["kind"] == "ppe"
    answer = client.post("/api/ai/ask", json={"question": "who has no helmet?"}).json()
    assert answer["answer"].startswith("Two people")
    assert client.post("/api/ai/ask", json={}).status_code == 400
    assert client.get("/api/ai/search", params={"q": "x", "since": "nonsense"}).status_code == 400


def test_a_bedrock_outage_is_a_502_not_a_crash(tmp_path):
    bedrock = FakeBedrock()
    worker, db = make_worker(bedrock=bedrock)
    add_event(db, "ppe")
    worker.run_once()
    bedrock.fail_embed = True
    client = authed_client(build_app(tmp_path, worker))
    assert client.get("/api/ai/search", params={"q": "helmet"}).status_code == 502
    assert client.post("/api/ai/ask", json={"question": "x"}).status_code == 502
