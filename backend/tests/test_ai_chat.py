"""Phase J steps 3-6: assistant chat with tools, memory, camera status history, suggested
questions, plain-English alert search and the daily digest - against an in-memory MongoDB
and a scripted fake Bedrock (no AWS, no cost)."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from datetime import UTC, date, datetime, timedelta

import mongomock
import pytest
from conftest import authed_client, seed_admin

from perimeter.ai.alert_search import AlertSearch
from perimeter.ai.bedrock import LazyClient
from perimeter.ai.camera_log import CameraStatusLog
from perimeter.ai.chat import ChatService, _history
from perimeter.ai.digest import DigestService
from perimeter.ai.memory import MemoryStore
from perimeter.ai.queries import alert_stats, event_query, in_hours, local_tz
from perimeter.ai.services import AIServices
from perimeter.ai.suggestions import suggestions
from perimeter.ai.tools import ToolBox, ToolContext
from perimeter.api.app import create_app
from perimeter.auth.models import Role, User
from perimeter.auth.passwords import hash_password
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.settings import AISettings, Settings, load_settings

DIMS = 256
OPERATOR = "operator@test.local"
OPERATOR_PASSWORD = "operator-password"


def ai(**overrides):
    return AISettings(**{"enabled": True, "api_key": "k", "embed_dims": DIMS, **overrides})


def word_vector(text: str) -> list[float]:
    vector = [0.0] * DIMS
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeBedrock:
    """Embeddings from words; `converse` replies from a queue (or a default); each
    `converse_stream` call plays the next script: a list of ("text", str) and
    ("tool", name, input) steps."""

    def __init__(self, streams=None, replies=None):
        self.streams = list(streams or [])
        self.replies = list(replies or [])
        self.stream_calls: list[dict] = []
        self.converse_calls: list[dict] = []

    def invoke_model(self, modelId, body, contentType, accept):  # noqa: N803
        text = json.loads(body)["inputText"]
        return {"body": io.BytesIO(json.dumps({"embedding": word_vector(text)}).encode())}

    def converse(self, **kwargs):
        self.converse_calls.append(kwargs)
        reply = self.replies.pop(0) if self.replies else "[]"
        if isinstance(reply, dict):  # a forced tool call
            content = [{"toolUse": {"toolUseId": "t1", "name": "set_filters", "input": reply}}]
        else:
            content = [{"text": reply}]
        return {"output": {"message": {"content": content}}}

    def converse_stream(self, **kwargs):
        self.stream_calls.append(json.loads(json.dumps(kwargs, default=str)))
        script = self.streams.pop(0) if self.streams else [("text", "ok")]
        events = [{"messageStart": {"role": "assistant"}}]
        stop = "end_turn"
        for index, step in enumerate(script):
            if step[0] == "text":
                events.append({"contentBlockDelta": {"contentBlockIndex": index,
                                                     "delta": {"text": step[1]}}})
            else:
                stop = "tool_use"
                events.append({"contentBlockStart": {"contentBlockIndex": index, "start": {
                    "toolUse": {"toolUseId": f"use{index}", "name": step[1],
                                "type": "tool_use"}}}})
                raw = json.dumps(step[2])
                for piece in (raw[: len(raw) // 2], raw[len(raw) // 2:]):
                    events.append({"contentBlockDelta": {"contentBlockIndex": index,
                                                         "delta": {"toolUse": {"input": piece}}}})
            events.append({"contentBlockStop": {"contentBlockIndex": index}})
        events.append({"messageStop": {"stopReason": stop}})
        return {"stream": iter(events)}


class FakeStore:
    def read(self, key):
        return b"\xff\xd8jpeg-bytes"

    def uri(self, key):
        return None


def now():
    return datetime.now(UTC)


def add_event(db, kind="fire", ts=None, camera_id="cam_01", test_video=False, **extra):
    doc = {"kind": kind, "camera_id": camera_id, "zone_name": "Gate 2",
           "message": ("[TEST VIDEO] " if test_video else "") + f"{kind} detected",
           "ts": ts or now(), "severity": "high", "snapshot_path": "snapshots/x.jpg", **extra}
    return str(db["events"].insert_one(doc).inserted_id)


def fresh_db():
    return mongomock.MongoClient()["perimeter_test"]


def cameras():
    return [{"id": "cam_01", "name": "Gate", "enabled": True, "state": "live",
             "modules": ["fire"], "boundaries": ["Gate 2"]}]


def build(db=None, bedrock=None, settings=None, with_memory=True, camera_log=None):
    db = db if db is not None else fresh_db()
    bedrock = bedrock or FakeBedrock()
    settings = settings or ai()
    lazy = LazyClient(settings, bedrock)
    memory = MemoryStore(settings, db, lazy) if with_memory else None
    tools = ToolBox(db, FakeStore(), cameras, memory=memory, camera_log=camera_log)
    chat = ChatService(settings, db, lazy, tools, memory)
    return chat, db, bedrock, memory


def run_turn(chat, message, **kwargs):
    return list(chat.stream(kwargs.pop("user_id", "admin@test.local"), message, **kwargs))


# -- queries -----------------------------------------------------------------------------------


def test_event_query_filters_by_type_zone_time_and_source():
    db = fresh_db()
    add_event(db, "fire")
    add_event(db, "fire", test_video=True)
    add_event(db, "ppe", zone_name="Bay 4")
    add_event(db, "fire", ts=now() - timedelta(days=3))
    since = (now() - timedelta(days=1)).isoformat()
    count = db["events"].count_documents
    assert count(event_query(kind="fire", since=since)) == 2
    assert count(event_query(kind="fire", since=since, source="real")) == 1
    assert count(event_query(source="test")) == 1
    assert count(event_query(zone="bay")) == 1
    with pytest.raises(ValueError, match="source"):
        event_query(source="bogus")


def test_hours_filter_wraps_past_midnight():
    at = lambda hour: datetime(2026, 10, 27, hour, tzinfo=local_tz())  # noqa: E731
    assert in_hours(at(23), 22, 6) and in_hours(at(3), 22, 6) and not in_hours(at(12), 22, 6)
    assert in_hours(at(9), 8, 17) and not in_hours(at(17), 8, 17)
    assert in_hours(at(5), None, None)


def test_stats_by_day_list_days_without_alerts():
    db = fresh_db()
    base = datetime.now(local_tz()).replace(hour=12, minute=0, second=0, microsecond=0)
    for days_ago in (4, 2, 0):  # every other day
        add_event(db, "fire", ts=(base - timedelta(days=days_ago)).astimezone(UTC))
    since = (base - timedelta(days=4)).replace(hour=0).isoformat()
    result = alert_stats(db["events"], ["day"], since=since)
    assert [r["count"] for r in result["rows"]] == [1, 0, 1, 0, 1]


def test_stats_by_hour_use_local_time():
    db = fresh_db()
    local = datetime.now(local_tz()).replace(hour=2, minute=30)
    add_event(db, "fire", ts=local.astimezone(UTC))
    rows = alert_stats(db["events"], ["kind", "hour"])["rows"]
    assert rows == [{"kind": "fire", "hour": "02:00", "count": 1}]
    with pytest.raises(ValueError, match="cannot group"):
        alert_stats(db["events"], ["colour"])


# -- camera status history ---------------------------------------------------------------------


def test_camera_log_keeps_only_changes_and_reports_outages():
    db = fresh_db()
    states = {"cam_01": "live"}
    log = CameraStatusLog(db, lambda: dict(states), interval_s=30)
    t0 = datetime(2026, 10, 27, 1, 0, tzinfo=UTC)
    log.sample(t0)
    log.sample(t0 + timedelta(seconds=30))
    states["cam_01"] = "lost"
    log.sample(t0 + timedelta(minutes=1))
    states["cam_01"] = "live"
    log.sample(t0 + timedelta(minutes=61))
    assert db["camera_status"].count_documents({}) == 3
    periods = log.history(t0, t0 + timedelta(hours=2))
    assert [(p["state"], p["minutes"]) for p in periods] == [("lost", 60.0)]


def test_time_the_server_was_off_shows_as_no_record():
    db = fresh_db()
    log = CameraStatusLog(db, lambda: {"cam_01": "live"}, interval_s=30)
    t0 = datetime(2026, 10, 27, 1, 0, tzinfo=UTC)
    log.sample(t0)
    log.close_open_rows()  # restart
    log.sample(t0 + timedelta(minutes=45))
    periods = log.history(t0, t0 + timedelta(hours=1))
    assert [(p["state"], p["minutes"]) for p in periods] == [("no_record", 45.0)]


# -- memory ------------------------------------------------------------------------------------


def test_memory_scopes_visibility_and_duplicates():
    _chat, db, _bedrock, memory = build()
    memory.add("Camera cam_02 is switched off from 2 to 3 am for cleaning", "system",
               user_id="a@x")
    memory.add("Prefers short answers", "user", user_id="a@x")
    event_id = add_event(db)
    memory.add("This fire alert was a welding test", "alert", user_id="a@x", event_id=event_id)
    again = memory.add("Camera cam_02 is switched off from 2 to 3 am for cleaning daily",
                       "system", user_id="b@x")
    assert db["ai_memories"].count_documents({"scope": "system"}) == 1  # merged, not doubled
    assert "daily" in again["text"]
    assert {m["scope"] for m in memory.visible("a@x")} == {"system", "user", "alert"}
    assert {m["scope"] for m in memory.visible("b@x")} == {"system", "alert"}
    with pytest.raises(ValueError, match="alert id"):
        memory.add("x", "alert", user_id="a@x")


def test_recall_puts_the_alerts_own_memories_first_and_respects_users():
    _chat, db, _bedrock, memory = build()
    event_id = add_event(db)
    memory.add("Gate 2 floods when it rains", "system", user_id="a@x")
    memory.add("Responsible for gate 2", "user", user_id="a@x")
    memory.add("False alarm from sunlight", "alert", user_id="a@x", event_id=event_id)
    found = memory.recall("gate 2", "b@x", event_id=event_id)
    assert found[0]["text"] == "False alarm from sunlight"
    assert all(m["scope"] != "user" for m in found)  # b@x never sees a@x's own memory


def test_only_an_admin_or_the_author_deletes_shared_memories():
    _chat, _db, _bedrock, memory = build()
    site = memory.add("Night shift starts at 22:00", "system", user_id="a@x")
    with pytest.raises(PermissionError):
        memory.delete(site["id"], "b@x", is_admin=False)
    assert memory.delete(site["id"], "admin@x", is_admin=True)
    own = memory.add("Likes tables", "user", user_id="a@x")
    with pytest.raises(PermissionError):
        memory.delete(own["id"], "admin@x", is_admin=True)  # private to its owner
    assert memory.delete(own["id"], "a@x", is_admin=False)


def test_extraction_keeps_durable_facts_from_a_conversation():
    bedrock = FakeBedrock(replies=[json.dumps([
        {"scope": "system", "text": "cam_03 faces the loading bay"},
        {"scope": "nonsense", "text": "dropped"},
    ])])
    _chat, db, _bedrock, memory = build(bedrock=bedrock)
    saved = memory.extract([{"role": "user", "content": "cam_03 faces the loading bay"}],
                           "a@x")
    assert [m["text"] for m in saved] == ["cam_03 faces the loading bay"]
    assert db["ai_memories"].find_one()["source"] == "auto"


def test_the_memory_index_is_rebuilt_from_mongodb():
    db = fresh_db()
    lazy = LazyClient(ai(), FakeBedrock())
    first = MemoryStore(ai(), db, lazy)
    first.add("Gate 2 floods when it rains", "system", user_id="a@x")
    second = MemoryStore(ai(), db, lazy)  # e.g. the index folder was deleted
    assert second.recall("gate", "a@x") == []
    assert second.sync_index() == 1
    assert second.recall("gate", "a@x")[0]["text"] == "Gate 2 floods when it rains"


# -- tools -------------------------------------------------------------------------------------


def test_list_alerts_shows_the_video_summary_but_never_the_verdict():
    _chat, db, _bedrock, _memory = build()
    event_id = add_event(db, "fire", clip_path="clips/x.mp4")
    db["video_analyses"].insert_one({"_id": event_id, "status": "done", "analysis": {
        "summary": "Flames grow on the left", "verdict": "likely_real"}})
    tools = ToolBox(db, FakeStore(), cameras)
    result = tools.run("list_alerts", ToolContext("a@x"), {"kind": "fire"})
    text = json.dumps(result.data)
    assert result.data["alerts"][0]["video_summary"] == "Flames grow on the left"
    assert "likely_real" not in text and "verdict" not in text


def test_alert_tools_default_to_the_chats_own_alert():
    _chat, db, _bedrock, _memory = build()
    event_id = add_event(db, "ppe")
    tools = ToolBox(db, FakeStore(), cameras)
    ctx = ToolContext("a@x", event_id=event_id)
    assert tools.run("alert_details", ctx, {}).data["id"] == event_id
    snapshot = tools.run("look_at_snapshot", ctx, {})
    assert snapshot.images and any("image" in block for block in snapshot.content())
    with pytest.raises(ValueError, match="not found"):
        tools.run("alert_details", ToolContext("a@x"), {"event_id": "nope"})


def test_camera_outages_tool_counts_start_hours():
    db = fresh_db()
    states = {"cam_01": "live"}
    log = CameraStatusLog(db, lambda: dict(states))
    day = datetime.now(local_tz()).replace(hour=2, minute=0, second=0, microsecond=0)
    for days_ago in (2, 1):
        t = (day - timedelta(days=days_ago)).astimezone(UTC)
        states["cam_01"] = "live"
        log.sample(t - timedelta(minutes=1))
        states["cam_01"] = "lost"
        log.sample(t)
        states["cam_01"] = "live"
        log.sample(t + timedelta(minutes=60))
    tools = ToolBox(db, FakeStore(), cameras, camera_log=log)
    data = tools.run("camera_outages", ToolContext("a@x"), {}).data
    assert data["per_camera"]["cam_01"]["start_hours"] == {"02:00": 2}
    assert data["per_camera"]["cam_01"]["minutes"] == 120.0


def test_the_help_guide_lists_every_alert_type():
    tools = ToolBox(fresh_db(), FakeStore(), cameras)
    guide = tools.run("product_help", ToolContext("a@x"), {}).data
    assert "Boundaries" in guide and "fire:" in guide and "ppe:" in guide


# -- chat --------------------------------------------------------------------------------------


def test_a_reply_streams_and_the_conversation_is_kept():
    bedrock = FakeBedrock(streams=[[("text", "Hello "), ("text", "there.")]])
    chat, _db, _bedrock, _memory = build(bedrock=bedrock, with_memory=False)
    events = run_turn(chat, "hi")
    assert [e["type"] for e in events] == ["conversation", "text", "text", "done"]
    convo = chat.get(events[0]["id"], "admin@test.local")
    assert [m["content"] for m in convo["messages"]] == ["hi", "Hello there."]
    assert chat.list("admin@test.local")[0]["title"] == "hi"
    assert chat.get(events[0]["id"], "someone-else") is None


def test_the_model_looks_things_up_with_tools_before_answering():
    bedrock = FakeBedrock(streams=[
        [("text", "Let me check. "), ("tool", "alert_stats", {"group_by": ["kind"]})],
        [("text", "There were 2 fire alerts.")],
    ])
    chat, db, _bedrock, _memory = build(bedrock=bedrock, with_memory=False)
    add_event(db, "fire")
    add_event(db, "fire")
    events = run_turn(chat, "how many fire alerts?")
    tool_events = [e for e in events if e["type"] == "tool"]
    assert tool_events == [{"type": "tool", "name": "alert_stats", "label": "Counting alerts"}]
    second = bedrock.stream_calls[1]["messages"]
    assert second[-2]["content"][-1]["toolUse"] == {
        "toolUseId": "use1", "name": "alert_stats", "input": {"group_by": ["kind"]}}
    result = second[-1]["content"][0]["toolResult"]
    assert '"total": 2' in result["content"][0]["text"]
    assert events[-1] == {"type": "done", "message_id": events[-1]["message_id"],
                          "tools": ["alert_stats"]}


def test_a_failing_tool_is_reported_to_the_model_not_the_user():
    bedrock = FakeBedrock(streams=[
        [("tool", "alert_details", {"event_id": "missing"})],
        [("text", "I could not find that alert.")],
    ])
    chat, _db, _bedrock, _memory = build(bedrock=bedrock, with_memory=False)
    events = run_turn(chat, "open alert missing")
    result = bedrock.stream_calls[1]["messages"][-1]["content"][0]["toolResult"]
    assert result["status"] == "error" and "not found" in result["content"][0]["text"]
    assert events[-1]["type"] == "done"


def test_a_follow_up_sends_the_earlier_turns():
    bedrock = FakeBedrock(streams=[[("text", "A.")], [("text", "B.")]])
    chat, _db, _bedrock, _memory = build(bedrock=bedrock, with_memory=False)
    first = run_turn(chat, "one")
    run_turn(chat, "two", conversation_id=first[0]["id"])
    sent = bedrock.stream_calls[1]["messages"]
    assert [(m["role"], m["content"][0]["text"]) for m in sent] == [
        ("user", "one"), ("assistant", "A."), ("user", "two")]


def test_history_is_always_user_first_and_alternating():
    stored = [{"role": "assistant", "content": "orphan"}, {"role": "user", "content": "q1"},
              {"role": "user", "content": "q2"}, {"role": "assistant", "content": ""}]
    assert [(m["role"], m["content"][0]["text"]) for m in _history(stored, 20)] == [
        ("user", "q1\n\nq2"), ("assistant", "(no reply)")]


def test_an_alert_chat_knows_its_alert_and_its_memories():
    bedrock = FakeBedrock(streams=[[("text", "It is the gate camera.")]])
    chat, db, _bedrock, memory = build(bedrock=bedrock)
    event_id = add_event(db, "fire")
    memory.add("This one was a welding test", "alert", user_id="a@x", event_id=event_id)
    events = run_turn(chat, "what is this?", event_id=event_id)
    system = bedrock.stream_calls[0]["system"][0]["text"]
    assert event_id in system and "This one was a welding test" in system
    assert chat.list("admin@test.local", event_id=event_id)[0]["id"] == events[0]["id"]
    assert chat.list("admin@test.local") == []  # alert chats stay under their alert


def test_a_bedrock_outage_ends_the_stream_with_an_error():
    class Down(FakeBedrock):
        def converse_stream(self, **kwargs):
            raise RuntimeError("bedrock is down")

    chat, _db, _bedrock, _memory = build(bedrock=Down(), with_memory=False)
    events = run_turn(chat, "hi")
    assert events[-1]["type"] == "error" and "bedrock is down" in events[-1]["message"]


def test_each_reply_feeds_long_term_memory():
    bedrock = FakeBedrock(streams=[[("text", "Noted.")]], replies=[json.dumps(
        [{"scope": "user", "text": "Looks after Bay 4"}])])
    chat, db, _bedrock, _memory = build(bedrock=bedrock)
    run_turn(chat, "I look after Bay 4", user_id="a@x")
    assert db["ai_memories"].find_one({"scope": "user"})["text"] == "Looks after Bay 4"


# -- suggestions -------------------------------------------------------------------------------


def test_suggestions_follow_the_alert_type_and_recent_events():
    db = fresh_db()
    ppe = add_event(db, "ppe", clip_path="clips/x.mp4")
    assert suggestions(db, event_id=ppe)[0] == "Who is missing a helmet?"
    future = add_event(db, "forklift_speeding", clip_path="clips/y.mp4")
    assert suggestions(db, event_id=future)[0] == "What happens in this clip?"
    assert suggestions(db, view="history")[0] == "Summarise this week's alerts by type"
    add_event(db, "fire")
    assert suggestions(db, view="history")[0] == "Tell me about today's fire alerts"
    assert len(suggestions(fresh_db(), view="cameras")) == 4


# -- plain-English search ----------------------------------------------------------------------


def test_alert_search_turns_words_into_filters_then_queries():
    db = fresh_db()
    night = datetime.now(local_tz()).replace(hour=23, minute=0) - timedelta(days=1)
    add_event(db, "fire", ts=night.astimezone(UTC))
    add_event(db, "fire", ts=night.replace(hour=13).astimezone(UTC))
    add_event(db, "ppe", ts=night.astimezone(UTC))
    bedrock = FakeBedrock(replies=[{"kind": "fire", "hour_from": 22, "hour_to": 6,
                                    "explanation": "Fire alerts at night"}])
    search = AlertSearch(ai(), db, LazyClient(ai(), bedrock), cameras)
    result = search.search("fire at night")
    assert result["explanation"] == "Fire alerts at night"
    assert len(result["events"]) == 1 and result["events"][0]["kind"] == "fire"
    sent = bedrock.converse_calls[0]
    assert sent["toolConfig"]["toolChoice"] == {"tool": {"name": "set_filters"}}


# -- daily digest ------------------------------------------------------------------------------


def test_the_digest_counts_yesterday_and_spots_same_hour_repeats():
    db = fresh_db()
    today = datetime.now(local_tz()).date()
    yesterday = today - timedelta(days=1)
    at = lambda d, h: datetime(d.year, d.month, d.day, h, 5, tzinfo=local_tz()).astimezone(UTC)  # noqa: E731
    for days_ago in range(1, 6):  # 02:05 on five days running
        add_event(db, "boundary", ts=at(today - timedelta(days=days_ago), 2))
    add_event(db, "fire", ts=at(yesterday, 14))
    add_event(db, "fire", ts=at(yesterday, 15), test_video=True)
    service = DigestService(ai(), db, LazyClient(ai(), FakeBedrock()))
    stats = service.stats(yesterday)
    assert stats["total"] == 2 and stats["by_type"] == {"boundary": 1, "fire": 1}
    assert stats["video_test_alerts"] == 1
    assert {"kind": "boundary", "camera": "cam_01", "hour": "02:00",
            "days_of_last_8": 5} in stats["recurring_same_hour"]


def test_the_digest_is_made_once_a_day_after_its_time_and_emailed():
    db = fresh_db()
    sent = []
    service = DigestService(ai(digest_time="08:00"), db,
                            LazyClient(ai(), FakeBedrock(replies=["Quiet day."])),
                            send=sent.append, from_addr="p@x", recipients=lambda: ["boss@x"])
    morning = datetime.now(local_tz()).replace(hour=7, minute=59)
    assert service.run_due(morning) is None
    made = service.run_due(morning.replace(hour=8, minute=0))
    assert made["text"] == "Quiet day." and made["emailed_to"] == ["boss@x"]
    assert sent[0]["To"] == "boss@x" and "Quiet day." in sent[0].get_content()
    assert service.run_due(morning.replace(hour=9)) is None  # not twice
    assert service.latest()[0]["day"] == (morning.date() - timedelta(days=1)).isoformat()


def test_a_failed_digest_email_still_keeps_the_digest():
    def broken(_message):
        raise OSError("smtp down")

    service = DigestService(ai(), fresh_db(), LazyClient(ai(), FakeBedrock(replies=["x"])),
                            send=broken, recipients=lambda: ["a@x"])
    made = service.generate(date(2026, 10, 26))
    assert made["emailed_to"] == [] and "smtp down" in made["email_error"]


def test_digest_settings_are_validated(tmp_path):
    cfg = tmp_path / "app.yaml"
    cfg.write_text('ai:\n  digest_time: "8am"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="digest_time"):
        load_settings(config_path=cfg, env_file=tmp_path / ".env")


# -- routes ------------------------------------------------------------------------------------


def build_app(tmp_path, bedrock=None):
    db = fresh_db()
    registry = CameraRegistry(db["cameras"])
    zones = ZoneStore(db["zones"], refresh_interval=0)
    users = seed_admin(db["users"])
    users.create(User(id=OPERATOR, email=OPERATOR, role=Role.OPERATOR,
                      password_hash=hash_password(OPERATOR_PASSWORD, rounds=4)))
    settings = Settings(env_file=tmp_path / ".env")
    manager = PipelineManager(settings, registry, zones, "unused-model-path.onnx")
    chat, _db, bedrock, memory = build(db=db, bedrock=bedrock)
    lazy = LazyClient(ai(), bedrock)
    services = AIServices(
        ai=ai(), db=db, chat=chat, memory=memory,
        search=AlertSearch(ai(), db, lazy, cameras),
        digest=DigestService(ai(), db, lazy),
    )
    app = create_app(settings, zones, registry, manager, users, ai_services=services)
    return app, db, bedrock


def sse(response):
    lines = response.text.splitlines()
    return [json.loads(line[6:]) for line in lines if line.startswith("data: ")]


def test_chat_streams_over_the_api_and_conversations_are_private(tmp_path):
    app, _db, _bedrock = build_app(tmp_path, FakeBedrock(streams=[[("text", "Hi!")]]))
    admin = authed_client(app)
    response = admin.post("/api/ai/chat", json={"message": "hello"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse(response)
    assert events[1] == {"type": "text", "text": "Hi!"} and events[-1]["type"] == "done"
    convo_id = events[0]["id"]
    assert admin.get(f"/api/ai/conversations/{convo_id}").json()["messages"][1]["content"] == "Hi!"
    operator = authed_client(app, OPERATOR, OPERATOR_PASSWORD)
    assert operator.get(f"/api/ai/conversations/{convo_id}").status_code == 404
    assert operator.get("/api/ai/conversations").json() == {"conversations": []}
    assert admin.delete(f"/api/ai/conversations/{convo_id}").status_code == 204


def test_memories_over_the_api(tmp_path):
    app, _db, _bedrock = build_app(tmp_path)
    operator = authed_client(app, OPERATOR, OPERATOR_PASSWORD)
    admin = authed_client(app)
    site = admin.post("/api/ai/memories", json={"text": "Shift change at 06:00",
                                                "scope": "system"}).json()
    mine = operator.post("/api/ai/memories", json={"text": "Watches Bay 4"}).json()
    assert mine["scope"] == "user"
    assert {m["text"] for m in operator.get("/api/ai/memories").json()["memories"]} == {
        "Shift change at 06:00", "Watches Bay 4"}
    assert operator.delete(f"/api/ai/memories/{site['id']}").status_code == 403
    assert admin.delete(f"/api/ai/memories/{site['id']}").status_code == 204
    bad = operator.post("/api/ai/memories", json={"text": "x", "scope": "planet"})
    assert bad.status_code == 400


def test_digests_search_and_suggestions_over_the_api(tmp_path):
    bedrock = FakeBedrock(replies=["A quiet day.", {"kind": "fire", "explanation": "Fire"}])
    app, db, _bedrock = build_app(tmp_path, bedrock)
    add_event(db, "fire")
    admin = authed_client(app)
    operator = authed_client(app, OPERATOR, OPERATOR_PASSWORD)
    assert operator.post("/api/ai/digests/run", json={}).status_code == 403
    made = admin.post("/api/ai/digests/run", json={"day": "2026-10-26"}).json()
    assert made["text"] == "A quiet day." and made["emailed_to"] == []
    assert operator.get("/api/ai/digests").json()["digests"][0]["day"] == "2026-10-26"
    found = operator.get("/api/ai/alert-search", params={"q": "fire"}).json()
    assert found["explanation"] == "Fire" and len(found["events"]) == 1
    assert operator.get("/api/ai/suggestions", params={"view": "cameras"}).json()["suggestions"]
    status = operator.get("/api/ai/status").json()
    assert status["chat"] and status["memory"] and status["digest"]
