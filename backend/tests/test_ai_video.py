"""Phase J foundation: AI settings, the alert-type catalogue, and Pegasus video analysis -
all against a fake Bedrock client (no AWS, no cost)."""

from __future__ import annotations

import io
import json
import os

import cv2
import numpy as np
import pytest
from botocore.exceptions import ClientError

from perimeter.ai import catalog
from perimeter.ai.bedrock import AIUnavailableError, runtime_client
from perimeter.ai.video import (
    GENERIC_FOCUS,
    PegasusAnalyzer,
    S3UnreadableError,
    analysis_prompt,
    reencode_for_analysis,
)
from perimeter.alerts.cooldown import ALERT_KINDS
from perimeter.evidence.store import LocalEvidenceStore
from perimeter.settings import AISettings, load_settings

ANALYSIS = {"summary": "A fire burns on the left.", "people_count": 0,
            "activities": ["fire burning"], "verdict": "likely_real", "findings": "grows"}


def client_error(code, message):
    return ClientError({"Error": {"Code": code, "Message": message}}, "InvokeModel")


class FakeBedrock:
    def __init__(self, s3_error=None, reply=None):
        self.s3_error, self.reply, self.calls = s3_error, reply or ANALYSIS, []

    def invoke_model(self, modelId, body, contentType, accept):  # noqa: N803 - boto3's names
        request = json.loads(body)
        self.calls.append((modelId, request))
        location = request["mediaSource"].get("s3Location")
        if location is not None and "bucketOwner" not in location:  # the real API's rule
            raise client_error("ValidationException",
                               "Invalid JSON: s3Location requires both uri and bucketOwner")
        if location is not None and self.s3_error is not None:
            raise self.s3_error
        message = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        payload = {"message": message, "input_tokens": 190, "output_tokens": 45}
        return {"body": io.BytesIO(json.dumps(payload).encode())}


class FakeStore:
    def __init__(self, data=b"small-clip", remote=True):
        self.data, self.remote = data, remote

    def uri(self, key):
        return f"s3://evid/{key}" if self.remote else None

    def read(self, key):
        return self.data


EVENT = {"id": "e1", "kind": "fire", "message": "Fire detected", "zone_name": "Bay 4",
         "camera_id": "cam_01", "clip_path": "clips/cam_01/2026-10-27/e1.mp4"}
NOT_READABLE = client_error("ValidationException", "Provided S3Location not found")
OWNER = {"s3_bucket_owner": "139713843552"}


def ai(**overrides):
    return AISettings(**{"enabled": True, "api_key": "k", **overrides})


def make_clip(path, frames=40, fps=30.0, size=(1280, 720)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


# -- settings ----------------------------------------------------------------------------


def test_settings_defaults_and_the_key_comes_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BEDROCK_API_KEY", "secret-key")
    monkeypatch.delenv("PERIMETER_S3_BUCKET_OWNER", raising=False)
    cfg = tmp_path / "app.yaml"
    cfg.write_text("ai:\n  enabled: true\n", encoding="utf-8")
    s = load_settings(config_path=cfg, env_file=tmp_path / ".env").ai
    assert s.configured and s.api_key == "secret-key"
    assert s.video_input == "auto" and s.inline_max_mb == 10.0
    assert s.video_model == "global.twelvelabs.pegasus-1-5-v1:0"


def test_settings_reject_an_unknown_video_input(tmp_path):
    cfg = tmp_path / "app.yaml"
    cfg.write_text("ai:\n  video_input: ftp\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ai.video_input"):
        load_settings(config_path=cfg, env_file=tmp_path / ".env")


def test_runtime_client_needs_ai_enabled_and_a_key():
    with pytest.raises(AIUnavailableError, match="disabled"):
        runtime_client(AISettings(enabled=False, api_key="k"))
    with pytest.raises(AIUnavailableError, match="BEDROCK_API_KEY"):
        runtime_client(AISettings(enabled=True, api_key=""))


def test_runtime_client_uses_the_bedrock_key_as_a_bearer_token(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client = runtime_client(ai(api_key="abc"))
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "abc"
    assert client.meta.region_name == "ap-south-1"


# -- catalogue -----------------------------------------------------------------------------


def test_every_alert_type_is_in_the_catalogue():
    assert {row["kind"] for row in catalog.catalog()} == set(ALERT_KINDS)


def test_a_future_alert_type_gets_a_generic_description():
    assert "forklift_speeding" in catalog.describe_kind("forklift_speeding")


# -- prompts -------------------------------------------------------------------------------


def test_prompt_uses_the_alert_types_own_focus():
    assert "grows or spreads" in analysis_prompt(EVENT)
    assert "Bay 4" in analysis_prompt(EVENT)


def test_a_future_alert_type_is_analysed_with_the_generic_focus():
    prompt = analysis_prompt({**EVENT, "kind": "forklift_speeding"})
    assert GENERIC_FOCUS in prompt


# -- how the clip reaches the model ----------------------------------------------------------


def test_auto_reads_from_s3_when_bedrock_can():
    fake = FakeBedrock()
    result = PegasusAnalyzer(ai(**OWNER), FakeStore(), fake).analyze(EVENT)
    media = fake.calls[0][1]["mediaSource"]
    assert media == {"s3Location": {"uri": "s3://evid/" + EVENT["clip_path"],
                                    "bucketOwner": "139713843552"}}
    assert result.input_mode == "s3" and len(fake.calls) == 1


def test_auto_falls_back_to_inline_when_bedrock_cannot_read_s3():
    """Once the bucket owner is set but the key still lacks S3 read access."""
    fake = FakeBedrock(s3_error=NOT_READABLE)
    result = PegasusAnalyzer(ai(**OWNER), FakeStore(), fake).analyze(EVENT)
    assert "s3Location" in fake.calls[0][1]["mediaSource"]
    assert "base64String" in fake.calls[1][1]["mediaSource"]
    assert result.input_mode == "inline" and result.verdict == "likely_real"


def test_auto_without_a_bucket_owner_goes_straight_to_inline():
    """Today's .env: no PERIMETER_S3_BUCKET_OWNER, and Bedrock rejects an S3 location
    without one - so no doomed S3 attempt is made."""
    fake = FakeBedrock()
    result = PegasusAnalyzer(ai(), FakeStore(), fake).analyze(EVENT)
    assert len(fake.calls) == 1 and "base64String" in fake.calls[0][1]["mediaSource"]
    assert result.input_mode == "inline"


def test_s3_mode_says_clearly_when_bedrock_cannot_read_the_bucket():
    fake = FakeBedrock(s3_error=NOT_READABLE)
    with pytest.raises(S3UnreadableError):
        PegasusAnalyzer(ai(video_input="s3", **OWNER), FakeStore(), fake).analyze(EVENT)
    assert len(fake.calls) == 1  # no silent fallback in s3 mode


def test_s3_mode_without_a_bucket_owner_names_the_missing_setting():
    with pytest.raises(S3UnreadableError, match="PERIMETER_S3_BUCKET_OWNER"):
        PegasusAnalyzer(ai(video_input="s3"), FakeStore(), FakeBedrock()).analyze(EVENT)


def test_the_real_apis_lowercase_s3location_message_triggers_the_fallback():
    fake = FakeBedrock(s3_error=client_error(
        "ValidationException", "Invalid JSON: s3Location requires both uri and bucketOwner"))
    assert PegasusAnalyzer(ai(**OWNER), FakeStore(), fake).analyze(EVENT).input_mode == "inline"


def test_other_errors_are_not_mistaken_for_an_s3_problem():
    fake = FakeBedrock(s3_error=client_error("ThrottlingException", "Too many requests"))
    with pytest.raises(ClientError, match="Too many requests"):
        PegasusAnalyzer(ai(**OWNER), FakeStore(), fake).analyze(EVENT)


def test_inline_mode_never_tries_s3():
    fake = FakeBedrock()
    PegasusAnalyzer(ai(video_input="inline"), FakeStore(), fake).analyze(EVENT)
    assert list(fake.calls[0][1]["mediaSource"]) == ["base64String"]


def test_a_local_store_goes_inline_directly(tmp_path):
    store = LocalEvidenceStore(tmp_path, "snapshots", "clips")
    (tmp_path / EVENT["clip_path"]).parent.mkdir(parents=True)
    (tmp_path / EVENT["clip_path"]).write_bytes(b"local-clip")
    fake = FakeBedrock()
    result = PegasusAnalyzer(ai(), store, fake).analyze(EVENT)
    assert result.input_mode == "inline" and len(fake.calls) == 1


def test_an_oversized_clip_is_re_encoded_smaller_before_sending(tmp_path):
    big = make_clip(tmp_path / "big.mp4")
    fake = FakeBedrock()
    settings = ai(video_input="inline", inline_max_mb=0.0001)
    result = PegasusAnalyzer(settings, FakeStore(big), fake).analyze(EVENT)
    sent = __import__("base64").b64decode(fake.calls[0][1]["mediaSource"]["base64String"])
    assert result.input_mode == "inline_reencoded" and len(sent) < len(big)


def test_reencode_lowers_height_and_frame_rate(tmp_path):
    out = tmp_path / "out.mp4"
    out.write_bytes(reencode_for_analysis(make_clip(tmp_path / "in.mp4"), height=360, fps=6.0))
    cap = cv2.VideoCapture(str(out))
    height, frames = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert height == 360
    assert frames <= 10  # 40 frames at 30 fps -> ~1.3 s at 6 fps


def test_an_alert_without_a_clip_is_refused():
    with pytest.raises(ValueError, match="no saved clip"):
        PegasusAnalyzer(ai(), FakeStore(), FakeBedrock()).analyze({**EVENT, "clip_path": None})


# -- reading the answer ------------------------------------------------------------------------


def test_structured_answer_is_parsed_and_the_request_asks_for_json():
    fake = FakeBedrock()
    result = PegasusAnalyzer(ai(), FakeStore(), fake).analyze(EVENT)
    assert fake.calls[0][1]["responseFormat"]["type"] == "json_schema"
    assert (result.summary, result.people_count, result.verdict) == (
        "A fire burns on the left.", 0, "likely_real")
    assert result.input_tokens == 190


def test_an_unexpected_verdict_becomes_unsure():
    fake = FakeBedrock(reply={**ANALYSIS, "verdict": "definitely"})
    assert PegasusAnalyzer(ai(), FakeStore(), fake).analyze(EVENT).verdict == "unsure"


def test_a_non_json_answer_is_kept_as_the_summary():
    fake = FakeBedrock(reply="Fire on the left side.")
    result = PegasusAnalyzer(ai(), FakeStore(), fake).analyze(EVENT)
    assert result.summary == "Fire on the left side." and result.verdict == "unsure"


def test_ask_returns_free_text_without_a_schema():
    fake = FakeBedrock(reply="Two people, no helmets.")
    answer = PegasusAnalyzer(ai(), FakeStore(), fake).ask(EVENT, "Who has no helmet?")
    assert answer == "Two people, no helmets."
    assert "responseFormat" not in fake.calls[0][1]
