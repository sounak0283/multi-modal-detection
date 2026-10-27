"""Video analysis of alert clips with TwelveLabs Pegasus on Bedrock (Phase J).

How a clip reaches the model (`ai.video_input`):
  s3     - Bedrock reads it straight from the evidence bucket: full quality, nothing uploaded
           from this machine, no size problem. Needs the Bedrock key's identity to have
           s3:GetObject on the bucket's clips/ (a separate AWS account today, so it is denied).
  inline - the clip bytes are sent in the request. Over `inline_max_mb` they are re-encoded to
           `inline_height` @ `inline_fps` first: measured on a 43 MB, 34 fps clip that Bedrock
           rejected ("Input is too long"), 720p @ 6 fps came to 1.4 MB and answered fastest.
  auto   - s3, falling back to inline when Bedrock cannot read the bucket.

Questions are chosen per alert type with a generic default, so a new detection module's
alerts are analysed without changing this file. The stored evidence clip is never modified.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2

from perimeter.ai.catalog import describe_kind
from perimeter.settings import AISettings

log = logging.getLogger("perimeter.ai.video")

VERDICTS = ("likely_real", "likely_false", "unsure")

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "people_count": {"type": "integer"},
        "activities": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "findings": {"type": "string"},
    },
    "required": ["summary", "people_count", "activities", "verdict", "findings"],
}

# What to look for, per alert type. Anything not listed uses GENERIC_FOCUS.
FOCUS = {
    "fire": "findings: where the fire is in the frame, how big, and whether it grows or spreads.",
    "smoke": "findings: where the smoke is, its source if visible, and whether it is spreading.",
    "crowd": "findings: how many people are grouped, what they are doing, and whether it looks "
             "like a normal gathering or a problem (argument, fight, panic).",
    "ppe": "findings: go through each visible person separately - what they wear (shirt "
           "colour), whether THEY wear a helmet and a vest, and what they do. Say 'not "
           "visible' when a head is hidden or unclear.",
    "access": "findings: who entered the restricted area, what they did there, and whether "
              "they left.",
    "boundary": "findings: who crossed the boundary, in which direction, and what they did.",
    "welding": "findings: whether the bright light is welding work or something else (fire); "
               "whether the welder is visible, and if so what THEY wear on the head and face; "
               "then each other person separately with their own helmet status.",
}
GENERIC_FOCUS = "findings: anything in the clip relevant to why this alert was raised."


class S3UnreadableError(RuntimeError):
    """Bedrock could not read the clip from S3 (usually: no read permission)."""


@dataclass
class VideoAnalysis:
    summary: str
    people_count: int | None
    activities: list[str]
    verdict: str
    findings: str
    model: str
    input_mode: str  # s3 | inline | inline_reencoded
    seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analysis_prompt(event: dict[str, Any]) -> str:
    kind = str(event.get("kind") or "alert")
    where = f" in zone {event['zone_name']}" if event.get("zone_name") else ""
    return (
        f"This CCTV clip raised a '{kind}' alert{where}: \"{event.get('message', '')}\". "
        f"{describe_kind(kind)} "
        "summary: 1-2 sentences describing what happens in the clip. "
        "people_count: how many people are visible. activities: short phrases. "
        "verdict: likely_real if the clip shows what the alert claims, likely_false if it "
        "clearly does not, unsure otherwise. "
        "Describe each person by what they wear; never attribute one person's equipment to "
        "another, and say 'not visible' rather than guess. "
        f"{FOCUS.get(kind, GENERIC_FOCUS)}"
    )


def reencode_for_analysis(data: bytes, height: int, fps: float) -> bytes:
    """A smaller copy for the model: scaled down to `height` (never up) and sampled to `fps`
    over the clip's full duration. H.264 first; MPEG-4 if no H.264 encoder is available."""
    with tempfile.TemporaryDirectory(prefix="perimeter-ai-") as tmp:
        src, dst = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
        src.write_bytes(data)
        for fourcc in ("avc1", "mp4v"):
            # Frames are streamed one at a time: a 10 s, 34 fps 1080p clip held whole in
            # memory would be ~2 GB.
            if _reencode_file(src, dst, fourcc, height, fps):
                return dst.read_bytes()
        raise RuntimeError("no video encoder available to re-encode the clip")


def _reencode_file(src: Path, dst: Path, fourcc: str, height: int, fps: float) -> bool:
    cap = cv2.VideoCapture(str(src))
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        step = max(1.0, src_fps / fps)
        writer = None
        size: tuple[int, int] = (0, 0)
        index, next_keep = 0, 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                src_h, src_w = frame.shape[:2]
                scale = min(1.0, height / src_h)
                size = (int(src_w * scale) // 2 * 2, int(src_h * scale) // 2 * 2)
                writer = cv2.VideoWriter(
                    str(dst), cv2.VideoWriter_fourcc(*fourcc), src_fps / step, size
                )
                if not writer.isOpened():
                    return False
            if index >= next_keep:
                if frame.shape[1::-1] != size:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                writer.write(frame)
                next_keep += step
            index += 1
        if writer is None:
            raise ValueError("clip has no readable frames")
        writer.release()
        return dst.is_file() and dst.stat().st_size > 0
    finally:
        cap.release()


def _is_s3_unreadable(exc: Exception) -> bool:
    text = str(exc).lower()
    code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
    return "s3location" in text or "s3 location" in text or (
        code == "AccessDeniedException" and "s3" in text
    )


class PegasusAnalyzer:
    def __init__(self, ai: AISettings, store: Any, client: Any = None) -> None:
        self.ai = ai
        self.store = store
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from perimeter.ai.bedrock import runtime_client

            self._client = runtime_client(self.ai)
        return self._client

    # -- public ------------------------------------------------------------------

    def analyze(self, event: dict[str, Any]) -> VideoAnalysis:
        """Structured analysis of one alert's clip."""
        started = time.perf_counter()
        out, mode = self._invoke(event, {
            "inputPrompt": analysis_prompt(event),
            "temperature": 0,
            "responseFormat": {"type": "json_schema", "jsonSchema": ANALYSIS_SCHEMA},
        })
        try:
            fields = json.loads(out.get("message") or "{}")
        except json.JSONDecodeError:
            fields = {"summary": out.get("message", ""), "verdict": "unsure"}
        verdict = fields.get("verdict") if fields.get("verdict") in VERDICTS else "unsure"
        return VideoAnalysis(
            summary=str(fields.get("summary", "")),
            people_count=fields.get("people_count"),
            activities=[str(a) for a in fields.get("activities") or []],
            verdict=verdict,
            findings=str(fields.get("findings", "")),
            model=self.ai.video_model,
            input_mode=mode,
            seconds=round(time.perf_counter() - started, 2),
            input_tokens=out.get("input_tokens"),
            output_tokens=out.get("output_tokens"),
            raw=out,
        )

    def ask(self, event: dict[str, Any], question: str) -> str:
        """A free-text answer to one question about an alert's clip."""
        out, _mode = self._invoke(event, {"inputPrompt": question, "temperature": 0})
        return str(out.get("message", ""))

    # -- internals ---------------------------------------------------------------

    def _invoke(self, event: dict[str, Any], body: dict[str, Any]) -> tuple[dict[str, Any], str]:
        key = event.get("clip_path")
        if not key:
            raise ValueError(f"alert {event.get('id')} has no saved clip")

        mode = self.ai.video_input
        uri = self.store.uri(key) if hasattr(self.store, "uri") else None
        is_s3 = bool(uri) and str(uri).startswith("s3://")
        # Bedrock rejects an S3 location without the bucket owner's account id (verified),
        # so without one there is nothing to try - auto goes straight to inline.
        if mode == "s3" and is_s3 and not self.ai.s3_bucket_owner:
            raise S3UnreadableError(
                "ai.video_input is s3 but PERIMETER_S3_BUCKET_OWNER (the evidence bucket's "
                "AWS account id) is not set in .env"
            )
        if mode in ("s3", "auto") and is_s3 and self.ai.s3_bucket_owner:
            location = {"uri": uri, "bucketOwner": self.ai.s3_bucket_owner}
            try:
                return self._call({**body, "mediaSource": {"s3Location": location}}), "s3"
            except Exception as exc:  # noqa: BLE001 - decide below whether to fall back
                if not _is_s3_unreadable(exc):
                    raise
                if mode == "s3":
                    raise S3UnreadableError(str(exc)) from exc
                log.info("Bedrock cannot read %s from S3 - sending the clip inline", uri)

        data = self.store.read(key)
        input_mode = "inline"
        if len(data) > self.ai.inline_max_mb * 1e6:
            data = reencode_for_analysis(data, self.ai.inline_height, self.ai.inline_fps)
            input_mode = "inline_reencoded"
        media = {"base64String": base64.b64encode(data).decode("ascii")}
        return self._call({**body, "mediaSource": media}), input_mode

    def _call(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self.client.invoke_model(
            modelId=self.ai.video_model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())
