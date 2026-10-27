"""Frame RAG over alert clips (Phase J step 2b).

Video search without Bedrock reading S3: this machine pulls a few keyframes out of each
clip and the chat model (Claude, India route) writes a one-line caption per frame. Each
caption is stored with its alert type ("fire alert, zone Bay 4: ...") and embedded by
`embed_model` (Titan Text v2, India route). When the clip also has a Pegasus analysis, its
summary is stored as one more "whole clip" entry - Pegasus sees motion (a fire growing),
which still frames cannot.

A question is embedded the same way, the nearest entries come back with their alert,
camera, time and caption, and Claude answers from them, looking at the frames themselves
(re-read from the clip: frame images are never stored, only captions and vectors).

Measured on real clips (2026-10-27): Titan Multimodal's image vectors never won against a
typed question and its text side ranked "person without a helmet" onto fire frames; Titan
Text v2 over the captions put all six PPE frames first. Hence text vectors.
"""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
import cv2

from perimeter.ai.bedrock import EMBED_MAX_CHARS, converse_text, embed_text
from perimeter.ai.catalog import describe_kind
from perimeter.settings import AISettings
from perimeter.timefmt import local_time

log = logging.getLogger("perimeter.ai.frames")

COLLECTION_NAME = "alert_frames"
WHOLE_CLIP = -1.0  # offset_s of the entry holding a clip's Pegasus summary
# An answer reads this many candidates' captions and lets Claude pick the relevant ones:
# embeddings rank "without a helmet" next to "wearing a helmet" (measured), Claude does not.
ANSWER_CANDIDATES = 12
ANSWER_IMAGES = 6  # the best few also get their picture re-read from the clip

ANSWER_SYSTEM = (
    "You answer questions about CCTV footage for a site-safety system. You are given "
    "entries retrieved from saved alert clips: single frames (with the picture) and "
    "whole-clip video summaries, each with its alert, camera and time. They are search "
    "results, so some may not be relevant - use only the ones that are. Answer only from "
    "these entries. If they do not show the answer, say so plainly. Refer to entries by "
    "number, e.g. [2]. Be brief and factual."
)


@dataclass
class FrameHit:
    event_id: str
    camera_id: str
    kind: str
    zone_name: str
    ts: float  # alert time, epoch seconds
    offset_s: float  # where in the clip; WHOLE_CLIP for the video summary
    caption: str
    score: float  # cosine similarity, higher is closer
    clip_path: str = ""

    @property
    def whole_clip(self) -> bool:
        return self.offset_s == WHOLE_CLIP

    def to_dict(self) -> dict[str, Any]:
        out = {**asdict(self), "time": local_time(self.ts), "whole_clip": self.whole_clip}
        out.pop("clip_path")  # a storage key, not for the dashboard
        return out


# -- reading frames out of a clip ------------------------------------------------------


def _with_clip_file(data: bytes, fn):
    with tempfile.TemporaryDirectory(prefix="perimeter-ai-") as tmp:
        path = Path(tmp) / "clip.mp4"
        path.write_bytes(data)
        return fn(path)


def _encode(frame, height: int) -> bytes:
    src_h, src_w = frame.shape[:2]
    if src_h > height:
        frame = cv2.resize(
            frame, (int(src_w * height / src_h), height), interpolation=cv2.INTER_AREA
        )
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not encode a frame")
    return jpeg.tobytes()


def _frame_count(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:  # some containers do not record it - count by reading
            while cap.grab():
                total += 1
        return total, fps
    finally:
        cap.release()


def extract_keyframes(data: bytes, count: int, height: int) -> list[tuple[float, bytes]]:
    """`count` frames spread evenly over the clip, as `(seconds into the clip, jpeg)`.
    Frames are streamed, never all held in memory."""

    def run(path: Path) -> list[tuple[float, bytes]]:
        total, fps = _frame_count(path)
        if total <= 0:
            raise ValueError("clip has no readable frames")
        wanted = sorted({min(total - 1, int((i + 0.5) * total / count)) for i in range(count)})
        out: list[tuple[float, bytes]] = []
        cap = cv2.VideoCapture(str(path))
        try:
            index = 0
            for target in wanted:
                while index < target and cap.grab():
                    index += 1
                ok, frame = cap.read()
                if not ok:
                    break
                index += 1
                out.append((round(target / fps, 2) if fps else 0.0, _encode(frame, height)))
        finally:
            cap.release()
        if not out:
            raise ValueError("clip has no readable frames")
        return out

    return _with_clip_file(data, run)


def frames_at(data: bytes, offsets: list[float], height: int) -> dict[float, bytes]:
    """The frames nearest the given offsets (seconds), as jpegs keyed by offset."""

    def run(path: Path) -> dict[float, bytes]:
        found: dict[float, bytes] = {}
        cap = cv2.VideoCapture(str(path))
        try:
            for offset in sorted(set(offsets)):
                cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000.0)
                ok, frame = cap.read()
                if ok:
                    found[offset] = _encode(frame, height)
        finally:
            cap.release()
        return found

    return _with_clip_file(data, run)


# -- the vector index -----------------------------------------------------------------------


def epoch(value: Any) -> float:
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=UTC)).timestamp()
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        moment = datetime.fromisoformat(value)
        return (moment if moment.tzinfo else moment.replace(tzinfo=UTC)).timestamp()
    return 0.0


def entry_text(event: dict[str, Any], caption: str) -> str:
    """What gets embedded: the caption with the alert's own type and zone in front, so a
    question about "fire" finds a fire alert even when its still frames only look bright."""
    zone = f", zone {event['zone_name']}" if event.get("zone_name") else ""
    return f"{event.get('kind') or 'alert'} alert{zone}: {caption}"[:EMBED_MAX_CHARS]


class FrameIndex:
    """Chroma collection of frame vectors. `persist_dir=None` is in-memory (tests); Chroma
    shares one in-memory store per process, so each such index gets its own collection."""

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        settings = chromadb.Settings(anonymized_telemetry=False)
        if persist_dir is None:
            self._client = chromadb.EphemeralClient(settings=settings)
            name = f"{COLLECTION_NAME}-{uuid.uuid4().hex}"
        else:
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
            name = COLLECTION_NAME
        self._collection = self._client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def has_event(self, event_id: str) -> bool:
        return bool(self._collection.get(where={"event_id": str(event_id)}, limit=1)["ids"])

    def frame_count(self, event_id: str) -> int:
        """How many frame entries (not counting the summary) a clip already has."""
        found = self._collection.get(where={"event_id": str(event_id)}, include=["metadatas"])
        return sum(1 for meta in found["metadatas"] if meta.get("offset_s", 0.0) != WHOLE_CLIP)

    def summary_caption(self, event_id: str) -> str | None:
        found = self._collection.get(ids=[f"{event_id}:summary"], include=["metadatas"])
        return found["metadatas"][0].get("caption") if found["ids"] else None

    def set_summary(self, event: dict[str, Any], caption: str, vector: list[float] | None) -> None:
        """Replace just the clip's summary entry (None removes it); its frames are kept."""
        entry_id = f"{event['id']}:summary"
        self._collection.delete(ids=[entry_id])
        if vector is None:
            return
        self._collection.add(
            ids=[entry_id],
            embeddings=[vector],
            metadatas=[{**self._base(event), "offset_s": WHOLE_CLIP, "caption": caption}],
        )

    @staticmethod
    def _base(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_id": str(event["id"]),
            "camera_id": str(event.get("camera_id") or ""),
            "kind": str(event.get("kind") or ""),
            "zone_name": str(event.get("zone_name") or ""),
            "ts": epoch(event.get("ts")),
            "clip_path": str(event.get("clip_path") or ""),
        }

    def replace_event(
        self, event: dict[str, Any], entries: list[tuple[float, str, list[float]]]
    ) -> None:
        """Store one clip's entries as `(offset_s, caption, vector)`, replacing any it had."""
        event_id = str(event["id"])
        self.remove_event(event_id)
        if not entries:
            return
        base = self._base(event)
        self._collection.add(
            ids=[
                f"{event_id}:summary" if offset == WHOLE_CLIP else f"{event_id}:{i}"
                for i, (offset, _caption, _vector) in enumerate(entries)
            ],
            embeddings=[vector for _offset, _caption, vector in entries],
            metadatas=[
                {**base, "offset_s": float(offset), "caption": caption}
                for offset, caption, _vector in entries
            ],
        )

    def remove_event(self, event_id: str) -> None:
        self._collection.delete(where={"event_id": str(event_id)})

    def purge_before(self, cutoff_epoch: float) -> None:
        self._collection.delete(where={"ts": {"$lt": float(cutoff_epoch)}})

    def query(
        self,
        vector: list[float],
        limit: int,
        event_id: str | None = None,
        camera_id: str | None = None,
        kind: str | None = None,
        since: float | None = None,
    ) -> list[FrameHit]:
        total = self.count()
        if total == 0:
            return []
        conditions: list[dict[str, Any]] = []
        if event_id:
            conditions.append({"event_id": str(event_id)})
        if camera_id:
            conditions.append({"camera_id": str(camera_id)})
        if kind:
            conditions.append({"kind": str(kind)})
        if since is not None:
            conditions.append({"ts": {"$gte": float(since)}})
        where = (
            None if not conditions
            else conditions[0] if len(conditions) == 1
            else {"$and": conditions}
        )
        result = self._collection.query(
            query_embeddings=[vector], n_results=min(limit, total), where=where
        )
        return [
            FrameHit(
                event_id=str(meta["event_id"]),
                camera_id=str(meta.get("camera_id", "")),
                kind=str(meta.get("kind", "")),
                zone_name=str(meta.get("zone_name", "")),
                ts=float(meta.get("ts", 0.0)),
                offset_s=float(meta.get("offset_s", 0.0)),
                caption=str(meta.get("caption", "")),
                score=round(1.0 - float(distance), 4),
                clip_path=str(meta.get("clip_path", "")),
            )
            for meta, distance in zip(
                (result.get("metadatas") or [[]])[0],
                (result.get("distances") or [[]])[0],
                strict=False,
            )
        ]


# -- indexing, search and answers over Bedrock ----------------------------------------------


def _parse_captions(text: str, count: int) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    captions: list[str] = []
    if start != -1 and end > start:
        try:
            captions = [str(c).strip() for c in json.loads(text[start:end + 1])]
        except json.JSONDecodeError:
            captions = []
    captions = captions[:count]
    return captions + [""] * (count - len(captions))


class FrameRAG:
    def __init__(self, ai: AISettings, store: Any, index: FrameIndex, client: Any = None) -> None:
        self.ai = ai
        self.store = store
        self.index = index
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from perimeter.ai.bedrock import runtime_client

            self._client = runtime_client(self.ai)
        return self._client

    # -- indexing --------------------------------------------------------------

    def index_event(self, event: dict[str, Any], summary: str = "", force: bool = False,
                    people_count: int | None = None) -> int:
        """Caption + embed one alert's keyframes (and its Pegasus `summary`, if any) into
        the index. Returns the clip's entries in the index.

        A clip's frames are embedded ONCE: indexing an already-indexed clip again (a retry,
        "Analyse again", a restart) only refreshes its summary entry when the summary
        changed - one embedding instead of a caption call plus every frame - and adds
        nothing. `force` re-does the frames (e.g. after a captioning change)."""
        key = event.get("clip_path")
        if not key:
            raise ValueError(f"alert {event.get('id')} has no saved clip")
        frames = self.index.frame_count(str(event["id"]))
        if frames and not force:
            summary = summary.strip()
            if summary != (self.index.summary_caption(str(event["id"])) or ""):
                vector = self.embed(entry_text(event, summary)) if summary else None
                self.index.set_summary(event, summary, vector)
            else:
                log.info("frames of alert %s already indexed - not embedded again", event["id"])
            return frames + (1 if summary else 0)
        keyframes = extract_keyframes(
            self.store.read(key), self.ai.frames_per_clip, self.ai.frame_height
        )
        captions = self._caption(event, [jpeg for _offset, jpeg in keyframes], summary.strip(),
                                 people_count)
        entries = [
            (offset, caption, self.embed(entry_text(event, caption)))
            for (offset, _jpeg), caption in zip(keyframes, captions, strict=True)
            if caption
        ]
        if summary.strip():
            entries.append(
                (WHOLE_CLIP, summary.strip(), self.embed(entry_text(event, summary.strip())))
            )
        self.index.replace_event(event, entries)
        return len(entries)

    def _caption(self, event: dict[str, Any], jpegs: list[bytes], summary: str = "",
                 people_count: int | None = None) -> list[str]:
        kind = str(event.get("kind") or "alert")
        # Measured on two empty fire clips: captions invented "one person wearing a helmet
        # and vest" (dark, smoky room) and read a jacket hanging on a locker as "a person in
        # dark clothing". Hence: "no people" unless clearly visible, plus the video model's
        # summary and people count (it watched the whole clip, motion included).
        context = f" A video model that watched the whole clip said: {summary}" if summary else ""
        if people_count is not None:
            context += (f" It counted {people_count} people in the clip; treat clothing, "
                        "statues or shapes as objects, not people.")
        prompt = (
            f"These {len(jpegs)} images are frames, in order, from a CCTV clip that raised a "
            f"'{kind}' alert. {describe_kind(kind)}{context} For each frame write one short "
            "factual sentence about what is clearly visible: people (how many, what they "
            "do), fire, smoke, vehicles or notable objects. If no person is clearly visible, "
            "say 'no people'. Only for a person you can clearly see, say 'helmet' or 'no "
            "helmet' and 'vest' or 'no vest'. Never guess at shapes in dark or blurred "
            f"areas. Reply with only a JSON array of {len(jpegs)} strings."
        )
        content: list[dict[str, Any]] = [
            {"image": {"format": "jpeg", "source": {"bytes": jpeg}}} for jpeg in jpegs
        ]
        content.append({"text": prompt})
        return _parse_captions(self._converse(content, max_tokens=800), len(jpegs))

    def embed(self, text: str) -> list[float]:
        return embed_text(self.client, self.ai, text)

    def _converse(
        self, content: list[dict[str, Any]], max_tokens: int, system: str | None = None
    ) -> str:
        return converse_text(self.client, self.ai, content, max_tokens, system)

    # -- search and answers ------------------------------------------------------

    def search(self, query: str, limit: int = 8, **filters: Any) -> list[FrameHit]:
        """Entries best matching a plain-English `query`. Filters: event_id, camera_id,
        kind, since (epoch seconds)."""
        query = query.strip()
        if not query:
            return []
        return self.index.query(self.embed(query), limit=limit, **filters)

    def answer(
        self, question: str, limit: int = ANSWER_CANDIDATES, **filters: Any
    ) -> dict[str, Any]:
        """Claude's answer to `question` from the best-matching entries, with sources."""
        hits = self.search(question, limit=limit, **filters)
        if not hits:
            return {"answer": "No analysed footage matches that yet.", "sources": []}
        images = self._reread(hits[:ANSWER_IMAGES])
        content: list[dict[str, Any]] = []
        for number, hit in enumerate(hits, start=1):
            zone = f", zone {hit.zone_name}" if hit.zone_name else ""
            where = (
                "video summary of the whole clip" if hit.whole_clip
                else f"frame {hit.offset_s:.1f} s into the clip"
            )
            content.append({"text": (
                f"[{number}] {hit.kind} alert {hit.event_id}, camera {hit.camera_id}{zone}, "
                f"{local_time(hit.ts)}, {where}: {hit.caption}"
            )})
            jpeg = images.get((hit.event_id, hit.offset_s))
            if jpeg:
                content.append({"image": {"format": "jpeg", "source": {"bytes": jpeg}}})
        content.append({"text": f"Question: {question}"})
        reply = self._converse(content, max_tokens=600, system=ANSWER_SYSTEM)
        return {"answer": reply.strip(), "sources": [hit.to_dict() for hit in hits]}

    def _reread(self, hits: list[FrameHit]) -> dict[tuple[str, float], bytes]:
        """The hit frames' pictures, read back out of their clips (one read per clip). A
        clip that cannot be read just leaves its frames to their captions."""
        offsets: dict[tuple[str, str], list[float]] = {}
        for hit in hits:
            if hit.clip_path and not hit.whole_clip:
                offsets.setdefault((hit.event_id, hit.clip_path), []).append(hit.offset_s)
        images: dict[tuple[str, float], bytes] = {}
        for (event_id, key), wanted in offsets.items():
            try:
                found = frames_at(self.store.read(key), wanted, self.ai.frame_height)
            except Exception:  # noqa: BLE001
                log.warning("could not re-read frames of event %s", event_id, exc_info=True)
                continue
            images.update({(event_id, offset): jpeg for offset, jpeg in found.items()})
        return images
