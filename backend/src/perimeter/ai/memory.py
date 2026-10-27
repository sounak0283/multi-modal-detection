"""Long-term memory for the assistant (Phase J step 5).

Three scopes, as decided:
  system - facts about the site, shared by everyone ("cam_02 is switched off 2-3 am for
           cleaning"); any user's chat can add one, only an admin deletes one.
  user   - one person's preferences and context ("looks after Bay 4"); only they see it.
  alert  - a conclusion about one alert ("this fire alert was a welding test").

MongoDB (`ai_memories`) is the record; a Chroma index in `ai.memory_dir` finds the ones
relevant to a question (Titan Text v2 vectors). The index is rebuilt from MongoDB if it
goes missing or out of step, so deleting the folder loses nothing.

Memories come from two places: the `remember` chat tool (asked for, or obvious), and a
cheap after-each-reply pass (`extract`) that keeps durable facts from the conversation.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb

from perimeter.ai.bedrock import LazyClient, converse_text, embed_text
from perimeter.settings import AISettings

log = logging.getLogger("perimeter.ai.memory")

MEMORIES = "ai_memories"
SCOPES = ("system", "user", "alert")
MAX_TEXT_CHARS = 500
# A new memory this close to an existing one (same scope and owner) replaces its text
# rather than piling up near-duplicates.
DUPLICATE_SIMILARITY = 0.9

EXTRACT_PROMPT = """You maintain the long-term memory of a site-safety assistant.
From the conversation below, list only DURABLE facts that the USER stated or confirmed
and that will still be true and useful next week:
- system: facts about the site, cameras, zones, schedules or recurring situations
  (e.g. "Camera cam_02 is switched off from 2 to 3 am daily for cleaning").
- user: this user's role, responsibilities or preferences (e.g. "Prefers short answers").
- alert: a conclusion about one specific alert, with its id (e.g. "Alert 6acf... was a
  planned welding test, not a fire").
Do NOT keep: anything the assistant worked out from the data (counts, statistics,
patterns, video findings - they can be looked up again and go out of date), questions,
greetings, the user's role or permissions, or anything you are unsure of. Most
conversations have nothing to keep: then reply []. Already remembered (do not repeat):
{known}

Reply with only a JSON array (possibly empty) of objects:
{{"scope": "system" | "user" | "alert", "text": "...", "event_id": "<alert id or null>"}}

Conversation:
{conversation}"""


def _now() -> datetime:
    return datetime.now(UTC)


def _public(doc: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k != "_id"}
    out["id"] = doc["_id"]
    for key in ("created_at", "updated_at"):
        if isinstance(out.get(key), datetime):
            out[key] = out[key].isoformat()
    return out


class MemoryStore:
    def __init__(
        self,
        ai: AISettings,
        db: Any,
        client: LazyClient,
        persist_dir: str | Path | None = None,
    ) -> None:
        self.ai = ai
        self.docs = db[MEMORIES]
        self._client = client
        settings = chromadb.Settings(anonymized_telemetry=False)
        if persist_dir is None:
            self._chroma = chromadb.EphemeralClient(settings=settings)
            name = f"memories-{uuid.uuid4().hex}"
        else:
            self._chroma = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
            name = "memories"
        self._index = self._chroma.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def _embed(self, text: str) -> list[float]:
        return embed_text(self._client.get(), self.ai, text)

    def sync_index(self) -> int:
        """Re-embed every memory if the index does not match MongoDB. Returns the count."""
        ids = [d["_id"] for d in self.docs.find({}, {"_id": 1})]
        if self._index.count() == len(ids) and (
            not ids or len(self._index.get(ids=ids)["ids"]) == len(ids)
        ):
            return len(ids)
        existing = self._index.get()["ids"]
        if existing:
            self._index.delete(ids=existing)
        for doc in self.docs.find({}):
            self._index_doc(doc)
        log.info("memory index rebuilt from MongoDB (%d memories)", len(ids))
        return len(ids)

    def _index_doc(self, doc: dict[str, Any], vector: list[float] | None = None) -> None:
        self._index.upsert(
            ids=[doc["_id"]],
            embeddings=[vector or self._embed(doc["text"])],
            metadatas=[{
                "scope": doc["scope"],
                "user_id": doc.get("user_id") or "",
                "event_id": doc.get("event_id") or "",
            }],
        )

    # -- writing -------------------------------------------------------------------------

    def add(
        self,
        text: str,
        scope: str,
        user_id: str | None = None,
        event_id: str | None = None,
        source: str = "chat",
    ) -> dict[str, Any]:
        text = " ".join(str(text).split())[:MAX_TEXT_CHARS]
        if not text:
            raise ValueError("a memory needs some text")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        if scope == "user" and not user_id:
            raise ValueError("a user memory needs the user")
        if scope == "alert" and not event_id:
            raise ValueError("an alert memory needs the alert id")
        owner = user_id if scope == "user" else None
        vector = self._embed(text)
        duplicate = self._closest(vector, scope, owner, event_id if scope == "alert" else None)
        now = _now()
        if duplicate is not None:
            self.docs.update_one({"_id": duplicate}, {"$set": {"text": text, "updated_at": now}})
            doc = self.docs.find_one({"_id": duplicate})
        else:
            doc = {
                "_id": uuid.uuid4().hex,
                "scope": scope,
                "text": text,
                "user_id": owner,
                "event_id": event_id if scope == "alert" else None,
                "created_by": user_id,
                "source": source,
                "created_at": now,
                "updated_at": now,
            }
            self.docs.insert_one(doc)
        self._index_doc(doc, vector)
        return _public(doc)

    def _closest(self, vector: list[float], scope: str, owner: str | None,
                 event_id: str | None) -> str | None:
        if self._index.count() == 0:
            return None
        where: dict[str, Any] = {"$and": [
            {"scope": scope}, {"user_id": owner or ""}, {"event_id": event_id or ""},
        ]}
        result = self._index.query(query_embeddings=[vector], n_results=1, where=where)
        ids, distances = result["ids"][0], result["distances"][0]
        if ids and 1.0 - distances[0] >= DUPLICATE_SIMILARITY:
            return ids[0]
        return None

    def delete(self, memory_id: str, user_id: str, is_admin: bool) -> bool:
        doc = self.docs.find_one({"_id": memory_id})
        if doc is None:
            return False
        own = doc.get("user_id") == user_id or doc.get("created_by") == user_id
        allowed = own if doc["scope"] == "user" else (is_admin or own)
        if not allowed:
            raise PermissionError("only an admin, or whoever added it, can delete this memory")
        self.docs.delete_one({"_id": memory_id})
        self._index.delete(ids=[memory_id])
        return True

    # -- reading ---------------------------------------------------------------------------

    def visible(self, user_id: str, scope: str | None = None,
                event_id: str | None = None) -> list[dict[str, Any]]:
        """What this user may see: system and alert memories, plus their own."""
        query: dict[str, Any] = {"$or": [
            {"scope": {"$in": ["system", "alert"]}}, {"scope": "user", "user_id": user_id},
        ]}
        if scope:
            query = {"$and": [query, {"scope": scope}]}
        if event_id:
            query = {"$and": [query, {"event_id": event_id}]}
        return [_public(d) for d in self.docs.find(query).sort("updated_at", -1)]

    def recall(self, query: str, user_id: str, event_id: str | None = None,
               limit: int = 6) -> list[dict[str, Any]]:
        """The memories most relevant to `query` that this user may see; an alert chat's
        own alert memories always come first."""
        found: dict[str, dict[str, Any]] = {}
        if event_id:
            for doc in self.docs.find({"scope": "alert", "event_id": event_id}).limit(limit):
                found[doc["_id"]] = doc
        if self._index.count() and query.strip():
            where = {"$or": [
                {"scope": "system"}, {"scope": "alert"},
                {"$and": [{"scope": "user"}, {"user_id": user_id}]},
            ]}
            result = self._index.query(
                query_embeddings=[self._embed(query)],
                n_results=min(limit, self._index.count()), where=where,
            )
            ids = [i for i in result["ids"][0] if i not in found]
            for doc in self.docs.find({"_id": {"$in": ids}}):
                found[doc["_id"]] = doc
        return [_public(d) for d in list(found.values())[:limit]]

    # -- learning from a conversation --------------------------------------------------------

    def extract(self, conversation: list[dict[str, str]], user_id: str,
                event_id: str | None = None) -> list[dict[str, Any]]:
        """Keep the durable facts from the latest exchange. Returns the memories saved."""
        text = "\n".join(f"{m['role']}: {m['content']}" for m in conversation[-6:])
        if not text.strip():
            return []
        known = self.recall(text[-1500:], user_id, event_id, limit=10)
        prompt = EXTRACT_PROMPT.format(
            known="\n".join(f"- [{m['scope']}] {m['text']}" for m in known) or "(nothing yet)",
            conversation=text[-6000:],
        )
        reply = converse_text(self._client.get(), self.ai, [{"text": prompt}], max_tokens=500)
        saved: list[dict[str, Any]] = []
        for item in _parse_items(reply):
            scope = item.get("scope")
            alert_id = item.get("event_id") or (event_id if scope == "alert" else None)
            try:
                saved.append(self.add(item.get("text", ""), scope, user_id=user_id,
                                      event_id=alert_id, source="auto"))
            except ValueError:
                continue
        return saved


def _parse_items(text: str) -> list[dict[str, Any]]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict)]
