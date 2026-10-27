"""The assistant (Phase J steps 3-5): one chat for surveillance questions and product help,
plus a chat of its own under every alert's video/snapshot.

A reply streams back as events (`text`, `tool`, `done`, `error`) while Claude calls tools
(tools.py) to look things up. Conversations are kept per user in `ai_conversations`; the
last `ai.chat_history_messages` go back to the model each turn, and the long-term
memories relevant to the question are added to the instructions. After each reply a
background pass keeps any durable facts (memory.py).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from perimeter.ai.bedrock import LazyClient
from perimeter.ai.catalog import catalog
from perimeter.ai.queries import now_local
from perimeter.ai.tools import ToolBox, ToolContext, _event_row, _visible_analysis
from perimeter.settings import AISettings
from perimeter.timefmt import local_time

log = logging.getLogger("perimeter.ai.chat")

CONVERSATIONS = "ai_conversations"
MAX_TOOL_ROUNDS = 8
MAX_MESSAGE_CHARS = 4000
REPLY_MAX_TOKENS = 1500
TITLE_CHARS = 60

SYSTEM = """You are the assistant built into Perimeter, a site-safety system that watches
cameras for fire, smoke, boundary crossings, crowds, missing protective equipment (PPE),
restricted-area access and more. You help with two things:
1. Surveillance: what happened, when, where, how often, patterns over days and hours,
   cameras going offline, what a video shows.
2. Product help: how to use the dashboard (use the product_help tool).

Rules:
- Facts come from tools. Never invent alerts, counts, times or camera names; if the
  tools do not show it, say so.
- Times are the site's local time. Right now it is {now}.
- Any "how do I..." question about the dashboard: call product_help first, and answer
  from it. Never say something cannot be done without checking the guide.
- For patterns ("every day", "every other day", "at a fixed time") use alert_stats
  grouped by day and/or hour, and camera_outages for cameras going missing.
- Alerts whose message starts with [TEST VIDEO] come from uploaded test videos, not real
  cameras. Say so when it matters; use source="real" for questions about the real site.
- When you mention an alert, give its time and camera.
- Answer briefly and plainly. Short paragraphs or bullet points; **bold** sparingly.
  Do not announce what you are about to look up ("I'll check...") - just answer.
- If the video model (ask_video, video analysis) and still frames disagree about what is
  in a clip, trust the video model and say the frames were unclear.
- Remember durable facts the user tells you about the site or themselves with the
  remember tool. Do not save guesses.

Alert types:
{kinds}
{memories}{alert}"""

ALERT_CONTEXT = """
This chat is about one alert - "this alert", "this video" and "this snapshot" mean it:
{details}
Use look_at_snapshot / look_at_frames to see it, and ask_video for what happens in the
clip over time. Use its id with other tools."""


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


class ChatService:
    def __init__(
        self,
        ai: AISettings,
        db: Any,
        client: LazyClient,
        tools: ToolBox,
        memory: Any = None,
        background: ThreadPoolExecutor | None = None,
    ) -> None:
        self.ai = ai
        self.conversations = db[CONVERSATIONS]
        self.events = db["events"]
        self.analyses = db["video_analyses"]
        self._client = client
        self.tools = tools
        self.memory = memory
        self._background = background

    # -- conversations -----------------------------------------------------------------

    def list(self, user_id: str, event_id: str | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"user_id": user_id, "event_id": event_id or None}
        docs = self.conversations.find(query, {"messages": 0}).sort("updated_at", -1).limit(50)
        return [self._summary(d) for d in docs]

    def get(self, conversation_id: str, user_id: str) -> dict[str, Any] | None:
        doc = self.conversations.find_one({"_id": conversation_id, "user_id": user_id})
        if doc is None:
            return None
        return {**self._summary(doc), "messages": [
            {**m, "ts": _iso(m.get("ts"))} for m in doc.get("messages", [])
        ]}

    def delete(self, conversation_id: str, user_id: str) -> bool:
        result = self.conversations.delete_one({"_id": conversation_id, "user_id": user_id})
        return result.deleted_count > 0

    @staticmethod
    def _summary(doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": doc["_id"],
            "title": doc.get("title"),
            "event_id": doc.get("event_id"),
            "created_at": _iso(doc.get("created_at")),
            "updated_at": _iso(doc.get("updated_at")),
        }

    # -- one turn ------------------------------------------------------------------------

    def stream(
        self,
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        event_id: str | None = None,
        is_admin: bool = False,
    ) -> Iterator[dict[str, Any]]:
        message = str(message or "").strip()[:MAX_MESSAGE_CHARS]
        if not message:
            yield {"type": "error", "message": "Type a question first."}
            return
        doc = self._conversation(user_id, conversation_id, event_id, message)
        yield {"type": "conversation", "id": doc["_id"], "title": doc["title"]}

        messages = _history(doc.get("messages", []), self.ai.chat_history_messages)
        messages.append({"role": "user", "content": [{"text": message}]})
        ctx = ToolContext(user_id=user_id, is_admin=is_admin, event_id=doc.get("event_id"))
        system = self._system(message, ctx)

        reply_parts: list[str] = []
        used_tools: list[str] = []
        try:
            for _round in range(MAX_TOOL_ROUNDS):
                # Text from separate rounds would otherwise run together mid-sentence.
                separator = "\n\n" if "".join(reply_parts).strip() else ""
                text, tool_uses, stop = yield from self._model_round(system, messages, separator)
                reply_parts.append(separator + text if text else "")
                assistant_content: list[dict[str, Any]] = []
                if text:
                    assistant_content.append({"text": text})
                assistant_content += [{"toolUse": use} for use in tool_uses]
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                if stop != "tool_use" or not tool_uses:
                    break
                results = []
                for use in tool_uses:
                    used_tools.append(use["name"])
                    yield {"type": "tool", "name": use["name"],
                           "label": self.tools.label(use["name"])}
                    results.append({"toolResult": self._run_tool(ctx, use)})
                messages.append({"role": "user", "content": results})
            else:
                reply_parts.append("\n\n(I stopped after several lookups - try a narrower "
                                   "question.)")
        except Exception as exc:  # noqa: BLE001 - the user gets a message, not a broken stream
            log.exception("chat turn failed")
            yield {"type": "error", "message": f"The AI service did not answer ({exc})."}
            return

        reply = "".join(reply_parts).strip() or "(no reply)"
        message_id = self._save_turn(doc["_id"], message, reply, used_tools)
        yield {"type": "done", "message_id": message_id, "tools": used_tools}
        self._learn(doc, message, reply, user_id)

    def _model_round(self, system: str, messages: list[dict[str, Any]], separator: str = ""):
        """Stream one model response. Yields text events (the first prefixed with
        `separator`); returns (text, tool uses, stop)."""
        response = self._client.get().converse_stream(
            modelId=self.ai.chat_model,
            system=[{"text": system}],
            messages=messages,
            toolConfig={"tools": self.tools.specs()},
            inferenceConfig={"maxTokens": REPLY_MAX_TOKENS, "temperature": 0.2},
        )
        text_parts: list[str] = []
        uses: dict[int, dict[str, Any]] = {}
        stop = "end_turn"
        for event in response["stream"]:
            if "contentBlockStart" in event:
                start = event["contentBlockStart"]
                use = start.get("start", {}).get("toolUse")
                if use:
                    # Only these keys go back to the model (the stream adds a "type").
                    uses[start["contentBlockIndex"]] = {
                        "toolUseId": use["toolUseId"], "name": use["name"], "input": "",
                    }
            elif "contentBlockDelta" in event:
                block = event["contentBlockDelta"]
                delta = block.get("delta", {})
                if "text" in delta:
                    lead = separator if not text_parts else ""
                    text_parts.append(delta["text"])
                    yield {"type": "text", "text": lead + delta["text"]}
                elif "toolUse" in delta and block.get("contentBlockIndex") in uses:
                    uses[block["contentBlockIndex"]]["input"] += delta["toolUse"].get("input", "")
            elif "messageStop" in event:
                stop = event["messageStop"].get("stopReason", stop)
        tool_uses = []
        for use in uses.values():
            try:
                parsed = json.loads(use["input"]) if use["input"] else {}
            except json.JSONDecodeError:
                parsed = {}
            tool_uses.append({**use, "input": parsed if isinstance(parsed, dict) else {}})
        return "".join(text_parts), tool_uses, stop

    def _run_tool(self, ctx: ToolContext, use: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.tools.run(use["name"], ctx, use["input"])
            return {"toolUseId": use["toolUseId"], "content": result.content()}
        except Exception as exc:  # noqa: BLE001 - the model is told, and can try otherwise
            log.warning("tool %s failed: %s", use["name"], exc)
            return {"toolUseId": use["toolUseId"], "status": "error",
                    "content": [{"text": f"{type(exc).__name__}: {exc}"}]}

    # -- prompt --------------------------------------------------------------------------

    def _system(self, message: str, ctx: ToolContext) -> str:
        kinds = "\n".join(f"- {row['kind']}: {row['description']}" for row in catalog())
        memories = ""
        if self.memory is not None:
            try:
                found = self.memory.recall(message, ctx.user_id, ctx.event_id)
            except Exception:  # noqa: BLE001 - answering without memory beats not answering
                log.warning("memory recall failed", exc_info=True)
                found = []
            if found:
                lines = "\n".join(
                    f"- [{_scope_label(m)}] {m['text']}" for m in found
                )
                memories = f"\nWhat you remember (from earlier conversations):\n{lines}\n"
        alert = ""
        if ctx.event_id:
            alert = ALERT_CONTEXT.format(details=self._alert_details(ctx.event_id))
        return SYSTEM.format(now=local_time(now_local()), kinds=kinds, memories=memories,
                             alert=alert)

    def _alert_details(self, event_id: str) -> str:
        from bson import ObjectId

        doc = self.events.find_one({"_id": ObjectId(event_id)})
        if doc is None:
            return f"(alert {event_id} no longer exists)"
        row = _event_row(doc)
        analysis = _visible_analysis(self.analyses.find_one({"_id": event_id}))
        if analysis:
            row["video_analysis"] = analysis
        return json.dumps(row, default=str, ensure_ascii=False)

    # -- storage -------------------------------------------------------------------------

    def _conversation(self, user_id: str, conversation_id: str | None, event_id: str | None,
                      first_message: str) -> dict[str, Any]:
        if conversation_id:
            doc = self.conversations.find_one({"_id": conversation_id, "user_id": user_id})
            if doc is not None:
                return doc
        if event_id:
            from bson import ObjectId
            from bson.errors import InvalidId

            try:
                found = self.events.find_one({"_id": ObjectId(event_id)}, {"_id": 1})
            except (InvalidId, TypeError):
                found = None
            if found is None:
                event_id = None
        now = _now()
        doc = {
            "_id": uuid.uuid4().hex,
            "user_id": user_id,
            "event_id": event_id or None,
            "title": first_message[:TITLE_CHARS],
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        self.conversations.insert_one(doc)
        return doc

    def _save_turn(self, conversation_id: str, question: str, reply: str,
                   tools: list[str]) -> str:
        now = _now()
        message_id = uuid.uuid4().hex
        self.conversations.update_one({"_id": conversation_id}, {
            "$push": {"messages": {"$each": [
                {"role": "user", "content": question, "ts": now},
                {"id": message_id, "role": "assistant", "content": reply, "ts": now,
                 "tools": tools},
            ]}},
            "$set": {"updated_at": now},
        })
        return message_id

    def _learn(self, doc: dict[str, Any], question: str, reply: str, user_id: str) -> None:
        if self.memory is None or not reply:
            return
        turn = [*({"role": m["role"], "content": m["content"]} for m in
                  doc.get("messages", [])[-4:]),
                {"role": "user", "content": question}, {"role": "assistant", "content": reply}]

        def run() -> None:
            try:
                self.memory.extract(turn, user_id, doc.get("event_id"))
            except Exception:  # noqa: BLE001 - memory is best effort
                log.warning("memory extraction failed", exc_info=True)

        if self._background is not None:
            self._background.submit(run)
        else:
            run()


def _history(stored: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """The last `limit` stored messages in the shape Bedrock needs: starting with the user
    and strictly alternating (an empty or missing message would otherwise break that)."""
    out: list[dict[str, Any]] = []
    for item in stored[-limit:]:
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        text = str(item.get("content") or "").strip() or "(no reply)"
        if out and out[-1]["role"] == role:
            out[-1]["content"][0]["text"] += "\n\n" + text
        elif out or role == "user":
            out.append({"role": role, "content": [{"text": text}]})
    if out and out[-1]["role"] == "user":  # the new question follows; never two in a row
        out.pop()
    return out


def _scope_label(memory: dict[str, Any]) -> str:
    if memory["scope"] == "alert":
        return f"alert {memory.get('event_id')}"
    return {"system": "site", "user": "about this user"}.get(memory["scope"], memory["scope"])
