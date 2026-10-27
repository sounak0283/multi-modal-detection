"""Routes for the AI features (Phase J): video analysis, footage search, the assistant
chat (streamed), its memory, the daily digest and plain-English alert search.

Kept out of `app.py`, like the Video test routes. Every route answers 503 while AI is not
configured (`ai.enabled` off or no `BEDROCK_API_KEY`), so the rest of the API is
unaffected. Any signed-in user may use them (the plan: all users can use the AI features);
admin-only: generating a digest on demand, deleting a site-wide memory.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from perimeter.ai.suggestions import suggestions
from perimeter.ai.worker import AnalysisError
from perimeter.auth.models import Role

log = logging.getLogger("perimeter.api.ai")

MAX_RESULTS = 20
MAX_QUESTION_CHARS = 1000


def add_ai_routes(
    router: APIRouter,
    *,
    require_any_role: Callable[..., Any],
    services: Any | None,
    parse_since: Callable[[str | None], datetime | None],
) -> None:
    def need(part: str) -> Any:
        found = getattr(services, part, None) if services is not None else None
        if found is None:
            raise HTTPException(503, "AI features are not configured.")
        return found

    def need_rag():
        rag = need("worker").rag
        if rag is None:
            raise HTTPException(503, "Footage search (ai.frame_rag) is switched off.")
        return rag

    def call_ai(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except HTTPException:
            raise
        except (ValueError, PermissionError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - Bedrock trouble is a 502, not a crash
            log.exception("AI request failed")
            raise HTTPException(502, "The AI service did not answer. Try again shortly.") from exc

    def filters(camera_id: str | None, kind: str | None, since: str | None) -> dict[str, Any]:
        since_dt = parse_since(since)
        return {
            "camera_id": camera_id or None,
            "kind": kind or None,
            "since": since_dt.timestamp() if since_dt else None,
        }

    def is_admin(user: Any) -> bool:
        return getattr(user, "role", None) == Role.ADMIN

    # -- status + suggestions ------------------------------------------------------------

    @router.get("/ai/status")
    def ai_status(_user: Any = Depends(require_any_role)) -> dict[str, Any]:
        if services is None:
            return {"available": False}
        worker = services.worker
        return {
            "available": True,
            "chat": services.chat is not None,
            "memory": services.memory is not None,
            "digest": services.digest is not None and services.ai.digest_enabled,
            "auto_analyze_kinds": list(services.ai.auto_analyze_kinds),
            "frame_search": bool(worker and worker.rag),
            "indexed_entries": worker.rag.index.count() if worker and worker.rag else 0,
        }

    @router.get("/ai/suggestions")
    def ai_suggestions(view: str | None = None, event_id: str | None = None,
                       _user: Any = Depends(require_any_role)) -> dict[str, Any]:
        if services is None:
            return {"suggestions": []}
        return {"suggestions": suggestions(services.db, view, event_id)}

    # -- video analysis ------------------------------------------------------------------

    @router.post("/ai/events/{event_id}/analyze", status_code=202)
    def analyze_event(event_id: str, _user: Any = Depends(require_any_role)) -> dict[str, Any]:
        try:
            return need("worker").request(event_id)
        except AnalysisError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.get("/ai/events/{event_id}/analysis")
    def event_analysis(event_id: str, _user: Any = Depends(require_any_role)) -> dict[str, Any]:
        found = need("worker").status(event_id)
        if found is None:
            raise HTTPException(404, "this alert has not been analysed")
        return found

    # -- footage search (frame RAG) ------------------------------------------------------

    @router.get("/ai/search")
    def search_footage(
        q: str,
        camera_id: str | None = None,
        kind: str | None = None,
        since: str | None = None,
        limit: int = 8,
        _user: Any = Depends(require_any_role),
    ) -> dict[str, Any]:
        rag = need_rag()
        found = filters(camera_id, kind, since)
        hits = call_ai(lambda: rag.search(
            q[:MAX_QUESTION_CHARS], limit=max(1, min(limit, MAX_RESULTS)), **found
        ))
        return {"results": [hit.to_dict() for hit in hits]}

    @router.post("/ai/ask")
    def ask_footage(payload: dict[str, Any], _user: Any = Depends(require_any_role)) -> dict:
        rag = need_rag()
        question = str(payload.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "question is required")
        found = filters(payload.get("camera_id"), payload.get("kind"), payload.get("since"))
        if payload.get("event_id"):
            found["event_id"] = str(payload["event_id"])
        return call_ai(lambda: rag.answer(question[:MAX_QUESTION_CHARS], **found))

    # -- plain-English alert search --------------------------------------------------------

    @router.get("/ai/alert-search")
    def alert_search(q: str, limit: int = 200,
                     _user: Any = Depends(require_any_role)) -> dict[str, Any]:
        search = need("search")
        if not q.strip():
            raise HTTPException(400, "type what you are looking for")
        return call_ai(lambda: search.search(q[:MAX_QUESTION_CHARS], limit))

    # -- chat ------------------------------------------------------------------------------

    @router.post("/ai/chat")
    def chat(payload: dict[str, Any], user: Any = Depends(require_any_role)) -> StreamingResponse:
        service = need("chat")

        def events() -> Iterator[str]:
            stream = service.stream(
                user.id, str(payload.get("message") or ""),
                conversation_id=payload.get("conversation_id") or None,
                event_id=payload.get("event_id") or None,
                is_admin=is_admin(user),
            )
            for event in stream:
                yield f"data: {json.dumps(event, default=str)}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/ai/conversations")
    def list_conversations(event_id: str | None = None,
                           user: Any = Depends(require_any_role)) -> dict[str, Any]:
        return {"conversations": need("chat").list(user.id, event_id)}

    @router.get("/ai/conversations/{conversation_id}")
    def get_conversation(conversation_id: str,
                         user: Any = Depends(require_any_role)) -> dict[str, Any]:
        found = need("chat").get(conversation_id, user.id)
        if found is None:
            raise HTTPException(404, "conversation not found")
        return found

    @router.delete("/ai/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str, user: Any = Depends(require_any_role)) -> None:
        if not need("chat").delete(conversation_id, user.id):
            raise HTTPException(404, "conversation not found")

    # -- memory ------------------------------------------------------------------------------

    @router.get("/ai/memories")
    def list_memories(scope: str | None = None, event_id: str | None = None,
                      user: Any = Depends(require_any_role)) -> dict[str, Any]:
        return {"memories": need("memory").visible(user.id, scope or None, event_id or None)}

    @router.post("/ai/memories", status_code=201)
    def add_memory(payload: dict[str, Any], user: Any = Depends(require_any_role)) -> dict:
        memory = need("memory")
        return call_ai(lambda: memory.add(
            str(payload.get("text") or ""), str(payload.get("scope") or "user"),
            user_id=user.id, event_id=payload.get("event_id") or None, source="user",
        ))

    @router.delete("/ai/memories/{memory_id}", status_code=204)
    def delete_memory(memory_id: str, user: Any = Depends(require_any_role)) -> None:
        memory = need("memory")
        try:
            found = memory.delete(memory_id, user.id, is_admin(user))
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        if not found:
            raise HTTPException(404, "memory not found")

    # -- daily digest ------------------------------------------------------------------------

    @router.get("/ai/digests")
    def list_digests(_user: Any = Depends(require_any_role)) -> dict[str, Any]:
        return {"digests": need("digest").latest()}

    @router.post("/ai/digests/run")
    def run_digest(payload: dict[str, Any] | None = None,
                   user: Any = Depends(require_any_role)) -> dict[str, Any]:
        if not is_admin(user):
            raise HTTPException(403, "only an admin can make a digest on demand")
        digest = need("digest")
        raw = (payload or {}).get("day")
        try:
            day = date.fromisoformat(raw) if raw else None
        except ValueError as exc:
            raise HTTPException(400, "day must be YYYY-MM-DD") from exc
        if day is None:
            from perimeter.ai.queries import now_local

            day = now_local().date()
        email = bool((payload or {}).get("email", False))
        return call_ai(lambda: digest.generate(day, email=email))
