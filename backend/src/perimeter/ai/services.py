"""Builds and runs every AI service together (Phase J), so main.py and the API deal with
one object. Each part is optional; anything missing makes its routes answer 503."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from perimeter.ai.alert_search import AlertSearch
from perimeter.ai.bedrock import LazyClient
from perimeter.ai.camera_log import CameraStatusLog
from perimeter.ai.chat import ChatService
from perimeter.ai.digest import DigestService
from perimeter.ai.frames import FrameIndex, FrameRAG
from perimeter.ai.inspect_people import PeopleInspector
from perimeter.ai.memory import MemoryStore
from perimeter.ai.tools import ToolBox
from perimeter.ai.video import PegasusAnalyzer
from perimeter.ai.worker import AnalysisWorker
from perimeter.settings import AISettings

log = logging.getLogger("perimeter.ai")


@dataclass
class AIServices:
    ai: AISettings
    db: Any
    worker: AnalysisWorker | None = None
    chat: ChatService | None = None
    memory: MemoryStore | None = None
    digest: DigestService | None = None
    search: AlertSearch | None = None
    camera_log: CameraStatusLog | None = None
    background: ThreadPoolExecutor | None = None
    _started: list[Any] = field(default_factory=list)

    def start(self) -> None:
        if self.memory is not None and self.background is not None:
            self.background.submit(self._sync_memory)
        for part in (self.camera_log, self.worker, self.digest):
            if part is None:
                continue
            try:
                part.start()
                self._started.append(part)
            except Exception:  # noqa: BLE001 - one part failing leaves the others running
                log.exception("could not start %s", type(part).__name__)

    def _sync_memory(self) -> None:
        try:
            self.memory.sync_index()
        except Exception:  # noqa: BLE001
            log.exception("could not sync the memory index")

    def stop(self) -> None:
        for part in reversed(self._started):
            try:
                part.stop()
            except Exception:  # noqa: BLE001
                log.exception("could not stop %s", type(part).__name__)
        if self.background is not None:
            self.background.shutdown(wait=False, cancel_futures=True)


def build_ai_services(
    ai: AISettings,
    db: Any,
    store: Any,
    root: Path,
    cameras: Callable[[], list[dict[str, Any]]],
    camera_states: Callable[[], dict[str, str]] | None = None,
    retention_days: int = 30,
    send_email: Callable[[EmailMessage], None] | None = None,
    from_addr: str = "",
    digest_recipients: Callable[[], list[str]] | None = None,
    client: Any = None,
) -> AIServices:
    lazy = LazyClient(ai, client)
    services = AIServices(ai=ai, db=db,
                          background=ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-bg"))

    if camera_states is not None:
        services.camera_log = CameraStatusLog(db, camera_states, ai.camera_log_seconds,
                                              retention_days)

    analyzer = PegasusAnalyzer(ai, store, client)
    rag = None
    if ai.frame_rag:
        rag = FrameRAG(ai, store, FrameIndex(root / ai.frame_index_dir), client)
    services.worker = AnalysisWorker(ai, db, analyzer, rag, retention_days=retention_days)

    try:
        services.memory = MemoryStore(ai, db, lazy, root / ai.memory_dir)
    except Exception:  # noqa: BLE001 - chat still works without long-term memory
        log.exception("could not open the memory index - chat runs without memory")

    inspector = PeopleInspector(ai, lazy, root) if ai.vision_model else None
    tools = ToolBox(db, store, cameras, analyzer=analyzer, rag=rag, memory=services.memory,
                    camera_log=services.camera_log, frame_height=ai.frame_height,
                    inspector=inspector)
    services.chat = ChatService(ai, db, lazy, tools, services.memory, services.background)
    services.search = AlertSearch(ai, db, lazy, cameras)
    services.digest = DigestService(ai, db, lazy, send=send_email, from_addr=from_addr,
                                    recipients=digest_recipients,
                                    camera_log=services.camera_log)
    return services
