"""Face gallery, backed by Chroma (Expansion Plan Phase F; PLATFORM_EXPANSION_PLAN.md
section 7).

Each enrolled person contributes up to 5 embeddings - one per captured pose (front,
left, right, up, down) - rather than a single vector. A face rarely presents
straight-on in a real boundary-crossing frame, and one canonical enrollment photo
matched poorly against an angled live capture. Storing one vector per pose and
matching against all of them (a plain nearest-neighbour query already does this - it
returns whichever vector, from whichever person and whichever pose, is closest) fixes
that without any extra aggregation logic.

Chroma's HNSW index handles the nearest-neighbour search; at the scale this product
targets (tens to low hundreds of enrolled people, so hundreds to low thousands of
vectors) that is comfortably fast without needing anything heavier. Rebuilt wholesale
on every enrollment/removal, mirroring `ZoneStore`'s reload-on-write shape - no
incremental patching to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import numpy as np

EMBEDDING_DIM = 128
DEFAULT_COSINE_THRESHOLD = 0.363  # OpenCV's own documented default for SFace
COLLECTION_NAME = "faces"


@dataclass(frozen=True)
class PersonRecord:
    id: str
    name: str
    external_id: str
    active: bool
    embeddings: list[list[float]] = field(default_factory=list)


class FaceGallery:
    """Nearest-neighbour match, across every pose of every active enrolled person.

    `persist_dir=None` (the default, and what every test uses) keeps the index
    in-memory only - each `FaceGallery()` instance is then fully isolated, which is
    exactly what a test wants. Production passes `settings.identity.gallery_dir` so
    the index survives a restart without needing to re-embed every enrolled person's
    photos from scratch.
    """

    def __init__(
        self, threshold: float = DEFAULT_COSINE_THRESHOLD, persist_dir: str | Path | None = None
    ) -> None:
        self.threshold = threshold
        # anonymized_telemetry=False: this index stores biometric face embeddings
        # (PLAN.md section 10.3) - nothing about it should phone home by default.
        settings = chromadb.Settings(anonymized_telemetry=False)
        self._client = (
            chromadb.EphemeralClient(settings=settings)
            if persist_dir is None
            else chromadb.PersistentClient(path=str(persist_dir), settings=settings)
        )
        self._collection = self._get_or_create_collection()
        self._count = 0

    def _get_or_create_collection(self):
        return self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def rebuild(self, persons: list[PersonRecord]) -> None:
        """Wholesale rebuild: drop every stored vector and re-add from `persons`."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._get_or_create_collection()

        ids: list[str] = []
        vectors: list[list[float]] = []
        metadatas: list[dict[str, str]] = []
        for person in persons:
            if not person.active:
                continue
            for pose_index, embedding in enumerate(person.embeddings):
                if not embedding:
                    continue
                ids.append(f"{person.id}:{pose_index}")
                vectors.append([float(v) for v in embedding])
                metadatas.append({"person_id": person.id, "name": person.name})

        self._count = len(ids)
        if ids:
            self._collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)

    def match(self, embedding: np.ndarray) -> tuple[str, str] | None:
        """Best match above `threshold` across every pose of every active person, or
        `None`. Returns `(person_id, name)`."""
        if self._count == 0:
            return None
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            return None

        result = self._collection.query(
            query_embeddings=[embedding.astype(np.float32).tolist()],
            n_results=1,
        )
        distances = result.get("distances") or [[]]
        metadatas = result.get("metadatas") or [[]]
        if not distances[0]:
            return None

        similarity = 1.0 - float(distances[0][0])
        if similarity < self.threshold:
            return None
        meta = metadatas[0][0]
        return meta["person_id"], meta["name"]
