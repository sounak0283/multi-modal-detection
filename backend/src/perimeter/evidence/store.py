"""Evidence storage (Expansion Plan Phase C).

`LocalEvidenceStore` is the only implementation today - no AWS account is stood up for
this repo, so clips and snapshots live on local disk rather than S3. The key shape is
deliberately identical to the plan's S3 object-key example
(`clips/cam_01/2026-09-12/<event_id>.mp4`) so that a future `S3EvidenceStore` swap is a
drop-in: `EvidenceWriter` and the API routes only ever see a `key` string, never a path.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("perimeter.evidence.store")


class LocalEvidenceStore:
    def __init__(self, root: Path, snapshot_dir: str, clip_dir: str) -> None:
        self.root = root
        self.snapshot_dir = snapshot_dir
        self.clip_dir = clip_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, kind_dir: str, camera_id: str, event_id: str, suffix: str) -> str:
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        return f"{kind_dir}/{camera_id}/{date}/{event_id}{suffix}"

    def save_snapshot(self, camera_id: str, event_id: str, jpeg: bytes) -> str:
        key = self._key(self.snapshot_dir, camera_id, event_id, ".jpg")
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        return key

    def save_clip(self, camera_id: str, event_id: str, tmp_path: Path) -> str:
        key = self._key(self.clip_dir, camera_id, event_id, tmp_path.suffix)
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_path), path)
        return key

    def resolve(self, key: str) -> Path | None:
        """A real filesystem path for `key`, or `None` if it doesn't exist or would
        escape `root` - keys are always server-generated, never user input, but this is
        cheap defence in depth against a corrupted/malicious stored key."""
        candidate = (self.root / key).resolve()
        if self.root.resolve() not in candidate.parents:
            log.warning("evidence key %r resolved outside the store root - refusing", key)
            return None
        return candidate if candidate.is_file() else None
