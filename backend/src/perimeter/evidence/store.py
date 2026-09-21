"""Evidence storage (Expansion Plan Phase C).

Two interchangeable backends, chosen by `storage.backend` in config/app.yaml:
`LocalEvidenceStore` (disk, the default) and `S3EvidenceStore` (a private S3 bucket, with
time-limited presigned links for viewing). Both use the same key shape
(`clips/cam_01/2026-09-12/<event_id>.mp4`), and `EvidenceWriter` and the API routes only
ever see a `key` string, never a path, so swapping backends changes nothing above this file.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("perimeter.evidence.store")


CONTENT_TYPES = {".jpg": "image/jpeg", ".mp4": "video/mp4", ".avi": "video/x-msvideo"}


class LocalEvidenceStore:
    # Remote stores hand out presigned links; a local one serves the file itself.
    is_remote = False

    def presigned_url(self, key: str, expires_s: float) -> str | None:  # noqa: ARG002
        return None

    def uri(self, key: str) -> str | None:  # noqa: ARG002
        return None

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

    def delete(self, key: str) -> None:
        """Remove the file for `key`, if it exists. Idempotent - a key already deleted
        (or missing for any other reason) is not an error, since the caller's whole
        point is "make sure this is gone", not "assert it was here"."""
        path = self.resolve(key)
        if path is not None:
            path.unlink(missing_ok=True)

    def resolve(self, key: str) -> Path | None:
        """A real filesystem path for `key`, or `None` if it doesn't exist or would
        escape `root` - keys are always server-generated, never user input, but this is
        cheap defence in depth against a corrupted/malicious stored key."""
        candidate = (self.root / key).resolve()
        if self.root.resolve() not in candidate.parents:
            log.warning("evidence key %r resolved outside the store root - refusing", key)
            return None
        return candidate if candidate.is_file() else None


class S3EvidenceStore:
    """Evidence in a private S3 bucket. Objects are encrypted at rest (SSE-S3) and are
    never public - the only way to view one is a presigned URL, minted per request by the
    API (for a logged-in user) or once per alert email with a fixed expiry.

    `client` is injectable so tests use a fake instead of touching AWS. Credentials are
    never handled here: boto3's default chain (env vars, shared config, or an IAM role)
    supplies them.
    """

    is_remote = True

    def __init__(
        self,
        bucket: str,
        snapshot_dir: str,
        clip_dir: str,
        prefix: str = "",
        client: Any = None,
        region: str = "",
    ) -> None:
        self.bucket = bucket
        self.snapshot_dir = snapshot_dir
        self.clip_dir = clip_dir
        self.prefix = prefix.strip("/")
        self._client = client if client is not None else self._make_client(region)

    @staticmethod
    def _make_client(region: str = "") -> Any:
        import boto3
        from botocore.config import Config

        # region_name + virtual-hosted addressing are what make presigned links use the
        # bucket's own regional endpoint. Without them boto3 signs for a generic endpoint and
        # every video link fails with SignatureDoesNotMatch (found against a real bucket).
        return boto3.client(
            "s3",
            region_name=region or None,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
            ),
        )

    def _key(self, kind_dir: str, camera_id: str, event_id: str, suffix: str) -> str:
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        return f"{kind_dir}/{camera_id}/{date}/{event_id}{suffix}"

    def _object_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _put(self, key: str, source: Path | bytes, suffix: str) -> None:
        extra = {
            "ContentType": CONTENT_TYPES.get(suffix.lower(), "application/octet-stream"),
            "ServerSideEncryption": "AES256",
        }
        if isinstance(source, bytes):
            self._client.put_object(
                Bucket=self.bucket, Key=self._object_key(key), Body=source, **extra
            )
        else:
            self._client.upload_file(
                str(source), self.bucket, self._object_key(key), ExtraArgs=extra
            )

    def save_snapshot(self, camera_id: str, event_id: str, jpeg: bytes) -> str:
        key = self._key(self.snapshot_dir, camera_id, event_id, ".jpg")
        self._put(key, jpeg, ".jpg")
        return key

    def save_clip(self, camera_id: str, event_id: str, tmp_path: Path) -> str:
        """Upload `tmp_path` and leave it in place: a failed attempt is retried from the
        same file (the writer removes it once the clip is safely stored or given up on)."""
        key = self._key(self.clip_dir, camera_id, event_id, tmp_path.suffix)
        self._put(key, tmp_path, tmp_path.suffix)
        return key

    def presigned_url(self, key: str, expires_s: float) -> str | None:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._object_key(key)},
            ExpiresIn=int(expires_s),
        )

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._object_key(key)}"

    def delete(self, key: str) -> None:
        """Idempotent: S3 reports success for a key that is already gone."""
        self._client.delete_object(Bucket=self.bucket, Key=self._object_key(key))

    def resolve(self, key: str) -> Path | None:  # noqa: ARG002
        return None  # nothing on local disk; callers use presigned_url()


def build_evidence_store(settings: Any, backend_root: Path, client: Any = None) -> Any:
    """Store for `settings.storage.backend`. A bad S3 setup falls back to local disk with
    a loud error rather than leaving the process unable to save evidence at all."""
    storage = settings.storage
    if storage.backend == "s3":
        if not storage.s3_bucket:
            log.error(
                "storage.backend is s3 but PERIMETER_S3_BUCKET is not set - "
                "falling back to LOCAL evidence storage"
            )
        else:
            try:
                store = S3EvidenceStore(
                    storage.s3_bucket, storage.snapshot_dir, storage.clip_dir,
                    prefix=storage.s3_prefix, client=client, region=storage.s3_region,
                )
                log.info("evidence storage: S3 bucket %s", storage.s3_bucket)
                return store
            except Exception:  # noqa: BLE001 - never leave the process without evidence storage
                log.exception("could not set up S3 evidence storage - falling back to LOCAL")
    return LocalEvidenceStore(
        root=backend_root / storage.evidence_root,
        snapshot_dir=storage.snapshot_dir,
        clip_dir=storage.clip_dir,
    )
