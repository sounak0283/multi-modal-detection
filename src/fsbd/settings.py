"""Configuration loading, with secrets kept out of the repository.

Two sources, deliberately split:

* **`.env`** - secrets. Camera URLs (which embed passwords), alert tokens, broker
  credentials. Git-ignored, never committed, never logged unredacted.
* **`config/app.yaml`** - tuning constants. Cadences, thresholds, retention. Committed
  and reviewed, because these are engineering decisions that need history.

Precedence is environment variable > `.env` > `app.yaml` > code default, so a deployment
can override anything without editing a file that git tracks.

Credential leakage
------------------
An RTSP URL contains the camera password in plain text. It appears in log lines, error
messages, health endpoints and support bundles unless something actively strips it, so
`redact` exists and every code path that reports a source is expected to use it. This is
the kind of leak that survives for years in a log archive.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml
from dotenv import load_dotenv

log = logging.getLogger("fsbd.settings")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "app.yaml"
DEFAULT_ZONES = REPO_ROOT / "config" / "zones.yaml"

_CREDENTIAL_PATTERN = re.compile(r"^(?P<user>[^:/@]+)(?::(?P<password>[^@]*))?$")


def redact(source: str | int | None) -> str:
    """Strip credentials from a URL so it is safe to log or display.

    rtsp://admin:hunter2@10.0.0.5:554/s1  ->  rtsp://admin:***@10.0.0.5:554/s1

    Falls back to a conservative placeholder rather than the original string if parsing
    fails: an unparseable URL is exactly the case where a password is most likely to slip
    through unnoticed.
    """
    if source is None:
        return "<unset>"
    if isinstance(source, int):
        return f"device:{source}"
    if "://" not in source:
        return source  # a bare file path or device index carries no credentials

    try:
        parts = urlsplit(source)
        if not parts.netloc or "@" not in parts.netloc:
            return source
        userinfo, _, hostport = parts.netloc.rpartition("@")
        match = _CREDENTIAL_PATTERN.match(userinfo)
        if match is None:
            return f"{parts.scheme}://***@{hostport}{parts.path}"
        user = match.group("user")
        safe_netloc = f"{user}:***@{hostport}" if match.group("password") else f"{user}@{hostport}"
        return urlunsplit((parts.scheme, safe_netloc, parts.path, parts.query, ""))
    except ValueError:
        return "<unparseable source, redacted>"


def parse_source(raw: str) -> str | int:
    """A bare integer means a local camera index; anything else is a URL or file path."""
    try:
        return int(raw)
    except ValueError:
        return raw


@dataclass(frozen=True)
class CameraSettings:
    id: str = "cam_01"
    source: str | int = 0
    substream: str | None = None
    decode_fps: float = 12.0

    @property
    def inference_source(self) -> str | int:
        """Prefer the substream for inference - it removes almost all decode cost."""
        return self.substream if self.substream else self.source

    def __str__(self) -> str:  # never leak the password via an f-string
        return f"CameraSettings(id={self.id}, source={redact(self.source)})"


@dataclass(frozen=True)
class InferenceSettings:
    person_every_n: int = 2
    firesmoke_every_n: int = 8
    firesmoke_offset: int = 3
    input_size: int = 416
    precision: str = "fp32"
    intra_op_threads: int | None = None

    def person_fps(self, decode_fps: float) -> float:
        """The rate the tracker and boundary engine actually run at."""
        return decode_fps / self.person_every_n


@dataclass(frozen=True)
class BoundarySettings:
    default_min_frames: int = 4
    lost_track_seconds: float = 5.0
    person_conf: float = 0.30


@dataclass(frozen=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    boundary: BoundarySettings = field(default_factory=BoundarySettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    zones_path: Path = DEFAULT_ZONES
    raw: dict[str, Any] = field(default_factory=dict)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    return value if isinstance(value, dict) else {}


def _env_int(name: str, fallback: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return fallback
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer, ignoring", name, raw)
        return fallback


def load_settings(
    config_path: Path | str | None = None, env_file: Path | str | None = None
) -> Settings:
    """Load .env then app.yaml, with environment variables taking precedence."""
    load_dotenv(env_file or REPO_ROOT / ".env", override=False)

    path = Path(config_path or os.getenv("FSBD_CONFIG") or DEFAULT_CONFIG)
    data: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = loaded if isinstance(loaded, dict) else {}
    else:
        log.warning("config not found at %s, using defaults", path)

    camera_yaml = _section(data, "camera")
    inference_yaml = _section(data, "inference")
    boundary_yaml = _section(data, "boundary")

    raw_source = os.getenv("FSBD_CAMERA_SOURCE") or camera_yaml.get("source")
    substream = os.getenv("FSBD_CAMERA_SUBSTREAM") or camera_yaml.get("substream")

    camera = CameraSettings(
        id=os.getenv("FSBD_CAMERA_ID") or camera_yaml.get("id", "cam_01"),
        source=parse_source(str(raw_source)) if raw_source is not None else 0,
        substream=str(substream) if substream else None,
        decode_fps=float(camera_yaml.get("decode_fps", 12.0)),
    )

    inference = InferenceSettings(
        person_every_n=int(inference_yaml.get("person_every_n", 2)),
        firesmoke_every_n=int(inference_yaml.get("firesmoke_every_n", 8)),
        firesmoke_offset=int(inference_yaml.get("firesmoke_offset", 3)),
        input_size=int(inference_yaml.get("input_size", 416)),
        precision=str(inference_yaml.get("precision", "fp32")),
        intra_op_threads=inference_yaml.get("intra_op_threads"),
    )

    boundary = BoundarySettings(
        default_min_frames=int(boundary_yaml.get("default_min_frames", 4)),
        lost_track_seconds=float(boundary_yaml.get("lost_track_seconds", 5.0)),
        person_conf=float(boundary_yaml.get("person_conf", 0.30)),
    )

    api = ApiSettings(
        host=os.getenv("FSBD_API_HOST", "127.0.0.1"),
        port=_env_int("FSBD_API_PORT", 8000) or 8000,
    )

    zones_env = os.getenv("FSBD_ZONES")
    settings = Settings(
        camera=camera,
        inference=inference,
        boundary=boundary,
        api=api,
        zones_path=Path(zones_env) if zones_env else DEFAULT_ZONES,
        raw=data,
    )

    log.info(
        "settings loaded: camera=%s source=%s decode=%.1ffps person=%.1fHz",
        camera.id,
        redact(camera.inference_source),
        camera.decode_fps,
        inference.person_fps(camera.decode_fps),
    )
    return settings
