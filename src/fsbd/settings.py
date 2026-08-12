"""Configuration loading, split by what the value actually *is*.

Not "secret vs non-secret" - that is the wrong axis and produced the wrong answer at
first. The distinction that matters:

* **`.env`** - **per-deployment facts**: this camera, this machine, this site. Camera
  source and credentials, resolution, decode rate, autostart. These differ per install,
  so committing them would mean the repository carried one site's wiring as if it were
  product configuration, and every install would conflict on it. Git-ignored.
* **`config/app.yaml`** - **engineering constants**: how the algorithm behaves. Inference
  cadences, K-of-N gates, hysteresis defaults, retention. Identical across every install,
  and worth having in git history so a threshold change is reviewable.

`decode_fps` sits in `.env` because it describes what *this machine* can keep up with.
`person_every_n` sits in `app.yaml` because that is the tuning decision.

Precedence is environment variable > `.env` > `app.yaml` > code default.

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
from dotenv import load_dotenv, set_key

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
    """Which camera, at what resolution.

    `use_cctv` is a boolean rather than a three-way enum, matching the pattern proven in
    the sibling face-recognition deployment: false reads `source` (a webcam index, or a
    video file path for development), true reads `cctv_rtsp_url`. Keeping the file-path
    case inside the webcam branch avoids a third mode while still allowing development
    against recorded clips.
    """

    id: str = "cam_01"
    use_cctv: bool = False
    source: str | int = 0  # webcam index or video file path; used when use_cctv is False
    cctv_rtsp_url: str = ""  # used when use_cctv is True
    substream: str | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    decode_fps: float = 12.0
    autostart: bool = True

    @property
    def resolved_source(self) -> str | int:
        """The stream to open, per the use_cctv switch."""
        if self.use_cctv:
            if not self.cctv_rtsp_url:
                raise ValueError(
                    "FSBD_USE_CCTV is true but FSBD_CCTV_RTSP_URL is empty - "
                    "set it in .env or in the dashboard's Camera page."
                )
            return self.cctv_rtsp_url
        return parse_source(str(self.source))

    @property
    def inference_source(self) -> str | int:
        """Prefer the substream for inference - it removes almost all decode cost."""
        return self.substream if self.substream else self.resolved_source

    def __str__(self) -> str:  # never leak the password via an f-string
        return f"CameraSettings(id={self.id}, source={redact(self.resolved_source)})"


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
    # Which .env this was loaded from, so a save writes back to the same file rather
    # than always to the repository root.
    env_file: Path = REPO_ROOT / ".env"
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


def _env_float(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return fallback
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number, ignoring", name, raw)
        return fallback


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, fallback: bool) -> bool:
    """Accepts the spellings people actually type in a .env file.

    A typo must not silently mean False - `FSBD_USE_CCTV=Ture` disabling the CCTV feed
    without a word would be a genuinely confusing site visit.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return fallback
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    log.warning("%s=%r is not a boolean, using %s", name, raw, fallback)
    return fallback


def load_settings(
    config_path: Path | str | None = None, env_file: Path | str | None = None
) -> Settings:
    """Load .env then app.yaml, with environment variables taking precedence."""
    env_path = Path(env_file or REPO_ROOT / ".env")
    load_dotenv(env_path, override=False)

    path = Path(config_path or os.getenv("FSBD_CONFIG") or DEFAULT_CONFIG)
    data: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = loaded if isinstance(loaded, dict) else {}
    else:
        log.warning("config not found at %s, using defaults", path)

    inference_yaml = _section(data, "inference")
    boundary_yaml = _section(data, "boundary")

    # Camera comes from .env only - it is a per-deployment fact, never a committed one.
    substream = os.getenv("FSBD_CAMERA_SUBSTREAM")
    camera = CameraSettings(
        id=os.getenv("FSBD_CAMERA_ID", "cam_01"),
        use_cctv=_env_bool("FSBD_USE_CCTV", False),
        source=os.getenv("FSBD_CAMERA_SOURCE", "0"),
        cctv_rtsp_url=os.getenv("FSBD_CCTV_RTSP_URL", ""),
        substream=substream or None,
        width=_env_int("FSBD_CAMERA_WIDTH", 1280) or 1280,
        height=_env_int("FSBD_CAMERA_HEIGHT", 720) or 720,
        fps=_env_int("FSBD_CAMERA_FPS", 30) or 30,
        decode_fps=_env_float("FSBD_DECODE_FPS", 12.0),
        autostart=_env_bool("FSBD_AUTOSTART_CAMERA", True),
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
        env_file=env_path,
        raw=data,
    )

    log.info(
        "settings loaded: camera=%s source=%s %dx%d decode=%.1ffps person=%.1fHz",
        camera.id,
        describe_source(camera),
        camera.width,
        camera.height,
        camera.decode_fps,
        inference.person_fps(camera.decode_fps),
    )
    return settings


def describe_source(camera: CameraSettings) -> str:
    """A redacted, never-raising label for logs and the health endpoint.

    `resolved_source` raises when use_cctv is set with no URL, which is a legitimate
    half-configured state on a fresh install. Logging must not be the thing that breaks.
    """
    try:
        return redact(camera.inference_source)
    except ValueError:
        return "<cctv selected, no URL set>"


# -- writing back (dashboard Camera page) ---------------------------------

CAMERA_ENV_KEYS = {
    "id": "FSBD_CAMERA_ID",
    "use_cctv": "FSBD_USE_CCTV",
    "source": "FSBD_CAMERA_SOURCE",
    "cctv_rtsp_url": "FSBD_CCTV_RTSP_URL",
    "substream": "FSBD_CAMERA_SUBSTREAM",
    "width": "FSBD_CAMERA_WIDTH",
    "height": "FSBD_CAMERA_HEIGHT",
    "fps": "FSBD_CAMERA_FPS",
    "decode_fps": "FSBD_DECODE_FPS",
    "autostart": "FSBD_AUTOSTART_CAMERA",
}


def save_camera_settings(camera: CameraSettings, env_file: Path | str | None = None) -> Path:
    """Persist camera settings to `.env`, creating it if absent.

    `.env` and not `app.yaml`: every field here is a per-deployment fact, and one of them
    is a password. Writing to the git-ignored file means no camera credential can ever
    reach a tracked file, which is a stronger guarantee than remembering to redact.

    os.environ is updated too, so a subsequent load_settings() in the same process sees
    the new values - dotenv does not override already-set variables by design.
    """
    path = Path(env_file or REPO_ROOT / ".env")
    if not path.exists():
        path.write_text(
            "# Written by the dashboard. Per-deployment settings; never commit this file.\n",
            encoding="utf-8",
        )

    values = {
        "id": camera.id,
        "use_cctv": "true" if camera.use_cctv else "false",
        "source": str(camera.source),
        "cctv_rtsp_url": camera.cctv_rtsp_url,
        "substream": camera.substream or "",
        "width": str(camera.width),
        "height": str(camera.height),
        "fps": str(camera.fps),
        "decode_fps": str(camera.decode_fps),
        "autostart": "true" if camera.autostart else "false",
    }

    for field_name, env_key in CAMERA_ENV_KEYS.items():
        value = values[field_name]
        set_key(str(path), env_key, value, quote_mode="never")
        os.environ[env_key] = value

    log.info("camera settings saved to %s (source=%s)", path.name, describe_source(camera))
    return path
