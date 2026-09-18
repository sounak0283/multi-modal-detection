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

log = logging.getLogger("perimeter.settings")

# backend/, not the repository root: the backend is a self-contained deployable unit and
# everything it reads at runtime (config, .env, sql, models, the built web bundle) lives
# beneath it. The repository root holds only shared documentation and the frontend.
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = BACKEND_ROOT / "config" / "app.yaml"

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
    """A single camera's config, read from `.env`/the CLI.

    Since the Expansion Plan's Phase A, this is no longer how a running system's cameras
    are configured - that is `cameras.registry.CameraRegistry`, backed by MongoDB, which
    supports any number of cameras with independently-chosen detection modules. This
    dataclass survives only as the seed `main.py` uses to create the *first* camera in an
    empty registry on a fresh install (see `main.py`'s bootstrap step), so `.env`-based
    quick-start setup still works without requiring a dashboard visit first.

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
                    "PERIMETER_USE_CCTV is true but PERIMETER_CCTV_RTSP_URL is empty - "
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
    # PPE (Expansion Plan Phase H) runs every Nth DETECTION frame, not raw camera frame
    # - it needs tracked person boxes, which only exist on detection frames, so it is
    # gated on Pipeline.stats.detections_run rather than frame.seq the way fire/smoke's
    # own offset cadence is.
    ppe_every_n: int = 3
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
class FireSmokeSettings:
    """Fire/smoke detection tuning (Expansion Plan Phase E; PLAN.md §5).

    Engineering constants, same axis as `BoundarySettings` above - reviewable in git,
    identical across installs. `gate_k`/`gate_n`/`gate_iou` are the K-of-N temporal gate
    from PLAN.md §5.5: a candidate confirms once it has been seen in at least `gate_k` of
    its last `gate_n` processed frames.
    """

    conf_fire: float = 0.45
    # Lower than fire: indoors, smoke reaches the camera first (PLAN.md §5.4).
    conf_smoke: float = 0.35
    gate_k: int = 6
    gate_n: int = 10
    gate_iou: float = 0.3


@dataclass(frozen=True)
class CrowdSettings:
    """Crowd-formation tuning (Expansion Plan Phase G; PLATFORM_EXPANSION_PLAN.md
    sections 4.2/5). Same axis as `FireSmokeSettings` - engineering constants for the
    hand-rolled DBSCAN clustering, reviewable in git, identical across installs.
    """

    eps_px: float = 80.0
    min_samples: int = 2


@dataclass(frozen=True)
class IdentitySettings:
    """Identity/face recognition tuning (Expansion Plan Phase F; PLAN.md section 7).

    Same axis as `BoundarySettings`/`FireSmokeSettings` - engineering constants,
    reviewable in git, identical across installs. `enabled` is the process-wide kill
    switch, default off: biometric processing needs a consent story per deployment
    (PLAN.md section 10.3) that the rest of the product does not.
    """

    enabled: bool = False
    min_person_box_px: int = 120
    cosine_threshold: float = 0.363
    # Chroma's on-disk index (Expansion Plan Phase F redesign). Relative to
    # BACKEND_ROOT, same as storage.evidence_root - already covered by the
    # backend/data/** gitignore rule.
    gallery_dir: str = "data/identity_gallery"


@dataclass(frozen=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    # Exposes POST /api/dev/simulate-alert, which injects a fake fire/smoke event so the
    # dashboard popup can be exercised before the fire/smoke model exists. Off by default:
    # this API already has no authentication (README), so a fake-alarm endpoint should
    # not be reachable on a site that has not opted in.
    dev_tools: bool = False


@dataclass(frozen=True)
class AuthSettings:
    """Session auth (Expansion Plan Phase B).

    Per-deployment secrets, same axis as `StorageSettings.mongo_url` above: these live in
    `.env`, never in `app.yaml`. `cookie_secure` defaults False to match the
    `host: 127.0.0.1`, no-TLS quick-start default (`ApiSettings.host`) - a Secure cookie
    is silently dropped by the browser over plain http, which would make login appear to
    succeed and then every subsequent request look logged-out. Set it True behind a
    reverse proxy terminating TLS.
    """

    session_secret: str = ""
    session_max_age_s: int = 12 * 60 * 60
    cookie_secure: bool = False
    admin_bootstrap_email: str = ""
    admin_bootstrap_password: str = ""


@dataclass(frozen=True)
class AlertSettings:
    """SMTP credentials (Expansion Plan Phase D).

    A per-deployment secret, same axis as `StorageSettings.mongo_url` - lives in `.env`.
    Routing *policy* (which rules exist, their severity floor/cooldown/recipients) is
    dashboard-managed and lives in MongoDB (`alerts/config_store.py:AlertConfigStore`),
    not here - unlike cameras/zones' bootstrap fields, there is nothing to seed from
    `.env` for alert rules, since an empty rule list is a perfectly valid starting state.
    """

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_use_tls: bool = True

    @property
    def smtp_enabled(self) -> bool:
        return bool(self.smtp_host)


@dataclass(frozen=True)
class StorageSettings:
    """MongoDB Atlas connection and retention (Expansion Plan Phase A.1).

    The connection string is a per-deployment fact containing a password, so it lives in
    .env. `retention_days` is a policy decision that PLAN.md section 10.3 requires be
    stated and enforced, so it lives in app.yaml where it is reviewable.
    """

    mongo_url: str = ""
    mongo_db: str = "perimeter"
    retention_days: int = 30

    # Evidence capture (Expansion Plan Phase C). Reserved in app.yaml since Phase A;
    # wired up here for the first time.
    ring_buffer_seconds: float = 15.0
    post_roll_seconds: float = 5.0
    evidence_root: str = "data/evidence"
    snapshot_dir: str = "snapshots"
    clip_dir: str = "clips"

    @property
    def enabled(self) -> bool:
        return bool(self.mongo_url)


@dataclass(frozen=True)
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    boundary: BoundarySettings = field(default_factory=BoundarySettings)
    firesmoke: FireSmokeSettings = field(default_factory=FireSmokeSettings)
    crowd: CrowdSettings = field(default_factory=CrowdSettings)
    identity: IdentitySettings = field(default_factory=IdentitySettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    # Which .env this was loaded from, so a save writes back to the same file rather
    # than always to the repository root.
    env_file: Path = BACKEND_ROOT / ".env"
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

    A typo must not silently mean False - `PERIMETER_USE_CCTV=Ture` disabling the CCTV feed
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
    env_path = Path(env_file or BACKEND_ROOT / ".env")
    # utf-8-sig, not utf-8: Notepad and PowerShell's Set-Content both write UTF-8 *with
    # a BOM* on Windows, and plain utf-8 decoding turns the first key into
    # "﻿PERIMETER_DATABASE_URL" - so the first setting in the file silently does not
    # exist. Measured: a .env written by Set-Content lost PERIMETER_DATABASE_URL entirely and
    # the app reported "not set" while the file plainly showed it. Reading as utf-8-sig
    # strips a BOM when present and is identical otherwise.
    load_dotenv(env_path, override=False, encoding="utf-8-sig")

    path = Path(config_path or os.getenv("PERIMETER_CONFIG") or DEFAULT_CONFIG)
    data: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = loaded if isinstance(loaded, dict) else {}
    else:
        log.warning("config not found at %s, using defaults", path)

    inference_yaml = _section(data, "inference")
    boundary_yaml = _section(data, "boundary")

    # Camera comes from .env only - it is a per-deployment fact, never a committed one.
    substream = os.getenv("PERIMETER_CAMERA_SUBSTREAM")
    camera = CameraSettings(
        id=os.getenv("PERIMETER_CAMERA_ID", "cam_01"),
        use_cctv=_env_bool("PERIMETER_USE_CCTV", False),
        source=os.getenv("PERIMETER_CAMERA_SOURCE", "0"),
        cctv_rtsp_url=os.getenv("PERIMETER_CCTV_RTSP_URL", ""),
        substream=substream or None,
        width=_env_int("PERIMETER_CAMERA_WIDTH", 1280) or 1280,
        height=_env_int("PERIMETER_CAMERA_HEIGHT", 720) or 720,
        fps=_env_int("PERIMETER_CAMERA_FPS", 30) or 30,
        decode_fps=_env_float("PERIMETER_DECODE_FPS", 12.0),
        autostart=_env_bool("PERIMETER_AUTOSTART_CAMERA", True),
    )

    inference = InferenceSettings(
        person_every_n=int(inference_yaml.get("person_every_n", 2)),
        firesmoke_every_n=int(inference_yaml.get("firesmoke_every_n", 8)),
        firesmoke_offset=int(inference_yaml.get("firesmoke_offset", 3)),
        ppe_every_n=int(inference_yaml.get("ppe_every_n", 3)),
        input_size=int(inference_yaml.get("input_size", 416)),
        precision=str(inference_yaml.get("precision", "fp32")),
        intra_op_threads=inference_yaml.get("intra_op_threads"),
    )

    boundary = BoundarySettings(
        default_min_frames=int(boundary_yaml.get("default_min_frames", 4)),
        lost_track_seconds=float(boundary_yaml.get("lost_track_seconds", 5.0)),
        person_conf=float(boundary_yaml.get("person_conf", 0.30)),
    )

    firesmoke_yaml = _section(data, "firesmoke")
    firesmoke = FireSmokeSettings(
        conf_fire=float(firesmoke_yaml.get("conf_fire", 0.45)),
        conf_smoke=float(firesmoke_yaml.get("conf_smoke", 0.35)),
        gate_k=int(firesmoke_yaml.get("gate_k", 6)),
        gate_n=int(firesmoke_yaml.get("gate_n", 10)),
        gate_iou=float(firesmoke_yaml.get("gate_iou", 0.3)),
    )

    crowd_yaml = _section(data, "crowd")
    crowd = CrowdSettings(
        eps_px=float(crowd_yaml.get("eps_px", 80.0)),
        min_samples=int(crowd_yaml.get("min_samples", 2)),
    )

    identity_yaml = _section(data, "identity")
    identity = IdentitySettings(
        enabled=bool(identity_yaml.get("enabled", False)),
        min_person_box_px=int(identity_yaml.get("min_person_box_px", 120)),
        cosine_threshold=float(identity_yaml.get("cosine_threshold", 0.363)),
        gallery_dir=str(identity_yaml.get("gallery_dir", "data/identity_gallery")),
    )

    api = ApiSettings(
        host=os.getenv("PERIMETER_API_HOST", "127.0.0.1"),
        port=_env_int("PERIMETER_API_PORT", 8000) or 8000,
        dev_tools=_env_bool("PERIMETER_DEV_TOOLS", False),
    )

    storage_yaml = _section(data, "storage")
    storage = StorageSettings(
        # URL from .env (it holds a password); retention from app.yaml (it is policy).
        mongo_url=os.getenv("PERIMETER_MONGO_URL", ""),
        mongo_db=os.getenv("PERIMETER_MONGO_DB", "perimeter"),
        retention_days=int(storage_yaml.get("retention_days", 30)),
        ring_buffer_seconds=float(storage_yaml.get("ring_buffer_seconds", 15.0)),
        post_roll_seconds=float(storage_yaml.get("post_roll_seconds", 5.0)),
        evidence_root=str(storage_yaml.get("evidence_root", "data/evidence")),
        snapshot_dir=str(storage_yaml.get("snapshot_dir", "snapshots")),
        clip_dir=str(storage_yaml.get("clip_dir", "clips")),
    )

    auth = AuthSettings(
        session_secret=os.getenv("PERIMETER_SESSION_SECRET", ""),
        session_max_age_s=_env_int("PERIMETER_SESSION_MAX_AGE_S", 12 * 60 * 60) or 12 * 60 * 60,
        cookie_secure=_env_bool("PERIMETER_COOKIE_SECURE", False),
        admin_bootstrap_email=os.getenv("PERIMETER_ADMIN_EMAIL", ""),
        admin_bootstrap_password=os.getenv("PERIMETER_ADMIN_PASSWORD", ""),
    )

    alerts = AlertSettings(
        smtp_host=os.getenv("PERIMETER_SMTP_HOST", ""),
        smtp_port=_env_int("PERIMETER_SMTP_PORT", 587) or 587,
        smtp_user=os.getenv("PERIMETER_SMTP_USER", ""),
        smtp_password=os.getenv("PERIMETER_SMTP_PASSWORD", ""),
        smtp_from=os.getenv("PERIMETER_SMTP_FROM", ""),
        smtp_use_tls=_env_bool("PERIMETER_SMTP_USE_TLS", True),
    )

    settings = Settings(
        camera=camera,
        inference=inference,
        boundary=boundary,
        firesmoke=firesmoke,
        crowd=crowd,
        identity=identity,
        api=api,
        storage=storage,
        auth=auth,
        alerts=alerts,
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
    "id": "PERIMETER_CAMERA_ID",
    "use_cctv": "PERIMETER_USE_CCTV",
    "source": "PERIMETER_CAMERA_SOURCE",
    "cctv_rtsp_url": "PERIMETER_CCTV_RTSP_URL",
    "substream": "PERIMETER_CAMERA_SUBSTREAM",
    "width": "PERIMETER_CAMERA_WIDTH",
    "height": "PERIMETER_CAMERA_HEIGHT",
    "fps": "PERIMETER_CAMERA_FPS",
    "decode_fps": "PERIMETER_DECODE_FPS",
    "autostart": "PERIMETER_AUTOSTART_CAMERA",
}


def save_camera_settings(camera: CameraSettings, env_file: Path | str | None = None) -> Path:
    """Persist camera settings to `.env`, creating it if absent.

    `.env` and not `app.yaml`: every field here is a per-deployment fact, and one of them
    is a password. Writing to the git-ignored file means no camera credential can ever
    reach a tracked file, which is a stronger guarantee than remembering to redact.

    os.environ is updated too, so a subsequent load_settings() in the same process sees
    the new values - dotenv does not override already-set variables by design.
    """
    path = Path(env_file or BACKEND_ROOT / ".env")
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
