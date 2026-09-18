"""Camera domain model (Expansion Plan, Phase A.2).

One document per camera in MongoDB replaces the single `.env`-configured camera. The
field shape mirrors `settings.CameraSettings` deliberately - `resolved_source` /
`inference_source` are copied nearly verbatim - so `Pipeline` and `capture.probe` do not
need to know or care whether a camera came from `.env` (legacy single-camera dev mode)
or from the registry.

`enabled_modules` is the mechanism that makes per-camera module selection real: a
`Pipeline` only builds the detector/monitor objects whose module name appears here.
Every module is opt-in, `"boundary"` included - a camera with none selected only
captures and streams. `"boundary"` (person detection + tracking + zone rules) is what
crowd, PPE, phone and identity consume, so enabling any of those enables it too;
fire/smoke works on whole frames and runs without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from perimeter.settings import parse_source, redact


class SourceType(Enum):
    WEBCAM = "webcam"
    FILE = "file"
    RTSP = "rtsp"


MODULE_BOUNDARY = "boundary"
MODULE_CROWD = "crowd"
MODULE_PPE = "ppe"
MODULE_PHONE = "phone"
MODULE_FIRE_SMOKE = "fire_smoke"
MODULE_IDENTITY = "identity"

ALL_MODULES = frozenset(
    {MODULE_BOUNDARY, MODULE_CROWD, MODULE_PPE, MODULE_PHONE, MODULE_FIRE_SMOKE, MODULE_IDENTITY}
)
# Modules that consume person tracks and zone membership from the boundary module.
REQUIRES_BOUNDARY = frozenset({MODULE_CROWD, MODULE_PPE, MODULE_PHONE, MODULE_IDENTITY})

MIN_DIMENSION = 64
MAX_DIMENSION = 7680  # 8K; beyond this is a typo, not a camera


@dataclass(frozen=True)
class Camera:
    """One camera's configuration and which detection modules run on it.

    RTSP credentials live on this document (not `.env` - there can be many cameras),
    which is a real change from the single-camera "credentials never touch a tracked
    file" guarantee `.env` provided. The API layer is responsible for redacting
    `rtsp_url` on every read (`to_public_dict` does this) and for restricting writes to
    admin-only routes once Phase B lands.
    """

    id: str
    name: str = ""
    source_type: SourceType = SourceType.WEBCAM
    source: str | int = 0  # webcam index or file path; used when source_type != RTSP
    rtsp_url: str = ""  # used when source_type == RTSP
    substream: str | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    decode_fps: float = 12.0
    autostart: bool = True
    enabled: bool = True
    enabled_modules: frozenset[str] = field(default_factory=frozenset)

    @property
    def resolved_source(self) -> str | int:
        if self.source_type is SourceType.RTSP:
            if not self.rtsp_url:
                raise ValueError(
                    f"camera {self.id!r} is set to RTSP but has no rtsp_url configured."
                )
            return self.rtsp_url
        return parse_source(str(self.source))

    @property
    def inference_source(self) -> str | int:
        """Prefer the substream for inference - it removes almost all decode cost."""
        return self.substream if self.substream else self.resolved_source

    def __str__(self) -> str:  # never leak the password via an f-string
        try:
            source = redact(self.resolved_source)
        except ValueError:
            source = "<rtsp selected, no URL set>"
        return f"Camera(id={self.id}, source={source})"

    def to_dict(self) -> dict[str, Any]:
        """Full representation for internal use (Mongo storage) - includes the raw RTSP URL."""
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type.value,
            "source": self.source,
            "rtsp_url": self.rtsp_url,
            "substream": self.substream,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "decode_fps": self.decode_fps,
            "autostart": self.autostart,
            "enabled": self.enabled,
            "enabled_modules": sorted(self.enabled_modules),
        }

    def to_public_dict(self) -> dict[str, Any]:
        """API-facing representation - the RTSP URL is redacted, never echoed in full."""
        payload = self.to_dict()
        payload["rtsp_url_redacted"] = redact(self.rtsp_url) if self.rtsp_url else ""
        payload["rtsp_url_set"] = bool(self.rtsp_url)
        payload.pop("rtsp_url")
        return payload


def describe_source(camera: Camera) -> str:
    """A redacted, never-raising label for logs and the health endpoint.

    Mirrors `settings.describe_source` - `resolved_source` raises when RTSP is selected
    with no URL set, which is a legitimate half-configured state right after a camera is
    created from the dashboard. Logging must not be the thing that breaks.
    """
    try:
        return redact(camera.inference_source)
    except ValueError:
        return "<rtsp selected, no URL set>"


def camera_from_dict(data: dict[str, Any]) -> Camera:
    """Parse and validate one camera document. Raises ValueError with an actionable message."""
    if not isinstance(data, dict):
        raise ValueError("camera must be a mapping")

    camera_id = str(data.get("id", "")).strip()
    if not camera_id:
        raise ValueError("camera is missing 'id'")

    try:
        source_type = SourceType(str(data.get("source_type", "webcam")).strip().lower())
    except ValueError:
        valid = ", ".join(t.value for t in SourceType)
        raise ValueError(
            f"camera {camera_id!r}: invalid source_type {data.get('source_type')!r} "
            f"(expected: {valid})"
        ) from None

    rtsp_url = str(data.get("rtsp_url", "")).strip()
    if source_type is SourceType.RTSP and not rtsp_url:
        raise ValueError(f"camera {camera_id!r}: source_type is rtsp but rtsp_url is empty")

    modules = {str(m).strip().lower() for m in data.get("enabled_modules", [])}
    unknown = modules - ALL_MODULES
    if unknown:
        raise ValueError(f"camera {camera_id!r}: unknown module(s) {sorted(unknown)}")
    if modules & REQUIRES_BOUNDARY:
        modules.add(MODULE_BOUNDARY)

    def as_int(key: str, fallback: int, low: int, high: int) -> int:
        if key not in data or data[key] in (None, ""):
            return fallback
        value = int(data[key])
        if not low <= value <= high:
            raise ValueError(f"camera {camera_id!r}: {key} must be between {low} and {high}")
        return value

    def as_float(key: str, fallback: float, low: float, high: float) -> float:
        if key not in data or data[key] in (None, ""):
            return fallback
        value = float(data[key])
        if not low <= value <= high:
            raise ValueError(f"camera {camera_id!r}: {key} must be between {low} and {high}")
        return value

    substream = data.get("substream")
    return Camera(
        id=camera_id,
        name=str(data.get("name", "")),
        source_type=source_type,
        source=data.get("source", 0),
        rtsp_url=rtsp_url,
        substream=str(substream).strip() or None if substream else None,
        width=as_int("width", 1280, MIN_DIMENSION, MAX_DIMENSION),
        height=as_int("height", 720, MIN_DIMENSION, MAX_DIMENSION),
        fps=as_int("fps", 30, 1, 240),
        decode_fps=as_float("decode_fps", 12.0, 1.0, 120.0),
        autostart=bool(data.get("autostart", True)),
        enabled=bool(data.get("enabled", True)),
        enabled_modules=frozenset(modules),
    )
