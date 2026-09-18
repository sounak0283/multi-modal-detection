"""Entry point: runs one pipeline per enabled camera and serves the dashboard.

    perimeter-dashboard                      # uses .env + config/app.yaml + MongoDB
    python -m perimeter.main --source 0      # override/seed the first camera for a quick test

The API binds to 127.0.0.1 by default. Every route now requires a logged-in session
(Expansion Plan Phase B) - see PERIMETER_ADMIN_EMAIL/PERIMETER_ADMIN_PASSWORD below for the first
account. Putting the dashboard on 0.0.0.0 still needs a reverse proxy terminating TLS in
front of it (set PERIMETER_COOKIE_SECURE=true once one is), since login itself is only as safe
as the transport it runs over.

MongoDB is a hard requirement, unlike the old optional PostgreSQL persistence
------------------------------------------------------------------------------
In the single-camera version, a database outage was non-fatal: alerts stopped being
recorded but the camera kept detecting, because zones lived in a YAML file independent of
the database. Since the Expansion Plan's Phase A, camera and zone configuration
themselves live in MongoDB - there is no local fallback describing which cameras to run
or what their zones are, so a MongoDB that cannot be reached at startup means there is
nothing to run. This trades the old graceful degradation for a startup-time fast failure
instead of a confusing "0 cameras, 0 zones, silently doing nothing" runtime state.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import uvicorn

from perimeter.alerts.bus import AlertBus
from perimeter.alerts.config_store import AlertConfigStore
from perimeter.alerts.cooldown import CooldownGate
from perimeter.alerts.router import AlertRouter
from perimeter.alerts.sinks.smtp import EmailSink
from perimeter.api.app import create_app
from perimeter.auth.models import Role, User, normalise_email
from perimeter.auth.passwords import hash_password
from perimeter.auth.registry import UserRegistry
from perimeter.boundary.zones import ZoneStore
from perimeter.cameras.manager import PipelineManager
from perimeter.cameras.registry import CameraRegistry
from perimeter.evidence.store import LocalEvidenceStore
from perimeter.evidence.writer import EvidenceWriter
from perimeter.identity.gallery import FaceGallery, PersonRecord
from perimeter.identity.resolver import IdentityResolver
from perimeter.identity.sface import FaceEmbedder
from perimeter.identity.yunet import FaceDetector
from perimeter.settings import BACKEND_ROOT, load_settings
from perimeter.store.db import Database

log = logging.getLogger("perimeter")

DEFAULT_MODEL = Path("models/yolox_person/yolox_nano.onnx")
DEFAULT_FIRESMOKE_MODEL = Path("models/firesmoke/model.onnx")
DEFAULT_PPE_MODEL = Path("models/ppe/model.onnx")
DEFAULT_YUNET_MODEL = Path("models/yunet/face_detection_yunet_2023mar.onnx")
DEFAULT_SFACE_MODEL = Path("models/sface/face_recognition_sface_2021dec.onnx")
RETENTION_SWEEP_INTERVAL_S = 24 * 60 * 60  # once a day is enough for a day-granularity policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="Seed/override the first camera's source.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--firesmoke-model",
        type=Path,
        default=DEFAULT_FIRESMOKE_MODEL,
        help="Fire/smoke weights (Expansion Plan Phase E). Optional - unlike --model, a "
        "missing file here does not stop the dashboard from starting; fire/smoke "
        "detection just stays inactive until one is placed there.",
    )
    parser.add_argument(
        "--ppe-model",
        type=Path,
        default=DEFAULT_PPE_MODEL,
        help="PPE compliance weights (Expansion Plan Phase H). Optional - same posture "
        "as --firesmoke-model: a missing file does not stop the dashboard from "
        "starting, PPE checking just stays inactive. See backend/models/ppe/README.md.",
    )
    parser.add_argument(
        "--yunet-model",
        type=Path,
        default=DEFAULT_YUNET_MODEL,
        help="YuNet face detection weights (Expansion Plan Phase F). Only used when "
        "identity.enabled is true in config/app.yaml.",
    )
    parser.add_argument(
        "--sface-model",
        type=Path,
        default=DEFAULT_SFACE_MODEL,
        help="SFace face recognition weights (Expansion Plan Phase F).",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Serve the API/dashboard without starting any camera pipelines.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _bootstrap_first_camera(
    registry: CameraRegistry, settings, source_override: str | None
) -> None:
    """Seed one camera from `.env`/`--source` if the registry is empty.

    Keeps the `.env`-based quick-start experience working on a fresh install without
    requiring a dashboard visit first - the same role `CameraSettings` played before
    multi-camera, now expressed as the first row in the registry instead of the only
    camera the process knows about.
    """
    if registry.list():
        return

    camera = settings.camera
    source = source_override if source_override is not None else camera.source
    registry.upsert_from_dict(
        {
            "id": camera.id,
            "name": camera.id,
            "source_type": "rtsp" if camera.use_cctv else "webcam",
            "source": source,
            "rtsp_url": camera.cctv_rtsp_url,
            "substream": camera.substream,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "decode_fps": camera.decode_fps,
            "autostart": camera.autostart,
            "enabled": True,
            "enabled_modules": [],
        }
    )
    log.info("seeded first camera %r into an empty registry", camera.id)


def _bootstrap_first_admin(users: UserRegistry, settings) -> bool:
    """Seed one admin account from `PERIMETER_ADMIN_EMAIL`/`PERIMETER_ADMIN_PASSWORD` if the
    `users` collection is empty - the same "seed from `.env` on a fresh install" role
    `_bootstrap_first_camera` plays for cameras.

    Returns False (rather than raising) when the collection is empty and no bootstrap
    credentials are configured, so `main()` can fail fast with an actionable message:
    an unreachable dashboard (no way to ever log in) is worse than a clear startup error.
    """
    if users.list():
        return True

    email = normalise_email(settings.auth.admin_bootstrap_email)
    password = settings.auth.admin_bootstrap_password
    if not email or not password:
        return False

    users.upsert(
        User(id=email, email=email, password_hash=hash_password(password), role=Role.ADMIN)
    )
    log.info("seeded first admin account %r into an empty users collection", email)
    return True


def _build_email_sink(settings) -> EmailSink | None:
    """`None` when SMTP isn't configured - `AlertRouter.dispatch` already logs a warning
    per dispatch if a dashboard-managed rule wants email anyway (Expansion Plan Phase D).
    Email is an optional notification channel; unlike MongoDB, its absence is never fatal.
    """
    if not settings.alerts.smtp_enabled:
        return None
    return EmailSink(
        host=settings.alerts.smtp_host,
        port=settings.alerts.smtp_port,
        user=settings.alerts.smtp_user,
        password=settings.alerts.smtp_password,
        from_addr=settings.alerts.smtp_from,
        use_tls=settings.alerts.smtp_use_tls,
    )


def _build_identity_resolver(
    settings, yunet_model: Path, sface_model: Path, database: Database
) -> IdentityResolver | None:
    """`None` when identity is disabled or the weights are missing - soft-fail exactly
    like Phase E's fire/smoke model check, never fatal (Expansion Plan Phase F). One
    instance is shared across every camera (see `pipeline.Pipeline`'s docstring)."""
    if not settings.identity.enabled:
        return None
    if not yunet_model.is_file() or not sface_model.is_file():
        log.warning(
            "identity enabled but model file(s) missing (%s, %s) - module inactive. "
            "See docs/phases/PHASE_F.md.",
            yunet_model, sface_model,
        )
        return None
    try:
        detector = FaceDetector(yunet_model)
        embedder = FaceEmbedder(sface_model)
    except Exception:  # noqa: BLE001 - any load failure degrades, never crashes
        log.exception("could not load identity models - module inactive")
        return None

    gallery = FaceGallery(
        threshold=settings.identity.cosine_threshold,
        persist_dir=BACKEND_ROOT / settings.identity.gallery_dir,
    )
    persons = [
        PersonRecord(
            id=p["id"], name=p["name"], external_id=p.get("external_id", ""),
            active=p.get("active", True), embeddings=p.get("embeddings", []),
        )
        for p in database.list_persons()
    ]
    gallery.rebuild(persons)
    log.info(
        "identity detection active (%d enrolled person(s), %d active)",
        len(persons), sum(1 for p in persons if p.active),
    )
    return IdentityResolver(
        detector, embedder, gallery, min_person_box_px=settings.identity.min_person_box_px
    )


def _start_retention_sweeper(database: Database, retention_days: int) -> threading.Thread:
    """Enforce `storage.retention_days` (Expansion Plan §8 / PLAN.md §10.3).

    `Database.purge_older_than` has existed since the Postgres implementation but nothing
    ever called it - this is the first thing that does.
    """

    def _run() -> None:
        while True:
            try:
                cutoff = datetime.now(UTC) - timedelta(days=retention_days)
                database.purge_older_than(cutoff)
            except Exception:  # noqa: BLE001 - a failed sweep must not kill the process
                log.exception("retention sweep failed")
            threading.Event().wait(RETENTION_SWEEP_INTERVAL_S)

    thread = threading.Thread(target=_run, name="retention-sweep", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    settings = load_settings()

    if args.source is not None:
        settings = replace(
            settings,
            camera=replace(settings.camera, use_cctv=False, source=args.source),
        )

    if not settings.storage.enabled:
        log.error(
            "PERIMETER_MONGO_URL is not set - cameras and zones now live in MongoDB, "
            "so there is nothing to run. See README for setup."
        )
        return 1

    if not settings.auth.session_secret:
        settings = replace(
            settings, auth=replace(settings.auth, session_secret=secrets.token_urlsafe(32))
        )
        log.warning(
            "PERIMETER_SESSION_SECRET is not set - generated an ephemeral one for this run. "
            "Every session will be logged out on the next restart; set it in .env for "
            "sessions to survive one."
        )

    try:
        database = Database(settings.storage.mongo_url, settings.storage.mongo_db)
        database.ensure_indexes()
    except Exception:
        log.exception("could not connect to MongoDB at startup")
        return 1

    registry = CameraRegistry(database.db["cameras"])
    store = ZoneStore(database.db["zones"])
    _bootstrap_first_camera(registry, settings, args.source)

    users = UserRegistry(database.db["users"])
    if not _bootstrap_first_admin(users, settings):
        log.error(
            "no admin account exists and PERIMETER_ADMIN_EMAIL/PERIMETER_ADMIN_PASSWORD are not "
            "set - there would be no way to log in. Set both in .env for the first run. "
            "See README for setup."
        )
        return 1

    alerts = AlertBus(database=database)
    alerts.start()
    _start_retention_sweeper(database, settings.storage.retention_days)

    if not args.model.is_file():
        log.error("model not found: %s\nSee README for the download step.", args.model)
        return 1

    # Unlike the person model above, this is never fatal (Expansion Plan Phase E) - no
    # trained weights exist yet for most installs, and fire/smoke is one optional module
    # among several, not something the rest of the product depends on.
    firesmoke_model_path = str(args.firesmoke_model) if args.firesmoke_model.is_file() else None
    if firesmoke_model_path is None:
        log.info(
            "fire/smoke detection unavailable - no model at %s. See "
            "backend/models/firesmoke/README.md.",
            args.firesmoke_model,
        )

    # Same posture as fire/smoke above (Expansion Plan Phase H) - no trained weights
    # exist yet for most installs, and PPE is one optional module among several.
    ppe_model_path = str(args.ppe_model) if args.ppe_model.is_file() else None
    if ppe_model_path is None:
        log.info(
            "PPE compliance checking unavailable - no model at %s. See "
            "backend/models/ppe/README.md.",
            args.ppe_model,
        )

    identity_resolver = _build_identity_resolver(
        settings, args.yunet_model, args.sface_model, database
    )

    manager = PipelineManager(
        settings, registry, store, str(args.model),
        alerts=alerts, firesmoke_model_path=firesmoke_model_path,
        identity_resolver=identity_resolver, ppe_model_path=ppe_model_path,
    )

    evidence_store = LocalEvidenceStore(
        root=BACKEND_ROOT / settings.storage.evidence_root,
        snapshot_dir=settings.storage.snapshot_dir,
        clip_dir=settings.storage.clip_dir,
    )
    evidence_writer = EvidenceWriter(
        database=database,
        store=evidence_store,
        ring_buffer_lookup=lambda cam_id: (
            pipeline.ring_buffer if (pipeline := manager.get(cam_id)) else None
        ),
        post_roll_seconds=settings.storage.post_roll_seconds,
    )
    evidence_writer.start()
    alerts.sinks.append(evidence_writer.on_alert)

    alert_config_store = AlertConfigStore(database.db["alert_config"])
    email_sink = _build_email_sink(settings)
    if email_sink is not None:
        email_sink.start()
    alert_router = AlertRouter(alert_config_store, CooldownGate(), email_sink)
    alerts.sinks.append(alert_router.dispatch)

    if not args.no_pipeline:
        manager.start()
    else:
        log.info("--no-pipeline: serving the API/dashboard without starting any cameras")

    app = create_app(
        settings,
        store,
        registry,
        manager,
        users,
        database=database,
        alerts=alerts,
        evidence_store=evidence_store,
        alert_config_store=alert_config_store,
        identity_resolver=identity_resolver,
    )
    host = args.host or settings.api.host
    port = args.port or settings.api.port

    log.info("dashboard on http://%s:%d", host, port)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        manager.stop()
        alerts.stop()
        evidence_writer.stop()
        if email_sink is not None:
            email_sink.stop()
        database.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
