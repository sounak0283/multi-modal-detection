"""Entry point: runs the pipeline and serves the dashboard.

    fsbd-dashboard                      # uses .env + config/app.yaml
    python -m fsbd.main --source 0      # override the camera for a quick test

The API binds to 127.0.0.1 by default. It has no authentication - it is a commissioning
tool, and anyone who can reach it can redraw the zones that arm the site. Putting it on
0.0.0.0 needs a reverse proxy with auth in front of it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import uvicorn

from fsbd.api.app import create_app
from fsbd.boundary.zones import ZoneStore
from fsbd.pipeline import Pipeline
from fsbd.settings import describe_source, load_settings

log = logging.getLogger("fsbd")

DEFAULT_MODEL = Path("models/yolox_person/yolox_nano.onnx")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="Override FSBD_CAMERA_SOURCE.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--zones", type=Path, default=None)
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Serve the editor without opening the camera (useful for zone review).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    settings = load_settings()

    # --source is a development override: it forces the webcam/file branch regardless of
    # FSBD_USE_CCTV, so `--source clip.mp4` works without editing .env.
    if args.source is not None:
        settings = replace(
            settings,
            camera=replace(settings.camera, use_cctv=False, source=args.source),
        )
    if args.zones is not None:
        settings = replace(settings, zones_path=args.zones)

    store = ZoneStore(settings.zones_path, camera_id=settings.camera.id)

    autostart = settings.camera.autostart and not args.no_pipeline
    if not autostart and not args.no_pipeline:
        log.info("FSBD_AUTOSTART_CAMERA is false - serving the dashboard without a camera")

    pipeline: Pipeline | None = None
    if autostart:
        if not args.model.is_file():
            log.error(
                "model not found: %s\nSee README for the download step.", args.model
            )
            return 1
        pipeline = Pipeline(settings, store, str(args.model))
        pipeline.start()

    app = create_app(settings, store, pipeline)
    host = args.host or settings.api.host
    port = args.port or settings.api.port

    log.info(
        "dashboard on http://%s:%d  (camera %s)", host, port, describe_source(settings.camera)
    )
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        if pipeline is not None:
            pipeline.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
