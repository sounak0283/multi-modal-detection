# Fire, Smoke & Boundary Detection

Fire and smoke detection plus boundary intrusion alerting for fixed-mount cameras.
CPU-only, permissive licences throughout, intended for commercial deployment.

- **[PLAN.md](PLAN.md)** — architecture, phases, and the reasoning behind each decision
- **[NOTICE.md](NOTICE.md)** — licence and provenance record for every model, dataset and dependency

> **Not a certified fire alarm.** This system provides supplementary situational
> awareness. It is not a substitute for certified fire detection under UL 268 or EN 54.
> See PLAN.md section 10.1.

## Status

**Phase 0 complete.** Repo skeleton, licence gates, and the site recorder that feeds
negative collection. No detection pipeline yet — see PLAN.md section 9 for the phase plan.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
pip install -e .
```

Requires Python 3.11+. Verified on 3.14 with OpenCV 4.14 and onnxruntime 1.28.

## Site recorder (Phase 0)

Fire-model false alarms are decided by site-specific negatives — welding sparks,
forklift beacons, machinery steam, sunlight through skylights. Those depend on the
customer's operations happening, not on developer time, so **recording starts before
anything else is built** and the footage is mined later.

```bash
# RTSP camera, 4 fps, 15-minute segments, capped at 100 GB
python tools/record_site.py --source "rtsp://user:pass@10.0.0.5:554/stream1" \
    --outdir recordings --disk-budget-gb 100

# Local webcam, also dumping a JPEG every 30 s for direct dataset mining
python tools/record_site.py --source 0 --still-interval 30

# Smoke-test against a video file
python tools/record_site.py --source clip.mp4 --outdir /tmp/rec --fps 2
```

Output layout:

```
recordings/
  2026-08-11/
    seg_090000.mp4        # 15 min at 4 fps
    seg_091500.mp4
  stills/
    2026-08-11/
      still_090030_123456.jpg
```

Records at low fps and re-encodes deliberately: the purpose is harvesting still frames
for a training set, not evidence video. Disk is a hard budget with oldest-first pruning,
and the capture loop reconnects with exponential backoff.

**Recordings contain customer premises footage.** They are git-ignored and fall under
the retention policy in PLAN.md section 10.3.

## Licence gates

This product is sold, so a copyleft dependency reaching a customer is a commercial
incident. Two gates run in CI ([.github/workflows/licence-gate.yml](.github/workflows/licence-gate.yml)):

```bash
# Gate 1 - Python packages
pip-licenses --fail-on="GPL;AGPL;LGPL;CC-BY-NC;CC-BY-SA;Proprietary;Unknown" \
    --ignore-packages opencv-python shapely fsbd

# Gate 2 - models and datasets (pip-licenses is blind to these)
python tools/check_notice.py --assets models data --notice NOTICE.md
```

Gate 2 exists because `pip-licenses` reads package metadata only and cannot see an
`.onnx` file or a dataset directory — which is exactly where the real risk sits. It also
enforces a `manifest.json` in every `models/<name>/`, so that "which shipped weights are
affected?" stays answerable if a dataset's terms later prove different from what was
recorded.

Nothing enters `models/` or `data/` without a NOTICE.md row **recorded at download time**.

## Tests

```bash
pytest -q
ruff check src tools tests
```
