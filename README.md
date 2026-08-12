# Fire, Smoke & Boundary Detection

Fire and smoke detection plus boundary intrusion alerting for fixed-mount cameras.
CPU-only, permissive licences throughout, intended for commercial deployment.

- **[PLAN.md](PLAN.md)** — architecture, phases, and the reasoning behind each decision
- **[NOTICE.md](NOTICE.md)** — licence and provenance record for every model, dataset and dependency

> **Not a certified fire alarm.** This system provides supplementary situational
> awareness. It is not a substitute for certified fire detection under UL 268 or EN 54.
> See PLAN.md section 10.1.

## Status

**Phases 0–2 complete.** Licence gates, site recorder, person detection, and tracking.
No boundary engine yet — see PLAN.md section 9 for the phase plan.

| Phase | Deliverable | Result |
|---|---|---|
| 0 | Repo, licence gates, site recorder | ✅ |
| 1 | Capture, YOLOX ONNX person detection, fps gate | ✅ **39 fps**, gate passed |
| 2 | ByteTrack tracking, ID-stability measurement | ✅ churn 2.0, 0 % fragments |
| 3 | Zone dashboard + boundary state machine | ✅ 4 zone types, live events |
| 4 | Alerts: SQLite, snapshot, clip, Telegram | next |

## Models

Weights are **not** in git — they are recorded in [NOTICE.md](NOTICE.md) and
`models/*/manifest.json` with SHA-256. Fetch the person detector:

```bash
curl -L -o models/yolox_person/yolox_nano.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
```

`manifest.json` is load-bearing for correctness, not just licensing: it records each
model's expected colour order. Megvii's exports expect BGR; OpenCV Zoo's re-export of
the same architecture expects RGB, and the wrong choice costs ~25 % recall **silently**.
A model without a manifest entry fails at load rather than under-detecting quietly.

## Configuration

Split by **what the value is**, not by whether it happens to be secret:

| File | Contains | Committed? |
|---|---|---|
| `.env` | **per-deployment facts** — camera source and credentials, resolution, decode rate, autostart | ❌ git-ignored |
| `config/app.yaml` | **engineering constants** — inference cadences, K-of-N gates, hysteresis defaults | ✅ yes |
| `config/zones.yaml` | zone polygons, drawn on one specific camera view | ❌ git-ignored |

Camera settings are per-deployment facts — camera index `0` on a dev laptop, an RTSP URL
at the customer site. Committing them would mean the repository carried one installation's
wiring as if it were product configuration, and every site would conflict on it. Templates
(`.env.example`, `config/zones.example.yaml`) are committed instead.

`decode_fps` lives in `.env` because it describes what *this machine* can keep up with.
`person_every_n` lives in `app.yaml` because that is the tuning decision.

```bash
cp .env.example .env
cp config/zones.example.yaml config/zones.yaml   # or just draw zones in the dashboard
```

Precedence is environment variable → `.env` → `app.yaml` → code default.

### Camera source: one boolean

```bash
FSBD_USE_CCTV=false          # false → laptop webcam, true → site CCTV
FSBD_CAMERA_SOURCE=0         # webcam index, or a video file path for development
FSBD_CCTV_RTSP_URL=          # rtsp://user:pass@192.168.1.64:554/stream1
```

An RTSP URL embeds the camera password in plain text, so every path that logs or displays
a source runs it through `redact()` first — otherwise it lands in log archives, the
`/api/health` response, and support bundles. The dashboard treats the URL as
**write-only**: you can replace it, never read it back.

## Alert storage (PostgreSQL)

Every alert is written to PostgreSQL and read back by the dashboard's **History** view.

```bash
# 1. edit the password inside sql/setup.sql, then run it as a superuser
psql -U postgres -f sql/setup.sql

# 2. put the URL in .env
FSBD_DATABASE_URL=postgresql://fsbd_app:YOURPASS@localhost:5432/fsbd
```

The schema is applied automatically at startup and is idempotent. `sql/schema.sql` is the
readable reference.

Alert wording is `Someone entered <zone>` / `Someone left <zone>`, with a timestamp. The
sentence is **stored on the row** rather than rebuilt at read time, so history always
shows what was actually sent even after the wording changes.

**Detection outranks persistence.** If PostgreSQL is unreachable the system keeps
detecting, the dashboard keeps showing alerts from memory, and the alert bus retries with
backoff. The header shows storage as a **separate indicator** from the camera — "alerts
firing but not recorded" needs a different response from "site unwatched".

> Driver note: this uses **pg8000** (BSD-3-Clause), not psycopg. `psycopg2` is
> LGPL-with-exceptions and `psycopg` v3 is LGPL-3.0 — the obvious choice would have failed
> the licence gate. See NOTICE.md §3.

### Ready for face recognition, not doing it yet

The schema carries a `persons` table and `identity_status` / `identity_name` /
`person_id` columns on `events`, created now and unused. Adding YuNet + SFace later
becomes "populate the table and set three columns" instead of a schema migration against
a table that by then holds live alert history.

`alerts/messages.py` has the matching seam: `subject_for()` returns `"Someone"` today and
a person's name once identity lands. A face found but not matched reads *"An unrecognised
person"*; **no usable face reads the same as having no identity layer at all** — outdoors
at a gate that is the common case, and rendering abstention as an accusation would be
worse than saying nothing.

## Dashboard

```bash
fsbd-dashboard                          # uses .env
python -m fsbd.main --source clip.mp4   # or point it at a file
```

Then open <http://127.0.0.1:8000>.

**React 19 + Vite 8 + Tailwind 4**, built to `web/dist` and served by the same FastAPI
process — one service on one port, no separate web server to install and patch on a
customer's box.

```bash
cd frontend
npm install
npm run build        # → web/dist, served at /
npm run dev          # or: Vite on :5173, proxying /api to :8000
npm run licences     # npm licence gate (pip-licenses cannot see node_modules)
```

Four views in a sidebar shell:

- **Live view** — MJPEG stream with boundaries, tracked IDs and foot points drawn on it
- **Boundaries** — freezes a frame; pick a type, click to place vertices, double-click or
  Enter to close, drag handles to adjust, right-click a handle to delete. Properties
  panel for name, classes, events, direction, severity, hysteresis and active hours
- **Camera** — webcam/CCTV toggle, resolution, fps, **Test connection**, save and
  hot-reconnect without restarting
- **Alert history** — every recorded alert with type and boundary filters, plus counts

The status rail shows **camera and recording as separate indicators**, because they fail
independently: a dead camera means the site is unwatched, a dead database means alerts
are firing but not being kept.

### UI end-to-end test

```bash
python -m fsbd.main --source clip.mp4 --port 8081 &
python tools/ui_e2e.py http://127.0.0.1:8081 ./shots
```

Drives the real browser: draws a boundary by clicking the canvas, saves, verifies the
stored coordinates match where it clicked, then waits for alerts to fire. It is the only
test that exercises canvas coordinate mapping, and it has already earned its keep — it
caught a transparent hint overlay silently swallowing clicks in the bottom-left of the
frame, which cost a vertex on any boundary drawn near that corner.

Needs `pip install playwright && playwright install chromium` (dev only).
- Four zone types: **Zone** (entry/exit), **Tripwire** (directional), **Exclusion**
  (suppress detections), **Fire ROI** (bias fire/smoke confidence)
- Per-zone rules: name, classes, events, direction, severity, hysteresis, schedule
- **Save** writes `config/zones.yaml`; the engine hot-reloads without a restart

> ⚠️ The API binds to `127.0.0.1` and has **no authentication**. It is a commissioning
> tool — anyone who can reach it can redraw the zones that arm the site. Exposing it on
> `0.0.0.0` needs a reverse proxy with auth in front.

**Exclusion masks are the highest value-per-hour feature here.** The worst false-alarm
sources are fixed in place — the welding bay, the beacon on the forklift charger, the
skylight that throws sunlight at 16:00. Masking them does more for the false-alarm rate
than another twenty thousand training images.

**Schedules matter almost as much.** A warehouse zone without one fires four hundred
times during the working day and gets switched off in week one.

### Reading the live overlay

The preview labels *why* a track is in the state it is, not just that it exists — which
is the question an installer is actually asking:

| Label | Meaning |
|---|---|
| `#7 0.82 44x118px` | tracked normally; confidence and box size |
| `#7 0.60 SMALL 18x34px (id needs 120)` | too small for the identity layer |
| `#7 MASKED (exclusion zone)` | suppressed by an exclusion mask |

The bar along the top carries fps, inference ms, tracked count, events and camera id.
If the feed drops, the stream shows a **rendered "CAMERA DISCONNECTED" frame with a live
timestamp** rather than freezing on the last good image — on a quiet scene those two look
identical, and one of them means the site is unprotected.

**Test connection** reports the *actual* resolution, not the requested one. A camera that
silently ignores a 1280×720 request changes what every pixel threshold means.

## Benchmark and preview tools

```bash
# Phase 1 gate - inference throughput on this machine
python tools/benchmark_fps.py --models models/yolox_person/*.onnx --source clip.mp4

# Phase 2 - track stability. Run against REAL site footage, not a test clip.
python tools/track_preview.py --source clip.mp4 --out annotated.mp4
```

`track_preview` reports churn ratio and fragment rate, which is what decides whether the
boundary state machine's hysteresis is sufficient in a given space. Warehouse racking
produces far more occlusion than open ground, so this must be rerun per site.

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
