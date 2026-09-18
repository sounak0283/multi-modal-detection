# Fire, Smoke & Boundary Detection

Fire and smoke detection plus boundary intrusion alerting for fixed-mount cameras.
CPU-only, permissive licences throughout, intended for commercial deployment.

- **[PLAN.md](PLAN.md)** — architecture, phases, and the reasoning behind each decision
- **[NOTICE.md](NOTICE.md)** — licence and provenance record for every model, dataset and dependency
- **[docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md)** — which remaining phases need an
  actual GPU training run vs. ship against pretrained weights, and current status of each

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

| File / store | Contains | Committed? |
|---|---|---|
| `.env` | **per-deployment secrets** — MongoDB/AWS connection strings, the dev-bootstrap camera's source and credentials | ❌ git-ignored |
| `config/app.yaml` | **engineering constants** — inference cadences, K-of-N gates, hysteresis defaults | ✅ yes |
| MongoDB `cameras`/`zones` collections | camera registry and zone polygons for every camera | n/a — a database, not a file, since Expansion Plan Phase A |
| `config/zones.example.yaml` | reference documentation of the zone shape only | ✅ yes, but no longer read by the running app |

`decode_fps` lives in `.env` because it describes what *this machine* can keep up with.
`person_every_n` lives in `app.yaml` because that is the tuning decision. Zones are drawn
in the dashboard's **Boundaries** page, not edited as a file.

```bash
cp .env.example .env
# then set PERIMETER_MONGO_URL before first run - see "Storage (MongoDB)" below
```

Precedence is environment variable → `.env` → `app.yaml` → code default.

### Camera source: legacy single-camera bootstrap only

Since the [Expansion Plan](PLATFORM_EXPANSION_PLAN.md)'s Phase A, cameras are configured
through the **Cameras** page (or the `/api/cameras` API), backed by MongoDB — any number
of them, each with its own module selection. The `.env` variables below only matter on a
brand-new install with an empty camera registry: `main.py` seeds exactly one camera from
them so `perimeter-dashboard` still "just works" without a dashboard visit first.

```bash
PERIMETER_USE_CCTV=false          # false → laptop webcam, true → site CCTV
PERIMETER_CAMERA_SOURCE=0         # webcam index, or a video file path for development
PERIMETER_CCTV_RTSP_URL=          # rtsp://user:pass@192.168.1.64:554/stream1
```

An RTSP URL embeds the camera password in plain text, so every path that logs or displays
a source runs it through `redact()` first — otherwise it lands in log archives, the
`/api/health` response, and support bundles. Once a camera exists in the registry, its
RTSP URL is edited from the Cameras page and is **write-only**: you can replace it, never
read it back.

## Storage (MongoDB)

Camera configuration, zone definitions and every alert live in MongoDB — self-hosted or
Atlas. See `docs/phases/PHASE_A.md` for the full migration record and rationale.

```bash
PERIMETER_MONGO_URL=mongodb://localhost:27017     # or an Atlas mongodb+srv:// connection string
PERIMETER_MONGO_DB=perimeter
```

Indexes are created automatically at startup and the operation is idempotent. Unlike the
project's original PostgreSQL-based design, **MongoDB is a hard requirement at startup**:
`perimeter-dashboard` will not start without it, because camera and zone configuration live
there too, not just alert history — there is no local file it can fall back to describing
which cameras to run. A MongoDB outage *after* startup is still non-fatal for an
already-running camera: the pipeline keeps detecting and the dashboard keeps showing
alerts from its in-memory buffer while the alert bus retries inserts with backoff.

Alert wording is `Someone entered <zone>` / `Someone left <zone>`, with a timestamp. The
sentence is **stored on the row** rather than rebuilt at read time, so history always
shows what was actually sent even after the wording changes.

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

### Video evidence (Expansion Plan Phase C)

Every alert now carries a snapshot and an MP4 clip, shown in Alert History under a new
Evidence column. A per-camera ring buffer (`storage.ring_buffer_seconds` in
`config/app.yaml`, default 15s) holds recent JPEG frames; on an alert, `EvidenceWriter`
waits `storage.post_roll_seconds` (default 5s) so the buffer picks up some "after"
footage too, then muxes whatever's buffered into a clip and writes both under
`storage.evidence_root` (default `backend/data/evidence`). See
`docs/phases/PHASE_C.md` for the full record.

Storage is **local filesystem, not S3** — no AWS account is stood up for this repo.
`evidence/store.py`'s `LocalEvidenceStore` is written behind the same key-based interface
the plan's S3 design describes, so swapping in a real `S3EvidenceStore` later only
touches that one module and `main.py`'s wiring.

### Email alerts (Expansion Plan Phase D)

Alert routing rules (who gets emailed, minimum severity, cooldown) are managed from the
dashboard's **Alerts** page (admin-only) — not a config file. `PUT`/`GET
/api/alert-config` back it, stored in MongoDB's `alert_config` collection and picked up
live, no restart needed. SMTP credentials are the one part that still lives in `.env`
(a per-deployment secret, same axis as the Mongo URL):

```bash
PERIMETER_SMTP_HOST=
PERIMETER_SMTP_PORT=587
PERIMETER_SMTP_USER=
PERIMETER_SMTP_PASSWORD=
PERIMETER_SMTP_FROM=              # defaults to PERIMETER_SMTP_USER if unset
```

Both a configured SMTP host **and** at least one enabled rule on the Alerts page are
required for email to actually go out — either alone is a logged no-op. The Alerts page
also has a **system sound** toggle (with its own severity floor) that plays a short beep
in any open, already-interacted-with dashboard tab when a new alert arrives — email is
the reliable, always-on channel; the sound is a convenience on top of it, since browsers
require a prior click on the page before they'll allow audio to play automatically. See
`docs/phases/PHASE_D.md` for the full record.

### Fire & smoke detection (Expansion Plan Phase E)

The software path is fully built — a K-of-N temporal gate (`PLAN.md` §5.5), per-class
confidence thresholds, `fire_roi`/`exclusion` zone awareness, alert publishing — and as
of 2026-09-16 **`backend/models/firesmoke/model.onnx` ships real trained weights**: a
40-epoch YOLOX-Nano fine-tune on D-Fire (see `backend/models/firesmoke/README.md` and
`docs/MODEL_TRAINING.md` for the training run, hardware used, and — importantly — what
validation is still outstanding before this should be treated as deployment-ready:
false-alarm rate on real site video and site-negative testing per `PLAN.md` §5.4/§9.3
have not been done yet). Any camera with the `fire_smoke` module enabled now runs live
fire/smoke detection; if `model.onnx` is ever removed, it falls back to logging that it's
inactive rather than a startup failure. See `docs/phases/PHASE_E.md` for the software
integration record.

### Identity & restricted-zone access control (Expansion Plan Phase F / F.1)

Unlike fire/smoke, this ships against real pretrained weights — YuNet (face detection)
and SFace (face recognition), both from the OpenCV Zoo, no GPU training needed. Off by
default (`identity.enabled: false` in `config/app.yaml` — biometric processing needs a
per-deployment consent story, `PLAN.md` §10.3). When enabled, a confirmed boundary ENTRY
resolves the person's face (`known:<name>` / `unknown_face` / `no_face`) and, for any
zone with `authorized_person_ids` set, anyone not on that list raises a second,
`critical`-severity, unconditionally-delivered alert on top of the normal ENTRY log.
Enrol people from the dashboard's **People** page (admin-only); pick who's authorised per
zone from the boundary editor. See `docs/phases/PHASE_F.md` for the full record.

### Crowd formation (Expansion Plan Phase G)

No model — hand-rolled DBSCAN clustering (`scipy.spatial.cKDTree`) over already-tracked
person foot points, distinguishing a genuine huddle from people merely spread across a
wide zone. Turn on the `crowd` module for a camera on the Cameras page, then set a
threshold (and optional hysteresis) on a boundary zone in the editor — a zone with no
threshold set never evaluates crowding. Fires once per formation, not on every frame a
crowd persists. See `docs/phases/PHASE_G.md` for the full record.

### PPE detection (Expansion Plan Phase H)

The software path is fully built — a whole-person-crop multi-label classifier wrapper,
an explicit present/absent/indeterminate threshold, per-zone/per-item hysteresis, alert
publishing — but **no model ships in this repo**. Unlike fire/smoke, this needs a
genuinely new model architecture, not a fine-tune of the existing person detector; see
`backend/models/ppe/README.md` and `PLATFORM_EXPANSION_PLAN.md` §5 for the recipe
(SH17 + Construction-PPE). Until a real `model.onnx` is placed at
`backend/models/ppe/`, any camera with the `ppe` module enabled just logs that it's
inactive and runs everything else normally — this is not a startup failure. Pick
required items (helmet/vest/gloves/shoes/glasses) per zone in the boundary editor. See
`docs/phases/PHASE_H.md` for the full record.

## Auth

Every route requires a logged-in session (Expansion Plan Phase B). On first run, if the
`users` collection is empty, the process seeds one admin account from `.env` and refuses
to start without it:

```bash
PERIMETER_ADMIN_EMAIL=you@example.com
PERIMETER_ADMIN_PASSWORD=choose-something-long
```

Manage further accounts from the dashboard's **Accounts** page (admin-only) once logged
in — two roles: `admin` (full read/write) and `operator` (read-only, no configuration
changes). The session is a signed, httpOnly cookie (`itsdangerous`, not a JWT); set
`PERIMETER_SESSION_SECRET` in `.env` so sessions survive a restart, otherwise a random one is
generated each boot and every session is logged out on the next restart. See
`docs/phases/PHASE_B.md` for the full implementation record.

## Dashboard

```bash
perimeter-dashboard                          # uses .env
python -m perimeter.main --source clip.mp4   # or point it at a file
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

A sidebar shell behind a login page (see "Auth" above), each view aware of the camera
switcher in the header where it applies (Live view, Boundaries):

- **Live view** — MJPEG stream with boundaries, tracked IDs and foot points drawn on it,
  scoped to whichever camera is selected
- **Boundaries** — freezes a frame; pick a type, click to place vertices, double-click or
  Enter to close, drag handles to adjust, right-click a handle to delete. Properties
  panel for name, classes, events, direction, severity, hysteresis and active hours.
  Saving requires the `admin` role
- **Cameras** — add/remove any number of cameras, each with its own source, resolution,
  fps, **Test connection**, and a per-camera detection-module selection. Every module
  is opt-in, boundary included — a new camera runs no detection until you select one.
  Crowd, PPE, phone and identity need boundary's person tracking and switch it on with
  them; fire & smoke runs on its own. Adding, editing and removing a camera requires the
  `admin` role — an `operator` can view this page but not change anything on it
- **Alert history** — every recorded alert with type, boundary and camera filters, plus
  counts, and an Evidence column with a snapshot/clip lightbox (Expansion Plan Phase C)
- **Alerts** — `admin`-only. Email routing rules (recipients, minimum severity, cooldown)
  and the browser system-sound setting (Expansion Plan Phase D)
- **Accounts** — `admin`-only. Create/edit/deactivate accounts, assign the `admin` or
  `operator` role, reset a password

The status rail shows **camera and recording as separate indicators**, because they fail
independently: a dead camera means the site is unwatched, a dead database means alerts
are firing but not being kept.

### UI end-to-end test

```bash
python -m perimeter.main --source clip.mp4 --port 8081 &
export PERIMETER_E2E_EMAIL=you@example.com PERIMETER_E2E_PASSWORD=...   # an admin account
python tools/ui_e2e.py http://127.0.0.1:8081 ./shots cam_01
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

> ⚠️ The API binds to `127.0.0.1` by default. Every route requires a logged-in session
> (Expansion Plan Phase B — see "Auth" below), but exposing it on `0.0.0.0` still needs a
> reverse proxy terminating TLS in front (`PERIMETER_COOKIE_SECURE=true` once one is) — auth
> over plain http on an open network still exposes credentials and session cookies.

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

## Layout

```
backend/          Python: detection pipeline, API, storage
  src/perimeter/       the package
    cameras/      camera domain model, MongoDB registry, PipelineManager
    auth/         user model, MongoDB registry, session cookies (Expansion Plan Phase B)
    alerts/       AlertBus, EmailSink, cooldown gate, AlertConfigStore (Phase D)
    boundary/     zone model, MongoDB-backed ZoneStore, boundary state machine,
                  crowd.py (hand-rolled DBSCAN + per-zone hysteresis, Phase G)
    capture/      capture thread, drop-to-latest slot, evidence ring buffer
    evidence/     LocalEvidenceStore + EvidenceWriter (Expansion Plan Phase C)
    detect/       person + fire/smoke detectors, K-of-N temporal gate (Phase E);
                  ppe.py (whole-crop multi-label classifier wrapper, Phase H)
    identity/     YuNet + SFace wrappers, in-memory gallery, IdentityResolver (Phase F)
    store/        MongoDB event store + persons collection
  tests/          545 tests
  tools/          benchmark, recorder, UI end-to-end
  config/         app.yaml (committed); zones.example.yaml is reference only -
                  live zones are in MongoDB, not a file, since Expansion Plan Phase A
  models/         ONNX weights + manifest.json provenance; firesmoke/ is software-ready,
                  no weights ship (Expansion Plan Phase E); yunet/ + sface/ ship real
                  pretrained weights (Expansion Plan Phase F)
  web/dist/       built dashboard, served by FastAPI
  data/evidence/  clips + snapshots (Expansion Plan Phase C, local disk for now)
  .env            per-deployment secrets (ignored) - Mongo/AWS creds, dev bootstrap camera

frontend/         React 19 + Vite 8 + Tailwind 4
  src/            components, pages (incl. Cameras.jsx), design tokens
  scripts/        npm licence gate

NOTICE.md                    licence and provenance record for the whole product
PLAN.md                      original v1 architecture and phase plan
PLATFORM_EXPANSION_PLAN.md   multi-camera/multi-module expansion - architecture, models, phases
docs/phases/                 a completion report per expansion phase, written as it ships
```

`backend/` is a **self-contained deployable unit** — everything it reads at runtime
lives beneath it, including the built dashboard. That is why the frontend builds *into*
the backend rather than beside it: a deployment stays one process on one port, with no
separate web server to install and keep patched.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows

cd backend
pip install -r requirements-dev.txt
pip install -e .

cd ../frontend
npm install && npm run build      # → backend/web/dist
```

Requires Python 3.11+ and Node 20+. Verified on Python 3.14 with OpenCV 4.14 and
onnxruntime 1.28, and Node 26 with React 19.

Run it:

```bash
cd backend
perimeter-dashboard                    # → http://127.0.0.1:8000
```

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
# Gate 1 - Python packages                                    (from backend/)
pip-licenses --fail-on="GPL;AGPL;LGPL;CC-BY-NC;CC-BY-SA;Proprietary;Unknown" \
    --ignore-packages opencv-python shapely perimeter

# Gate 2 - models and datasets, invisible to pip-licenses      (from backend/)
python tools/check_notice.py --root .. \
    --assets backend/models backend/data --notice NOTICE.md

# Gate 3 - npm packages, invisible to both of the above       (from frontend/)
npm run licences
```

Three gates because no single tool sees everything: `pip-licenses` reads Python package
metadata, which cannot see an `.onnx` file, a dataset directory, or `node_modules` —
and the frontend is the half that compiles into the bundle a customer receives.

Gate 2 exists because `pip-licenses` reads package metadata only and cannot see an
`.onnx` file or a dataset directory — which is exactly where the real risk sits. It also
enforces a `manifest.json` in every `models/<name>/`, so that "which shipped weights are
affected?" stays answerable if a dataset's terms later prove different from what was
recorded.

Nothing enters `models/` or `data/` without a NOTICE.md row **recorded at download time**.

## Tests

```bash
cd backend
pytest -q                      # 545 tests
ruff check src tools tests
```

All `python tools/...` commands in this file run from `backend/`.
