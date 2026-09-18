# Phase A — MongoDB Migration + Multi-Camera Architecture

- **Status:** ✅ Complete
- **Date:** 12 September 2026
- **Plan reference:** [`PLATFORM_EXPANSION_PLAN.md`](../../PLATFORM_EXPANSION_PLAN.md) §10, Phase A
- **Depends on:** nothing (foundation phase)
- **Blocks:** every later phase (B–J)

## 1. Summary

Replaced the single-camera, PostgreSQL-backed runtime with a multi-camera,
MongoDB-backed one. Any number of cameras can now be registered, each independently
selecting which detection modules run on it (`boundary` is always on; `crowd`, `ppe`,
`phone`, `fire_smoke`, `identity` are reserved toggles for Phases E–I). Camera and zone
configuration moved off `.env`/`zones.yaml` into MongoDB collections, and a
`PipelineManager` now runs one `Pipeline` instance per enabled camera, reconciling
against the registry as cameras are added, edited, disabled or removed — no restart
required.

All 297 backend tests pass, `ruff` is clean, the frontend builds cleanly, and the full
stack was manually verified end-to-end in a real browser against an in-memory
(`mongomock`) database (see §6).

## 2. What was implemented

### Backend — new files

| File | Purpose |
|---|---|
| `backend/src/perimeter/cameras/models.py` | `Camera` dataclass, `SourceType`/module-name constants, `camera_from_dict()` validation, `describe_source()` |
| `backend/src/perimeter/cameras/registry.py` | `CameraRegistry` — MongoDB CRUD over the `cameras` collection, versioned like `ZoneStore` |
| `backend/src/perimeter/cameras/manager.py` | `PipelineManager` — reconciles running `Pipeline` instances against the registry (start/stop/reconfigure), polling every 5s plus an immediate `notify_camera_changed()` hook from the API |
| `backend/tests/test_store_mongo.py` | Direct unit tests for the new `Database` class (18 tests) |
| `backend/tests/test_cameras_registry.py` | Direct unit tests for `Camera`/`CameraRegistry`/`PipelineManager` (20 tests) |

### Backend — rewritten files

| File | What changed |
|---|---|
| `backend/src/perimeter/store/db.py` | Full rewrite: `pg8000`/PostgreSQL → `pymongo`/MongoDB. Same public method contract (`insert_event`, `recent_events`, `event_counts`, `zone_entry_counts`, `healthy`, `purge_older_than`, `mark_delivered`) so `AlertBus` and the API needed minimal changes. Event ids are now MongoDB ObjectId strings, not integers. `migrate()` → `ensure_indexes()`. |
| `backend/src/perimeter/boundary/zones.py` | `ZoneStore` moved from YAML-file-backed (mtime hot-reload) to MongoDB-backed (throttled poll, `refresh_interval` now configurable — tests use `0` for determinism). `Zone` gained five inert fields reserved for later phases: `authorized_person_ids`, `crowd_threshold`, `crowd_min_frames`, `ppe_required`, `phone_restricted`. Added `Severity.CRITICAL` (for Phase F.1). |
| `backend/src/perimeter/pipeline.py` | `Pipeline.__init__`/`.reconfigure()` now take a `cameras.models.Camera` instead of reading camera fields off `Settings.camera` — every `self.settings.camera.*` reference replaced with `self.camera.*`. |
| `backend/src/perimeter/main.py` | Bootstraps `Database` + `CameraRegistry` + `ZoneStore` + `PipelineManager` instead of one `Pipeline`. MongoDB is now a **hard requirement at startup** (see §4). Seeds a first camera from `.env`/`--source` into an empty registry so single-camera quick-start still works. Wires up the daily retention sweep (`Database.purge_older_than`, previously dead code even under Postgres). |
| `backend/src/perimeter/api/app.py` | Routes restructured around `/api/cameras` (list/create/get/update/delete/test) and per-camera `/api/cameras/{id}/zones`, `/snapshot`, `/stream`, `/track-states`. `/api/events*` gained an optional `camera_id` filter and now requires MongoDB configured (no more in-memory fallback — see §4). `/api/health` is now a system-wide summary (`database` + a `cameras: [...]` list) rather than one camera's stats. |
| `backend/src/perimeter/settings.py` | `StorageSettings.database_url` → `mongo_url`/`mongo_db`. `CameraSettings` narrowed to "dev-mode bootstrap seed" only (see its docstring). Removed the now-unused `zones_path`/`DEFAULT_ZONES`. |
| `backend/src/perimeter/alerts/bus.py` | Comments updated from PostgreSQL to MongoDB wording; no logic changes (it was already storage-agnostic, duck-typed against `database.insert_event`). |
| `backend/tests/test_boundary_engine.py`, `test_zones.py`, `test_alerts.py`, `test_camera_api.py`, `test_api.py`, `test_camera_settings.py` | Migrated off YAML-file/`.env`-camera fixtures onto `mongomock` collections and the new `Camera`/`CameraRegistry`/`PipelineManager` API surface. Removed tests for things that no longer exist (`parse_dsn`, `dump_zones`/`parse_zones`, `_camera_from_payload`). |

### Backend — removed

- `backend/sql/` (`schema.sql`, `setup.sql`) — MongoDB has no schema to migrate; camera and zone config no longer live in files/SQL at all.
- `pg8000` dependency.

### Frontend — new files

| File | Purpose |
|---|---|
| `frontend/src/pages/Cameras.jsx` | Replaces `CameraSettings.jsx`. Camera list + add/edit/delete detail panel, including the per-camera detection-module `ChipGroup`. |

### Frontend — rewritten/updated files

| File | What changed |
|---|---|
| `frontend/src/api.js` | Every camera/zone/video/event call is now camera-scoped (`listCameras`, `createCamera`, `getCamera(id)`, `updateCamera(id, ...)`, `deleteCamera(id)`, `getZones(cameraId)`, `saveZones(cameraId, zones)`, `streamUrl(cameraId, bust)`, `snapshotUrl(cameraId, bust)`, `events({..., cameraId})`, etc.). |
| `frontend/src/App.jsx` | Owns `cameras`/`cameraId` state, polls the camera list, renders a camera switcher in the header for Live/Boundaries, shows a "no cameras yet" empty state, and remounts `LiveView`/`Boundaries` (`key={cameraId}`) on camera switch (see §4 for why). |
| `frontend/src/components/Shell.jsx` | Nav: `camera` → `cameras`. `StatusRail` rewritten for the new multi-camera `/api/health` shape (`N/M cameras running` instead of one camera's feed state / fps / tracked count). |
| `frontend/src/pages/LiveView.jsx` | Takes a `cameraId` prop; per-camera stats now come from `api.getCamera(cameraId)` instead of the old global `/api/health`. |
| `frontend/src/pages/Boundaries.jsx` | Takes a `cameraId` prop for the snapshot backdrop. |
| `frontend/src/pages/AlertHistory.jsx` | Added a camera filter dropdown; removed the now-unreachable "PostgreSQL down, showing in-memory alerts" fallback path (`/api/events` errors outright now instead of degrading — see §4) and simplified `deriveStats` accordingly. |

### Frontend — removed

- `frontend/src/pages/CameraSettings.jsx` — superseded by `Cameras.jsx`.

### Documentation

- `PLATFORM_EXPANSION_PLAN.md` — no changes this phase (already written before implementation started).
- `README.md` — camera/storage sections rewritten for MongoDB + multi-camera; repo layout tree updated.
- `NOTICE.md` — `pg8000`/PostgreSQL rows replaced with `pymongo`/`mongomock`/MongoDB Atlas rows in the shipped-dependency table; the excluded-driver note reframed as historical.
- `backend/.env.example` — `PERIMETER_DATABASE_URL` → `PERIMETER_MONGO_URL`/`PERIMETER_MONGO_DB`; camera variables reframed as "dev-mode bootstrap only".
- `backend/tools/ui_e2e.py` — updated to hit `/api/cameras/{id}/zones` instead of the removed `/api/zones`.

## 3. New dependencies

| Package | Where | Licence |
|---|---|---|
| `pymongo>=4.8,<5` | Runtime | Apache-2.0 |
| `mongomock>=4.1,<5` | Dev/test only, never shipped | BSD-3-Clause |

`pg8000` was removed from both `pyproject.toml` and `requirements.txt`.

## 4. Errors found and fixed during implementation

These were real defects caught before they shipped, not just typos:

1. **`PipelineManager.notify_camera_changed()` would open real camera hardware from a
   unit test.** The first version called `self._reconcile()` unconditionally, which
   constructs a real `Pipeline` (opens the camera, loads the ONNX model) the moment a
   camera is created via the API — including under `--no-pipeline` and inside API unit
   tests that never call `manager.start()`. Fixed by adding a `self._started` guard;
   `notify_camera_changed()` and the poll loop are now no-ops until `start()` has been
   called. Covered by `test_notify_camera_changed_is_a_noop_before_start`.
2. **The frontend camera switcher wouldn't refresh promptly.** `usePolling` keeps its
   `setInterval` running with whatever closure it started with and only updates a ref for
   the *next* tick — switching cameras would keep showing the previous camera's stats
   until the next poll fired (up to 10s later for the summary poll). Fixed by keying
   `LiveView`/`Boundaries` on `cameraId` in `App.jsx` (`key={cameraId}`), forcing a clean
   remount and an immediate fetch on switch.
3. **Stray dead code** left behind while drafting `camera_from_dict` (an empty
   `for identity_gated in (...): pass` loop) — removed before it shipped.
4. **Cascading test breakage from three removed APIs** (`store.db.parse_dsn`,
   `boundary.zones.dump_zones`/`parse_zones`, `api.app._camera_from_payload`) required
   migrating `test_alerts.py`, `test_zones.py`, `test_camera_api.py`, and `test_api.py`
   rather than just patching imports — the fixtures themselves were file-based/YAML and
   needed rebuilding around `mongomock` and the new `Camera`/`CameraRegistry` API. All
   pre-existing test *intent* was preserved; nothing was deleted to make it pass except
   tests for formats that no longer exist (e.g. the old `{"cameras": {...}}` top-level
   YAML document shape).

## 5. Deliberate behavioural changes (read before deploying)

- **MongoDB is now required at startup, not optional.** In the original PostgreSQL
  design, a database outage was non-fatal because zones lived in a YAML file independent
  of it. Now camera and zone configuration live in MongoDB too, so there is nothing to
  run without it — `perimeter-dashboard` logs an error and exits (code 1) if `PERIMETER_MONGO_URL`
  is unset or unreachable at boot. A MongoDB outage *after* startup remains non-fatal for
  already-running cameras (unchanged `AlertBus` retry/backoff behaviour).
- **`/api/events` no longer has an in-memory fallback.** The old version served from the
  pipeline's in-memory ring when PostgreSQL was down. With N cameras and no guarantee a
  given camera's in-process ring still exists (it can be deleted from the registry while
  its history should remain queryable), MongoDB is the only source of truth for history
  now; the endpoint returns 503 instead of degrading. `/api/events/live` (in-memory,
  "what's happening right now") is unaffected and still works with no database at all.
- **RTSp credentials move from `.env` to MongoDB documents.** Necessary for multi-camera
  (`.env` can only hold one camera's secret); redact-on-read behaviour is preserved
  (`to_public_dict()` never returns the raw URL), and access to camera documents should
  be admin-only once Phase B (auth) lands — until then, the same "no auth yet, don't
  expose this on the open network" caveat that already applied to `.env` applies to the
  `cameras` collection.

## 6. Verification

**Automated:**

```
297 passed, 1 skipped in ~24s   (backend/tests/, pytest)
ruff check src/ tests/ → All checks passed!
npm run build → builds cleanly, 234.71 kB bundle
```

The one skip is `test_missing_build_says_so_rather_than_showing_a_blank_page`, which
`pytest.skip`s *because* the frontend is built (it tests the opposite condition) — not a
gap.

**Manual, in a real browser:** booted the actual FastAPI app (`create_app` with a
`mongomock`-backed `Database`/`CameraRegistry`/`ZoneStore`, `PipelineManager.start()`
deliberately not called since there's no real camera hardware on the dev box) against
the production frontend build, seeded two cameras (one with `crowd`+`identity` modules
enabled) and one zone, and drove it with Playwright:

- **Live view** — camera switcher shows both cameras by name; occupancy panel correctly
  shows the seeded "Loading Bay" zone at 0/0; status rail correctly reads "0/2 running"
  and "not recording" (no pipeline started, no database configured, both as expected for
  this harness).
- **Cameras** — list shows both cameras; selecting "Front gate" shows its form populated
  correctly, with **Crowd formation** and **Identity / face recognition** chips
  highlighted, matching what was seeded.
- **Boundaries** — correctly loads and displays the one seeded zone ("Loading Bay") for
  the selected camera.
- **Alert history** — "All cameras" filter present; correctly shows
  "Could not load history: event storage is not configured" (matches §5's new 503
  behaviour) rather than crashing or silently showing stale data.
- Only console errors were the three expected 503s (snapshot/stream/events, all because
  no pipeline or database was actually running in this harness) — no React errors, no
  broken layout.

The one-off smoke scripts used for this (`tools/_phase_a_smoke.py`,
`tools/_phase_a_playwright.py`) were deleted after use; they are not part of the test
suite.

**Update, same day:** real MongoDB Atlas credentials were provided after this phase was
first written up. `Database(url, db_name).ensure_indexes()` / `.healthy()` were run
against the live cluster and connected successfully (`PERIMETER_MONGO_URL` in `backend/.env`,
never committed). The "not verified" note below now applies only to running against a
real camera/RTSP source.

**Not verified in this phase:** running against a real camera/RTSP source with
`PipelineManager.start()` actually opening hardware end to end (needs a camera on the
deployment box or network, not just database credentials).

## 7. Known limitations / carried forward to later phases

- `ZoneStore.save()` is not transactional (`delete_many` + `insert_many`, not wrapped in
  a MongoDB multi-document transaction). Scoped to one camera's zones at a time; a torn
  write is recoverable by re-saving from the dashboard, which holds the full desired
  state in memory. Noted as an accepted trade-off in the original plan, not revisited
  here.
- No authentication yet — camera CRUD, zone edits, and the dev-tools alert-simulation
  endpoint are all open to anyone who can reach the API, same posture as before this
  phase (Phase B addresses this next).
- Only `enabled_modules` bookkeeping exists for crowd/PPE/phone/fire-smoke/identity —
  `Pipeline` does not yet build any detector for them (they don't exist yet). Enabling
  one on a camera today reserves the choice; it has no runtime effect until its owning
  phase ships.
- The retention sweep (`main.py`'s daily `purge_older_than` call) has not been exercised
  against a real MongoDB deployment over a real 24-hour cycle — only unit-tested against
  `mongomock` with a synthetic cutoff.
