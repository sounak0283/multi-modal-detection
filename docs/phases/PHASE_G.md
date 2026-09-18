# Phase G — Crowd formation

- **Status:** ✅ Complete
- **Date:** 15 September 2026
- **Plan reference:** [`PLATFORM_EXPANSION_PLAN.md`](../../PLATFORM_EXPANSION_PLAN.md) §4.2/§5
- **Depends on:** Phase A (MongoDB, `BoundaryEngine`'s zone geometry cache), Phase F (`engine.zone_by_id()`)
- **Blocks:** nothing — H/I/J are independent of this phase

## 1. Summary

`Zone.crowd_threshold`/`crowd_min_frames` have existed on the `Zone` dataclass since
Phase A — fully parsed, persisted, round-tripped by tests — but nothing ever consumed
them. `MODULE_CROWD` was likewise already a registered, selectable camera module, and
the Cameras page already offered "Crowd formation" as a toggle. This phase is what
wires all of it up for the first time, the same shape Phase E did for `fire_smoke` and
Phase F did for `identity`.

No model — crowd formation is DBSCAN clustering over already-tracked person foot
points, hand-rolled over `scipy.spatial.cKDTree` rather than adding scikit-learn for one
algorithm (`PLATFORM_EXPANSION_PLAN.md` §5's own reasoning). The reason to cluster
rather than just read `BoundaryEngine.zone_occupancy()`'s headcount: a zone can hold
many people spread across a wide area (not a crowd) or fewer people tightly bunched in
one corner (a genuine huddle) — density-connectivity is what tells the two apart.

All 483 backend tests pass, `ruff` is clean, and the frontend was rebuilt with a new
"Crowd formation" fieldset in the zone editor.

## 2. What was implemented

### Backend — new files

| File | Purpose |
|---|---|
| `boundary/crowd.py` | `CrowdMonitor` — hand-rolled DBSCAN (`_dbscan_cluster_sizes`, classic core-point/density-connected flood fill over `cKDTree.query_ball_point`) plus a per-zone hysteresis state machine mirroring `BoundaryEngine`'s confirmed/candidate design: fires once when the largest cluster reaches `threshold` and holds for `min_frames` frames, then requires the cluster to drop below threshold again before it can re-fire |
| `test_crowd_monitor.py`, `test_crowd_settings.py`, `test_crowd_pipeline.py` | DBSCAN clustering + hysteresis (pure logic, no model), `app.yaml` parsing, and zone-membership/publish wiring against a real `Pipeline` |

### Backend — changed files

| File | What changed |
|---|---|
| `settings.py` | `CrowdSettings` (`eps_px`, `min_samples`) — a new `crowd:` app.yaml block |
| `config/app.yaml` | New `crowd:` block with defaults and rationale for `eps_px` |
| `boundary/engine.py` | `crowd_membership()` — same `_compile()`-iteration shape as `suppressed_mask`/`confidence_delta`; zone id → boolean mask of which foot points fall inside it, scoped to `POLYGON` zones only (matching `zone_occupancy()`'s existing scoping) that declare `crowd_threshold` |
| `alerts/messages.py` | `crowd_message()` |
| `pipeline.py` | `Pipeline` builds an optional `CrowdMonitor` when `"crowd"` is in `camera.enabled_modules` — unconditional (no file to load, no soft-fail needed, unlike fire/smoke/identity); `_process_crowd()` runs every detection frame, filters out exclusion-masked foot points (an exclusion zone suppresses detections for every module, not just boundary), and publishes through the existing `AlertBus` for every zone `crowd_membership()` and `CrowdMonitor.update()` confirm |
| `cameras/manager.py` | No change needed — `enabled_modules` was already in `_RECONNECT_FIELDS` since Phase E, so toggling `crowd` on a live camera already triggers `Pipeline.reconfigure()`, which now also rebuilds `crowd_monitor` |

### Frontend

| File | What changed |
|---|---|
| `components/ZoneProperties.jsx` | New "Crowd formation" fieldset, same collapsible-toggle shape as the existing "Active hours" schedule block, shown only for `polygon` zones: a threshold input and an optional hysteresis input (same "detection frames, not camera frames" hint the boundary hysteresis field already carries) |

`Cameras.jsx`'s `crowd` module toggle already existed before this phase — nothing
needed there.

## 3. New dependencies

None — `scipy` is already a dependency (`boundary/geometry.py`/`shapely`-adjacent code).

## 4. Design notes

- **Why exclusion-masked points are stripped before clustering:** `zones.py`'s own
  docstring describes exclusion zones as suppressing "detections inside this region" in
  general terms, not scoped to boundary alone — a queue line or waiting area marked as
  an exclusion zone should not itself be read as a crowd.
- **Why `CrowdMonitor` needs no soft-fail construction:** unlike `FireSmokeDetector`
  (an ONNX model file that may not exist yet) or `IdentityResolver` (two model files
  plus a MongoDB-backed gallery), `CrowdMonitor` has no external dependency at all — it
  activates the instant the module flag is set, same posture `BoundaryEngine` itself
  has always had.
- **Why crowd zones are scoped to `POLYGON` type only:** matches `zone_occupancy()`'s
  existing scoping comment — crowd formation is an "area people occupy" concept, which
  `exclusion`/`fire_roi`/`tripwire` zones are not.

## 5. Verification

**Automated:**

```
483 passed, 4 skipped in ~37s   (backend/tests/, pytest)
ruff check src/ tools/ tests/ → All checks passed!
npm run build (frontend/) → clean
```

**Manual:** exercised via the pipeline-level tests directly (`_process_crowd` called
against a hand-built `TrackedDetections`), the same approach Phase F's access-control
tests used — no live camera footage of an actual crowd was available in this
environment.

## 6. Known limitations / carried forward

- `eps_px`/`min_samples` are camera-resolution-relative pixel constants, not
  perspective-corrected — a very wide-angle or heavily tilted camera may need a
  site-specific `eps_px` tuned higher or lower than the 80px default; this is
  documented in `config/app.yaml`'s comment, same as `FireSmokeSettings`'s thresholds.
- No live-footage verification of clustering accuracy was possible in this
  environment — the DBSCAN primitive and hysteresis state machine are fully covered by
  unit tests with synthetic point clouds, but a real multi-person clustering scenario
  is a physical, on-site verification step, the same caveat Phase E/F's accuracy
  claims carry.
