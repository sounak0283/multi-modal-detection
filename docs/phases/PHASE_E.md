# Phase E — Fire & Smoke Detection (software integration)

- **Status:** ✅ Software integration complete; model training not started
- **Date:** 15 September 2026
- **Plan reference:** `PLAN.md` §4/§5, [`PLATFORM_EXPANSION_PLAN.md`](../../PLATFORM_EXPANSION_PLAN.md) Phase E
- **Depends on:** Phase A (MongoDB), boundary engine (`fire_roi`/`exclusion` zones, present since PLAN.md's original design)
- **Blocks:** nothing — every other remaining phase (F–J) is independent of this one

## 1. Summary

Fire/smoke is the product's headline feature, but the actual detector needs a GPU,
curated datasets (D-Fire/FASDD), and site-negative footage collected on a real
site — none of which a coding session can produce. What this phase delivers is the
**complete software integration**, running against a placeholder model path, exactly as
`PLATFORM_EXPANSION_PLAN.md`'s own risk register describes: *"software integration ships
independent of model readiness (testable with stub weights); per-module go-live gated on
measured recall/precision, not on the phase's code being merged."* When real trained
weights are placed at `backend/models/firesmoke/model.onnx`, fire/smoke detection
activates on every restart with **no further code changes**.

Most of the ground was already prepared and simply unconsumed: `config/app.yaml`'s
`firesmoke:` block (`conf_fire`, `conf_smoke`, `gate_k`, `gate_n`, `gate_iou`) had never
been read by `settings.py`; `BoundaryEngine.suppressed_mask()`/`confidence_delta()` were
already class-agnostic and fully able to serve fire/smoke, just never called for
anything but `"person"`; `alerts/messages.py:firesmoke_message()` existed and was tested
for wording but never called from a publish path; the frontend zone editor already fully
supported `fire_roi`/`exclusion` zone types. This phase is what wires all of it together
for the first time — **no frontend changes were needed**.

All 427 backend tests pass (403 from before this phase + 27 new, 3 of which always skip
in this environment pending real weights), `ruff` is clean, and the graceful-degradation
behaviour (camera keeps running normally when fire/smoke is enabled but no model
exists) was verified against a real running pipeline.

## 2. What was implemented

### Backend — new files

| File | Purpose |
|---|---|
| `detect/firesmoke.py` | `FireSmokeDetector` — mirrors `detect/person.py:PersonDetector`'s shape over the shared `YoloxOnnx` wrapper; class ids `{0: fire, 1: smoke}` per `PLAN.md` §5.3's training recipe. Returns raw, class-filtered detections only — no per-class/zone policy, same separation `PersonDetector` already has from `BoundaryEngine` |
| `detect/temporal_gate.py` | `TemporalGate` — the K-of-N candidate-confirmation algorithm from `PLAN.md` §5.5, implemented exactly as documented (IoU ≥ `gate_iou` joins a candidate, confirms once hit count reaches `gate_k` within the last `gate_n` processed frames), plus the box-area-growth severity escalation rule ("fire grows, brake lights and sunsets do not") |
| `models/firesmoke/README.md` | Documents the expected model shape and how to drop a real one in — no weights ship |
| Five new test files | `test_temporal_gate.py`, `test_firesmoke_settings.py`, `test_firesmoke_pipeline.py`, `test_firesmoke_integration.py` (always-skip until real weights exist, mirrors `test_detect_integration.py`) |

### Backend — changed files

| File | What changed |
|---|---|
| `settings.py` | `FireSmokeSettings` — the `firesmoke:` app.yaml block, parsed for the first time |
| `boundary/engine.py` | `fire_roi_zone_at()` — a small new helper (same `_compile()`-iteration shape as `confidence_delta`) so an alert can name a real zone ("Fire detected in Server Room") |
| `pipeline.py` | `Pipeline` gains an optional `firesmoke_model_path`; conditionally builds `FireSmokeDetector`+`TemporalGate` when `"fire_smoke"` is in `camera.enabled_modules` **and** the model actually loads — any failure degrades to `None`, logged, never fatal; a second, independently-offset cadence check in `_run()`; `_process_firesmoke()`/`_publish_firesmoke()` apply the zone-aware policy (exclusion suppression, fire_roi sensitivity delta) and publish through the existing `AlertBus` |
| `cameras/manager.py` | `PipelineManager` threads `firesmoke_model_path` through; `enabled_modules` added to `_RECONNECT_FIELDS` (see §4) |
| `main.py` | New `--firesmoke-model` flag; unlike the person model, a missing file is **not fatal** — logged once, `None` passed down |
| `test_boundary_engine.py`, `test_cameras_registry.py` | Extended coverage for `suppressed_mask`/`confidence_delta`/`fire_roi_zone_at` with `"fire"`/`"smoke"` classes (previously person-only); `FakePipeline` test double updated for the new constructor parameter |

### No frontend changes

Confirmed via research before starting: `frontend/src/lib/zones.js` and
`ZoneProperties.jsx` already fully support `fire_roi`/`exclusion` zone types,
`applies_to`, and `conf_delta`; `Cameras.jsx`'s `fire_smoke` module toggle already
existed. Nothing new to build or wire up.

## 3. New dependencies

None.

## 4. A real gap found and fixed along the way

While wiring `enabled_modules` for the first time (`Pipeline` never read it before this
phase), found that `PipelineManager._reconcile()` only reconfigures a running `Pipeline`
when a **connection-related** field changes (`_RECONNECT_FIELDS`: source/resolution/fps)
— any other field change, including `enabled_modules`, was silently invisible to an
already-running camera until the whole process restarted. This would have made the
Cameras page's `fire_smoke` toggle a silent no-op on a live camera. Fixed by adding
`enabled_modules` to `_RECONNECT_FIELDS`, with a test
(`test_changing_enabled_modules_reconfigures_in_place`) proving it now takes effect.

## 5. Verification

**Automated:**

```
427 passed, 3 skipped in ~35s   (backend/tests/, pytest)
ruff check src/ tools/ tests/ → All checks passed!
```

**Manual:** `tools/_phase_e_smoke.py` (deleted after use) started a real
`PipelineManager` with a camera that has `fire_smoke` enabled and
`--firesmoke-model models/firesmoke/model.onnx` (which does not exist). Confirmed the
exact log line: `WARNING perimeter.pipeline: fire/smoke enabled for cam_01 but no usable
model at models/firesmoke/model.onnx (...) - module inactive`, that
`pipeline.firesmoke_detector is None`, that the boundary engine and person detector
loaded and ran normally regardless, and that `manager.stop()` shut down cleanly.

**Not verified (cannot be, without real weights):** actual fire/smoke detection
accuracy, the K-of-N gate's real-world false-alarm rate, the area-growth escalation
against genuine footage. `test_firesmoke_pipeline.py` stands in with the real
person-detection weights (same architecture/wrapper) to prove the *construction and
wiring* path works end to end; `test_firesmoke_integration.py` activates automatically
the moment real weights land, per `models/firesmoke/README.md`.

## 6. Known limitations / carried forward

- No model exists. Training is a separate, GPU-and-dataset workstream — see
  `models/firesmoke/README.md` and `PLAN.md` §5 for the recipe.
- The `fire_roi`/`exclusion` frontend UI has never been exercised against a real
  fire/smoke detection in a live browser (no model to trigger one) — only its config
  storage/rendering was verified in earlier phases.
- No commissioning/positive-test procedure (`PLAN.md` §9.3 — smoke machine, controlled
  tray burn) has been run; that's a physical, on-site step for whenever real weights and
  a deployment exist.
