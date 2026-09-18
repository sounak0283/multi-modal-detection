# Phase H — PPE detection (software integration only)

- **Status:** ✅ Software integration complete; model training not started
- **Date:** 15 September 2026
- **Plan reference:** [`PLATFORM_EXPANSION_PLAN.md`](../../PLATFORM_EXPANSION_PLAN.md) §5/§10/§11
- **Depends on:** Phase A (MongoDB), Phase G (`BoundaryEngine.crowd_membership`'s sibling shape, `zone_by_id`)
- **Blocks:** nothing — I/J are independent of this phase

## 1. Summary

PPE is the one remaining module in the expansion that needs a genuinely new model
architecture — not a YOLOX fine-tune like fire/smoke, but "a new lightweight,
person-crop, region-constrained multi-label model... helmet/vest/gloves/shoes/glasses
per zone requirement, with an explicit 'indeterminate' state" trained on SH17 +
Construction-PPE (`PLATFORM_EXPANSION_PLAN.md` §5), described there as "the largest
net-new data/architecture workstream" of the whole expansion. No GPU/training infra is
attached to this environment, so — exactly like Phase E's fire/smoke — this phase ships
the **complete software integration against a placeholder model path**, per the risk
register's own mitigation: *"software integration ships independent of model readiness
(testable with stub weights); per-module go-live gated on measured recall/precision, not
on the phase's code being merged."* When real trained weights are placed at
`backend/models/ppe/model.onnx`, PPE compliance checking activates on every restart with
no further code change.

`Zone.ppe_required` has existed on the dataclass since Phase A — persisted and
round-tripped by tests, but never validated against a known item set and never
consumed. `MODULE_PPE` was already a registered, selectable camera module and the
Cameras page already offers "PPE compliance" as a toggle.

All 507 backend tests pass, `ruff` is clean, and the frontend was rebuilt with a new
"PPE required" fieldset in the zone editor.

## 2. What was implemented

### Backend — new files

| File | Purpose |
|---|---|
| `detect/ppe.py` | `PPE_ITEMS` (the fixed, ordered item set); `PPEClassifier` — a thin `onnxruntime` wrapper, deliberately **not** built on `YoloxOnnx` (that wrapper decodes YOLOX's anchor-grid output; a whole-person-crop multi-label classifier is an unrelated I/O shape); `classify_state()` — the pure threshold function producing the explicit `present`/`absent`/`indeterminate` three-way state; `PPEMonitor` — per-(zone, track, item) hysteresis, same confirmed/age shape `CrowdMonitor` (Phase G) established, plus stale-entry eviction mirroring `BoundaryEngine._evict_stale` |
| `models/ppe/README.md` | Documents the expected model shape and how to drop a real one in — no weights ship |
| `test_ppe_classifier.py`, `test_ppe_settings.py`, `test_ppe_pipeline.py` | Threshold logic + hysteresis (pure, no model), `app.yaml` parsing, and zone-membership/publish wiring against a real `Pipeline` with a fake classifier injected |

### Backend — changed files

| File | What changed |
|---|---|
| `boundary/zones.py` | `PPE_ITEMS` — closes a real gap: `ppe_required` previously accepted any free-form lowercased string with no validation at all; now validated the same way `applies_to` already is |
| `boundary/engine.py` | `ppe_membership()` — identical `_compile()`-iteration shape to `crowd_membership()` (Phase G), POLYGON-only, scoped to zones with a non-empty `ppe_required` |
| `settings.py` | `InferenceSettings.ppe_every_n` — PPE checking runs once every N **detection** frames (gated on `Pipeline.stats.detections_run`, not the raw camera-frame counter fire/smoke's own offset cadence uses, since PPE inherently needs tracked person boxes which only exist on detection frames) |
| `alerts/messages.py` | `ppe_message()` |
| `pipeline.py` | `Pipeline` gains an optional `ppe_model_path`; `_build_ppe()` mirrors `_build_firesmoke()`'s soft-fail construction exactly; `_process_ppe()` crops the *whole* tracked person box (not the top-35% face crop identity uses), classifies once per person per check, and calls `PPEMonitor.update()` per required item; a confirmed violation publishes through the existing `AlertBus` |
| `cameras/manager.py` | Threads `ppe_model_path` through to each `Pipeline`, same shape as `firesmoke_model_path`/`identity_resolver` |
| `main.py` | New `--ppe-model` flag; not fatal if the file is missing, same posture as `--firesmoke-model` |

### Frontend

| File | What changed |
|---|---|
| `components/ZoneProperties.jsx` | New "PPE required" fieldset for `polygon` zones — a `ChipGroup` over the five items, writing `ppe_required`, with an inline note that checking stays inactive until real weights are placed |

`Cameras.jsx`'s `ppe` module toggle already existed before this phase — nothing needed
there.

## 3. New dependencies

None — `onnxruntime` is already a dependency (`detect/yolox_onnx.py`); `PPEClassifier`
uses it directly rather than through the YOLOX-specific wrapper.

## 4. Design notes

- **Why `PPEClassifier` isn't built on `YoloxOnnx`:** that wrapper's own docstring is
  explicit that it decodes YOLOX's anchor-grid detector output. A whole-person-crop
  multi-label classifier (no boxes, no NMS, a fixed-size input, a fixed-length sigmoid
  output vector) has nothing in common with that shape - reusing it would have meant
  bending an unrelated abstraction rather than writing the small, honest wrapper this
  phase ships instead.
- **Why the "indeterminate" state is a runtime threshold, not a model output:**
  documented explicitly in `models/ppe/README.md` so whoever trains the real model
  doesn't over-engineer a three-way classification head — `classify_state()`'s two
  thresholds (0.35/0.65 defaults) are the tunable surface, adjustable against measured
  validation performance without retraining.
- **Why PPE's cadence is keyed off the detection-frame counter, not `frame.seq`:**
  unlike fire/smoke (which runs independent of person tracking), PPE inherently needs
  tracked person boxes, which only exist on detection frames — there's no risk of PPE's
  cadence colliding with fire/smoke's raw-frame offset the way two raw-frame cadences
  could, so no extra offset arithmetic was needed.
- **Why exclusion-masked people are never checked:** same reasoning `_process_crowd`
  already established (Phase G) - an exclusion zone suppresses detections for every
  module, not just boundary.

## 5. Verification

**Automated:**

```
507 passed, 4 skipped in ~48s   (backend/tests/, pytest)
ruff check src/ tools/ tests/ → All checks passed!
npm run build (frontend/) → clean
```

**Manual:** Pipeline-level tests inject a fake `PPEClassifier` directly onto a real
`Pipeline` (constructed with real person-detection weights) to exercise
`_process_ppe`/`_publish_ppe` end to end — zone membership, hysteresis confirm/reset,
exclusion masking, and alert publishing all verified without needing real PPE weights.

## 6. Known limitations / carried forward

- No model exists. Training is a separate, GPU-and-dataset workstream — see
  `models/ppe/README.md` and `PLATFORM_EXPANSION_PLAN.md` §5 for the recipe, and
  `NOTICE.md`'s process for confirming SH17/Construction-PPE's licences before training
  starts.
- Unlike Phase E's fire/smoke construction test (which stands in the real person
  weights, since both share the `YoloxOnnx` architecture), `PPEClassifier`'s
  construction against a real, loadable ONNX file could not be end-to-end tested in
  this environment — no `onnx` model-authoring package is available to synthesise a
  minimal valid classifier file, and no PPE weights exist to test against directly. The
  missing-file soft-fail path and the classify/hysteresis/publish logic (via a fake
  classifier) are both fully covered; only the literal `onnxruntime.InferenceSession(...)`
  construction call against a real file is unverified.
- No PPE-required frontend UI has been exercised against a real classification in a
  live browser (no model to trigger one) — only its config storage/rendering was
  verified, same caveat Phase E's `fire_roi`/`exclusion` zone UI carried.
