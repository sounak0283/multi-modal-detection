# PPE model integration log — `demo_package.zip` → Phase H PPE module

Branch `feature/ppe-model-integration`, 2026-09-28. Every result below is from a command
actually run; key output lines are quoted.

## 1. Investigation

### 1A. The package

Extracted to `third_party/demo_package/` (zip left untouched; binaries git-ignored).

| File | Size | What it is |
|---|---|---|
| `models/ppe_final_C_v2_320.onnx` | 3,653,119 | PPE detector, YOLOX-Nano, input `[1,3,320,320]` float, output `[1,2100,7]` |
| `models/person_yolox_nano_416.onnx` | 3,659,407 | COCO person detector |
| `samples/test_input.mp4` / `annotated_output.mp4` | 6.5 / 8.3 MB | 600-frame 960×720 10 fps **slideshow** of held-out test images |
| `config.json`, `demo.py`, `model_card.md`, `README.md`, `requirements.txt` | — | thresholds, reference runner, metrics |

- **Format:** ONNX (PyTorch-exported). Runtime: onnxruntime, CPU. No torch/ultralytics needed.
- **Classes:** `0 head` (bare head = NO HELMET), `1 helmet`. No vest/gloves/shoes/glasses.
- **Pre-processing:** letterbox to 320, pad 114 top-left, raw 0-255 float, BGR, NCHW; output
  is the undecoded YOLOX grid (1600+400+100 anchors × 4 box + objectness + 2 classes).
- **Crop:** person box padded x 0.10 / top 0.12 / bottom 0.04; people < 96 px tall skipped.
- **Thresholds:** head 0.45, helmet 0.55, NMS IoU 0.45, person 0.30.
- **The package's person model is byte-identical to the project's `yolox_nano.onnx`**
  (both SHA-256 `c789161e…0b7d`) — the live pipeline's person boxes are exactly what this
  PPE model was trained on, so only stage 2 needed integrating.
- Standalone run in the project venv: `frames 600, final FPS 24.1, NO HELMET tracks 102`.
- **Parity:** the project's existing `YoloxOnnx` wrapper vs the package's own runner on 87
  real person crops: 0 count mismatches, identical classes, score diff 0.000000, boxes within
  0.01 px. (An initial 9.4 px figure was my own comparison sorting tied scores differently.)

### 1B. The project

- Entry points: `perimeter-server` (FastAPI) → `PipelineManager` → one `Pipeline` per camera
  (live, `capture_realtime=True`, drop-to-latest); Video test → the same `Pipeline` class
  with `capture_realtime=False` (every frame, deterministic).
- Modules: boundary (person + ByteTrack), crowd, identity, fire_smoke, welding, ppe;
  phone is not implemented. All run on the same frame in one inference thread.
- Existing PPE module (`detect/ppe.py`): `PPEClassifier.classify(crop) -> {item: prob}` for
  5 items, bucketed by `classify_state` (≥0.65 present / ≤0.35 absent / else indeterminate),
  `PPEMonitor` hysteresis (3 consecutive "absent"), zones' `ppe_required`, `_publish_ppe`
  alert. **No weights existed** — PPE was inactive.

**Baseline (commit `ddf8d82`):**

| Check | Result |
|---|---|
| `pytest tests` | 730 passed, 2 skipped |
| `ruff check` | clean |
| Video: fire+welding clip (fire_smoke, welding) | `welding_spark: 1` |
| Video: fire+smoke clip (fire_smoke) | `smoke: 2, fire: 2` |
| Video: crowd clip (boundary, crowd, identity) | `crowd formed 36 / dispersed 36, entry 14, exit 25` |
| Live, simulated (looping sample, 5 modules), baseline worktree | 9.88 fps, crowd 13/13, entry 41, exit 43, memory flat |
| Live, real webcam (5 modules), baseline worktree | 12.01 fps, 6.00 det/s, memory flat, no alerts |
| Pre-existing issues | `cam_02` RTSP camera unreachable (retry loop); `cam_01` disabled in the saved camera config (left as is); webcam needs ~40 s to deliver its first frame (OpenCV native backend) |

### 1C. Gap analysis

| New model | Existing PPE module expects | Bridge |
|---|---|---|
| Detector: head/helmet **boxes** | Classifier: one **probability per item** | Map the chosen head's class+score into the existing bands: helmet → [0.65,1], head → [0,0.35], none → 0.5 |
| Helmet only | 5 items | Other items return 0.5 (indeterminate) and are skipped with a one-time warning. Previously a missing item defaulted to **0.0 = "absent" = a false violation** |
| Trained on **padded** crops, ≥96 px people | Pipeline cropped the bare box | `classify_person(frame, box)` lets each backend crop its own way |
| Several heads can fall in one padded crop | one person per crop | Heads must be in the person's x-range and upper half; nearest the centre wins; head under a stronger helmet box = helmet (package rule) |
| 3-of-5-frames smoothing in the demo | `PPEMonitor` 3 consecutive checks | Kept `PPEMonitor` (indeterminate neither advances nor resets) |
| Custom YOLOX decoder | `YoloxOnnx` already shared | Reused it — proven identical above |

## 2. Plan

1. Branch + baseline commit.
2. `PPESettings` + `ppe:` block in `app.yaml` with `backend: helmet_detector | classifier`.
3. `detect/ppe_helmet.py:PPEHelmetDetector` adapter on `YoloxOnnx`; CUDA if available else CPU.
4. Additive `classify_person` / `supported_items` / `PPEResult` in `detect/ppe.py`; keep
   `classify()` and every signature unchanged; pipeline falls back to `classify(crop)` for any
   object that only has that (existing tests' fakes).
5. Pipeline: backend selection in `_build_ppe`, unsupported-item handling, head overlay.
6. `main.py`: default model path follows the configured backend; `--ppe-model` still wins.
7. Manifest, NOTICE row, SHA-verified install tool, README.
8. Tests, then video/live/regression runs.

## 3. Test cycles

**Cycle 1 — unit/integration:** 17 adapter tests + pipeline/settings tests → 756 passed,
2 skipped; ruff flagged 2 long lines (fixed). Licence gate passed.

**Cycle 2 — video mode (real server, `/test-video` API, PPE sample, zone requires helmet):**
`finished stats={'frames': 600, 'detections': 300} alerts={'boundary:exit': 26, 'boundary:entry': 22, 'ppe:helmet': 10}`.
**Failure found by inspecting the recorded annotated stream:** stale HELMET/NO HELMET boxes
drawn on walls and on a waist, with only one person tracked. Root cause: overlay marks
expired on a 1.5 s wall-clock TTL, but Video test runs at ~45 fps, so 1.5 s spanned several
scenes; marks also stayed at fixed coordinates and outlived their track. Fix: marks are
stored relative to the person box, drawn only while that track is in the current frame, and
aged by detection count. Regression tests added. Remaining cosmetic lag: a mark is redrawn
from the last PPE check (≤1 s), so it can shift if the person box changes shape in between;
drawn on the same frame, the adapter's head boxes sit exactly on the helmets (checked
visually on 4 sample frames). Alert decisions are always made on the measured frame.

**Cycle 3 — after the fix:**

| Test | Result |
|---|---|
| Video, PPE only (boundary+ppe) | 600/600 frames, `ppe:helmet 10`, annotated MP4 + stills saved, overlay correct |
| Video, all 6 modules on the PPE sample | `ppe:helmet 10`, entry 22 / exit 26 (identical to PPE-only — no interference), crowd 8/8 |
| Video regression matrix vs baseline | **identical**: welding 1; fire 2 / smoke 2; crowd 36/36, entry 14, exit 25 |
| Live simulated (looping sample, 6 modules, 100 s) | 9.88 fps steady (native 10), 4.88 det/s, private memory end−start +3.7 MB (flat), `stop()` 0.02 s, threads 1→1, 0 warnings, `ppe:helmet 14` |
| Live simulated, same 5 modules as baseline | **identical to baseline**: crowd 13/13, entry 41, exit 43, 9.88 fps |
| Live **real webcam** (6 modules, 150 s) | first frame at 45 s (device open — same as baseline), then 95 s at 12.00 fps / 6.00 det/s, memory +1.1 MB (flat), `stop()` 0.32 s, 0 warnings; bare-headed person → confirmed NO HELMET overlay + `ppe:helmet` alert |
| Shared frame never modified by PPE | new test, passes |
| Server log over all runs | 0 warnings/errors (apart from the pre-existing RTSP retry) |
| Fresh checkout (`git worktree`) | install tool verifies SHA-256 and installs; licence gate passes; `691 passed, 70 skipped` (skips = person model not downloaded there, as in any fresh checkout) |
| Final full suite | `759 passed, 2 skipped` (baseline 730 + 29 new; same 2 skips); ruff clean; licence gate passed |

Note on the real-webcam run: it also raised one `welding_spark` soft alert in an office. The
spark heuristic reads the raw frame and never the overlay (`pipeline.py` `_process_sparks`
uses `frame.image`; drawing is on a copy), and PPE provably does not modify that frame — so
this is the pre-existing welding heuristic reacting to a bright scene, not an integration
effect.

## 4. Definition of Done

- [x] Model extracted, analysed, specs documented (§1A, `models/ppe/manifest.json`)
- [x] Integrated via the existing interface (`classify` contract; pipeline, zones, alerts unchanged)
- [x] PPE works in video mode — real server, alerts + saved annotated output
- [x] PPE works in live mode — **real webcam** and a simulated looping stream
- [x] All other modules unchanged in both modes — identical counts vs baseline
- [x] No new errors/warnings; clean startup; pipeline `stop()` clean, no leaked threads.
      Not tested: graceful shutdown of the whole `perimeter-server` process (I stop it with
      `taskkill /F`); that code path was not touched.
- [x] Dependencies: none added — onnxruntime/opencv/numpy already pinned in compatible ranges
- [x] Revert switch: `ppe.backend: classifier`
- [x] All existing tests pass; 29 new tests
- [x] Committed on `feature/ppe-model-integration`
