# Model Training Tracker

Living document. Every phase that adds a detection capability gets a row here the moment
it's planned, noting whether it needs an actual GPU training run (dataset curation +
training infra, a separate workstream from the software integration) or ships against
pretrained weights. Updated as each phase is built — check here before starting any new
phase to know whether "build the software" is the whole job or only half of it.

## Summary

| Phase | Capability | GPU training required? | Status |
|---|---|---|---|
| E | Fire & smoke detection | **Yes** | ✅ **Trained 2026-09-16** — `backend/models/firesmoke/model.onnx` ships real weights. **Not yet validated on real site footage** (see below) — training-loss convergence only. |
| F / F.1 | Identity (face) + restricted-zone allow-list | No — pretrained (YuNet + SFace) | ✅ Complete ([PHASE_F.md](phases/PHASE_F.md)) |
| G | Crowd formation | No — rules only (DBSCAN), no model at all | ✅ Complete ([PHASE_G.md](phases/PHASE_G.md)) |
| H | PPE detection | **Yes** | Software integration done ([PHASE_H.md](phases/PHASE_H.md)); **no model trained yet** |
| I | Phone-near-ear | No — pretrained (RTMPose + existing phone class) | Not started |

One phase still needs a real training run: **H (PPE)**. Phase E now ships trained weights
(below). Everything else either needs no model or uses an already-pretrained,
commercially-licensed one.

## Phase E — Fire & smoke detection

**✅ Trained 2026-09-16.** `backend/models/firesmoke/model.onnx` (YOLOX-Nano, 2 classes,
416×416, 3.5MB) ships real weights fine-tuned from the COCO-pretrained `yolox_nano.pth`
checkpoint on D-Fire. Full provenance, training config and known limitations are in
`backend/models/firesmoke/manifest.json` and `NOTICE.md` §1/§2.

- **Deviated from the original recipe in two ways, both deliberate and recorded:**
  1. **Hardware.** `PLAN.md` §5.3 assumed a rented GPU (RunPod/Vast.ai/Lambda/Colab Pro).
     This run instead used a local **NVIDIA GTX 1650 (4GB VRAM)**, which was available on
     the development machine. Batch size (8) and epoch count (40 instead of 100–150) were
     reduced accordingly — `training/YOLOX/exps/custom/fire_nano.py` has the full exp
     config and the reasoning in its comments.
  2. **Dataset.** Used **D-Fire only**, obtained via a Kaggle mirror
     (`sayedgamal99/smoke-fire-detection-yolo`, CC0-1.0) rather than FASDD_CV — FASDD's
     licence was never confirmed (still unresolved, see `NOTICE.md` §2), and D-Fire alone
     was sufficient to get a working fine-tune. **No site negatives were collected or
     used** (`PLAN.md` §5.2) — this model has only ever seen D-Fire's own images.
- **Two real bugs hit and fixed during this run** (both documented in
  `training/exps/custom/fire_nano.py` and `training/watchdog_train.sh` comments, kept
  there since the fix is exactly what a future rerun needs to avoid repeating them):
  1. YOLOX's in-training COCO-mAP evaluation JIT-compiles a C++ extension
     (`yolox.layers.COCOeval_opt`) that needs an MSVC toolchain — absent on the training
     machine, it crashed training outright after only 5 of 40 epochs, and the crash was
     **silent**: `trainer.py`'s `try/except/finally` still logged "Training of experiment
     is done" despite the early exit. Fixed by disabling in-training eval entirely
     (acceptable — `PLAN.md` §5.4 already says not to tune this model on COCO mAP).
  2. `tools/train.py --resume` combined with `-c <checkpoint>` loads *that* checkpoint via
     a strict `load_state_dict` instead of `latest_ckpt.pth`, crashing on the 80→2 class
     head-shape mismatch. Fixed by omitting `-c` on resume.
  3. `tools/export_onnx.py` called the removed internal API `torch.onnx._export`
     (removed in torch 2.6) — patched to the public `torch.onnx.export`.
- **Sanity-checked, not yet validated.** Loaded through `backend/src/perimeter/detect/
  yolox_onnx.py`'s real `YoloxOnnx` wrapper and run against actual D-Fire train images:
  correctly detected fire (0.5–0.8 confidence) on fire-labelled images and smoke
  (0.57–0.61) on smoke-labelled images, confirming the class-order remap
  (`training/scripts/dfire_to_coco.py`, see its own comments — the Kaggle mirror's
  `data.yaml` uses the *opposite* class order from this project) is correct end-to-end.
  **This is not the same as the validation `PLAN.md` §5.4/§9.3 requires** — false-alarm
  rate on hours of real site video through the full pipeline (K-of-N gate included), and
  recall on the D-Fire test split (4,306 images, never scored — in-training eval was
  disabled) are both still outstanding before this model should be treated as
  deployment-ready. Do that before shipping to a customer.
- **Drop-in path (already done):** `backend/models/firesmoke/model.onnx` +
  `manifest.json` are in place. No code change needed — `main.py` checks for the file at
  every startup and activates the module automatically for any camera with `fire_smoke`
  enabled.
- **Software status:** ✅ complete (`docs/phases/PHASE_E.md`) — detector wrapper, K-of-N
  temporal gate, zone-aware policy (`fire_roi`/`exclusion`), alert publishing, all built
  and tested; now backed by real weights instead of a placeholder model path.

## Phase H — PPE detection

- **Datasets:** SH17 + Construction-PPE (per `PLATFORM_EXPANSION_PLAN.md` §5) — licences
  not yet verified against this repo's permissive-only policy (`NOTICE.md`'s CC0/MIT/
  Apache-2.0-only rule); confirm and record before training starts.
- **Recipe:** a new lightweight, person-crop, region-constrained multi-label classifier
  (helmet/vest/gloves/shoes/glasses per zone requirement), not the same architecture as
  the person/fire-smoke detector — this is a genuinely new model shape, the largest
  net-new data/architecture workstream of the six-module expansion per
  `PLATFORM_EXPANSION_PLAN.md` §5's own note.
- **Software status:** ✅ complete (`docs/phases/PHASE_H.md`) — `PPEClassifier` wrapper
  (a plain `onnxruntime` multi-label classifier, not built on the YOLOX wrapper),
  `classify_state()`'s present/absent/indeterminate thresholds, `PPEMonitor`'s per-item
  hysteresis, zone-aware `ppe_required` policy, alert publishing, all built and tested
  against fake classifier doubles (no real weights exist to test against).

## Phases that do NOT need training

- **F/F.1 — Identity:** ✅ complete. YuNet (face detection, MIT) + SFace (face
  recognition, Apache-2.0) — both pretrained, from the OpenCV Zoo, no training run.
  Weights downloaded and verified 2026-09-15 (`backend/models/yunet/`,
  `backend/models/sface/`, both loadable via OpenCV's own `cv2.FaceDetectorYN`/
  `cv2.FaceRecognizerSF` wrappers). Software integration:
  [docs/phases/PHASE_F.md](phases/PHASE_F.md).
- **G — Crowd formation:** ✅ complete. No model at all — hand-rolled DBSCAN clustering
  over already-tracked person foot-points (`scipy.spatial.cKDTree`), pure rules.
  Software integration: [docs/phases/PHASE_G.md](phases/PHASE_G.md).
- **I — Phone-near-ear:** RTMPose (pretrained, Apache-2.0) for wrist/elbow/ear
  keypoints + the *existing* COCO-pretrained person-detection weights filtered to class
  67 (`cell phone`) instead of class 0 — zero new training, per
  `PLATFORM_EXPANSION_PLAN.md` §5's own note ("software only, zero new training").
