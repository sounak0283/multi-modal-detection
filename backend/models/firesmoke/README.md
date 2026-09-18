# Fire/smoke model (Expansion Plan Phase E)

**Trained 2026-09-16.** `model.onnx` ships real weights: a 40-epoch fine-tune of the
Apache-2.0 COCO-pretrained `yolox_nano.pth` checkpoint on D-Fire. Software integration
(`src/perimeter/detect/firesmoke.py`, `src/perimeter/detect/temporal_gate.py`,
`Pipeline._process_firesmoke`) and the model itself are both in place — see
`manifest.json` for full provenance, training config, and `docs/MODEL_TRAINING.md`'s
Phase E section for the training-run writeup (including two real bugs hit and fixed
along the way).

`main.py` checks for `model.onnx` at startup; since it now exists, fire/smoke detection
activates automatically for any camera with the `fire_smoke` module enabled — no code
change or config flag needed.

## Before shipping to a customer: validate, don't just trust training loss

Training loss converged normally and a quick sanity check (running the exported ONNX
through the real `YoloxOnnx` runtime wrapper against actual D-Fire images) confirmed
correct fire/smoke class detection with sensible confidence (0.5–0.8 for fire, 0.57–0.61
for smoke). **That is not the validation `PLAN.md` §5.4 and §9.3 require:**

- **False-alarm rate**, measured by running hours of real, continuous site footage through
  the *full* pipeline including the K-of-N temporal gate (`PLAN.md` §5.4 — a held-out set
  of still images cannot produce an hourly rate, and pre-gate detections overstate it).
- **Recall on the D-Fire test split** (4,306 images, held out but never scored — the
  in-training COCO-mAP evaluator was disabled during this run, see below).
- **Site negatives** (`PLAN.md` §5.2 — welding sparks, forklift beacons, steam, glare,
  IR footage if applicable): this model has only ever seen D-Fire's own images. None of
  the site-specific false-alarm sources PLAN.md calls out have been tested against it.

Do all three before treating this checkpoint as deployment-ready.

## What's different from the original PLAN.md §5.3 recipe

- **Hardware:** trained locally on a single **GTX 1650 (4GB VRAM)** rather than rented GPU
  time. Batch size 8, 40 epochs (not 100–150) — see
  `training/YOLOX/exps/custom/fire_nano.py` for the exact config and reasoning.
- **Dataset:** D-Fire only, via a Kaggle mirror (CC0-1.0) rather than D-Fire + FASDD_CV —
  FASDD's licence was never confirmed. No site negatives collected or used yet.
- **In-training evaluation disabled.** YOLOX's periodic COCO-mAP eval needs a C++
  extension that JIT-compiles against an MSVC toolchain not present on the training
  machine; left enabled, it silently truncated training to 5 of 40 epochs (see
  `docs/MODEL_TRAINING.md` for the full incident). Not a real loss of signal per PLAN.md
  §5.4's own guidance to tune on false-alarm rate, not mAP — but it does mean recall was
  never automatically scored during or after training.

## Model shape

- **Two classes:** `{0: fire, 1: smoke}`, matching `PLAN.md` §5.3's convention and
  `src/perimeter/detect/firesmoke.py`'s `FIRE_CLASS_ID`/`SMOKE_CLASS_ID`. Note the
  training dataset's own YOLO labels used the *opposite* order (`data.yaml` declared
  `0: smoke, 1: fire`) — `training/scripts/dfire_to_coco.py` remaps this during the
  YOLO→COCO conversion, verified against the label files' own class-frequency
  distribution before training and against real detections on labelled images afterward.
- **416×416 input**, same YOLOX architecture/wrapper (`detect/yolox_onnx.py`) the person
  detector uses — same ONNX Runtime session shape, BGR colour order (`to_rgb: false`),
  same as the Megvii-style export convention.
- **`manifest.json`** records colour order, byte size, SHA-256, and the full training
  config/provenance chain.

## Retraining or improving this model

See `training/YOLOX/exps/custom/fire_nano.py` and `training/scripts/dfire_to_coco.py` for
the exp config and dataset conversion script used for this run — both are reusable for a
future retrain (e.g. once site negatives exist, or on better GPU hardware with more
epochs). `training/watchdog_train.sh` is a crash-resilient training supervisor (auto-
resumes from the last checkpoint on any crash, including this machine's specific MSVC/
Ninja gap) if training needs to run unattended again.
