# Model card - PPE helmet detector (final_C_v2)

## What it does
Two-stage, CPU-only: a COCO person detector (YOLOX-nano, 416 px) finds people >= 96 px tall; each person
crop (padded) goes to the PPE detector (YOLOX-nano, 320 px), which finds **head** (bare head = NO HELMET)
and **helmet** (a head wearing a hard hat, any colour). A simple IoU tracker raises NO HELMET only after a
bare head is seen in 3 of the last 5 frames. **Vest is not included** (see limits).

## Training
- Data: Hard Hat Workers (Dataverse, CC0), SHEL5K (Mendeley, CC BY 4.0), Roboflow Construction Site Safety
  v30 (CC BY 4.0); person crops made with the runtime person detector; label fixes (CSS hard-hat boxes
  extended to the helmeted-head extent; unreliable SHEL5K orphan-shell boxes reviewed/excluded).
  Provenance: repository NOTICE.md.
- Recipe C: YOLOX-nano, 320 px, fp32, batch 48, grayscale (p 0.2) + CCTV degradation (p 0.3) + IR
  simulation (p 0.05) augmentation, 60 epochs warm-started from the 20-epoch ablation model (arm C).
- Hardware: NVIDIA GTX 1650 4 GB, torch 2.6.0+cu124.

## Validation (val_v2, 4,607 held-out person crops, 6,962 boxes)

| Input condition | AP50:95 | AP50 | Head P / R | Helmet P / R |
|---|---|---|---|---|
| Colour | 61.8 | 96.0 | 93.4 / 90.2 | 96.9 / 93.3 |
| Grayscale | 60.0 | 94.5 | 92.5 / 87.3 | 96.1 / 90.5 |
| IR (simulated) | 58.5 | 93.1 | 93.0 / 83.9 | 96.5 / 89.0 |
| CCTV-degraded (simulated) | 56.7 | 92.4 | 90.8 / 83.6 | 96.4 / 87.4 |

P / R at the package thresholds, IoU 0.5, per frame (before the 3-of-5 smoothing).
Previous 20-epoch model (arm C) on the same set: AP50 95.6 / 93.9 / 92.6 / 91.6 (colour / gray / IR / CCTV).

## Thresholds (config.json)
- head 0.45: highest threshold keeping head recall >= 90 % on colour val (precision 93.4 %)
- helmet 0.55: F1-optimal on colour val
- person 0.30 (runtime person-detector confidence)

## Helmet recall by helmet colour (colour val, at thresholds)
red 94.8 %, yellow 95.4 %, white 92.0 %, blue 93.1 %, orange 91.3 %, **green 81.8 % (n=22), black/grey 73.9 % (n=115)**.

## Known limits
- **Black/grey helmets are weaker** (74 % recall colour, 63 % on CCTV-degraded input); green helmets have very
  little data (22 val examples).
- **Night/IR and CCTV** are simulated in validation, not real footage: head recall drops to ~84 %
  per frame. Real site footage is needed to confirm.
- **No vest detection**: pseudo-labelling every training image with the chosen vest teacher would have taken
  ~33 h on this GPU. Vest detection in grayscale/IR was also weak for every teacher tested.
- People shorter than 96 px are not checked; heads are judged per person crop.
- Training images are mostly web/stock-style site photos; accuracy on a specific camera must be verified.
- The sample video is a slideshow of held-out test images, not continuous CCTV.


## Package verification (demo_package_final)
- ONNX vs PyTorch (20 val images): PASS, 23 boxes, worst IoU 1.0, worst score diff 0.0
- End-to-end on samples/test_input.mp4, GPU hidden: 53.4 FPS (CPU, 4 threads, smoothed at end)
- Fresh venv (Python 3.12, requirements.txt only): frames 150, final FPS 50.1, NO HELMET tracks 21
- Fresh venv packages: numpy==2.5.3, onnxruntime==1.30.0, opencv-python==5.0.0.93
