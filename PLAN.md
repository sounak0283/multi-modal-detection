# Fire, Smoke & Boundary Detection — Project Plan

**Status:** planning only. No implementation yet.
**Constraint:** permissive licences only (MIT / Apache-2.0 / BSD / CC0). Must be safe to ship in a
closed-source commercial product with no copyleft obligation and no paid enterprise licence.
**Commercial:** confirmed — this product will be sold. All §10 gates are live requirements, not
precautions.

**Revision 2** — folds in a design review. Changes from rev 1: boundary state machine rewritten
(§6.3, two functional bugs), tracker cadence corrected (§4), temporal-gate arithmetic corrected
(§5.5), false-alarm methodology corrected (§5.4), INT8 downgraded from mandate to Phase-1 experiment
(§9), site-negative collection moved to Phase 0 (§9), boundary engine expanded to four zone types
with a dashboard spec (§6), health/tamper events added (§4, §8).

---

## 1. Scope

One camera, one process. Three capabilities on a live feed:

1. **Fire & smoke detection** — detect flame and smoke regions as two separate classes, alert on
   sustained detections.
2. **Boundary detection** — the installer draws zones on a live camera view from a browser dashboard.
   Four zone types (area, tripwire, exclusion mask, fire ROI). A tracked person crossing an active
   boundary emits an ENTRY or EXIT event.
3. **Identity tagging** — on a boundary event only, attempt face detection + recognition so the alert
   reads *"unknown person entered Loading Bay at 14:32"* rather than *"motion detected"*.

### Confirmed constraints

| Constraint | Value | Consequence |
|---|---|---|
| Hardware | CPU-only laptop / PC | No transformer detectors. Quantisation likely required — measure, don't assume (§9). |
| Cameras | 1 (single site) | One process, three threads. No orchestration, no message broker, SQLite not Postgres. |
| Camera mount | **Must be fixed, not PTZ** | Normalised zone coordinates are meaningless on a moving camera. Hard requirement. |
| Scene | Indoor (factory / warehouse / office) + some outdoor | Public fire datasets are outdoor-skewed. Site-specific negatives are the main mitigation. |
| Licensing | Permissive only | No Ultralytics. Fire/smoke model must be trained in-house on permissive data. |
| Commercial | **Yes — confirmed** | NOTICE file, licence CI gate, biometric consent story, non-certification disclaimer, liability cover. |

### Explicit non-goals (v1)

- Multi-camera / multi-site. Design must not *prevent* it, but do not build for it.
- Re-identification of the same person across cameras.
- Vehicle, weapon, PPE, or fall detection.
- Certified fire alarm functionality — see §10.

---

## 2. Why not the obvious stack

`YuNet` and `SFace` were the original idea, but neither can detect fire, smoke, or people.
YuNet is a **face** detector: it misses anyone facing away, wearing a helmet or hard hat, or standing
more than ~8–10 m from the camera. Built as the boundary trigger, someone walking in backwards is
invisible. They are kept, but demoted to the identity layer (§7).

`YOLOv8` / `YOLO11` are the natural choice for the person and fire detectors and are ruled out:
Ultralytics code **and** pretrained weights are AGPL-3.0. Same for the usual substitutes.

| Ruled out | Licence | Note |
|---|---|---|
| YOLOv5 / v8 / v11 (Ultralytics) | AGPL-3.0 | Network-served or distributed ⇒ must open the whole app. |
| YOLOv6 (Meituan), YOLOv7 | GPL-3.0 | |
| YOLO-NAS | Apache code, **non-commercial weights** | Worst trap — repo badge looks clean. |
| SORT (abewley) | GPL-3.0 | Catches people who assume all trackers are MIT. |
| PyQt5 / PyQt6 | GPL | Use a browser UI. PySide6 (LGPL) also works but browser avoids the question. |
| `python-telegram-bot` | LGPL-3 | Call the Telegram REST endpoint with `requests` instead. |
| RF-DETR XL / 2XL | PML 1.0 | Nano–Large are Apache-2.0; XL/2XL are not. |

**Any pretrained `fire_best.pt` found on GitHub, Kaggle or Roboflow Universe is almost certainly
fine-tuned from Ultralytics weights, and is therefore AGPL-derived.** Downloading one re-imports
exactly the problem being avoided. The fire/smoke model must be trained in-house (§5).

### AGPL §13 applies to the dashboard

The product serves a FastAPI web UI. AGPL-3.0 §13 triggers on **network interaction**, not only on
distribution — an AGPL component anywhere in the serving path would compel source release even
without shipping a binary. The no-Ultralytics rule is therefore load-bearing, not precautionary.

### Training infrastructure

Do **not** train on free Colab or Kaggle notebook tiers — their terms on commercial use are unclear
at best, and "we trained our commercial product on a free research tier" is a poor answer during
customer due diligence. Rent GPU time (RunPod / Vast.ai / Lambda) or use Colab Pro. A YOLOX-nano run
on D-Fire is a few hours on one modern GPU; cost is not a real constraint here.

---

## 3. Approved stack

### Runtime (shipped)

| Layer | Choice | Licence |
|---|---|---|
| Person detection | YOLOX-nano ONNX | Apache-2.0 |
| Tracking | ByteTrack via **`trackers`** (Roboflow) | Apache-2.0 |
| Fire & smoke | YOLOX-nano trained in-house | Apache-2.0 (arch) / ours (weights) |
| Face detection | YuNet | see model dir LICENSE |
| Face recognition | SFace | Apache-2.0 |
| Inference | `onnxruntime` | MIT |
| Imaging | `opencv-python` | Apache-2.0 |
| Geometry | `shapely` | BSD-3 |
| Store | SQLite (WAL mode) | public domain |
| API | `fastapi` + `uvicorn` | MIT / BSD-3 |
| UI | plain-JS canvas, no framework | — |
| Notify | `requests` → Telegram REST | Apache-2.0 |

### Training only (never shipped)

| Layer | Choice | Licence |
|---|---|---|
| Framework | PyTorch | BSD-3 |
| Trainer | YOLOX (Megvii) | Apache-2.0 |
| Quantisation | `onnxruntime.quantization` | MIT |

**Production contains no PyTorch.** On a CPU box this means ~2 s cold start instead of ~15 s,
a far smaller installer, and one consistent inference path (OpenCV + onnxruntime).

### Download sources

| Asset | URL |
|---|---|
| YOLOX repo + COCO weights | `https://github.com/Megvii-BaseDetection/YOLOX` |
| YOLOX ONNX export guide | `https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime` |
| YOLOX-S ONNX (ready-made) | `https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox` |
| YuNet ONNX | `https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet` |
| SFace ONNX | `https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface` |
| ByteTrack (reference impl) | `https://github.com/ifzhang/ByteTrack` |
| `supervision` | `https://github.com/roboflow/supervision` |

> **Verify each `opencv_zoo` model's own LICENSE file** before shipping. The zoo repo carries a
> top-level licence but individual model directories carry their own, and they are not all identical.

---

## 4. Architecture

```
              ┌────────────────────────────────────────────┐
  RTSP/USB →  │ DECODE THREAD                              │
              │  cv2.VideoCapture, 12–15 fps               │
              │  → drop-to-latest slot (single overwrite)  │
              │  → JPEG ring buffer, 15 s pre-roll         │
              │  → health watchdog (feed loss / obscured)  │
              └───────────────────┬────────────────────────┘
                                  ▼
              ┌────────────────────────────────────────────┐
              │ INFER THREAD (grabs freshest frame only)   │
              │                                            │
              │  n % 2 == 0 → YOLOX-nano person @416       │
              │              → ByteTrack update            │
              │              → boundary state machine      │
              │                                            │
              │  n % 8 == 3 → YOLOX-nano fire/smoke @416   │
              │              → K-of-N temporal gate        │
              │                                            │
              │  on event   → YuNet → SFace (identity)     │
              └───────────────────┬────────────────────────┘
                                  ▼  event queue
              ┌────────────────────────────────────────────┐
              │ ALERT THREAD                               │
              │  cooldown → SQLite + JPEG + 15 s clip      │
              │           → Telegram / MQTT / webhook      │
              └────────────────────────────────────────────┘
```

### Why drop-to-latest, not a queue

A `queue.Queue` between decode and inference fills when inference lags, and the pipeline drifts
permanently behind live — an alert about something that happened 40 seconds ago. A single-slot
overwrite always yields the freshest frame and silently discards the rest. This is the single most
important structural decision in the runtime.

### Why the alert thread is separate

A Telegram POST that blocks for 3 s while holding the inference loop stalls the entire pipeline.
Alert delivery, disk writes and DB commits all live off the hot path.

### Cadence: the two models must not collide

Person runs on `n % 2 == 0`, fire on **`n % 8 == 3`** — deliberately offset. Scheduling fire on
`n % 8 == 0` makes both models run back-to-back on the same frame, stacking 50–80 ms onto every
eighth frame and producing visible periodic jitter for no reason.

### The tracker runs on detection frames only

ByteTrack updates *from detections*. On skip frames there are no detections, and calling
`update_with_detections()` with an empty set does not interpolate — it ages tracks toward deletion
via the lost-track buffer, actively degrading tracking. **Do not call the tracker on skip frames at
all.** The boundary state machine therefore evaluates at the person-detection cadence (~6–8 Hz), not
at frame rate.

Consequence for §6: `min_frames: 4` means ~0.5–0.7 s of hysteresis, not 4/15 s. Tune with that in
mind.

**Occlusion tolerance must be configured in seconds, not frames.** ByteTrack's `lost_track_buffer`
counts *update calls*, and we update at the detection cadence rather than the library's default
assumption of 30 Hz. Left at defaults it would silently deliver a quarter of the intended tolerance.
`PersonTracker` takes `lost_track_seconds` and converts internally.

**Two implementation traps, both measured against `trackers` 2.6.0 (2026-08-11):**

- Detections the tracker has seen but not yet *confirmed* return the sentinel id **−1**, not `None`
  — `tracker_id` is an integer array and cannot hold `None`. Every unconfirmed detection in a frame
  carries the same −1, so passing them through collapses several different people into one track
  identity. Since §6.3 keys entry/exit state on track id, two strangers entering a zone would share
  one state and cancel each other's events. Unconfirmed detections must be dropped.
- Valid track ids start at **0**, not 1 (unlike the original ByteTrack). Filtering with `> 0` would
  silently drop the first person to appear after every camera start.

### Measured track stability

400 frames of pedestrian footage at 7.5 Hz detection cadence, `tools/track_preview.py`:

| Metric | Value |
|---|---|
| Unique track IDs | 16 |
| Peak concurrent tracks | 8 |
| Churn ratio (IDs / peak concurrent) | 2.0 |
| Track fragments (< 1 s) | 0 % |
| Median track length | 53 detection frames (~7 s) |

Stable enough that §6.3's born-settled initialisation is sufficient. **This must be rerun on the
customer's own footage** — warehouse racking produces far more occlusion than an open courtyard, and
churn is what decides whether the hysteresis holds.

### Camera health is an alert kind, not just a reconnect

A vision system that silently stops seeing is worse than no system, and for a sold product it is a
support and liability issue. The decode thread runs a watchdog emitting `kind='health'`:

| Subtype | Trigger |
|---|---|
| `feed_lost` | No frame decoded for > 10 s |
| `feed_restored` | Decoding resumes after a loss |
| `lens_obscured` | Frame variance collapses below threshold for > 30 s (covered, sprayed, unplugged light) |

### CPU thread contention

Set `cv2.setNumThreads(1)` and give onnxruntime `intra_op_num_threads = physical_cores - 1`.
OpenCV and ORT each grabbing every core is the classic cause of *"it got slower when I optimised it."*

### Latency budget (target, 4–8 core CPU)

**Measured 2026-08-11** on the dev machine (AMD Zen 2, 6 physical cores, ORT 1.28, 5 intra-op
threads), 30 timed passes over 768×576 pedestrian footage:

| Stage | Model | Input | Cadence | Median | Result |
|---|---|---|---|---|---|
| Person | **YOLOX-nano FP32** | 416×416 | every 2nd frame | **25.6 ms → 39 fps** | ✅ ~5× headroom |
| Person (alt) | YOLOX-tiny FP32 | 416×416 | every 2nd frame | 52.4 ms → 19 fps | ✅ affordable if nano's accuracy falls short |
| Person (ref) | YOLOX-S FP32 | 640×640 | — | 172.7 ms → 5.8 fps | ❌ too heavy |
| Person (ref) | YOLOX-S **INT8** | 640×640 | — | 2217 ms → 0.5 fps | ❌ see §9.2 |
| Fire/smoke | YOLOX-nano | 416×416 | every 8th frame, offset | ~26 ms expected | same architecture |
| Track + boundary | ByteTrack (Kalman) | — | **detection frames only** | < 2 ms | |
| Face | YuNet + SFace | 320 crop from **full-res** frame | on event only | ~15 + 20 ms | not yet measured |

Nano's stage split is 2.7 ms preprocess / 19.2 ms forward / 3.5 ms postprocess. Preprocess and
postprocess are ~24 % of the budget, so a faster model alone would yield less than it appears —
worth remembering before optimising the network further.

The headroom is real but should not be spent casually: it is what absorbs the fire/smoke model,
the tracker, JPEG encoding for the ring buffer, and the decode thread competing for the same cores.

A person walking at 1.5 m/s moves ~20 cm between person-inference frames and the Kalman filter
interpolates that comfortably. Someone *running* at 4 m/s moves ~60 cm — still fine for a tripwire
sign-flip, but it is the case to test, not the walking case.

### Ring buffer sizing

15 s × 12 fps × 720p **JPEG-encoded** ≈ 180 frames ≈ 10–15 MB. Store encoded, not raw — raw is
~900 MB.

15 s rather than 8 s because the fire temporal gate consumes 3–5 s reaching confirmation (§5.5). An
8 s buffer would leave only ~3 s of genuine pre-event footage. **Clips for fire events are cut from
the candidate's first hit, not from confirmation time** — otherwise the clip opens on an
already-established fire and shows the customer nothing about how it started.

---

## 5. Fire & smoke model

### 5.1 Datasets

| Dataset | Size | Licence | Link |
|---|---|---|---|
| **D-Fire** (primary) | 21,527 imgs, 26,557 boxes | **CC0 1.0** (see caveat) | `https://github.com/gaiasd/DFireDataset` |
| **FASDD_CV** (scale-up) | 95,314 samples | *verify on landing page* | `https://doi.org/10.57760/sciencedb.j00104.00103` |
| FASDD Kaggle mirror | — | see source | `https://www.kaggle.com/datasets/yuulind/fasdd-cv-coco` |
| Pyro-SDIS (**skip for v1**) | 29,537 train / 4,099 val | Apache-2.0 | `https://huggingface.co/datasets/pyronear/pyro-sdis` |

**D-Fire composition:** 1,164 fire-only · 5,867 smoke-only · 4,658 fire+smoke · **9,838 "none"**.
Images are 416×416 with YOLO-format normalised annotations and pre-split train/val/test.
That 9,838-image negative block is the most valuable part of the dataset — it is specifically
composed of objects that resemble fire or smoke (lamps, sun glare, red objects).

> **CC0 caveat.** The authors' CC0 declaration covers their contribution — the compilation and the
> annotations. The underlying images were web-sourced, and the authors could not waive third-party
> copyright they never held. In practice this is the same situation as COCO (which underpins the
> YOLOX weights) and training on it is standard and low-risk, but it is *not* the unqualified "clean"
> that CC0 usually implies. Record it accurately in NOTICE.md rather than overstating it.

**Pyro-SDIS is excluded for v1** — long-range wildfire plumes against open sky, wrong domain for a
warehouse. Its licence is clean, but Pyronear's *trained models* are YOLOv8/v11-based: use the data,
never the weights.

**Rejected:** DFS (siyuanwu) — no stated licence. FLAME1 — CC BY 4.0 but aerial and
classification-only. Any Roboflow Universe copy marked "unspecified" — treat as unusable.
Anything CC BY-NC (kills commercial) or CC BY-SA (may propagate share-alike to the model).

### 5.2 Site negatives — the actual differentiator

Public fire data is outdoor-skewed. Indoor fire detection data is genuinely scarce, and this is the
project's largest technical risk. Collect **500–1000 negatives from the real cameras**:

- welding and grinding sparks — the worst offender in any factory
- machinery steam, condensation, cleaning vapour
- forklift/vehicle headlights, rotating amber beacons, warning lights
- exit signs, red and orange safety markings, space heaters
- low-angle sunlight through skylights or roller doors
- fog, dusk and headlight glare on the outdoor views
- **night-time IR footage**, if the outdoor cameras switch to IR — greyscale IR will otherwise be
  entirely out-of-distribution for a model trained on daylight RGB

**Start this in Phase 0, not Phase 6.** These events depend on the customer's operations happening,
not on developer time — it is a calendar dependency you do not control. Get continuous recording
running at the site as early as site access can be negotiated, and mine the footage later. Treating
this as a collection sprint at Phase 6 makes the schedule hostage to whether anyone happens to weld
that fortnight.

### 5.3 Training recipe

1. Merge D-Fire + FASDD_CV to two classes `{0: fire, 1: smoke}`.
   Deduplicate by perceptual hash — both pull from overlapping web sources.
2. Keep **all** D-Fire "none" images as background images. Target ≥ 30 % negatives in train.
3. Add the site negatives from §5.2.
4. Train:
   `python -m yolox.tools.train -f exps/custom/fire_nano.py -d 1 -b 32 --fp16 -c yolox_nano.pth`
   100–150 epochs · `no_aug_epochs=15` · input 640 for training, export at 416 · mosaic on for the
   first 85 % of epochs.
5. Export ONNX → quantise → validate (§9 gate).
6. **Write `models/firesmoke/manifest.json`** recording dataset versions + content hashes, the
   training config, and the git commit. Without this there is no way to answer "which shipped weights
   are affected?" if a dataset's terms later turn out to be different than believed — which, for a
   sold product, is the question that matters.

### 5.4 Threshold tuning — do not optimise mAP

Tune for **false alarms per hour**, then report recall at that operating point. A model with
excellent mAP that cries wolf twice an hour will be switched off by the customer within a week, and a
detector nobody looks at has zero value.

**Measure it on continuous site video, running the full pipeline including the K-of-N gate.**
A held-out set of negative *images* cannot produce an hourly rate — there is no time axis — and
measuring pre-gate detections badly overstates the alert rate, since the temporal gate is precisely
what converts frequent per-frame false detections into rare alerts. The metric is: *run N hours of
recorded site footage end-to-end, count emitted alerts.*

Target ≤ 1 alert/hour/camera as a starting point, but treat it as negotiable with the customer —
24/day may still be too many for an unattended site, and 1/day may be achievable once exclusion masks
(§6.2) cover the fixed offenders.

**Weight recall toward smoke.** Indoors, smoke reaches the camera before flame does — a fire in an
aisle may be entirely occluded while smoke is visible at ceiling level. Run a lower confidence
threshold for the smoke class than for fire.

### 5.5 Temporal gate (K-of-N)

Maintain candidate blobs. A detection joins a candidate if IoU ≥ 0.3 with that candidate's last box.
**Alert when a candidate reaches ≥ 6 hits within the last 10 processed frames.**

At the ~2 fps fire cadence, 10 processed frames is a **5-second window**, and 6 hits means a
**minimum 3 s to alert**. Budget 3–5 s of detection latency, and size the ring buffer accordingly
(§4). If that latency is unacceptable to the customer, the lever is the fire cadence (`n % 8` → `n % 4`),
not K or N — shrinking the window is what false alarms are made of.

Escalate severity if box area grows monotonically over 5 s — fire grows, brake lights and sunsets do
not. This one rule removes more false alarms than any architecture change.

---

## 6. Boundary engine

### 6.1 Geometry

- Zones stored as **normalised 0–1 coordinates** so they survive a resolution change or a substream
  swap.
- Test the **foot point** = `((x1 + x2) / 2, y2)` — bottom-centre of the box — not the centroid.
  Centroids break under perspective: a tall person standing outside a floor zone has a centroid
  inside it.
- **Polygon:** `shapely` point-in-polygon on the foot point.
- **Tripwire:** sign of the 2-D cross product gives the side. A sign flip between consecutive
  evaluations is a crossing; the order of the flip gives ENTRY vs EXIT.
- **Validate polygons on save.** `shapely` will happily accept a self-intersecting bowtie and then
  return nonsense containment results. Reject non-simple polygons in the dashboard, not at runtime.

### 6.2 Zone types

Four types. The third is the highest-value item in this document relative to its cost.

| Type | Shape | Purpose |
|---|---|---|
| `polygon` | closed polygon | Person inside/outside → ENTRY / EXIT |
| `tripwire` | line or polyline | Directional crossing → A→B / B→A |
| `exclusion` | closed polygon | **Suppress** detections inside this region |
| `fire_roi` | closed polygon | Restrict or bias fire/smoke detection to a region |

**Exclusion masks are what will save the deployment.** The worst false-alarm sources are fixed in
place: the welding bay, the amber beacon on the forklift charger, the skylight that throws sunlight
at 16:00, a window looking onto a road. Let the installer draw a polygon around each and mark it
ignore-for-fire-but-still-track-people. This does more for false-alarm rate than another 20,000
training images, and it is hours of work against weeks.

**Fire ROI is the inverse:** mark the high-value area (server room, chemical store) and apply a
*lower* confidence threshold there, because a missed fire in that specific spot costs more than a
false alarm does.

### 6.3 State machine (per track ID) — corrected

Rev 1 contained two functional bugs. Both are fixed below; both are documented because the naive
implementation reintroduces them.

**Rev 3 (implementation):** the rev 2 pseudocode below was itself wrong — see bug 3. Two
states are required, not one plus a flag.

```
track_state = { confirmed: IN|OUT, candidate: IN|OUT, age: int }

on track creation:
    observed  = test(foot_point)
    confirmed = candidate = observed   # born already settled at its true position
    age       = min_frames             # the initial observation is NOT an event

on each detection-frame update:
    observed = test(foot_point)
    if observed != candidate:
        candidate = observed; age = 1
    else:
        age += 1

    if age >= min_frames and candidate != confirmed:
        emit(ENTRY if candidate == IN else EXIT)
        confirmed = candidate
```

**Bug 1 — a time-based cooldown swallows EXIT events.** Rev 1 guarded emission with
`now - last_event_ts > cooldown_s` at `cooldown_s: 30`. A person entering at t=0 and leaving at t=10
had their EXIT suppressed; and because the guard also used `state_age == min_frames` (exact
equality, true for one update only), the event never re-fired — it was lost permanently. Entering and
leaving inside 30 s is the *normal* case, so entry/exit pairing was broken by default. The
`emitted` flag replaces the timer: state transitions are inherently self-limiting, and hysteresis
already does the rate-limiting work.

**Bug 2 — uninitialised tracks fire phantom ENTRYs.** Rev 1 never defined the state of a new track.
ByteTrack reassigns IDs after occlusion, which in a warehouse with racking happens constantly. If a
new track defaults to `OUT`, *every person who appears already inside a zone fires a false ENTRY* —
including the same person who just acquired a new ID while standing still. Initialising from the
observed position with `emitted = True` means only a genuinely observed transition emits.

**Bug 3 — a single state plus an `emitted` flag fires spurious events.** Found while implementing
rev 2's pseudocode above. Sequence with `min_frames: 3`: a person is settled OUTSIDE, flickers
inside for one frame (candidate IN, age 1 — never confirmed), then returns outside. With one state,
the return to OUT looks like a fresh transition that has not been emitted, so it fires an **EXIT
with no matching ENTRY**. Comparing the settled candidate against the last *confirmed* state makes
the round trip the no-op it actually was. Both directions of this are covered by regression tests.

**The tradeoff, accepted deliberately:** someone who crosses the boundary while fully occluded and
re-emerges with a new track ID produces no event. That is the correct trade against ID-churn alert
floods, but it is a decision, not an oversight — record it in the customer-facing limitations.

`min_frames: 4` is the hysteresis. Note from §4 that the state machine runs at the person-detection
cadence (~6–8 Hz), so 4 frames ≈ 0.5–0.7 s, not 4/15 s.

### 6.4 Per-zone rules

Geometry alone is not enough. Each zone carries:

| Field | Purpose |
|---|---|
| `name` | Appears verbatim in the alert ("Loading Bay") — let the installer type it |
| `detect` | Which classes the zone applies to: `person`, `fire`, `smoke` |
| `events` | `entry`, `exit`, or both — most sites only care about entry |
| `direction` | Tripwire only: `a_to_b`, `b_to_a`, `both` |
| `schedule` | Active hours/days |
| `severity` | Routes to different sinks |
| `min_frames` | Per-zone hysteresis — a busy doorway needs more damping than a fenceline |

**`schedule` is close to as important as the exclusion mask.** A warehouse zone without one fires
400 times during the working day and gets switched off in week one.

### 6.5 Config schema

`config/zones.yaml`, keyed by camera ID even though there is only one camera today. Retrofitting a
camera ID into flat config later is annoying; adding it now costs nothing.

```yaml
cameras:
  cam_01:
    zones:
      - id: loading_bay
        name: "Loading Bay"
        type: polygon
        points: [[0.12,0.44],[0.61,0.41],[0.66,0.93],[0.08,0.95]]   # normalised 0–1
        detect: [person]
        events: [entry, exit]
        min_frames: 4
        schedule: {days: [mon,tue,wed,thu,fri], from: "18:00", to: "06:00"}
        severity: high

      - id: gate_line
        name: "Main Gate"
        type: tripwire
        points: [[0.20,0.70],[0.85,0.66]]
        direction: a_to_b          # a_to_b | b_to_a | both
        detect: [person]
        min_frames: 3
        severity: high

      - id: welding_bay
        name: "Welding Bay"
        type: exclusion
        points: [[0.70,0.30],[0.98,0.30],[0.98,0.75],[0.70,0.75]]
        applies_to: [fire, smoke]  # ignore fire here, still track people through it

      - id: server_room
        name: "Server Room"
        type: fire_roi
        points: [[0.05,0.10],[0.35,0.10],[0.35,0.40],[0.05,0.40]]
        applies_to: [fire, smoke]
        conf_delta: -0.08          # more sensitive in this high-value area
```

### 6.6 Dashboard — zone editor

Plain-JS canvas over a live snapshot (§8), no framework.

1. **Freeze a frame** from the live feed as the drawing backdrop.
2. **Pick a type**, then click to place vertices; double-click or Enter to close. Tripwires take two
   points, or N for a polyline.
3. **Drag vertices** to adjust; right-click a vertex to delete.
4. **Tripwire direction** renders as a perpendicular arrow — click it to cycle A→B / B→A / both.
5. **Rules panel** — name, classes, events, schedule, severity, `min_frames`.
6. **Save** → writes `zones.yaml`, and the engine **hot-reloads without restart**. Installers iterate
   ten to twenty times on site; a process restart per edit is miserable and turns a 20-minute
   commissioning into an afternoon.
7. **Live test mode** — overlay real detections and track IDs on the video, showing each track's
   IN/OUT state and flashing zones on event. This is how the installer confirms a zone works before
   leaving site, and it doubles as the sales demo.

Validation on save: reject self-intersecting polygons (§6.1), reject tripwires shorter than ~5 % of
frame width, warn if a `polygon` zone covers > 80 % of frame (almost always a mis-drawn zone).

---

## 7. Identity layer (YuNet + SFace)

**Invoked only on a boundary event.** Never on every frame — the cost is unjustifiable on CPU and the
privacy exposure is unnecessary.

1. Gate on size: skip if the person box is under ~120 px tall **in the original frame**. Below that
   the face is under ~40 px and YuNet returns nothing useful.
2. Crop the top ~35 % of the person bounding box **from the full-resolution frame**, never from the
   416 letterboxed inference input.
3. YuNet → take the largest face.
4. SFace → 128-d embedding.
5. Cosine similarity against `gallery.npz` (embeddings + names). OpenCV's default threshold ≈ 0.363.

Three outcomes, and all three must be first-class in the alert copy:

| Outcome | Meaning |
|---|---|
| `known:<name>` | Matched an enrolled person. |
| `unknown_face` | A face was found, no gallery match. |
| `no_face` | No usable face — back turned, helmet, too far, motion blur. |

**`no_face` must never be rendered as "intruder."** Outdoors at a gate it will be the common case.
Expect the layer to abstain often and design the UI around abstention rather than treating absence of
a match as evidence of anything.

Indoors, people are large in frame and often facing the camera, so this layer will actually earn its
place. Outdoors, treat any match as a bonus.

---

## 8. Repository layout

Implemented as an installable package under a `src/` layout — `src/fsbd/…` rather than a
bare `src/…` — so imports resolve unambiguously and `pip install -e .` works. Module
responsibilities are otherwise exactly as listed.

```
src/fsbd/
  record.py                       # Phase 0 site recorder (negative collection)
  capture/     rtsp.py            # VideoCapture wrapper, reconnect/backoff
               latest_slot.py     # single-slot drop-to-latest
               ring_buffer.py     # JPEG pre-roll buffer, 15 s
               health.py          # feed-loss / lens-obscured watchdog
  detect/      yolox_onnx.py      # shared ORT session wrapper, letterbox, NMS
               person.py
               firesmoke.py       # + K-of-N temporal gate
  track/       bytetrack.py       # supervision wrapper, detection-frame cadence only
               state.py           # per-track boundary state (§6.3)
  boundary/    geometry.py        # foot point, cross product, point-in-polygon
               engine.py          # state machine, hysteresis, exclusion masks
               zones.py           # YAML load/validate, normalised <-> pixel, hot-reload
               schedule.py        # per-zone active-hours evaluation
  identity/    yunet.py
               sface.py
               gallery.py         # enroll, match, delete-by-name
  alerts/      bus.py
               cooldown.py
               sinks/  telegram.py  mqtt.py  smtp.py  webhook.py
  api/         app.py             # FastAPI: /zones /events /stream /enroll /health
  store/       db.py              # SQLite schema + migrations, WAL mode
web/           index.html, zones.js, live.js      # plain JS canvas zone editor
models/        firesmoke/{model.onnx, manifest.json}, person/, yunet/, sface/
config/        zones.yaml, app.yaml
data/          dfire/  fasdd/  site_negatives/
tools/         quantize.py, benchmark_fps.py, export_onnx.py, replay_eval.py
NOTICE.md
THIRD_PARTY_LICENSES.md
PLAN.md
```

### SQLite concurrency

The alert thread writes while the API thread reads. Without care this throws `database is locked` in
production and nowhere else.

- `PRAGMA journal_mode=WAL` at startup
- `check_same_thread=False` on the connection
- **Single writer** — only the alert thread writes; the API reads through its own connection

### Event schema

```sql
CREATE TABLE events (
  id            INTEGER PRIMARY KEY,
  ts            REAL    NOT NULL,      -- unix epoch, float
  camera_id     TEXT    NOT NULL,
  kind          TEXT    NOT NULL,      -- 'boundary' | 'fire' | 'smoke' | 'health'
  subtype       TEXT,                  -- 'entry'|'exit' | 'feed_lost'|'feed_restored'|'lens_obscured'
  zone_id       TEXT,
  track_id      INTEGER,
  confidence    REAL,
  identity      TEXT,                  -- 'known:<name>' | 'unknown_face' | 'no_face' | NULL
  bbox          TEXT,                  -- JSON [x1,y1,x2,y2] normalised
  snapshot_path TEXT,
  clip_path     TEXT,
  delivered     INTEGER DEFAULT 0,
  delivery_attempts INTEGER DEFAULT 0,
  last_attempt_ts   REAL
);
CREATE INDEX idx_events_ts        ON events(ts DESC);
CREATE INDEX idx_events_undelivered ON events(delivered) WHERE delivered = 0;
```

The alert thread retries undelivered events with exponential backoff, capped — a Telegram outage must
not silently drop a fire alert, and must not spin forever either.

---

## 9. Phases

| # | Deliverable | Est. |
|---|---|---|
| 0 | Repo, `NOTICE.md`, CI licence gate, **and start continuous site recording** | 1 day + site access |
| 1 | Capture thread + drop-to-latest + YOLOX-nano ONNX + **fps benchmark: FP32 vs FP16 vs INT8** | 4 days |
| 2 | ByteTrack + on-screen IDs; verify IDs survive occlusion in the actual space | 2 days |
| 3 | Canvas zone editor (4 types) → YAML, hot-reload, state machine, entry/exit events | 6–7 days |
| 4 | Alert thread: SQLite WAL, snapshot, 15 s pre-roll clip, Telegram sink, retry loop | 3 days |
| 5 | Health watchdog: feed loss, lens obscured | 1 day |
| 6 | YOLOX training pipeline proven end-to-end on a D-Fire subset | 2–3 days |
| 7 | Full fire/smoke train + exclusion-mask tuning + threshold tuning on site video | 2–3 weeks |
| 8 | YuNet + SFace enrollment, identity tagging, default-off feature flag | 3 days |
| 9 | **Commissioning test** — smoke machine + controlled tray burn, customer sign-off | 2 days + scheduling |

**Phases 1–5 are a sellable product on their own** (intrusion detection with health monitoring).
Build them first and in that order. Phase 7 is where the calendar actually goes — dataset curation
and tuning, not modelling — which is exactly why site recording starts at Phase 0.

### 9.1 The one gate that matters

**Measure real fps at the end of Phase 1, before building anything else.** If YOLOX-nano at 416
lands under 8 fps on the target machine, the entire cadence table in §4 needs rebalancing — lower
decode fps, larger frame skip, or 320 input. Discovering that in week one is cheap. Discovering it in
week six means rebuilding the threading model.

### 9.2 Quantisation — SETTLED, and the answer is no

Rev 1 mandated INT8 as "roughly a 2–3× speedup for a small mAP cost." Rev 2 downgraded it to an
experiment. **The experiment has now run, and INT8 is a dead end on this hardware.**

Measured 2026-08-11, YOLOX-S @640 on AMD Zen 2:

| Precision | Median forward | Throughput | vs FP32 |
|---|---|---|---|
| FP32 | 158 ms | 5.8 fps | — |
| INT8 | 2206 ms | 0.5 fps | **0.01× — 14× slower** |

Not merely "less than 2–3×" — an order of magnitude *worse*. Zen 2 has AVX2 but no AVX512-VNNI, so
ORT has no fast INT8 convolution kernel and falls back to a path dominated by per-operator
quantise/dequantise conversion. The quoted speedups in the literature quietly assume VNNI.

**Decision: ship FP32.** It is unnecessary anyway — YOLOX-nano FP32 at 416 already delivers 39 fps
against an 8 fps requirement (§4). Quantisation was solving a problem that the correct model size
had already solved.

Revisit only if the deployment CPU is confirmed to be Intel with AVX512-VNNI, or if the fire/smoke
model turns out heavier than nano. The benchmark tool (`tools/benchmark_fps.py`) makes rerunning
this a single command on any candidate machine, and it should be rerun on the actual target box
before deployment rather than assumed from these numbers.

### 9.3 Site validation — positives, not just negatives

There is no point tuning false alarms if nothing proves the detector fires on real fire *in this
building*. Staging a fire casually is not an option, so use the standard fire-industry commissioning
approach:

- **Smoke machine / smoke pellets** at several distances and heights — the same equipment used to
  commission conventional smoke detection
- **Small controlled tray burn** with fire safety present and customer approval
- Record every test; these clips become the regression set for future model versions

This is also the customer's acceptance criterion, so it is a commercial deliverable as much as a
technical one. Phase 9 requires scheduling with the site — book it early.

---

## 10. Commercial gates that are not about licensing

### 10.1 This is not a certified fire alarm

A vision system is **not** a certified fire detector under UL 268 or EN 54. It must be sold and
documented as *supplementary situational awareness*, never as a replacement for certified detection.
Put the disclaimer in the UI, the installation docs, and the contract. Marketing it otherwise creates
liability that is difficult to insure and may be unlawful in some jurisdictions.

### 10.2 Product liability

The §10.1 disclaimer is necessary but not sufficient. Selling a product that customers may rely on for
life-safety carries exposure a UI notice alone does not cover:

- Contractual limitation of liability in the sales agreement
- Product liability insurance appropriate to the deployment scale
- Documented, dated evidence of the §9.3 commissioning tests per site
- A stated position on what happens when the system is offline (see the health events in §4 — the
  ability to prove the customer was notified of an outage is a liability control, not a feature)

Flagged as a pre-sale checklist item. Requires qualified legal and insurance advice, not engineering.

### 10.3 Biometric data

The identity layer is biometric processing regardless of how permissive the model licence is.
Relevant regimes include India's **DPDP Act 2023**, the EU **GDPR Art. 9**, and Illinois **BIPA**.
Requirements typically include informed consent, a stated retention period, and a deletion path.

Practical design consequences:

- Ship the identity layer as a **default-off feature flag**. A customer who has not done the consent
  paperwork simply never enables it, and the rest of the product still works.
- `gallery.npz` needs a documented delete-by-name path, exposed in the API and the UI.
- Log a retention policy for snapshots and clips; default to something short (e.g. 30 days) and make
  it configurable.
- Do not store face crops beyond what is needed to produce the embedding.

This is a note on where to seek qualified advice, not legal advice — the specific obligations depend
on jurisdiction and deployment, and should be reviewed by a lawyer before the first commercial sale.

### 10.4 Licence hygiene from day one

- `NOTICE.md` records every model, dataset and dependency with its licence and source URL.
- `THIRD_PARTY_LICENSES.md` carries the full licence texts required for redistribution
  (Apache-2.0 §4 requires retaining the notice; BSD and MIT require the copyright line).
- CI gate: `pip-licenses --fail-on="GPL;AGPL;LGPL;CC-BY-NC;CC-BY-SA"` on every build.
- **A second gate that fails if anything in `models/` or `data/` lacks a NOTICE.md row.**
  `pip-licenses` reads Python package metadata only — it cannot see an `.onnx` file, a dataset, or a
  vendored source file, which are the three places this project's actual licensing risk lives.
  The dependency gate alone creates false confidence.
- Record the licence of every dataset **at download time**, not at publication time.

---

## 11. Open questions to resolve before Phase 1

1. **Is the camera fixed or PTZ?** Normalised zones are meaningless on a moving camera. If PTZ, zones
   must bind to presets and the whole boundary design changes. Treat a fixed mount as a hard
   requirement in the quotation.
2. Camera model and whether it exposes a **substream** — a 640×360 substream for inference plus the
   main stream for recording removes the decode cost almost entirely and is the single cheapest
   performance win available.
3. Exact CPU: core count and **AVX2 vs AVX-512/VNNI** — determines the realistic fps ceiling and
   decides §9.2.
4. Whether alerts go to Telegram, MQTT into an existing BMS/SCADA, or email. Build one sink in
   Phase 4; the interface supports the rest.
5. Whether the outdoor views run **IR at night** — greyscale IR will degrade a fire model trained
   purely on daylight RGB. If yes, night-time site negatives are mandatory (§5.2).
6. When can site recording start? This gates Phase 7, which is the critical path.
