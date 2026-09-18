# PPE model (Expansion Plan Phase H)

Software integration for PPE compliance checking is complete
(`src/perimeter/detect/ppe.py`, `Pipeline._process_ppe`/`_publish_ppe`), but **no
trained weights ship in this repo**. This is the one remaining module that needs a
genuinely new model architecture rather than a fine-tune of the existing person/
fire-smoke detector — training needs a GPU, curated datasets, and (per
`PLATFORM_EXPANSION_PLAN.md` §5) is "the largest net-new data/architecture workstream"
of the whole expansion; see that document's §5 for the dataset list (SH17 +
Construction-PPE) and `NOTICE.md`'s process for confirming their licences before
training starts.

Until a real model is placed here, PPE checking stays a clean, logged no-op: `main.py`
checks for `model.onnx` at startup and, if it's absent, logs once and starts every
camera's pipeline without it — boundary detection and every other module keep working
normally.

## Expected shape, once trained

- **Input:** a whole-person crop (not a face crop — PPE items span the whole body),
  resized to a fixed size (128×128 documented as the default in
  `detect/ppe.py:DEFAULT_INPUT_SIZE`; adjust the wrapper if the trained model uses a
  different size), `NCHW`, colour order to be confirmed and recorded in `manifest.json`
  the same way `models/yolox_person/manifest.json` records BGR vs RGB per file — this is
  load-bearing for correctness, not just licensing (see that manifest's own note on how
  silently getting colour order wrong costs measured recall).
- **Output:** one sigmoid presence probability per item, in the exact order
  `detect/ppe.py:PPE_ITEMS` declares: `("helmet", "vest", "gloves", "shoes",
  "glasses")` — a plain multi-label classifier, not an object detector (no boxes, no
  NMS). `PPEClassifier` is a thin `onnxruntime` wrapper, deliberately **not** built on
  `detect/yolox_onnx.py:YoloxOnnx` (that wrapper decodes YOLOX's specific anchor-grid
  output; a whole-crop multi-label classifier has a different I/O shape entirely).
- **The explicit "indeterminate" state** `PLATFORM_EXPANSION_PLAN.md` §5 calls for is
  not a third model output — it's a confidence-gap threshold applied at runtime
  (`detect/ppe.py:classify_state`, default: below 0.35 is "absent", above 0.65 is
  "present", the gap between is "indeterminate"). Tune those two thresholds against
  measured validation performance once real weights exist; they do not need to be
  baked into the model itself.
- A **`manifest.json`** next to `model.onnx`, matching `models/yolox_person/manifest.json`'s
  shape — colour order, input size, and (for an in-house checkpoint) the dataset
  versions/hashes trained on, the training config, and the git commit.
- A `NOTICE.md` §1 row updated from the current `☐` (reserved, unchecked) to a real
  entry recording that provenance — per that file's own "recorded at download/training
  time, not after" rule.

## Placing a model

```bash
perimeter-dashboard --ppe-model models/ppe/model.onnx
# or just drop the file at the default path above - it's picked up automatically
```

No code change is needed once the file exists at this path — `main.py` re-checks on
every startup.
