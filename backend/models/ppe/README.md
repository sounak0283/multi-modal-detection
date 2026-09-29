# PPE model (Expansion Plan Phase H)

Two backends, selected by `ppe.backend` in `config/app.yaml`:

| `ppe.backend` | Model file | What it checks |
|---|---|---|
| `helmet_detector` (default) | `ppe_final_C_v2_320.onnx` | **helmet only** — YOLOX-Nano head/helmet detector on a padded person crop |
| `classifier` | `model.onnx` (no weights exist yet) | the original multi-label interface: helmet / vest / gloves / shoes / glasses |

Weights are git-ignored (NOTICE.md §1). Install the helmet detector from the package — the
file is checked against the SHA-256 in `manifest.json` before it is written:

```bash
python tools/install_ppe_model.py --zip "path/to/demo_package.zip"
```

Nothing else is needed: `main.py` picks up the configured path on every start (`--ppe-model`
still overrides it). A missing or unloadable model logs once and leaves PPE inactive; every
other module keeps working.

## How the helmet detector fits the pipeline

`detect/ppe_helmet.py:PPEHelmetDetector` keeps the classifier's contract — a
`{item: probability}` dict per person — so zones, `PPEMonitor`'s hysteresis, alerts,
evidence and e-mail are unchanged:

- helmet on the head → helmet probability ≥ 0.65 → `present`
- bare head → ≤ 0.35 → `absent` → a violation once seen on 3 consecutive PPE checks
- no head found, or the person is under 96 px tall → 0.5 → `indeterminate` (never a violation)
- vest / gloves / shoes / glasses → always `indeterminate`. A zone that requires one of them
  logs a warning once and never alerts for it — this model cannot see those items.

To use it: enable the `ppe` module on a camera (or Video test), and on a boundary of type
Zone tick **helmet** under required PPE.

## Accuracy and limits (from the package's model_card.md)

Colour val AP50 96.0; head P/R 93.4/90.2, helmet P/R 96.9/93.3. Black/grey helmets are
weaker (74 % recall; 63 % on CCTV-like input), green has little data. Night/IR and CCTV are
only simulated in validation. Trained mostly on web/stock site photos — verify on each
camera before relying on it.
