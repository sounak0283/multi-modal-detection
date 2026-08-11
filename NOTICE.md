# NOTICE

Third-party models, datasets and dependencies used by this project, with licence and source.

**Maintenance rule:** every entry is recorded **at the time the asset is first downloaded**, not at
release time. Reconstructing provenance after training is miserable and error-prone. Nothing enters
`models/`, `data/` or `requirements.txt` without a row in this file.

**Policy:** permissive licences only — MIT, Apache-2.0, BSD-2/3-Clause, CC0, CC BY.
Explicitly excluded: GPL, AGPL, LGPL, CC BY-NC, CC BY-SA, and any "research use only" or
"non-commercial" term, on code, weights, or data.

**MPL-2.0 — conditionally accepted.** MPL-2.0 is *file-level* copyleft: the obligation
attaches to the MPL-licensed files themselves, not to code that merely links against them.
Distributing an unmodified MPL component alongside proprietary code imposes only a notice
requirement, so MPL-2.0 dependencies are accepted **provided we never modify them**. If we
ever patch an MPL file, the modified file must be published. Two shipped transitive
dependencies are affected (§3): `certifi` and `tqdm`. Recorded here so the decision is
deliberate rather than accidental — it was found by running the CI gate, not by review.

---

## 1. Models — shipped in the product

| Asset | Directory | Version | Licence | Source | Verified |
|---|---|---|---|---|---|
| YOLOX-Nano / Tiny — person detection | `models/yolox_person` | release 0.1.1rc0 | **Apache-2.0** | https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0 | ☑ 2026-08-11 |
| YOLOX-S — person detection (benchmark reference) | `models/yolox_person` | 2022nov | **Apache-2.0** | https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox | ☑ 2026-08-11 |
| YuNet — face detection | *(Phase 8)* | 2023mar | **MIT** — © 2020 Shiqi Yu | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | ☑ 2026-08-11 |
| SFace — face recognition | *(Phase 8)* | 2021dec | **Apache-2.0** | https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface | ☑ 2026-08-11 |
| Fire/smoke detector | *(Phase 7)* | — | **Ours** — trained in-house | see §2 for training data provenance | ☐ |

Every model directory ships a `manifest.json` recording licence, source, SHA-256 per weight file,
and — for in-house checkpoints — the dataset versions and hashes it was trained on, the training
config, and the git commit. Without this there is no way to answer *"which shipped weights are
affected?"* if a dataset's terms later prove different from what is recorded here, which for a
product being sold is the question that actually matters. Enforced by CI (§6.1).

The manifest is also **load-bearing for correctness**, not only for licensing: it records each
model's expected colour order. Megvii's own exports expect BGR while OpenCV Zoo's re-export of the
same architecture expects RGB, and feeding the wrong one degrades recall by roughly a quarter
without raising any error. A model dropped into `models/` without a manifest fails at load.

### Verification record — 2026-08-11

Each `opencv_zoo` model directory carries its own `LICENSE`, and they are **not** all identical, so
each was read directly rather than inferred from the repository's top-level licence:

- `models/object_detection_yolox/LICENSE` → Apache License 2.0
- `models/face_detection_yunet/LICENSE` → **MIT License**, © 2020 Shiqi Yu <shiqi.yu@gmail.com>
- `models/face_recognition_sface/LICENSE` → Apache License 2.0

YuNet being MIT rather than Apache-2.0 is the reason this check mattered. The YOLOX release weights
are covered by the YOLOX repository's Apache-2.0 licence; the ONNX files are published as release
assets of that same repository.

### Deliberately excluded

Recorded so the decision is not silently reversed by a future contributor.

| Excluded | Licence | Reason |
|---|---|---|
| Ultralytics YOLOv5 / v8 / v11 (code + weights) | AGPL-3.0 | Network use or distribution triggers full source disclosure. |
| YOLOv6 (Meituan), YOLOv7 | GPL-3.0 | Copyleft. |
| YOLO-NAS | Apache code, non-commercial weights | Repo badge looks clean; the weights are not. |
| SORT (abewley) | GPL-3.0 | Copyleft. |
| RF-DETR XL / 2XL (`rfdetr_plus`) | PML 1.0 | Nano–Large are Apache-2.0; XL/2XL are not. |
| PyQt5 / PyQt6 | GPL | Browser UI used instead. |
| `python-telegram-bot` | LGPL-3 | Telegram REST called directly with `requests`. |
| Any third-party pretrained fire/smoke checkpoint | AGPL-derived in practice | Almost all are fine-tuned from Ultralytics weights. |

---

## 2. Datasets — used for training

Training data provenance determines whether the resulting weights can be sold. Every dataset used to
produce a shipped model must appear here with its licence and any attribution obligation.

| Dataset | Size | Licence | Attribution required | Source | Downloaded | Verified |
|---|---|---|---|---|---|---|
| D-Fire | 21,527 images / 26,557 boxes | CC0 1.0 Universal *(compilation — see caveat)* | No (citation requested, not required) | https://github.com/gaiasd/DFireDataset | | ☐ |
| FASDD_CV | 95,314 samples | *verify at source* | *verify* | https://doi.org/10.57760/sciencedb.j00104.00103 | | ☐ |
| Site negatives (`data/site_negatives`) | ~500–1000 images | **Ours** — captured on site | n/a | internal — customer premises | | n/a |

> **Action:** FASDD's licence must be confirmed on the Science Data Bank landing page before any
> FASDD-trained weights are shipped. If it carries an attribution requirement, the citation goes in
> §5 below. If the terms turn out to be non-commercial, drop FASDD and train on D-Fire alone.

> **D-Fire CC0 caveat — do not overstate this.** The authors' CC0 declaration covers *their*
> contribution: the compilation and the annotations. The underlying images were web-sourced, and the
> authors could not waive third-party copyright they never held. This is the same situation as COCO
> (which underpins the YOLOX weights we ship), and training on such data is standard industry practice
> and low-risk — but it is **not** the unqualified "clean" that CC0 normally implies, and this
> document should not claim otherwise. Recorded accurately here so that a future reader evaluating
> risk sees the real position.

### Requested citation — D-Fire

Not a licence obligation under CC0, but requested by the authors and worth honouring:

> Pedro Vinícius Almeida Borges de Venâncio, Adriano Chaves Lisboa, Adriano Vilela Barbosa.
> *An automatic fire detection system based on deep convolutional neural networks for low-power,
> resource-constrained devices.* Neural Computing and Applications, 2022.

### Evaluated and not used

| Dataset | Licence | Reason |
|---|---|---|
| Pyro-SDIS | Apache-2.0 | Clean licence, wrong domain — long-range wildfire plumes against sky. May be reconsidered for an outdoor-only variant. Note: Pyronear's *trained models* are YOLOv8/v11-based and must not be used. |
| FLAME1 | CC BY 4.0 | Aerial, classification/segmentation only. |
| DFS (siyuanwu) | **No stated licence** | Unusable — absence of a licence is not permission. |
| Roboflow Universe fire datasets | Mixed / often unspecified | Treat "unspecified" as unusable. Individual CC BY sets may be reconsidered case by case. |

---

## 3. Runtime dependencies — shipped

| Package | Licence | Notes |
|---|---|---|
| `opencv-python` | Apache-2.0 | Bundled FFmpeg binaries are LGPL — dynamically linked, no source obligation on our code, but note it in `THIRD_PARTY_LICENSES.md`. |
| `onnxruntime` | MIT | |
| `numpy` | BSD-3-Clause | |
| `shapely` | BSD-3-Clause | GEOS backend is LGPL, dynamically linked. |
| `supervision` (ByteTrack) | MIT | |
| `fastapi` | MIT | |
| `uvicorn` | BSD-3-Clause | |
| `pydantic` | MIT | |
| `requests` | Apache-2.0 | |
| `PyYAML` | MIT | |
| SQLite | Public domain | Bundled with CPython. |
| `certifi` | **MPL-2.0** | Transitive via `requests`. File-level copyleft — acceptable **unmodified only**, see policy note above. |
| `tqdm` | **MPL-2.0 AND MIT** | Transitive via `supervision`. Same condition as `certifi`. |
| `numpy`, `urllib3`, `charset-normalizer`, `idna`, `annotated-types`, `pydantic-core`, `typing-extensions`, `starlette`, `anyio`, `sniffio`, `h11`, `click`, `colorama`, `defusedxml`, `flatbuffers`, `protobuf`, `sympy`, `mpmath`, `packaging` | MIT / BSD / Apache-2.0 / PSF | Transitive closure. Full inventory published by the CI job on every run. |

Verified against the installed environment on 2026-08-11 (Python 3.14, OpenCV 4.14.0,
onnxruntime 1.28.0, numpy 2.5.2, shapely 2.1.2, supervision 0.30.0). Nothing in the
shipped closure carries GPL, AGPL, LGPL, or a non-commercial term. The only non-permissive
entries are the two MPL-2.0 components above and the dynamically linked LGPL natives in §6.

## 4. Development / training dependencies — not shipped

Not distributed to customers, so obligations are lighter — but still recorded, because a
training-only dependency can taint the *weights* even when it never ships.

| Package | Licence |
|---|---|
| PyTorch | BSD-3-Clause |
| YOLOX (trainer) | Apache-2.0 |
| `pycocotools` | BSD-2-Clause |
| `pip-licenses` (CI gate) | MIT |

---

## 5. Attribution notices reproduced at distribution

Full licence texts required for redistribution live in `THIRD_PARTY_LICENSES.md`.

Apache-2.0 §4 requires retaining copyright, patent, trademark and attribution notices when
redistributing. BSD and MIT require the copyright line and disclaimer. CC0 requires nothing.

- [ ] `THIRD_PARTY_LICENSES.md` created and populated
- [ ] Apache-2.0 full text included once, with a list of the components it covers
- [ ] MIT / BSD copyright lines reproduced per component
- [ ] Any CC BY dataset attribution included, if FASDD or another CC BY set is used

---

## 6. CI enforcement

```
pip-licenses --format=markdown --with-urls \
  --fail-on="GPL;AGPL;LGPL;CC-BY-NC;CC-BY-SA;Proprietary;Unknown"
```

Run on every pull request. Blocking a non-compliant dependency at PR time is free; discovering one
after a customer deployment is not.

Note that `--fail-on="LGPL"` will flag transitively bundled LGPL binaries (FFmpeg via
`opencv-python`, GEOS via `shapely`). These are dynamically linked and acceptable — add explicit
allow-list exceptions with a comment explaining why, rather than removing the check.

Expect `--fail-on="Unknown"` to be noisy: Python package metadata is frequently absent or malformed.
Resolve each case by reading the actual LICENSE file and adding an allow-list entry with the finding
recorded — do not weaken the check to quieten it.

### 6.1 Second gate — assets, not just packages

`pip-licenses` reads **Python package metadata only**. It cannot see an `.onnx` file, a dataset
directory, or a vendored source file — which are precisely the three places this project's licensing
risk actually lives. The dependency gate alone creates false confidence.

A second CI check must fail the build if anything under `models/` or `data/` has no corresponding row
in this file:

```
python tools/check_notice.py --assets models/ data/ --notice NOTICE.md
```

Every model directory and every dataset directory must be listed in §1 or §2 above. An unlisted asset
is a build failure, not a warning.
