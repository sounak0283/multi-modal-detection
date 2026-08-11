# Third-Party Licences

Full licence texts for components redistributed with this product.

Apache-2.0 §4 requires retaining copyright, patent, trademark and attribution notices
when redistributing. BSD and MIT require the copyright line and disclaimer. CC0 requires
nothing.

**This file is populated before the first release.** The inventory and policy live in
[NOTICE.md](NOTICE.md); this file carries the verbatim texts those licences oblige us to
ship. Keeping them separate means the inventory stays readable while the obligation is
still discharged.

## Status

- [ ] Apache-2.0 full text included once, with the list of components it covers
- [ ] MIT full text + per-component copyright lines
- [ ] BSD-2-Clause / BSD-3-Clause full text + per-component copyright lines
- [ ] Model-specific licences from `opencv_zoo` model directories (verify each — they are
      not all identical, see NOTICE.md §1)
- [ ] Any CC BY dataset attribution, if FASDD or another CC BY set is used

## Components by licence

Populate from `pip-licenses --format=markdown --with-urls` plus the model rows in
NOTICE.md §1. The CI job publishes that inventory on every run.

### Apache-2.0

_Components:_ opencv-python, onnxruntime (see note), requests, YOLOX (architecture and
COCO-pretrained weights), SFace.

> Full Apache-2.0 text to be inserted here.

### MIT

_Components:_ onnxruntime, supervision, fastapi, pydantic, PyYAML, YuNet (verify against
the model directory's own LICENSE file).

> Full MIT text plus per-component copyright lines to be inserted here.

### BSD-3-Clause

_Components:_ numpy, shapely, uvicorn.

> Full BSD-3-Clause text plus per-component copyright lines to be inserted here.

### MPL-2.0

_Components:_ certifi (via requests), tqdm (via supervision).

File-level copyleft. We ship both **unmodified**, which imposes a notice obligation only.
Patching either file triggers a publication obligation on the modified file — see the
policy note in NOTICE.md. Do not vendor or patch these.

> Full MPL-2.0 text to be inserted here.

## Dynamically linked LGPL components

Recorded for transparency. Both are LGPL native libraries bundled inside otherwise
permissive wheels, dynamically linked, imposing no source obligation on our code. They
are the two deliberate `--ignore-packages` exceptions in the CI gate.

| Component | Bundled in | Licence |
|---|---|---|
| FFmpeg | `opencv-python` | LGPL-2.1+ |
| GEOS | `shapely` | LGPL-2.1 |

If either is ever statically linked, or the product ships a modified build, this
position must be re-examined before release.
