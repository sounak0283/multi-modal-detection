# Phase F / F.1 — Identity (face) + restricted-zone allow-list

- **Status:** ✅ Complete
- **Date:** 15 September 2026
- **Plan reference:** `PLAN.md` §7, [`PLATFORM_EXPANSION_PLAN.md`](../../PLATFORM_EXPANSION_PLAN.md) §4.3/§5/§6/§7/§8
- **Depends on:** Phase A (MongoDB), the boundary engine's ENTRY/EXIT events
- **Blocks:** nothing — G/H/I are independent of this phase

## 1. Summary

Every boundary ENTRY/EXIT event carried hardcoded `identity_status=None,
identity_name=None` since the alert-wording layer was written — `alerts/messages.py`'s
`subject_for`/`boundary_message` already knew how to render `known:<name>` /
`unknown_face` / `no_face`, just never received real values. This phase fills that in:
a confirmed ENTRY's person box is cropped, run through YuNet (face detection) then SFace
(128-d embedding), and matched against enrolled people by cosine similarity — all
gated to ENTRY events only, never every frame, per `PLAN.md` §7's own cost/privacy
reasoning.

F.1 builds on top of it: `Zone.authorized_person_ids` (a field reserved on the dataclass
since Phase A but never consumed) is now enforced. Anyone entering a zone that declares
an allow-list, and who isn't a positively-matched member of it — known-but-unlisted,
unknown_face, or no_face alike — raises a second, `critical`-severity `access` alert
alongside (never instead of) the normal ENTRY log, with unconditional delivery via the
existing `cooldown_seconds: 0` bypass (no `CooldownGate` changes needed).

Unlike fire/smoke (Phase E), this phase shipped against **real, downloaded, verified
pretrained weights** — YuNet and SFace from the OpenCV Zoo, both MIT/Apache-2.0, no GPU
training required at all. `docs/MODEL_TRAINING.md` tracks this as one of the "no training
needed" phases.

All backend tests pass (see §5), `ruff` is clean, and the frontend was rebuilt with the
new Persons page and the zone editor's allow-list picker (per the standing rule: any
backend change ships its frontend in the same pass).

## 2. What was implemented

### Backend — new files

| File | Purpose |
|---|---|
| `identity/yunet.py` | `FaceDetector` — wraps `cv2.FaceDetectorYN` (OpenCV's own reference decoder for YuNet's undecoded multi-scale SSD-style output; hand-decoding it was evaluated and rejected during research) |
| `identity/sface.py` | `FaceEmbedder` — wraps `cv2.FaceRecognizerSF`, running `alignCrop()` (YuNet's 5-point landmarks) before `feature()` on every call |
| `identity/gallery.py` | `FaceGallery` — in-memory cosine-similarity matrix over enrolled persons, rebuilt wholesale on any enrollment/removal (`PLATFORM_EXPANSION_PLAN.md` §7's own recommendation against a vector DB at this scale) |
| `identity/resolver.py` | `IdentityResolver` — the single entry point `Pipeline` calls: size-gates on the person box (`min_person_box_px`), crops the top 35% from the full-resolution frame, runs detection + embedding + gallery match, returns `known`/`unknown_face`/`no_face` |
| `models/yunet/`, `models/sface/` | Real downloaded weights + `manifest.json` (SHA256, licence, runtime notes) |
| Five new test files | `test_identity_settings.py`, `test_identity_gallery.py`, `test_identity_resolver.py` (fake detector/embedder doubles, no real cv2 load), `test_identity_pipeline.py` (access-control logic against a real `Pipeline`, `_publish`/`_check_access_control` exercised directly), `test_identity_integration.py` (real YuNet/SFace weights — runs for real, does not always skip, unlike fire/smoke's) |
| `test_persons_api.py` | `/api/persons` CRUD, admin-only writes, embedding never echoed back |

### Backend — changed files

| File | What changed |
|---|---|
| `settings.py` | `IdentitySettings` — the already-present `identity:` app.yaml block, parsed for the first time; `enabled` defaults `False` (biometric processing needs a per-deployment consent story, `PLAN.md` §10.3) |
| `store/db.py` | `insert_person`/`list_persons`/`get_person`/`update_person`/`delete_person` — the `persons` collection and its unique index existed since Phase A, unused until now; delete purges the embedding outright, no soft delete of biometric data |
| `boundary/engine.py` | `zone_by_id()` — small lookup helper (same `_compile()`-adjacent shape as `fire_roi_zone_at`), so the access-control check can get the full `Zone` from a `BoundaryEvent.zone_id` |
| `alerts/messages.py` | `unauthorized_access_message()` — reuses `subject_for` for consistent wording |
| `pipeline.py` | `Pipeline` gains an optional, **shared** `identity_resolver` (one instance across every camera, not per-camera like the fire/smoke detector — the models/gallery are stateless per call and expensive to load); `_process_detection` resolves identity only for confirmed ENTRY events on cameras with the `identity` module enabled; `_publish` threads the result into `boundary_message`; new `_check_access_control` fires the F.1 critical alert |
| `cameras/manager.py` | `PipelineManager` threads `identity_resolver` through to each `Pipeline` it starts |
| `main.py` | `_build_identity_resolver()` — soft-fails (never fatal) when `identity.enabled` is false or the weight files are missing, exactly like Phase E's fire/smoke check; loads the gallery from MongoDB at startup; new `--yunet-model`/`--sface-model` CLI flags |
| `api/app.py` | `GET/POST /api/persons`, `PATCH`/`DELETE /api/persons/{id}` — admin-only writes, any-role reads (matches the zones/users precedent); enrollment decodes a photo server-side and embeds it — the browser never computes or sees a face vector; `/api/health` gained `identity_enabled` |

### Frontend

| File | What changed |
|---|---|
| `pages/Persons.jsx` (new) | Admin-only enrolment page: list + toggle-active + remove, and a name+photo enrolment form (base64 upload, server-side embedding) — mirrors `Users.jsx`'s list/detail-panel shape |
| `components/ZoneProperties.jsx` | New "Restrict entry to" checklist (fetches `/api/persons` once per mount), shown whenever a zone emits events; empty selection means "allow anyone", matching the backend's `authorized_person_ids` semantics |
| `components/Shell.jsx` | New "People" nav entry (admin-only, alongside Alerts/Accounts) |
| `App.jsx` | Routes to `Persons`, passes `health.identity_enabled` through so the page can explain *why* enrolment is disabled rather than just hiding it |
| `api.js` | `listPersons`/`createPerson`/`updatePerson`/`deletePerson` |

`Cameras.jsx`'s `identity` module toggle and `cameras/models.py`'s `MODULE_IDENTITY`
already existed before this phase (reserved scaffolding) — nothing needed there.

## 3. New dependencies

None — `cv2.FaceDetectorYN`/`cv2.FaceRecognizerSF` ship in the `opencv-python` version
already pinned for the person/fire-smoke detectors.

## 4. Design notes

- **Why a shared resolver, not one per camera:** the fire/smoke detector is built once
  per `Pipeline` because its weights are typically small and per-camera state-free is
  cheap either way; the identity models plus an in-memory gallery matrix are heavier to
  construct and are entirely stateless with respect to which camera calls them, so one
  instance serves every camera in the process. `main.py` builds it once and
  `PipelineManager` threads the same object into every `Pipeline`.
- **Why access control is a second, separate alert:** `PLATFORM_EXPANSION_PLAN.md` §4.3
  is explicit that the normal ENTRY event still logs — F.1 adds a parallel critical
  alert, it does not replace or suppress the ordinary boundary log.
- **`no_face` is never rendered as "intruder":** `alerts/messages.py:subject_for`
  already enforced this (Phase D-era code, written ahead of the models existing);
  `no_face` reads as "Someone", identical to having no identity layer at all — outdoors
  at a gate this is the common case, not an anomaly.

## 5. Verification

**Automated:**

```
464 passed, 4 skipped in ~37s   (backend/tests/, pytest)
ruff check src/ tools/ tests/ → All checks passed!
```

One of the four skips (`test_embedding_dimension_is_128`) is expected: it looks for a
face in a synthetic gradient image (no real photograph ships in this repo), and YuNet
correctly finds nothing — itself a useful confirmation that the detector doesn't
hallucinate faces in noise. The other three skips are the pre-existing fire/smoke and
person-model-integration skips, unrelated to this phase.

**Manual:** `frontend npm run build` succeeded; `/api/persons` exercised via
`TestClient` (see `test_persons_api.py`) covering enrolment, listing without leaking the
embedding, deactivation and deletion, and the 503/403/400 edge cases.

## 6. Known limitations / carried forward

- No real face photograph was available in this environment to demonstrate an actual
  `known` match end-to-end against live camera footage — the resolver's gating,
  gallery-matching, and access-control logic are fully covered by unit/integration
  tests against the real weights (blank-frame and synthetic-crop cases), but a live
  recognition accuracy check is a physical, on-site verification step, same caveat
  Phase E's fire/smoke carries for its own accuracy.
- The process-wide `identity.enabled` switch requires a restart to take effect (same
  posture as the fire/smoke model path) — there is no live toggle from the dashboard.
- Retention (`storage.retention_days`) applies to event/evidence records as before;
  enrolled photos themselves are never stored, only the derived embedding.
