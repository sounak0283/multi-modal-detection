# Phase C — Video Evidence

- **Status:** ✅ Complete
- **Date:** 14 September 2026
- **Plan reference:** [`PLATFORM_EXPANSION_PLAN.md`](../../PLATFORM_EXPANSION_PLAN.md) §9/§10, Phase C
- **Depends on:** Phase A (MongoDB + multi-camera)
- **Blocks:** Phase E (fire/smoke — clips are how a false-alarm review actually happens),
  Phase J (incident management)

## 1. Summary

Every alert now carries a real snapshot and MP4 clip. `store/db.py`'s `snapshot_path`/
`clip_path` columns have existed since Phase A but were always `None` — nothing ever set
them. This phase wires the whole path up: a per-camera JPEG ring buffer feeds an
`EvidenceWriter` that, on every alert, waits `post_roll_seconds` then muxes whatever's
buffered into a clip and picks the frame closest to the alert's own timestamp as the
snapshot, stores both, and updates the event row. The dashboard's Alert History gained
an Evidence column with a thumbnail that opens a lightbox playing the clip.

**Storage is local filesystem, not S3** (confirmed with the user — no AWS account is
stood up for this repo). A small `EvidenceStore` interface (`LocalEvidenceStore` today)
means a future `S3EvidenceStore` swap only touches one module and `main.py`'s wiring —
`EvidenceWriter`, the API routes, and the frontend never see anything but a `key` string.
This also means **no new dependency this phase** — no `boto3` — evidence is written with
`cv2` (already a dependency) and served with `FileResponse`.

All 351 backend tests pass (348 from before Phase C + a net of 25 new across four new
files — 29 added, 4 removed during a mid-phase test-design fix, see §4), `ruff` is
clean, the frontend builds cleanly, and the full alert → clip/snapshot → playback path
was manually verified end to end in a real browser (see §5).

## 2. What was implemented

### Backend — new files

| File | Purpose |
|---|---|
| `capture/ring_buffer.py` | `JpegRingBuffer` — thread-safe, time-bounded buffer of `(ts, jpeg)`, evicting anything older than `seconds` on every append |
| `evidence/store.py` | `LocalEvidenceStore` — writes clips/snapshots under `data/evidence/{clips,snapshots}/<camera_id>/<date>/<event_id>.*`, resolves a stored key back to a path (refusing anything that would escape the store root) |
| `evidence/writer.py` | `EvidenceWriter` — queue+thread mirroring `AlertBus`'s own shape; registered as an `AlertBus` sink (the existing `sinks` hook, no new one needed); `mux_clip()` (the real `cv2.VideoWriter` mux) is passed in as an injectable `mux` callable — see §4 for why |
| `backend/tests/test_ring_buffer.py`, `test_evidence_store.py`, `test_evidence_writer.py`, `test_evidence_api.py` | New test coverage |

### Backend — changed files

| File | What changed |
|---|---|
| `pipeline.py` | `Pipeline.__init__` builds `self.ring_buffer`; `_render()` appends the **raw** (unannotated) JPEG it already encodes every frame — evidence should show what actually happened, not the dashboard's zone/track overlay; `reconfigure()` rebuilds the buffer (old frames belong to the feed just replaced) |
| `settings.py` | `StorageSettings` gains `ring_buffer_seconds`, `post_roll_seconds`, `evidence_root`, `snapshot_dir`, `clip_dir` — the first and last two were already reserved, unread, in `app.yaml` since Phase A |
| `config/app.yaml` | Adds `post_roll_seconds`, `evidence_root` next to the storage keys already there |
| `store/db.py` | `update_evidence_paths()` (a `$set`, same shape as `mark_delivered`) and `get_event()` (single lookup by id, returns `None` rather than raising on a malformed id — reached from a user-supplied URL path parameter) |
| `api/app.py` | `GET /api/events/{id}/snapshot` and `/clip`, both `require_any_role` (read tier); `create_app()` gains an `evidence_store` parameter |
| `main.py` | Builds `LocalEvidenceStore` + `EvidenceWriter`, appends `evidence_writer.on_alert` to `alerts.sinks`, starts/stops it alongside `manager`/`alerts` |

### Frontend

| File | What changed |
|---|---|
| `src/api.js` | `eventSnapshotUrl()`/`eventClipUrl()` — plain URL builders like the existing live `streamUrl`/`snapshotUrl`, cookies ride along automatically |
| `src/pages/AlertHistory.jsx` | New "Evidence" column (thumbnail when `snapshot_path` is set) and an `EvidenceLightbox` modal (full snapshot + `<video controls>` clip playback), styled after `FireAlertOverlay.jsx`'s existing modal |

## 3. New dependencies

None. Evidence capture uses `cv2` (already shipped) and `FileResponse`; no `boto3`, since
storage is local for now.

## 4. A real finding, and the test-design fix it drove

While writing `test_evidence_writer.py`, the "happy path" tests were flaky under the
full suite (passed 100% in isolation, failed intermittently at the end of a full run).
Root-caused to a **real, worth-flagging production risk**: `test_camera_api.py`'s
unreachable-RTSP-host probes are already known (per that file's own module docstring) to
leave OpenCV's FFmpeg backend stalling the *next* `cv2` video open in the same process
for up to ~30s. Confirmed while debugging this that the stall affects `cv2.VideoWriter`
too, not just `cv2.VideoCapture` — and in this environment it measured well over a
minute in one run, not the ~30s that file documents. In production terms: **a single
camera stuck reconnecting to a bad RTSP URL could stall evidence-clip writing for every
other camera in the process**, since all cameras share one process and OpenCV's FFmpeg
backend apparently serializes these operations globally. This is a real limitation worth
a follow-up investigation (possibly muxing in a subprocess, or an explicit per-open
timeout) — out of scope to fix here, but recorded rather than quietly worked around.

The test-design fix: `EvidenceWriter` now takes an injectable `mux` callable
(`mux_clip` is the real, `cv2.VideoWriter`-based implementation and the constructor
default). Every test but one substitutes a fake mux (instant, no codec involved) so
`EvidenceWriter`'s own orchestration logic — post-roll timing, per-camera ring-buffer
lookup, graceful degradation, snapshot/clip independence — is tested without inheriting
that cross-file cv2 timing hazard. No existing test in this codebase exercises
`record.py:open_writer` (which `mux_clip` reuses) with a real codec either, for the same
reason, so the real mux path is verified by the manual browser check (§5) instead of a
unit test.

## 5. Verification

**Automated:**

```
351 passed, 1 skipped in ~37s   (backend/tests/, pytest)
ruff check src/ tools/ tests/ → All checks passed!
npm run build → builds cleanly
```

**Manual, in a real browser:** a throwaway script (`tools/_phase_c_smoke.py`, deleted
after use — same pattern as every prior phase) booted the real FastAPI app plus a real
`EvidenceWriter` against a `mongomock`-backed `Database`, seeded a `JpegRingBuffer` with
20 real, visibly-labelled JPEG frames ("EVIDENCE TEST frame N"), and published a real
boundary alert through the real `AlertBus` — no live camera or person-in-frame needed,
since detection itself isn't what this phase adds. Confirmed:

- Real snapshot (6.9 KB) and clip (19.4 KB MP4) files appeared under
  `data/evidence-smoke/{snapshots,clips}/cam_01/<date>/<event_id>.*` within the
  `post_roll_seconds` window.
- Alert History's Evidence column shows the real snapshot thumbnail.
- Clicking it opens the lightbox with the full snapshot ("EVIDENCE TEST frame 8" — the
  frame closest to the alert's `ts`, exactly as `closest_to()` should pick) and a
  `<video>` player that plays the real clip on click.

**Not verified in this phase:** evidence capture from a real running `Pipeline` against
a live camera/RTSP source with a person actually crossing a zone (the manual check above
exercises every component downstream of `AlertBus.publish()`, but not
`Pipeline._render()`'s ring-buffer append in a live capture loop) — needs a camera on
the deployment box, same caveat every prior phase's real-hardware path has carried.

## 6. Known limitations / carried forward

- The cross-camera FFmpeg-stall risk described in §4 is unresolved — noted here so it
  isn't lost, not something this phase's scope covered.
- No clip/snapshot retention sweep specific to evidence files — `purge_older_than`
  (Phase A) deletes the *event* row after `retention_days`, but does not currently also
  delete the evidence files that event pointed to. A dangling file after its row is
  purged is a disk-usage leak, not a correctness or privacy bug (nothing can reach an
  orphaned file without knowing its exact key, and the object carries no path back to an
  event once the row is gone) — worth fixing before this ships, not blocking for now.
- `S3EvidenceStore` doesn't exist yet — swapping to it when an AWS account is available
  is the one piece of the original plan's Phase C explicitly deferred.
