# Phase D — Email Alerting

- **Status:** ✅ Complete
- **Date:** 15 September 2026
- **Plan reference:** [`PLATFORM_EXPANSION_PLAN.md`](../../PLATFORM_EXPANSION_PLAN.md) §9/§10, Phase D
- **Depends on:** Phase A (MongoDB), Phase C (evidence — shares `AlertBus.sinks`' pattern)
- **Blocks:** Phase F/F.1 (identity + allow-list — `cooldown_seconds: 0` critical-alert
  bypass exists now, ready for an unauthorized-access rule), Phase J (incident management)

## 1. Summary

Alerts now reach an inbox, not just the dashboard: an SMTP sink, routed and throttled by
per-rule severity floor and cooldown, plus a configurable browser alert sound. Built in
two passes. The first read routing rules from `config/app.yaml` at startup (a static,
restart-to-change file, matching this app's "engineering constant" convention). The user
then asked for this to be dashboard-managed instead — add/edit recipients, severity and
cooldown from a page in the UI, same as cameras/zones/accounts, plus a system sound
setting — which meant moving rules into MongoDB, editable live, the same shift zones went
through in Phase A. The revision kept everything reusable from the first pass
(`AlertRule`, `sink_rule_from_dict`, `CooldownGate`, `EmailSink`) and replaced only the
storage/wiring layer.

All 403 backend tests pass (376 from before this phase + a net 27 new, after removing 2
tests for a function deleted in the revision), `ruff` is clean, the frontend builds
cleanly, and the full save → persist → role-enforcement → validation path was verified
against a real running instance via `curl` (browser automation was unavailable this
session — see §5).

## 2. What was implemented

### Backend — new files

| File | Purpose |
|---|---|
| `alerts/cooldown.py` | `AlertRule`/`AlertConfig` + `sink_rule_from_dict`/`alert_config_from_dict` validators; `CooldownGate` (severity floor + per-(rule,camera,kind,zone) cooldown, `cooldown_seconds: 0` bypasses) |
| `alerts/config_store.py` | `AlertConfigStore` — MongoDB-backed, one document (`alert_config` collection), structurally copied from `boundary/zones.py:ZoneStore` (throttled reload, reject-bad-document-keep-previous) |
| `alerts/router.py` | `AlertRouter` — the one callable registered on `AlertBus.sinks`; reads live rules from the store on every dispatch rather than `AlertBus.sinks` ever being mutated after startup |
| `alerts/sinks/smtp.py` | `EmailSink` — its own queue+thread (mirrors `evidence/writer.py:EvidenceWriter`, since a hung SMTP connection would otherwise stall `AlertBus`'s one notify loop for every other sink); injectable `send` for testing, matching `EvidenceWriter`'s injectable `mux` |
| Five new test files | `test_cooldown.py`, `test_email_sink.py`, `test_alert_config_store.py`, `test_alert_router.py`, `test_alert_config_api.py` |

### Backend — changed files

| File | What changed |
|---|---|
| `settings.py` | `AlertSettings` — SMTP credentials only (`.env`); routing rules live in MongoDB, not settings |
| `main.py` | Builds `AlertConfigStore` + `EmailSink` (if `PERIMETER_SMTP_HOST` set) + `CooldownGate` + `AlertRouter`; appends `router.dispatch` to `alerts.sinks`; stops `EmailSink` in the shutdown `finally` block |
| `api/app.py` | `GET`/`PUT /api/alert-config` (`require_any_role` / `require_admin`); `create_app()` gains `alert_config_store` |
| `config/app.yaml` | The now-empty `alerts:` block removed — a comment points at where routing config moved (MongoDB/dashboard) |
| `.env.example` | SMTP section extended (`_FROM`, `_USE_TLS`), documented as required alongside a dashboard rule for email to actually go out |

### Frontend

| File | What changed |
|---|---|
| `src/pages/Alerts.jsx` (new) | Admin-only. System sound card (`sound_enabled` toggle + `sound_min_severity` select, saved immediately) + email rules list/detail panel (recipients, severity floor, cooldown, enabled) — one-document "hold the full list, PUT it whole" pattern like `Boundaries.jsx`/zones, not per-item CRUD like `Cameras.jsx`/`Users.jsx` |
| `src/components/AlertSoundPlayer.jsx` (new) | Always mounted; polls `/api/events/live` + `/api/alert-config`, plays a Web Audio beep (plain oscillator, no audio asset) on a new alert meeting the configured severity floor, dedup'd the same way `FireAlertOverlay.jsx` already dedups |
| `src/api.js` | `getAlertConfig()`/`saveAlertConfig()` |
| `src/components/Shell.jsx` | `Alerts` nav item, admin-only, with a new inline `BellIcon` |
| `src/App.jsx` | Mounts `<AlertSoundPlayer/>`, routes the `alerts` view |

## 3. New dependencies

None. `smtplib`/`email.message` are Python stdlib; the Web Audio API needs no library.

## 4. Design decisions worth recording

- **One document, not per-item CRUD, for alert rules.** Rules are edited together as a
  small set in one page — the same reasoning that already puts a camera's zones in one
  `ZoneStore.save(zones, camera_id)` call rather than per-zone CRUD. `AlertConfigStore`
  is a near-literal copy of `ZoneStore`'s shape for this reason.
- **`AlertBus.sinks` is never mutated after startup.** `AlertRouter.dispatch` (the one
  callable actually registered) re-reads current rules from `AlertConfigStore` on every
  call — `reload()` is throttled internally, so this is cheap, the same "call it
  liberally" pattern `BoundaryEngine`/the API routes already use against `ZoneStore`.
  This sidesteps ever needing a lock around `AlertBus.sinks` itself.
- **Cooldown key deliberately excludes `subtype`.** Entry and exit of the same zone share
  one cooldown window — a busy door's constant entry/exit churn is exactly the "routine
  boundary crossing" traffic the plan says a cooldown should throttle as one stream.
- **`EmailSink` gets its own thread, matching `EvidenceWriter`'s precedent from Phase C.**
  A synchronous send inside `AlertBus._notify()`'s loop would let a hung SMTP connection
  stall every other sink and every subsequent alert.
- **The browser alert sound is a convenience, not the backstop.** Documented in the UI
  copy itself: autoplay needs a prior user gesture (a known, unfixed browser limitation
  the plan's own risk register already calls out) — email is the reliable, always-on
  channel.
- **Rules aren't seeded from `.env`.** Unlike the first camera/first admin bootstrap, an
  empty rule list is a perfectly valid starting state — nothing to seed.

## 5. Verification

**Automated:**

```
403 passed, 1 skipped in ~35-40s   (backend/tests/, pytest)
ruff check src/ tools/ tests/ → All checks passed!
npm run build → builds cleanly
```

**Manual, against a real running instance, via `curl`** (browser automation was
unavailable this session — the MCP connection had dropped — so this substitutes direct
API calls for what would otherwise be a browser walkthrough):
`tools/_phase_d_smoke.py` (deleted after use) booted the real app against a
`mongomock`-backed database with a seeded admin and operator account. Confirmed: admin
login works; `GET /api/alert-config` returns the empty-list defaults
(`sound_enabled: true`, `sound_min_severity: "high"`); `PUT` with a two-recipient smtp
rule saves and a follow-up `GET` reflects it exactly, including a server-generated rule
`id`; an operator account can `GET` (200) but not `PUT` (403); a rule with an empty `to`
list is rejected with a clean `400` and the actionable message
(`"alert sink: type 'smtp' requires a non-empty 'to' list"`), not a 500.

**Not verified this phase:** an actual email arriving in an inbox (needs real SMTP
credentials, which the user will supply separately per their own answer earlier in this
work) and the browser sound/UI walkthrough (needs browser automation, unavailable this
session — worth a follow-up pass once it's back).

## 6. Known limitations / carried forward

- Only `type: "smtp"` is implemented; `telegram`/`mqtt`/`webhook` remain reserved,
  rejected-with-a-clear-message types in `sink_rule_from_dict`.
- No rate limit on `PUT /api/alert-config` itself (an admin mistake or a compromised
  admin session could spam-save) — not flagged as a real risk given the existing
  admin-only auth boundary, but noted for completeness.
- The browser sound has no per-user mute/volume control, only the one global
  `sound_enabled`/`sound_min_severity` pair — a per-viewer preference (e.g. `localStorage`)
  was considered and skipped as unnecessary scope for this phase.
