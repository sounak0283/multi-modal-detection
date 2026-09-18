# Phase B — Auth & RBAC

- **Status:** ✅ Complete
- **Date:** 14 September 2026
- **Plan reference:** [`PLATFORM_EXPANSION_PLAN.md`](../../PLATFORM_EXPANSION_PLAN.md) §8, Phase B
- **Depends on:** Phase A (MongoDB + multi-camera)
- **Blocks:** Phase C (evidence), Phase F/F.1 (identity + allow-list), Phase J (incident
  management) per the plan's sequencing diagram

## 1. Summary

Closed the "no authentication yet" gap flagged since Phase A: every API route now
requires a logged-in session, enforced by a signed, httpOnly cookie (`itsdangerous`, not
a JWT, per the plan's own design) and one of two roles - `admin` (full read/write) and
`operator` (read-only). A single `require_role()` dependency, built once per role
combination and reused across every route, is the one place that enforcement lives.
Camera CRUD, zone editing, user management, and the dev-only alert-simulation endpoint
are admin-only; every read route (cameras, zones, snapshot/stream, events, health) works
for both roles. `bcrypt` and `itsdangerous` - already added to `pyproject.toml` ahead of
this phase - are now actually wired in, and `NOTICE.md` carries their licence rows.

All 320 backend tests pass (297 from before Phase B + 23 new), `ruff` is clean, the
frontend builds cleanly, and the full login → RBAC → logout flow was manually verified
end to end in a real browser against a `mongomock`-backed app (see §6).

## 2. What was implemented

### Backend — new package `backend/src/perimeter/auth/`

| File | Purpose |
|---|---|
| `auth/models.py` | `Role` enum, `User` dataclass, `user_from_dict()` validation (email shape, 8-72 char password length), `to_public_dict()` (never echoes `password_hash`) |
| `auth/passwords.py` | `bcrypt` wrapper - `hash_password()`/`verify_password()`, cost factor overridable for tests |
| `auth/registry.py` | `UserRegistry` - MongoDB CRUD over `users`, structurally identical to `cameras/registry.py:CameraRegistry` |
| `auth/sessions.py` | `SessionManager` wrapping `itsdangerous.URLSafeTimedSerializer` - signs/verifies a token carrying only a user id |

### Backend — changed files

| File | What changed |
|---|---|
| `settings.py` | New `AuthSettings` (`session_secret`, `session_max_age_s`, `cookie_secure`, `admin_bootstrap_email/password`), read from `.env` |
| `main.py` | Generates an ephemeral session secret with a warning if `PERIMETER_SESSION_SECRET` is unset; `_bootstrap_first_admin()` seeds one admin from `PERIMETER_ADMIN_EMAIL`/`PERIMETER_ADMIN_PASSWORD` into an empty `users` collection, mirroring `_bootstrap_first_camera`; fails fast (exit 1) if the collection is empty and no bootstrap credentials are set |
| `api/app.py` | `create_app()` gains a required `users: UserRegistry` parameter; `get_current_user`/`require_role()` dependencies; `/api/auth/{login,logout,me}` and `/api/users*` routes; every existing route now carries `Depends(require_admin)` or `Depends(require_any_role)` |
| `.env.example` | Documents the four new auth variables |
| `NOTICE.md` | Adds `bcrypt` and `itsdangerous` rows to §3 |

### Frontend

| File | What changed |
|---|---|
| `src/pages/Login.jsx` (new) | Email/password form, inline error on failure |
| `src/pages/Users.jsx` (new) | Admin-only account list + detail panel (create/edit role/deactivate/reset password/delete), mirrors `Cameras.jsx`'s shape |
| `src/api.js` | `login`/`logout`/`me`/user CRUD calls; `setUnauthorizedHandler()` so any 401 drops the app back to the login screen, not just the one call that hit it |
| `src/App.jsx` | Boots by calling `/api/auth/me`; renders `<Login>` or `<Dashboard>`; passes `user`/role down for UI gating (save button, add/edit camera, dev-tools) |
| `src/components/Shell.jsx` | Account row (email, role, log out) in the sidebar; `Accounts` nav item shown only for `admin` |
| `src/pages/Cameras.jsx` | New `readOnly` prop hides "+ Add camera", "Save & reconnect", "Remove", and the dev-tools block for non-admins |

## 3. New dependencies

Already present in `pyproject.toml`/`requirements.txt` from before this phase (added
ahead of time, labelled "Auth (Expansion Plan Phase B)"); this phase is what wires them
in and records them in `NOTICE.md`:

| Package | Where | Licence |
|---|---|---|
| `bcrypt>=4.0,<5` | Runtime | Apache-2.0 |
| `itsdangerous>=2.1,<3` | Runtime | BSD-3-Clause |

No new dependencies were added during implementation.

## 4. Design decisions worth recording

- **The session cookie carries only a user id, never a role.** `get_current_user`
  re-reads the user from `UserRegistry` on every request. A role change or account
  deactivation therefore takes effect on the very next request from that session, not
  the next login - verified directly by `test_role_change_takes_effect_on_the_very_next_request`
  and `test_deactivating_a_user_logs_them_out_on_their_next_request`.
- **`SameSite=Lax` instead of a CSRF token scheme.** The app is same-origin,
  single-process, no CORS - Lax already blocks the cross-site POST case CSRF tokens
  exist to stop, so adding a second mechanism would be machinery the threat model
  doesn't need.
- **The last active admin cannot be deleted or demoted/deactivated.** Enforced
  server-side (`409`) in both `PUT` and `DELETE /api/users/{id}` - the one lockout
  footgun worth guarding against explicitly, since there is no other path back into the
  system once it happens.
- **`cookie_secure` defaults `False`.** Matches the existing `127.0.0.1`, no-TLS
  quick-start default - a `Secure` cookie is silently dropped by the browser over plain
  http, which would make login appear to succeed and then every following request look
  logged-out. Documented in `.env.example` to set `true` once a reverse proxy terminates
  TLS.
- **An unset `PERIMETER_SESSION_SECRET` degrades, it doesn't block startup.** A random one is
  generated and logged as a warning; every session is invalidated on the next restart,
  which is a UX cost, not a security hole (a regenerated secret fails safe). Contrast
  with `PERIMETER_ADMIN_EMAIL`/`PERIMETER_ADMIN_PASSWORD`, which **do** block startup when the
  `users` collection is empty - there, degrading gracefully would mean a dashboard
  nobody could ever log into.
- **Operator-role UI gating (hidden Save/Add/Remove buttons) is a UX nicety, not the
  security boundary.** The backend's `require_role()` checks are what actually stop a
  non-admin; the frontend changes just avoid showing a control that would 403 anyway.
  The zone-drawing canvas itself was deliberately left interactive for operators (only
  the Save action is gated) rather than reworking `BoundaryCanvas.jsx`/
  `ZoneProperties.jsx` - out of scope for what this phase needed.

## 5. Errors found and fixed during implementation

- **`ruff`'s B008 rule flagged every `Depends(require_role(...))` call** - `Depends()`
  as an argument default is FastAPI's own idiom, not the mutable-default bug B008 exists
  to catch, but ruff doesn't know that by default. Fixed properly rather than suppressed:
  `require_admin`/`require_any_role` are now built once (as `require_role(*ADMIN_ONLY)`/
  `require_role(*ALL_ROLES)`) instead of called inline at each of the ~20 routes, and
  `fastapi.Depends` was added to `[tool.ruff.lint.flake8-bugbear].extend-immutable-calls`
  for the remaining outer call.
- **Triggered a native `window.confirm()` dialog while manually testing the Users page's
  delete button**, which froze the browser-automation tab (confirm/alert/prompt block
  the CDP command queue). Recovered by abandoning the frozen tab and continuing in a
  fresh one rather than fighting it - `Users.jsx` intentionally reuses the same
  `window.confirm()` pattern `Cameras.jsx` already uses for delete, so this isn't a
  product defect, just a known limitation of the browser-automation tooling.

## 6. Verification

**Automated:**

```
320 passed, 1 skipped in ~38s   (backend/tests/, pytest - 297 prior + 23 new in test_auth.py)
ruff check src/ tests/ → All checks passed!
npm run build → builds cleanly
```

**Manual, in a real browser:** booted the real FastAPI app via a throwaway script
(`tools/_phase_b_smoke.py`, deleted after use - same pattern as Phase A's
`_phase_a_smoke.py`) against a `mongomock`-backed `Database`/registries, seeded one
admin and one operator account, and drove it with the Chrome browser tool:

- Wrong password shows the same generic "invalid email or password" error whether the
  email exists or not.
- Correct admin login lands on Live view; the sidebar shows the account row
  (`admin@example.com` / `Admin`) and an `Accounts` nav item.
- The Accounts page lists both seeded accounts with role badges and a working detail
  panel (email locked, password field write-only, role select, active toggle).
- Logging out and back in as the operator account: `Accounts` nav item is gone, the
  Cameras page shows no "+ Add camera" button and no "Save & reconnect"/"Remove" buttons
  on the detail panel (only "Test connection" remains), and the Boundaries page shows no
  "Save boundaries" action - matching the read-only role exactly.

**Not verified in this phase:** running against the real MongoDB Atlas cluster (Phase
A's `.env` already has working credentials, but this phase's manual pass deliberately
used `mongomock` instead, to avoid writing a bootstrap admin/test accounts into a shared
database from an unattended verification step - see the Phase A precedent for why
`mongomock` is preferred for this kind of pass); rate-limiting login attempts was
considered out of scope and is not implemented.

## 7. Known limitations / carried forward to later phases

- No login rate-limiting or lockout after repeated failed attempts.
- No password-reset-by-email flow - an admin resets another account's password directly
  from the Users page; there is no self-service "forgot password".
- No audit log of who changed what (camera credentials, zone edits, role changes) -
  `PLATFORM_EXPANSION_PLAN.md` doesn't call for one in Phase B, but it would be a natural
  companion to Phase J's incident-management work.
- The boundary-drawing canvas remains fully interactive for operators (only the Save
  action is hidden) rather than being made read-only end to end - see §4.
- Session cookies are not rotated/refreshed on activity (fixed `session_max_age_s` from
  login time, not a sliding window) - matches the plan's scope, not revisited here.
