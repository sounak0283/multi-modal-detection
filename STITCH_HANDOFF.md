# Perimeter UI Redesign — Integration Brief for Claude Code

## What this is

The Perimeter dashboard (fire, smoke, boundary, PPE, crowd and face-recognition monitoring) has been redesigned in Google Stitch. The export contains one folder per screen, each with static HTML/Tailwind (`code.html`) and a screenshot (`screen.png`). Your job is to **re-skin the existing app** to match these designs **without changing what the app does**.

Inputs placed in the repo by the user:

- `design/stitch/` — unzipped Stitch export (HTML + PNG per screen)
- `design/DESIGN.md` — Stitch's design-system description (colors, type, components)
- this file

## Ground rules

1. **Functionality and APIs stay exactly as they are.** Every existing route, API call, websocket/stream, form field, validation and permission check must keep working. Treat the Stitch HTML as a visual reference, not as source code to paste in.
2. **Do not invent backend features.** Stitch added decorative details that the app doesn't support (see "Ignore list"). Do not build endpoints or fake data for them. If a design element has no real data behind it, leave it out and list it in your final report.
3. **No hard-coded sample data.** Names like "Marcus Vance", "Site Alpha", fake emails, fps numbers, counts etc. in the HTML are placeholders. Bind everything to the real data the current UI already uses.
4. **Componentize.** Build shared primitives once (see "Component inventory") and reuse them on every page instead of copying markup per screen.
5. **Work page by page** in the order below, checking that each page still works before moving to the next. Commit after each page.

## Step 0 — Survey before changing anything

- Identify the frontend framework, styling approach (Tailwind? CSS modules? plain CSS?), router and where each page lives.
- Map each existing page to the Stitch screen in the table below.
- If the app isn't using Tailwind, either add it or translate the tokens into the existing styling system — pick whichever is less disruptive and say which you chose.
- Report the plan before starting large changes.

## Step 1 — Design tokens and theme

Extract from `DESIGN.md` and the Stitch HTML (Tailwind config is usually inlined in a `<script>` in each `code.html`):

> **Important — DESIGN.md conflicts with itself.** Its YAML front-matter is an auto-generated Material palette (e.g. `primary: #c0c1ff`, `surface: #101419`, `error: #ffb4ab`) that does **not** match the written spec in the same file. **The prose sections and the table below are the source of truth.** Use the front-matter only for typography, radii and spacing, which do agree. If a screen's inlined Tailwind config uses the Material values, normalise it to the values below.

Additional values from DESIGN.md prose: Surface Level 2 `#182230`, Level 2 border `#2A3749`, hover border `#2E3D4E`, primary hover `#4F46E5`, text primary `#F1F5F9`, text secondary `#94A3B8`, text muted `#64748B`. Buttons 32px high, table rows 36px, status pills and telemetry chips 20px, card radius 8px, chip radius 4–6px, 4px spacing rhythm, sidebar 240px (56px collapsed), right inspector 360px. Elevation via tonal steps and hairlines, no soft shadows except the one Level 2 shadow.

| Token | Value |
|---|---|
| Background | `#0B0F14` |
| Surface / raised | `#121821` |
| Border (hairline) | `#1F2833` |
| Primary / action | `#6366F1` (indigo) |
| Critical / fire / unauthorised | `#EF4444` |
| Warning / PPE missing / smoke | `#F59E0B` |
| Safe / entry / recognised | `#10B981` |
| Info / crowd | `#06B6D4` |
| UI font | Inter |
| Telemetry font (fps, ms, timestamps, track IDs, camera IDs) | JetBrains Mono |

Severity colors (Critical, High, Medium, Low) and event-pill colors (Entry, Exit, Helmet, Crowd formed, Dispersed, Fire, Smoke) must come from one shared map so they are identical on every page. Dark mode is the default; keep a light theme possible via CSS variables.

## Step 2 — App shell

- Sidebar grouped as **MONITOR** (Live view, Boundaries, Cameras), **REVIEW** (Alert history, Evidence, Recognition log), **CONFIGURE** (Video test, Alerts / Alert rules, People, Accounts, Settings). Keep existing route paths.
- Sidebar footer: real system health (cameras running count, recording status) and the signed-in user with role and Log out — wire to whatever the current footer uses.
- Top bar: page title, camera switcher where the page already has one, and the existing Assistant button (it may open a right-side drawer instead of a floating bubble, but must keep calling the same assistant API).

## Step 3 — Screen mapping

| Existing page | Stitch screen | Notes |
|---|---|---|
| Sign in | Perimeter — Sign In | Email + password only. Keep current auth flow. Tagline and status footer are cosmetic; show the version only if the app already exposes one. |
| Live view | Perimeter — Live View Console | Keep the current stream element and telemetry overlay (fps, ms, tracked, events, camera). Single/grid toggle only if multi-camera view already exists or is trivial to compose from existing streams. Occupancy, recent activity, legend bind to existing data. |
| Boundaries | Perimeter — Boundaries & Zones | Keep the existing polygon/tripwire/exclusion drawing logic and the full property set (model, detect People/Fire/Smoke, alert on Entry/Exit, restrict entry to people, severity, hysteresis, crowd formation, PPE required, active hours, delete). Restyle the inspector; tabs are optional. Keep the PPE "model not installed" notice driven by the real check. |
| Cameras | Perimeter — Cameras | Same fields as today (camera ID locked after creation, display name, Webcam/file vs CCTV/RTSP, device index/URL, resolution, camera fps, decode fps, auto-start, enabled, detection modules, Test connection, Save & reconnect, Remove). Module card states (requires Boundary, PPE model missing, Phone near ear not built) must reflect real backend state. |
| Alert history | Perimeter — Alert History & Triage | Keep natural-language search, KPI counts, date range, type/boundary/camera/limit filters, table columns and evidence thumbnails. Add a per-day chart only if it can be computed from data already fetched. |
| Evidence | Perimeter — Evidence Viewer | Keep list filters, video playback (including the "open in new tab" fallback for old non-H.264 clips), Analyse video, Ask chat with suggested questions, alert details. **Security:** never render the raw S3 path or the presigned URL in the UI. Show a masked path plus "Copy secure link" (with expiry) that copies the URL to the clipboard. |
| Recognition log | Perimeter — Recognition Log | Keep counts, filters and table. The enroll/assign modal is only in scope if People enrollment already has an API; otherwise link "Enroll" to the People page. |
| Video test | Perimeter — Video Test | Keep the existing upload → run → results flow and the "[TEST VIDEO]" tagging. Event timeline and progress only from data the backend already reports. |
| Alerts (rules) | Perimeter — Alert Rules & Policy | Restyle existing settings only. |
| People | Perimeter — People | Card grid + detail drawer over the existing people/face-sample data. |
| Accounts | Perimeter — Accounts | Existing user list, roles and invite/create flow. Show the permissions matrix only if roles/permissions exist in the backend. |
| Settings | Perimeter — Settings | Restyle existing settings sections only. Mask secrets (SMTP passwords, keys, bucket credentials). |

## Component inventory (build once, reuse)

AppShell, Sidebar, TopBar, PageHeader, Card / SectionCard with uppercase label, KPI tile (optional sparkline), StatusDot (with pulse for live), SeverityChip, EventPill, Toggle, SegmentedControl, Chip / ChipGroup (multi-select), Select, TextField (with locked state), DataTable (sticky header, row click, thumbnail cell, mono cells), Drawer, Modal, EmptyState (one line + primary action), Skeleton, VideoTile with telemetry overlay, Timeline with event markers, Toast.

## Ignore list — decorative things Stitch invented

Do not implement these unless the backend already supports them; list any you skip in the final report:

- Facility names, site codes, "Site Alpha", operator names, badge/RFID IDs, fake emails
- NVIDIA Orin / edge node hardware panels, GPU/thermal readouts (unless the backend reports them)
- "Export OSHA Incident Pack", "Assign Safety Officer", "Dismiss as False Positive" actions
- Benchmark JSON export, model registry links, per-model version numbers that aren't real
- Build/version strings like `v4.2.1-PROD`, latency badges, "All systems operational" if there's no health endpoint
- Extra cameras or stream counts beyond what the API returns

## Done means

- Every page matches its Stitch screen in layout, spacing, color and type at desktop width, and stays usable at tablet width.
- All existing flows still work: sign in/out, live stream, drawing and saving boundaries, camera add/edit/test/save/remove, alert filtering and search, evidence playback/analysis/ask, recognition log, video test, people, accounts, settings, assistant.
- No raw presigned URLs, AWS key IDs or secrets are visible anywhere in the UI.
- A short report listing: what changed per page, anything from the designs you skipped and why, and any follow-up backend work that would unlock skipped design elements.
