---
name: Perimeter Light Safety Console
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#464554'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#767586'
  outline-variant: '#c7c4d7'
  surface-tint: '#494bd6'
  primary: '#4648d4'
  on-primary: '#ffffff'
  primary-container: '#6063ee'
  on-primary-container: '#fffbff'
  inverse-primary: '#c0c1ff'
  secondary: '#6b38d4'
  on-secondary: '#ffffff'
  secondary-container: '#8455ef'
  on-secondary-container: '#fffbff'
  tertiary: '#b10e6b'
  on-tertiary: '#ffffff'
  tertiary-container: '#d23284'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e1e0ff'
  primary-fixed-dim: '#c0c1ff'
  on-primary-fixed: '#07006c'
  on-primary-fixed-variant: '#2f2ebe'
  secondary-fixed: '#e9ddff'
  secondary-fixed-dim: '#d0bcff'
  on-secondary-fixed: '#23005c'
  on-secondary-fixed-variant: '#5516be'
  tertiary-fixed: '#ffd9e4'
  tertiary-fixed-dim: '#ffb0cd'
  on-tertiary-fixed: '#3e0022'
  on-tertiary-fixed-variant: '#8c0053'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  display-sm:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.02em
  telemetry-lg:
    fontFamily: JetBrains Mono
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.02em
  telemetry-md:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
  telemetry-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-lg: 1.5rem
  margin: 1.5rem
  margin-sm: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system delivers a high-trust, alert-optimized, and joyful control interface for modern industrial safety and perimeter monitoring. While industrial consoles traditionally default to austere dark rooms and clinical utilitarianism, this system balances mission-critical clarity with an approachable, fatigue-reducing daytime aesthetic.

The visual style blends crisp modern SaaS minimalism with purposeful chromatic vitality:
- **Atmospheric Calm:** Soft tinted neutrals ease eye fatigue across 12-hour shifts while preserving daylight readability.
- **Chromatic Precision:** Rich semantic accents and energetic navigational tiles provide immediate affordance, spatial orientation, and triage speed without creating visual chaos.
- **Tactile Softness:** Fluid micro-radii, crisp hairline dividers, and whisper-soft atmospheric shadows yield a dependable, modern tool feel rather than an archaic SCADA utility.

## Colors

The palette establishes strict hierarchical zones: neutral canvases for sustained focus, radiant brand transitions for identity moments, and dedicated high-contrast semantic tones for situational awareness.

### Brand Gradient
- **Perimeter Aura:** Linear 135° from `#6366F1` (Indigo) through `#8B5CF6` (Violet) to `#EC4899` (Pink). Reserved for high-value actions, global state indicators, and visual orientation bars.

### Core Canvas & Surfacing
- **App Canvas:** `#F7F8FC`
- **Surface Elevation (Cards/Sheets):** `#FFFFFF`
- **Hairline Divider / Border:** `#E6E8F0`
- **Subtle Surface Tint:** `#EEF2F6`

### Text & Content Hierarchy
- **Text Primary:** `#0F172A` (Slate 900)
- **Text Secondary:** `#475569` (Slate 600)
- **Text Muted / Placeholder:** `#94A3B8` (Slate 400)
- **Text Inverse:** `#FFFFFF`

### Operational Status (Semantics)
- **Critical / Breach:** `#F43F5E` (Rose 500) | Soft fill: `#FFE4E6`
- **Warning / Degraded:** `#F59E0B` (Amber 500) | Soft fill: `#FEF3C7`
- **Safe / Secure:** `#10B981` (Emerald 500) | Soft fill: `#D1FAE5`
- **Info / Diagnostic:** `#0EA5E9` (Sky 500) | Soft fill: `#E0F2FE`

### Navigation & Metric Pastel Accents
- **Live View:** Foreground `#4F46E5`, Background `#EEF2FF`
- **Boundaries:** Foreground `#7C3AED`, Background `#EDE9FE`
- **Cameras:** Foreground `#0284C7`, Background `#E0F2FE`
- **Alert History:** Foreground `#E11D48`, Background `#FFE4E6`
- **Evidence Vault:** Foreground `#D97706`, Background `#FEF3C7`
- **Recognition Log:** Foreground `#0D9488`, Background `#CCFBF1`
- **Video Diagnostics:** Foreground `#DB2777`, Background `#FCE7F3`
- **Personnel & Access:** Foreground `#059669`, Background `#D1FAE5`
- **User Accounts:** Foreground `#2563EB`, Background `#DBEAFE`
- **Configuration:** Foreground `#475569`, Background `#F1F5F9`

## Typography

The type system implements a strict dual-family strategy:
- **Inter** handles visual infrastructure, titles, layout framing, and readable prose. Neutral grotesque proportions guarantee high legibility across dense dashboard configurations.
- **JetBrains Mono** is enforced for all numerical telemetry, zone identifiers, IP addresses, RTSP latency logs, and temporal timestamps. This ensures optical alignment and prevents layout shifting during live telemetry streams.

Headings feature tight tracking to maintain solid visual weight, while labels and telemetry counters maintain open spacing for rapid scanning under high operational stress.

## Layout & Spacing

The console relies on a fluid grid bounded by rigid navigational anchor panels. 
- **Fixed Sidebar Utility:** A consistent 72px collapsed / 240px expanded navigation column anchoring high-priority module switches.
- **Fluid Operations Canvas:** A 12-column adaptive layout built with 1rem gutters for standard screen configurations (laptop / field tablets) expanding to 1.5rem on fixed monitoring displays (1440px and above).
- **Component Rhythms:** Internal padding for container cards adheres to `space-lg` (1rem) on operational widgets and `space-xl` (1.5rem) on analytical overviews. Data-dense tables and telemetry rows compress to `space-sm` (0.5rem) vertical padding for maximum information visibility above the fold.

## Elevation & Depth

Visual hierarchy uses flat optical separation layered with low-amplitude shadows rather than dramatic skeuomorphic depth.

- **Level 0 (Floor):** `#F7F8FC` baseline canvas.
- **Level 1 (Card & Module Shells):** `#FFFFFF` surfaces bounded by a crisp 1px `#E6E8F0` border and an ambient foundation shadow: `0 1px 2px rgba(16, 24, 40, 0.06)`.
- **Level 2 (Hover & Active Contexts):** `#FFFFFF` cards gaining subtle focus through `0 4px 6px -1px rgba(16, 24, 40, 0.08), 0 2px 4px -2px rgba(16, 24, 40, 0.04)`.
- **Level 3 (Flyouts, Command Palettes & Modals):** Pure white container framed with `#E6E8F0` and elevated using `0 12px 24px -4px rgba(16, 24, 40, 0.12)`.
- **Alert Pulse Overlay:** During critical breach events, elevation is accompanied by an internal 2px inset border ring (`#F43F5E`) rather than heavy exterior drop shadows to preserve grid order.

## Shapes

The interface standardizes on an intentional 12px (`0.75rem`) border radius for all card enclosures, viewport containers, and modal layers. 

- **Containers & Cards:** 12px corner radius.
- **Interactive Controls (Inputs, Standard Buttons):** 8px corner radius (`space-sm` + `space-xs`), providing a distinct nested aesthetic inside 12px modules.
- **Badges, Status Pills & Nav Tiles:** Fully rounded pill forms (`9999px`) for contextual statuses; squircle tiles (10px radius) for sidebar navigation symbols.

## Components

### Buttons
- **Brand Primary:** Background styled with the linear gradient (`#6366F1` to `#EC4899`), white Inter semi-bold text, 8px border radius, with zero border. On hover, apply 92% opacity and subtle brightness lift.
- **Neutral Secondary:** `#FFFFFF` background, 1px `#E6E8F0` border, `#0F172A` text. Hover brings `#F8FAFC` background.
- **Critical Action:** `#F43F5E` background, `#FFFFFF` text. Used exclusively for perimeter lockdowns, siren triggers, or arming overrides.

### KPI & Diagnostic Tiles
- Constructed on `#FFFFFF` surfaces with 12px radius and the standard 1px border.
- Include a 40x40px rounded pastel icon container at top-left:
  - Sky (`#E0F2FE` bg / `#0284C7` icon)
  - Violet (`#EDE9FE` bg / `#7C3AED` icon)
  - Coral (`#FFE4E6` bg / `#E11D48` icon)
  - Amber (`#FEF3C7` bg / `#D97706` icon)
- Metrics rendered in bold display sizing with JetBrains Mono delta indicators (`+2.4%`, `-12ms`).

### Navigation Tiles
- Compact 44x44px square tiles with 10px rounded corners.
- Rendered in thematic soft pastels with matching saturated icons. Active state introduces a 2px left brand-gradient indicator and subtle `#6366F1` outline.

### Input Fields & Selects
- Height 40px, 8px radius, background `#FFFFFF`, border 1px `#E6E8F0`.
- Text styled with Inter 14px `#0F172A`; placeholder `#94A3B8`.
- Focus ring: 3px soft indigo halo `rgba(99, 102, 241, 0.15)` with a `#6366F1` border.

### Telemetry Badges & Chips
- Height 24px, pill-shaped (`9999px`), internal horizontal padding 8px.
- Uses JetBrains Mono 11px uppercase typography.
- Status styles map to semantic fills: Safe (`#D1FAE5` bg / `#065F46` text), Warning (`#FEF3C7` bg / `#92400E` text), Critical (`#FFE4E6` bg / `#9F1239` text).

### Live Video & Map Viewport Cards
- Framed in 12px white shell with an inset 1px inner border.
- Stream overlays (FPS, Camera UID, Zone Name) anchored using translucent `#0F172A` pills with white JetBrains Mono text and 4px backdrop blur.