---
name: Perimeter
colors:
  surface: '#101419'
  surface-dim: '#101419'
  surface-bright: '#36393f'
  surface-container-lowest: '#0a0e13'
  surface-container-low: '#181c21'
  surface-container: '#1c2025'
  surface-container-high: '#262a30'
  surface-container-highest: '#31353b'
  on-surface: '#e0e2ea'
  on-surface-variant: '#c7c4d7'
  inverse-surface: '#e0e2ea'
  inverse-on-surface: '#2d3136'
  outline: '#908fa0'
  outline-variant: '#464554'
  surface-tint: '#c0c1ff'
  primary: '#c0c1ff'
  on-primary: '#1000a9'
  primary-container: '#8083ff'
  on-primary-container: '#0d0096'
  inverse-primary: '#494bd6'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#4edea3'
  on-tertiary: '#003824'
  tertiary-container: '#00885d'
  on-tertiary-container: '#000703'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e1e0ff'
  primary-fixed-dim: '#c0c1ff'
  on-primary-fixed: '#07006c'
  on-primary-fixed-variant: '#2f2ebe'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#101419'
  on-background: '#e0e2ea'
  surface-variant: '#31353b'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 20px
  body-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  telemetry-lg:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 18px
  telemetry-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
  telemetry-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
  tracker-tag:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 0.75rem
  margin: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style
The design system operates at the intersection of mission-critical surveillance, developer-grade precision, and high-density industrial telemetry. Styled after modern control-room consoles with the refined restraint of minimalist developer tools, it avoids decorative clutter in favor of uncompromising data clarity, low cognitive fatigue during extended shifts, and immediate triage velocity.

The aesthetic fuses utilitarian dark-mode engineering with responsive operational optics. It evokes calm vigilance, uncompromising reliability, and institutional authority. Interfaces prioritize information density, rapid scanning paths, hairline delineation, and high-contrast status signifiers designed to remain legible across multi-monitor wall displays and workstation arrays.

## Colors
The palette is built strictly for dark-adapted viewing environments, anchoring visual mass to near-black slate tones while reserving luminous chroma for status, AI inferencing tracks, and operational alerts.

### Surface System
- **Base Canvas (`#0B0F14`)**: Viewport foundation and outer application frame.
- **Surface Level 1 (`#121821`)**: Primary card panels, stream grids, and container modules.
- **Surface Level 2 (`#182230`)**: Elevated dropdowns, active inspector rails, popovers, and interactive hover states.
- **Border / Hairline (`#1F2833`)**: Crisp structural bounding lines between controls and panel subdivisions.

### Signal & Diagnostic Roles
- **Electric Indigo (`#6366F1`)**: Primary focus rings, manual capture triggers, selected controls, system confirmation states.
- **Critical Crimson (`#EF4444`)**: Breached perimeters, thermal runaways, unauthorized access, fire/hazard events.
- **Warning Amber (`#F59E0B`)**: Missing PPE violations, anomalous speed/motion, smoke precursor flags, degraded sensor health.
- **Safe Emerald (`#10B981`)**: Verified clearance, authenticated personnel, nominal telemetry, stable connectivity.
- **Sensor Cyan (`#06B6D4`)**: LIDAR overlays, thermal point-clouds, optical track IDs, ambient telemetry readouts.

### Text Contrast
- **Text Primary (`#F1F5F9`)**: Headings, critical metrics, high-priority labels.
- **Text Secondary (`#94A3B8`)**: UI field labels, inactive column headers, secondary summaries.
- **Text Muted (`#64748B`)**: Telemetry units, disabled elements, hairline divider text.

## Typography
Typographic execution is divided strictly by cognitive mode: natural language comprehension (Inter) versus rapid numeric parsing (JetBrains Mono).

Inter provides compact proportions and neutral geometric glyphs for layouts, navigational landmarks, forms, and analytical narratives. JetBrains Mono is assigned strictly to dynamic telemetry values: camera IDs, latency timestamps, FPS counters, track confidence percentages, coordinates, and RTSP stream specs. 

All technical micro-labels and system tags (`tracker-tag`) must render uppercase with extended tracking (`letterSpacing: 0.08em`) to guarantee legibility when pinned over live video feeds or condensed within dense multi-row inspector sidebars.

## Layout & Spacing
The layout follows a dense, fluid command-center matrix designed to maximize screen real estate across ultra-wide monitors and multi-display wall arrays. 

- **Grid Structure**: Continuous multi-pane canvas. Left vertical control rail (collapsible to 56px icon mode, default 240px width), center multi-stream matrix (dynamic 1×1, 2×2, 3×3, or asymmetric hero layout), and right inspector/telemetry drawer (fixed 360px width).
- **Density Scaling**: A base 4px rhythm governs internal padding and gaps. Internal gaps between video panels use strict 8px (`space-sm`) channels to avoid dead screen space. 
- **Breakpoints**:
  - `Desktop Expanded (≥1680px)`: Full 3-column layout (Rail + Video Stream Matrix + Event Timeline + Telemetry Drawer) fully visible.
  - `Desktop Standard (1280px - 1679px)`: Side inspector switches to an overlaid modal drawer or slide-over layer.
  - `Mobile / Tablet Control (<1280px)`: Stream matrix collapses to single-column feeds with segmented card views and bottom floating telemetry sheets.

## Elevation & Depth
In a professional control dashboard, drop shadows and fuzzy ambient spreads create perceptual mud that interferes with visual vigilance. Elevation is communicated strictly through tonal step-ups and crisp hairline outlines.

- **Level 0 (Canvas Base `#0B0F14`)**: Foundational canvas; absorbs screen glare and anchors the viewport.
- **Level 1 (Panels & Feeds `#121821`)**: Primary bounded regions. Defined by an explicit `1px solid #1F2833` hairline border with 0px blur shadow.
- **Level 2 (Interactive & Modals `#182230`)**: Popovers, contextual menus, tooltips, and elevated tracker inspect panels. Separated by `1px solid #2A3749` and an ultra-subtle, directional drop shadow: `0 4px 12px rgba(0, 0, 0, 0.45)`.
- **Live Bounding HUD**: Bounding boxes drawn over live camera feeds use semi-translucent fills (`rgba(99, 102, 241, 0.12)`) bounded by a high-contrast 1.5px solid stroke with no shadow blurring.

## Shapes
Geometry is disciplined, utilitarian, and restrained. Structural containers, video viewports, inputs, and cards use an 8px (`0.5rem`) corner radius. Internal sub-elements like telemetry chips, segment buttons, and filter pills utilize 4px to 6px radii to maintain nested geometric harmony. 

Status badges and pulse dots are the sole circular exceptions: status dots use full pill geometries (`9999px`) to create clear contrast against the rigid rectangular UI grid.

## Components

### Buttons & Interactive Controls
- **Primary Action**: Solid `#6366F1` background, `#F1F5F9` text, 8px border radius, 32px height for high density. Hover shifts to `#4F46E5`. Active state scales down subtly (`scale(0.98)`).
- **Secondary / Ghost**: Background `#121821`, hairline border `1px solid #1F2833`, text `#94A3B8`. On hover: surface tints to `#182230`, border lightens to `#2E3D4E`, text updates to `#F1F5F9`.
- **Destructive / Emergency Stop**: Solid `#EF4444` background with white text, or an outline variant with `1px solid #EF4444/50` on `#121821` with `#EF4444` text.

### Badges, Pills & Indicators
- **Status Pills**: Compact (20px height), JetBrains Mono 11px uppercase. Built with a tinted dark base (`rgba(color, 0.12)`), a low-opacity border (`rgba(color, 0.3)`), and high-contrast foreground text (`color`).
- **Live Pulse Dot**: A 6px circular indicator preceding live feeds or active detections. Includes an animated concentric ping ring expanding to 14px with fading opacity (`animation: pulse 2s cubic-bezier(0, 0, 0.2, 1) infinite`).

### Telemetry Chips
- Compact micro-containers (`#182230/80` with `1px solid #1F2833`), 20px height, 4px border radius. Pairs an uppercase muted metric label (e.g., `FPS`, `LAT`, `ID`) in `#64748B` with a monospaced value in `#F1F5F9`.

### Segmented Switches
- Inset container in `#0B0F14` with a `1px solid #1F2833` shell. Segment items render flat against the background; active segment shifts to `#182230` with a subtle white keyline (`1px solid #334155`), snapping cleanly without bouncy animations.

### High-Density Data Tables
- Table rows locked to 36px height. Row borders rendered with hairline dividers (`1px solid #182230`). Header cells use 11px JetBrains Mono uppercase tracking in `#64748B`. Row hover invokes immediate background highlight to `#182230/70` with cursor feedback. Numeric values are right-aligned and rendered in JetBrains Mono.

### Live Stream HUD Overlays
- Bounding boxes over AI-detected objects render crisp 1.5px lines with an anchored tracker chip pinned to the top-left vertex. The chip displays the classification label, confidence score, and persistent tracking hash (e.g., `FORKLIFT_04 [98%] #A8F2`).
