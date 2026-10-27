"""How to use the dashboard - what the assistant answers product questions from.

The detection modules and alert types are read from the code (`ALL_MODULES`, the alert
catalogue), so a new module appears here without editing this file. The page guide below
describes the dashboard; keep it in step when a page changes.
"""

from __future__ import annotations

from perimeter.ai.catalog import catalog
from perimeter.cameras.models import ALL_MODULES

PAGES = """\
Dashboard pages (left sidebar). Operators see the first six; admins also see the rest.
- Live view: each camera's live picture with detections and boundaries drawn on it.
  "All cameras" (header) shows every camera in a grid; the header menu picks one camera.
- Boundaries: draw the areas (zones) and lines (tripwires) that raise alerts, per camera.
  Pick the camera in the header, draw on the picture, set the zone's properties on the
  right (name, type, what it detects, schedule), then "Save boundaries" (admin only).
  Changes apply without a restart.
- Cameras: add a camera (webcam/USB device or RTSP URL), test the connection, turn it on
  or off, and choose which detection modules run on it.
- Alert history: every recorded alert. Filter by period (Today / This week / This month /
  Last 6 months / All time), type, boundary and camera; the cards at the top count the
  same filtered window. Click a snapshot to see the evidence. Plain-English search: type
  what you want ("fire alerts on cam_01 last night") in the AI search box.
- Evidence: alerts with a saved video or snapshot. Pick one on the left to watch the clip
  and read every detail (time, camera, type, boundary, person, where it is stored). Below
  the video, "Ask about this alert" chats about that clip; "Analyse video" runs the AI
  video analysis if it has not run yet.
- Recognition log: every face identification (known / unrecognised / no face), with
  camera and time.
- Video test (admin): upload a video and run it through the detectors as if it were a
  live camera. Its alerts are marked [TEST VIDEO]. Can be switched off in Settings.
- Alerts (admin): email rules - who is emailed for which alert types and cameras, and
  the browser alert sound. Fire/smoke/crowd emails go out at once; the video follows.
- People (admin): enrol faces (several poses) so restricted zones know who is allowed.
- Accounts (admin): create sign-ins and choose the role: admin (can change everything)
  or operator (can watch and review, not change settings).
- Settings (admin): system-wide switches, e.g. the Video test section.

AI features:
- Assistant (button at the bottom right of every page): ask about alerts, cameras,
  footage and patterns, or how to use the dashboard. It remembers useful facts.
- Video analysis: fire, smoke, crowd, access and PPE alert clips are analysed
  automatically (any other alert on request) - a summary of what happens in the clip.
- Footage search: find moments across all analysed clips by describing them.
- Daily digest: a summary of yesterday's alerts, emailed each morning to admins.
- Memory: "What I remember" in the assistant lists what it has kept; delete anything
  that is wrong. Site-wide memories can only be deleted by an admin.
"""


def guide() -> str:
    kinds = "\n".join(f"- {row['kind']}: {row['description']}" for row in catalog())
    modules = ", ".join(sorted(ALL_MODULES))
    return f"{PAGES}\nDetection modules a camera can run: {modules}.\n\nAlert types:\n{kinds}\n"
