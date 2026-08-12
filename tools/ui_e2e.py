#!/usr/bin/env python3
"""Drive the dashboard the way an installer does, and check what actually happened.

Why this exists as a separate tool rather than a pytest case: it needs a running server
with a real camera source, which is an integration environment rather than a unit test.
It is the only test that exercises canvas coordinate mapping and pointer handling, and
it has already earned its place - it caught a transparent hint overlay silently
swallowing clicks in the bottom-left of the frame, so a boundary drawn near that corner
lost a vertex. No API-level test could have seen that.

Usage:
    python -m fsbd.main --source clip.mp4 --port 8081 &
    python tools/ui_e2e.py http://127.0.0.1:8081 ./shots

Requires the dev extra:  pip install playwright && playwright install chromium
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    import requests
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - dev-only tool
    print("needs: pip install playwright requests && playwright install chromium")
    raise SystemExit(2) from None

# Fractions of the frame. Deliberately includes the bottom-left corner, which is where
# the hint overlay sits - that is the regression this tool exists to catch.
POLYGON = [(0.30, 0.55), (0.95, 0.55), (0.95, 0.95), (0.30, 0.95)]
ALERT_TIMEOUT_S = 90


def main(argv: list[str]) -> int:
    base = argv[1] if len(argv) > 1 else "http://127.0.0.1:8081"
    out = Path(argv[2] if len(argv) > 2 else ".")
    out.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on(
            "console",
            lambda m: problems.append(f"console.{m.type}: {m.text}")
            if m.type == "error"
            else None,
        )
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))

        page.goto(base, wait_until="networkidle")
        page.wait_for_timeout(2500)

        # -- draw ----------------------------------------------------------
        page.click('.nav-item[data-view="zones"]')
        page.wait_for_timeout(1500)
        page.click('.tool[data-type="polygon"]')

        box = page.locator("#overlay").bounding_box()
        print(f"canvas: {box['width']:.0f}x{box['height']:.0f}")
        for fx, fy in POLYGON:
            page.mouse.click(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
            page.wait_for_timeout(120)

        page.click("#btn-finish")
        page.wait_for_timeout(600)
        page.fill("#f-name", "Walkway")
        page.wait_for_timeout(300)
        page.click("#btn-save")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "ui_boundary_saved.png"))

        # -- verify what was stored ----------------------------------------
        zones = requests.get(f"{base}/api/zones", timeout=10).json()["zones"]
        if not zones:
            problems.append("save persisted no boundary")
        else:
            points = zones[0]["points"]
            print(f"stored: {[[round(x, 3), round(y, 3)] for x, y in points]}")
            if len(points) != len(POLYGON):
                problems.append(
                    f"expected {len(POLYGON)} vertices, stored {len(points)} - "
                    f"something is swallowing clicks"
                )
            for (want_x, want_y), (got_x, got_y) in zip(POLYGON, points, strict=False):
                if abs(want_x - got_x) > 0.02 or abs(want_y - got_y) > 0.02:
                    problems.append(
                        f"vertex drifted: clicked ({want_x}, {want_y}), "
                        f"stored ({got_x:.3f}, {got_y:.3f})"
                    )

        # -- wait for the engine to react ----------------------------------
        print("waiting for alerts ...")
        deadline = time.time() + ALERT_TIMEOUT_S
        events: list[dict] = []
        while time.time() < deadline:
            events = requests.get(f"{base}/api/events?limit=10", timeout=10).json()["events"]
            if len(events) >= 2:
                break
            time.sleep(3)

        print(f"alerts: {len(events)}")
        for e in events[:6]:
            print(f"  {e['ts']}  {e['message']}  [{e['subtype']}]")
        if len(events) < 2:
            problems.append("no alerts fired after drawing the boundary")

        for view in ("live", "history"):
            page.click(f'.nav-item[data-view="{view}"]')
            page.wait_for_timeout(2000)
            page.screenshot(path=str(out / f"ui_{view}.png"))

        browser.close()

    print("--- problems ---")
    print("\n".join(problems) if problems else "none")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
