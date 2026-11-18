"""Drive the Scan panel in a real browser and assert what the user sees.

The point of this gate is one sentence: the lowest point of a sampled curve is
a sample, not the bottom of the well.  On a 12-point grid with a 0.052 A step
the sampled minimum jumps by 0.026 A when the grid slides half a step, while
the parabola vertex moves by 0.0003 A.  The panel now prints both, and this
check is what stops a later refactor from quietly printing only the sample
again -- which is exactly how the number came to be quoted to three decimals
in the first place.

Static selector checks cannot see it: they pass on a handler that throws the
moment it runs, and on a note that is never populated.

Run with the gate server up:

    python -m backend.browser_scan
"""
from __future__ import annotations

import os
import time

from playwright.sync_api import sync_playwright

BASE = os.environ.get("CHATDFT_URL", "http://127.0.0.1:8000")
FAILURES: list = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1500, "height": 980})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE, wait_until="networkidle", timeout=60000)
        check("page loads", page.title() != "")

        # reach it the way a user does, through the composer
        box = page.locator("#input")
        box.fill("scan the O-H bond of water")
        box.press("Enter")

        deadline = time.time() + 420
        while time.time() < deadline:
            cls = page.locator("#scanContent").get_attribute("class") or ""
            if "hidden" not in cls:
                break
            page.wait_for_timeout(1500)

        content = page.locator("#scanContent")
        cls = content.get_attribute("class") or ""
        check("scan panel is revealed", "hidden" not in cls)

        # the figure itself
        check("the curve has a canvas", page.locator("#scanChart").count() == 1)
        drawn = page.evaluate(
            "() => { const c = document.getElementById('scanChart');"
            " return !!(c && c.width > 0 && c.height > 0); }")
        check("the chart canvas has been sized", bool(drawn))

        rows = page.locator("#scanTable tbody tr").count()
        check("the table lists the sampled points", rows >= 8, f"{rows} rows")

        # --- the reason this gate exists --------------------------------
        note = page.locator("#scanNote").inner_text() if page.locator("#scanNote").count() else ""
        shown = "hidden" not in (
            page.locator("#scanNote").get_attribute("class") or "")
        check("the panel prints a note under the curve", bool(note) and shown,
              note[:120])
        check("the note gives the fitted minimum, not just the sample",
              "parabola vertex" in note.lower(), note[:140])
        check("the note still gives the lowest sampled point",
              "sampled point" in note.lower(), note[:140])
        check("the note states the grid step", "grid step" in note.lower(),
              note[:140])
        check("the note does not let a rigid scan pass for an equilibrium",
              "rigid" in note.lower() and "optimisation" in note.lower(),
              note[:160])

        # the narration carries the same caveat, since the figure is read
        # next to it
        body = page.locator("body").inner_text()
        check("the narrative quotes a grid step too", "sampled every" in body,
              "narration")

        real = [e for e in errors if "favicon" not in e.lower()]
        check("no console errors", not real, "; ".join(real[:3]))

        page.screenshot(path=os.path.join(os.path.dirname(__file__), "shot_scan.png"))
        print("  screenshot ->", os.path.join(os.path.dirname(__file__), "shot_scan.png"))
        browser.close()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: " + ", ".join(FAILURES))
        return 1
    print("\nscan panel check: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
