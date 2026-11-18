"""Drive the Series panel in a real browser and assert what the user sees.

The Series panel is the one place where a *comparison* is drawn, so it is also
the one place where a wrong scope or a missing provenance line does real
damage.  Static selector checks cannot see either: they pass happily on a
handler that throws the moment it runs.

Run with the gate server up:

    python -m backend.browser_series
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
        box.fill("compare the HOMO-LUMO gap of benzene, pyridine, furan and pyrrole")
        box.press("Enter")

        deadline = time.time() + 420
        while time.time() < deadline:
            cls = page.locator("#seriesContent").get_attribute("class") or ""
            if "hidden" not in cls:
                break
            page.wait_for_timeout(1500)

        content = page.locator("#seriesContent")
        check("series panel is revealed", "hidden" not in (content.get_attribute("class") or ""))

        sub = page.locator("#seriesSub").inner_text() if page.locator("#seriesSub").count() else ""
        check("the panel says how many were calculated", "calculated" in sub, sub[:90])

        # A//B: a comparison drawn without saying what geometry it sits on
        # cannot be reproduced or compared with anyone else's numbers.
        check("the panel names the geometry the numbers sit on",
              "geometry:" in sub, sub[:120])
        check("the geometry is the force-field conformer, not a DFT minimum",
              "MMFF94" in sub and "DFT" not in sub, sub[:120])

        note = page.locator("#seriesNote").inner_text() if page.locator("#seriesNote").count() else ""
        check("the note says the points are single points on those geometries",
              "MMFF94" in note, note[:140])

        # the figure itself
        check("the trend chart has a canvas", page.locator("#seriesChart").count() == 1)
        drawn = page.evaluate(
            "() => { const c = document.getElementById('seriesChart');"
            " return !!(c && c.width > 0 && c.height > 0); }")
        check("the chart canvas has been sized", bool(drawn))

        rows = page.locator("#seriesTable tbody tr").count()
        check("the table lists every molecule", rows >= 4, f"{rows} rows")

        # the property picker must not offer a column with no numbers in it
        opts = page.locator("#seriesProp option").count()
        check("the property picker offers real columns", opts >= 3, f"{opts} options")

        # switching property must redraw without throwing
        if opts >= 2:
            page.select_option("#seriesProp", index=1)
            page.wait_for_timeout(600)
            check("switching property does not raise", not errors, "; ".join(errors[:2]))

        real = [e for e in errors if "favicon" not in e.lower()]
        check("no console errors", not real, "; ".join(real[:3]))

        page.screenshot(path=os.path.join(os.path.dirname(__file__), "shot_series.png"))
        print("  screenshot ->", os.path.join(os.path.dirname(__file__), "shot_series.png"))
        browser.close()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: " + ", ".join(FAILURES))
        return 1
    print("\nseries panel check: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
