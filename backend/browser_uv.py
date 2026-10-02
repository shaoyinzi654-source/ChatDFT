"""Drive the UV-Vis panel in a real browser and assert what the user sees.

This gate exists for two things a static selector check cannot see, and one
of them is the defect the round-14 audit found.

1. The absorption curve is now on an absolute scale, and the pane offers both
   ordinates.  The two views have the *same shape* -- the absolute one is the
   relative one times a constant -- so a broken toggle that redraws the same
   numbers under a new axis label looks completely fine.  Only reading the
   axis title and the plotted values out of the live chart catches it.

2. The window now comes from the states.  Ethylene's first six roots all lie
   below 146 nm, so the old fixed 180-800 nm default drew nothing but
   Gaussian tails at 100% -- a figure that looks like a spectrum and is not
   one.  The pane has to report the window it used and warn when a band is
   cut off.

Run with the gate server up:

    python -m backend.browser_uv
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


def chart_box(page, cid="uvChart"):
    """The canvas's laid-out box, in CSS pixels."""
    return page.evaluate(
        """(cid) => {
            const c = document.getElementById(cid);
            if (!c) return null;
            const r = c.getBoundingClientRect();
            return {w: Math.round(r.width), h: Math.round(r.height)};
        }""", cid)


def chart_state(page, cid="uvChart"):
    """Axis title, dataset label and the plotted y values, from Chart.js."""
    return page.evaluate(
        """(cid) => {
            const c = Chart.getChart(document.getElementById(cid));
            if (!c) return null;
            return {
              ytitle: c.options.scales.y.title.text,
              label: c.data.datasets[0].label,
              data: c.data.datasets[0].data,
              x0: c.data.labels[0],
              x1: c.data.labels[c.data.labels.length - 1],
            };
        }""", cid)


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

        # Ethylene, not naphthalene: its bands are the ones a fixed window
        # would hide, so this is the molecule that exercises the new default.
        box = page.locator("#input")
        box.fill("TD-DFT of ethylene with 6 excited states")
        box.press("Enter")

        deadline = time.time() + 420
        while time.time() < deadline:
            # NB: `or ""` and not `or "hidden"`.  Removing the last class
            # leaves class="", which is falsy, so the obvious-looking default
            # makes this loop run for its full seven minutes and then report
            # the figure as hidden while the chart behind it is drawn.
            cls = page.locator("#uvWrap").get_attribute("class") or ""
            if "hidden" not in cls:
                break
            page.wait_for_timeout(1500)

        cls = page.locator("#uvWrap").get_attribute("class") or ""
        check("the absorption figure is revealed", "hidden" not in cls)

        page.locator('#resultTabs .tab[data-tab="spectrum"]').click()
        page.wait_for_timeout(600)

        rel = chart_state(page)
        check("the absorption chart exists", rel is not None)
        if not rel:
            browser.close()
            print("\n1 FAILURES: chart missing")
            return 1

        # --- the figure has a box, and is not still growing --------------
        # Every chart is built with maintainAspectRatio:false, so the canvas
        # fills its parent; a parent with no fixed height then grows to fit
        # the canvas, and the two feed each other.  Measured before the CSS
        # ceiling existed: the UV-Vis canvas reached 41,667 px tall -- its
        # markup says 240 -- and the results pane 6,870,677 px, with the
        # spectrum drawn as a single vertical line.
        #
        # chart_state() cannot see any of that.  It reads the Chart.js model,
        # and the model stayed perfectly correct throughout: right numbers,
        # right axis title, right window, nothing on screen.  One reading
        # cannot see it either, because early on the canvas is still the size
        # the markup asked for.  Only two readings, some seconds apart.
        box1 = chart_box(page)
        page.wait_for_timeout(4000)
        box2 = chart_box(page)
        check("the chart canvas has a sane box",
              bool(box1) and 120 <= box1["h"] <= 400 and box1["w"] >= 200,
              str(box1))
        check("the chart canvas is not still growing", box1 == box2,
              f"{box1} -> {box2}")

        # --- the window actually contains the bands ---------------------
        note = (page.locator("#uvNote").inner_text()
                if page.locator("#uvNote").count() else "")
        check("the pane reports the plotted window", "Window" in note, note[:140])
        check("the window is the one taken from the states",
              "from the states" in note, note[:140])
        # ethylene's S2 is the bright one, at ~137 nm -- inside a window that
        # comes from the states, far outside the old 180-800 nm default
        check("the window reaches below 180 nm", rel["x1"] < 180.0,
              f"longest wavelength {rel['x1']} nm")
        check("the window starts below 120 nm", rel["x0"] < 120.0,
              f"shortest wavelength {rel['x0']} nm")
        check("no band is reported as cut off", "cut off" not in note.lower(),
              note[:160])

        # --- the sum rule and the convergence caveat --------------------
        check("the pane states the sum rule residual", "\u222b" in note, note[:160])
        check("the pane states how much of the sum rule was captured",
              "Thomas" in note and "sum rule" in note, note[:200])

        # --- the absolute ordinate --------------------------------------
        check("the relative view is labelled as relative",
              "Relative" in rel["ytitle"], rel["ytitle"])
        rel_max = max(rel["data"])
        check("the relative view runs to 100", abs(rel_max - 100.0) < 0.5,
              f"max {rel_max}")

        page.locator('[data-uv-scale="epsilon"]').click()
        page.wait_for_timeout(600)
        abso = chart_state(page)
        check("switching the ordinate keeps a chart", abso is not None)
        if abso:
            check("the axis title switches to molar absorptivity",
                  "absorptivity" in abso["ytitle"], abso["ytitle"])
            check("the dataset label switches too",
                  "absorptivity" in abso["label"], abso["label"])
            check("the same number of points is plotted",
                  len(abso["data"]) == len(rel["data"]),
                  f"{len(abso['data'])} vs {len(rel['data'])}")
            # the whole point: the *shape* is unchanged, the *numbers* are not
            scale = max(abso["data"]) / max(rel["data"])
            check("the absolute ordinate is not the 0-100 one",
                  scale > 1.0, f"ratio {scale:.3g}")
            # ... and every point that carries signal is the relative one times
            # that constant.  Points below 1% of full scale are excluded: the
            # payload rounds the relative array to three decimals, so the tails
            # round to 0.000 while the absolute array keeps a small non-zero
            # value, and comparing those would test the rounding, not the scale.
            sig = [(a, r) for a, r in zip(abso["data"], rel["data"]) if r > 1.0]
            worst = max((abs(a - scale * r) / max(abs(a), 1e-30) for a, r in sig),
                        default=0.0)
            check("the two views differ only by that constant",
                  bool(sig) and worst < 1e-3,
                  f"worst relative deviation {worst:.2e} over {len(sig)} points")
            # ethylene S2 has f = 0.55; with FWHM 0.4 eV the absolute peak is
            # 2.3154e8 * f / (sigma_cm * sqrt(2pi)) ~ 3.7e4 L mol-1 cm-1
            check("the absolute peak is in the L mol-1 cm-1 range a paper prints",
                  1e3 < max(abso["data"]) < 1e6, f"peak {max(abso['data']):.4g}")

        # --- the peak table carries eps_max ------------------------------
        # The header is uppercased by CSS, so inner_text returns "ΕMAX" with a
        # capital epsilon -- an exact match on "eps" would fail on a column
        # that is plainly there.
        head = page.locator("#specTable thead").inner_text().upper()
        check("the transition table has an eps_max column",
              ("\u0395" in head or "EPS" in head) and "MAX" in head,
              head.replace("\n", " | ")[:140])

        # back to the relative view so the exported PNG is the familiar one
        page.locator('[data-uv-scale="relative"]').click()
        page.wait_for_timeout(400)
        back = chart_state(page)
        check("switching back restores the relative axis",
              back and "Relative" in back["ytitle"],
              back["ytitle"] if back else "no chart")

        real = [e for e in errors if "favicon" not in e.lower()]
        check("no console errors", not real, "; ".join(real[:3]))

        shot = os.path.join(os.path.dirname(__file__), "shot_uv.png")
        page.screenshot(path=shot)
        print("  screenshot ->", shot)
        browser.close()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: " + ", ".join(FAILURES))
        return 1
    print("\nUV-Vis panel check: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
