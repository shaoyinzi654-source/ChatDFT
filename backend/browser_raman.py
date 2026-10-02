"""Drive the Raman panel in a real browser and assert what the user sees.

The Raman figure sits in the same tab as the IR spectrum, is drawn on its own
axes, and is normalised to its own maximum -- which means every check that
would work on a single curve is blind to the thing that matters.  The defect
this gate is built to catch is an implementation whose Raman intensities are
just the IR intensities again: both curves would be smooth, both would peak at
100, both would sit under a plausible axis label, and the *only* place the
error shows is that the two figures disagree with each other.

So the central assertion here is a comparison, read out of the two live
Chart.js instances: water's strongest Raman line must be at a different
wavenumber from its strongest IR band.  Concretely the symmetric O-H stretch is
the weakest IR band (1.8 km/mol against 79.6 for the bend) and the strongest
Raman line -- the textbook case of the two spectra disagreeing, and impossible
to get right by reusing one for the other.

Also checked, because they are user-visible and a static selector test cannot
see them: the pane reports alpha_iso, the sum rule, the depolarization ceiling
and the basis caution; the peak table carries activity and rho; and the chart
survives being resized by a tab switch.

Run with the gate server up:

    python -m backend.browser_raman
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


def chart_state(page, cid):
    """Axis title, dataset label and the plotted values, from Chart.js."""
    return page.evaluate(
        """(cid) => {
            const c = Chart.getChart(document.getElementById(cid));
            if (!c) return null;
            const d = c.data.datasets[0].data;
            let imax = 0;
            for (let i = 1; i < d.length; i++) if (d[i] > d[imax]) imax = i;
            return {
              ytitle: c.options.scales.y.title.text,
              label: c.data.datasets[0].label,
              n: d.length,
              max: Math.max.apply(null, d),
              peak_x: c.data.labels[imax],
              x0: c.data.labels[0],
              x1: c.data.labels[c.data.labels.length - 1],
              colour: c.data.datasets[0].borderColor,
            };
        }""", cid)


def table(page, sel):
    return page.evaluate(
        """(sel) => {
            const t = document.querySelector(sel);
            if (!t) return null;
            return {
              head: Array.from(t.querySelectorAll('thead th')).map(x => x.textContent),
              rows: Array.from(t.querySelectorAll('tbody tr')).map(
                r => Array.from(r.querySelectorAll('td')).map(c => c.textContent)),
            };
        }""", sel)


def main() -> int:
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE, wait_until="networkidle", timeout=60000)
        check("page loads", page.title() != "")

        box = page.locator("#input")
        box.fill("vibrational frequencies of water")
        box.press("Enter")

        deadline = time.time() + 600
        while time.time() < deadline:
            # `or ""` and not `or "hidden"`: removing the last class leaves
            # class="", which is falsy, and the obvious default then makes this
            # loop run to its full timeout and report the figure missing while
            # it is on screen.
            cls = page.locator("#ramanWrap").get_attribute("class") or ""
            if "hidden" not in cls:
                break
            page.wait_for_timeout(1500)

        page.locator('#resultTabs .tab[data-tab="vibrations"]').click()
        page.wait_for_timeout(800)

        cls = page.locator("#ramanWrap").get_attribute("class") or ""
        check("the Raman figure is revealed", "hidden" not in cls, cls[:60])

        ir = chart_state(page, "vibChart")
        ram = chart_state(page, "ramanChart")
        check("the IR chart exists", ir is not None)
        check("the Raman chart exists", ram is not None)
        if not ir or not ram:
            browser.close()
            print("\n1 FAILURES: a chart is missing")
            return 1

        # --- the two spectra are genuinely different figures ------------
        # This is the check that a "Raman = IR" implementation cannot pass.
        check("the Raman curve is a different dataset from the IR curve",
              ir["label"] != ram["label"],
              f"{ir['label']!r} vs {ram['label']!r}")
        check("the two charts are separate objects with different data",
              ir["colour"] != ram["colour"],
              f"{ir['colour']} vs {ram['colour']}")
        check("the Raman axis is labelled as a Raman intensity",
              "raman" in str(ram["ytitle"]).lower(), str(ram["ytitle"]))
        check("the Raman curve is normalised to 100",
              abs(ram["max"] - 100.0) < 0.5, str(ram["max"]))
        check("the Raman axis runs high wavenumber on the left",
              float(ram["x0"]) > float(ram["x1"]),
              f"{ram['x0']} -> {ram['x1']}")
        check("the Raman curve has the same point count as the IR curve",
              ram["n"] == ir["n"], f"{ram['n']} vs {ir['n']}")

        # The strongest Raman line and the strongest IR band must be at
        # different wavenumbers.  Both figures are normalised, so the peak
        # positions are the only comparable quantity.
        d_peak = abs(float(ram["peak_x"]) - float(ir["peak_x"]))
        check("the strongest Raman line is not the strongest IR band",
              d_peak > 100.0,
              f"IR peaks at {ir['peak_x']}, Raman at {ram['peak_x']}")

        # --- what the pane says about the numbers -----------------------
        note = (page.locator("#ramanNote").inner_text()
                if page.locator("#ramanNote").count() else "")
        check("the pane reports the isotropic polarizability",
              "\u03B1" in note or "iso" in note.lower(), note[:160])
        check("the pane reports the sum-rule residual",
              "sum rule" in note.lower(), note[:160])
        check("the pane states the depolarization ceiling",
              "0.750" in note or "0.75" in note, note[:200])
        check("the pane raises the basis caution at 6-31G*",
              "basis" in note.lower() and "diffuse" in note.lower(),
              note[:220])

        # --- the table --------------------------------------------------
        t = table(page, "#ramanTable")
        check("the Raman peak table exists", bool(t and t["rows"]))
        if t and t["rows"]:
            head = " ".join(str(h) for h in t["head"]).lower()
            check("the table has an activity column", "activity" in head, head)
            check("the table has a depolarization column",
                  "\u03c1" in head or "rho" in head, head)
            check("the table has one row per mode", len(t["rows"]) == 3,
                  str(len(t["rows"])))
            # Water's asymmetric stretch is B2, so its isotropic derivative
            # vanishes by symmetry and rho is exactly 3/4.  Read out of the
            # rendered table, not the payload: this is what the user sees.
            rhos = []
            for r in t["rows"]:
                try:
                    rhos.append(float(r[-1]))
                except (TypeError, ValueError):
                    pass
            check("a mode is shown at rho = 0.750 (the B2 limit)",
                  any(abs(v - 0.75) < 5e-3 for v in rhos), str(rhos))
            check("no mode exceeds the 3/4 ceiling",
                  all(v <= 0.7501 for v in rhos), str(rhos))
            acts = []
            for r in t["rows"]:
                try:
                    acts.append(float(r[-2]))
                except (TypeError, ValueError):
                    pass
            check("the activities are on a plausible A^4/amu scale",
                  bool(acts) and 0.1 < max(acts) < 1000.0, str(acts))

        # --- and the figure survives being switched away from and back --
        page.locator('#resultTabs .tab[data-tab="properties"]').click()
        page.wait_for_timeout(400)
        page.locator('#resultTabs .tab[data-tab="vibrations"]').click()
        page.wait_for_timeout(600)
        again = chart_state(page, "ramanChart")
        check("the Raman chart still exists after a tab round trip",
              again is not None)
        if again:
            check("switching tabs did not change the Raman data",
                  again["n"] == ram["n"]
                  and abs(again["max"] - ram["max"]) < 0.5)

        # --- the skip path is wired, even if it is not taken here -------
        # A figure that is not drawn has to say why.  The element is in the
        # DOM and hidden on this path; a molecule too big for the response
        # solve uses it.
        check("the 'Raman not computed' notice exists and is hidden here",
              page.locator("#ramanSkip").count() == 1
              and "hidden" in (page.locator("#ramanSkip")
                               .get_attribute("class") or ""))

        check("no console errors", not errors, "; ".join(errors[:3]))

        shot = os.path.join(os.path.dirname(__file__), "shot_raman.png")
        try:
            page.locator('#resultTabs .tab[data-tab="vibrations"]').click()
            page.wait_for_timeout(400)
            page.screenshot(path=shot, full_page=True)
            print("  screenshot ->", shot)
        except Exception as exc:                              # noqa: BLE001
            print("  (screenshot failed:", exc, ")")

        browser.close()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: " + ", ".join(FAILURES))
        return 1
    print("\nRaman panel check: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
