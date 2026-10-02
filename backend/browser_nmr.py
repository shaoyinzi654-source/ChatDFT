"""Drive the NMR panel in a real browser and assert what the user sees.

The NMR figure is the one place in the app where the *axis direction* carries
meaning: a spectrum with high ppm on the right is not a spectrum anyone can
read.  Chart.js will happily draw it either way, the JSON is identical, and no
payload-level check can tell the two apart -- so the reversal is asserted here,
out of the live chart object.

The other thing only a browser can see is the level of theory.  The shielding
is computed at Hartree-Fock whatever functional was requested, and the panel
has to say so on the figure; if that chip goes missing the figure still looks
perfect and every absolute shielding in it is mislabelled.  So the chips are
read back from the DOM.

Water is the test molecule because it is its own 17O reference: a correct
wiring gives water a 17O shift of exactly 0.00 ppm, which is a number no
half-working implementation produces by accident.

Cost note: the first run on a cold reference cache also computes TMS (17 atoms,
nao = 98 at 6-31G*, about five minutes) for the 1H scale.  After that it is
read from data/nmr_cache.  Regenerate the cache with:

    python -m backend.gen_nmr_cache

Run with the gate server up:

    python -m backend.browser_nmr
"""
from __future__ import annotations

import os
import time

from playwright.sync_api import sync_playwright

BASE = os.environ.get("CHATDFT_URL", "http://127.0.0.1:8000")
FAILURES: list = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def chart_state(page, cid):
    """Axis titles, orientation and the plotted values, from Chart.js."""
    return page.evaluate(
        """(cid) => {
            const c = Chart.getChart(document.getElementById(cid));
            if (!c) return null;
            const d = c.data.datasets[0].data;
            let imax = 0;
            for (let i = 1; i < d.length; i++) if (d[i] > d[imax]) imax = i;
            return {
              xtitle: c.options.scales.x.title.text,
              ytitle: c.options.scales.y.title.text,
              reverse: c.options.scales.x.reverse === true,
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
        page.on("console",
                lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE, wait_until="networkidle", timeout=60000)
        check("page loads", page.title() != "")

        box = page.locator("#input")
        box.fill("NMR spectrum of water")
        box.press("Enter")

        # The reference cache makes this fast after the first run, but a cold
        # cache computes TMS and that takes minutes.
        deadline = time.time() + 1200
        while time.time() < deadline:
            cls = page.locator("#nmrContent").get_attribute("class") or ""
            if "hidden" not in cls:
                break
            page.wait_for_timeout(2000)

        page.locator('#resultTabs .tab[data-tab="nmr"]').click()
        page.wait_for_timeout(900)

        cls = page.locator("#nmrContent").get_attribute("class") or ""
        check("the NMR panel is revealed", "hidden" not in cls, cls[:60])

        nm = chart_state(page, "nmrChart")
        check("the NMR chart exists", nm is not None)
        if not nm:
            browser.close()
            print("\n1 FAILURES: the NMR chart is missing")
            return 1

        # --- the axis convention, which only the live chart knows -------
        check("the x axis is labelled as a chemical shift",
              "ppm" in str(nm["xtitle"]).lower(), str(nm["xtitle"]))
        check("the x axis is reversed (high ppm on the left)",
              nm["reverse"] is True, str(nm["reverse"]))
        check("the y axis is a relative intensity",
              "intensity" in str(nm["ytitle"]).lower(), str(nm["ytitle"]))
        check("the curve is normalised to 1",
              abs(float(nm["max"]) - 1.0) < 1e-6, str(nm["max"]))
        check("the curve has enough points to be a spectrum",
              int(nm["n"]) > 500, str(nm["n"]))
        # The grid is stored ascending and reversed by the chart.  A
        # pre-reversed array would come out reversed twice and read
        # high-to-low on the page.
        check("the stored ppm grid is ascending, not pre-reversed",
              float(nm["x0"]) < float(nm["x1"]),
              f"{nm['x0']} -> {nm['x1']}")

        # --- the level of theory has to be on the figure ----------------
        lvl = page.locator("#nmrLevel").inner_text()
        check("the panel states the SCF level", "RHF" in lvl, lvl[:200])
        check("the panel says the requested functional is not used",
              "hartree-fock" in lvl.lower()
              and ("not used" in lvl.lower() or "shielding" in lvl.lower()),
              lvl[:240])
        check("the panel reports the dE/dB consistency check",
              "dE/dB" in lvl or "dE/db" in lvl, lvl[:240])
        check("the panel names the reference compounds",
              "TMS" in lvl, lvl[:240])

        # --- the shifts --------------------------------------------------
        t = table(page, "#nmrTable")
        check("the shift table exists", bool(t and t["rows"]))
        if t and t["rows"]:
            head = " ".join(str(h) for h in t["head"]).lower()
            check("the table has a shift column", "shift" in head, head)
            check("the table has an absolute-shielding column",
                  "shielding" in head, head)
            check("the table has an integral column", "count" in head, head)
            els = [r[0] for r in t["rows"]]
            check("water shows one O and one H row",
                  any("O" in e for e in els) and any("H" in e for e in els),
                  str(els))
            # water is its own 17O reference, so its 17O shift is 0.00
            o_rows = [r for r in t["rows"] if "O" in r[0]]
            if o_rows:
                check("water's 17O shift is 0.00 ppm (it is its own reference)",
                      abs(float(o_rows[0][1])) < 0.01, str(o_rows[0][1]))
            # 1H of water is about 0.3 ppm vs TMS at this level
            h_rows = [r for r in t["rows"] if r[0].strip().startswith("H")]
            if h_rows:
                check("water's 1H shift is a small positive number",
                      -1.0 < float(h_rows[0][1]) < 2.0, str(h_rows[0][1]))

        # --- the tensors -------------------------------------------------
        tt = table(page, "#nmrTensor")
        check("the tensor table exists", bool(tt and tt["rows"]))
        if tt and tt["rows"]:
            check("the tensor table has one row per nucleus",
                  len(tt["rows"]) == 3, str(len(tt["rows"])))
            head = " ".join(str(h) for h in tt["head"])
            check("the tensor table shows principal values",
                  "11" in head and "22" in head and "33" in head, head)
            check("the tensor table shows a span and a skew",
                  "Span" in head and "Skew" in head, head)
            # every principal value must be a finite number, and the span
            # must be non-negative -- it is sigma33 - sigma11 by definition
            spans = []
            for r in tt["rows"]:
                try:
                    spans.append(float(r[4]))
                except (IndexError, TypeError, ValueError):
                    pass
            check("every span is a non-negative number",
                  bool(spans) and all(v >= -1e-9 for v in spans), str(spans))

        # --- one axis per isotope ---------------------------------------
        # A spectrum is acquired for one nucleus.  A 1H window is about 12 ppm
        # wide and a 17O window about 1000, so a single axis holding both is
        # not a spectrum; and the ppm width of the same 1 Hz line differs by
        # the ratio of the Larmor frequencies, so they do not even share a
        # linewidth.  The panel used to draw one axis for everything.
        seg = page.locator("#nmrIsotope")
        check("the isotope selector is present", seg.count() > 0)
        iso_btns = page.locator("#nmrIsotope [data-nmr-isotope]")
        labels = sorted(iso_btns.nth(i).inner_text().strip()
                        for i in range(iso_btns.count()))
        check("the selector offers 1H and 17O", labels == ["17O", "1H"],
              str(labels))
        active = page.locator("#nmrIsotope .active")
        check("the panel opens on the protons",
              active.count() > 0 and active.inner_text().strip() == "1H",
              active.inner_text().strip() if active.count() else "(none)")

        # --- the sticks are drawn, not merely promised ------------------
        # The caption used to say "the sticks are placed at the computed
        # isotropic shift of each group, with the integral as height" while
        # the chart only ever drew the envelope, and nothing read
        # spectrum.sticks at all.
        sticks = page.evaluate(
            """() => {
                const c = Chart.getChart(document.getElementById('nmrChart'));
                if (!c || !c.$nmrSticks) return null;
                return c.$nmrSticks.map(s => ({
                    d: s.delta_ppm, h: s.rel_intensity }));
            }""")
        check("the chart is handed the sticks to draw", bool(sticks),
              str(sticks))
        if sticks:
            check("the tallest stick is at the top of the axis",
                  abs(max(float(s["h"]) for s in sticks) - 1.0) < 1e-9,
                  str([s["h"] for s in sticks]))
            check("there is one stick per signal in the table",
                  len(sticks) == len(t["rows"]) if t and t["rows"] else False,
                  f"{len(sticks)} sticks, "
                  f"{len(t['rows']) if t and t['rows'] else 0} rows")

        # switching isotope must move the axis -- that is the whole point
        if iso_btns.count() > 1:
            page.locator('#nmrIsotope [data-nmr-isotope="17O"]').click()
            page.wait_for_timeout(700)
            nm2 = chart_state(page, "nmrChart")
            check("switching to 17O redraws the figure", nm2 is not None)
            if nm2:
                check("the 17O axis is a different window from the 1H one",
                      (float(nm2["x0"]) != float(nm["x0"])
                       or float(nm2["x1"]) != float(nm["x1"])),
                      f"{nm2['x0']}..{nm2['x1']} vs {nm['x0']}..{nm['x1']}")
                check("the 17O curve is still normalised",
                      abs(float(nm2["max"]) - 1.0) < 1e-6, str(nm2["max"]))
            note17 = (page.locator("#nmrNote").inner_text()
                      if page.locator("#nmrNote").count() else "")
            check("the note names the isotope on the axis", "17O" in note17,
                  note17[:160])
            page.locator('#nmrIsotope [data-nmr-isotope="1H"]').click()
            page.wait_for_timeout(700)

        # --- the note explains the broadening ---------------------------
        note = (page.locator("#nmrNote").inner_text()
                if page.locator("#nmrNote").count() else "")
        check("the note gives the linewidth and the spectrometer frequency",
              "Hz" in note and "MHz" in note, note[:200])
        # the resolution is what decides whether the drawn line is the one the
        # caption names, so it has to be on the figure
        check("the note reports the grid resolution",
              "points per" in note or "grid" in note, note[:240])

        # --- and the figure survives a tab round trip -------------------
        page.locator('#resultTabs .tab[data-tab="properties"]').click()
        page.wait_for_timeout(400)
        page.locator('#resultTabs .tab[data-tab="nmr"]').click()
        page.wait_for_timeout(700)
        again = chart_state(page, "nmrChart")
        check("the NMR chart still exists after a tab round trip",
              again is not None)
        if again:
            check("switching tabs did not change the NMR data",
                  again["n"] == nm["n"] and again["reverse"] is True)

        check("no console errors", not errors, "; ".join(errors[:3]))

        shot = os.path.join(os.path.dirname(__file__), "shot_nmr.png")
        try:
            page.locator('#resultTabs .tab[data-tab="nmr"]').click()
            page.wait_for_timeout(400)
            page.screenshot(path=shot, full_page=True)
            print("  screenshot ->", shot)
        except Exception as exc:                              # noqa: BLE001
            print("  (screenshot failed:", exc, ")")

        browser.close()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: " + ", ".join(FAILURES))
        return 1
    print("\nNMR panel check: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
