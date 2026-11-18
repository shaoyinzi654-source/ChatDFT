"""Drive the Design panel in a real browser and assert what the user sees.

Run with the gate server up:

    python -m backend.browser_design
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

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

        tab = page.locator('#resultTabs .tab[data-tab="design"]')
        check("Design tab exists", tab.count() == 1)

        # drive it from the composer, which is how a user reaches it
        box = page.locator("#input")
        box.fill("design 4 amines for CO2 capture, molecular weight below 160, no halogens")
        box.press("Enter")

        deadline = time.time() + 420
        seen_status = ""
        while time.time() < deadline:
            txt = page.locator("#designContent").get_attribute("class") or ""
            if "hidden" not in txt:
                break
            stage = page.locator("#jobStage").inner_text() if page.locator("#jobStage").count() else ""
            if stage and stage != seen_status:
                seen_status = stage
                print(f"        ... {stage}")
            page.wait_for_timeout(1500)

        content = page.locator("#designContent")
        check("design panel is revealed",
              "hidden" not in (content.get_attribute("class") or ""),
              seen_status)

        # re-resolve: the locator above was created before the run started
        state = page.evaluate(
            "() => ({url: location.href, tabs: document.querySelectorAll("
            "'#resultTabs .tab').length, design: !!document.querySelector("
            "'#resultTabs .tab[data-tab=\"design\"]'), active: (document."
            "querySelector('#resultTabs .tab.active') || {}).dataset})")
        print("        page state:", state)
        check("tabs are still in the document", state["tabs"] >= 10,
              str(state["tabs"]))
        check("tab auto-switched to Design",
              state["active"] and state["active"].get("tab") == "design",
              str(state["active"]))

        cards = page.locator(".design-card")
        n = cards.count()
        check("candidate cards rendered", n >= 1, f"{n} cards")
        if n:
            first = cards.first
            check("card shows a SMILES",
                  len(first.locator("code").inner_text().strip()) > 2,
                  first.locator("code").inner_text())
            check("card shows a score",
                  len(first.locator(".design-score em").inner_text().strip()) > 0,
                  first.locator(".design-score em").inner_text())
            check("card shows descriptor cells",
                  first.locator(".design-desc-cell").count() >= 3,
                  str(first.locator(".design-desc-cell").count()))
            check("card shows desirability bars",
                  first.locator(".design-bar-row").count() >= 1,
                  str(first.locator(".design-bar-row").count()))
            check("card has action buttons",
                  first.locator(".design-actions button").count() == 2)

            # the quality scores must be on the card, not only in the CSV:
            # they are the reason a candidate is worth making at all
            # `inner_text` is the *rendered* text and the label is styled
            # `text-transform: uppercase`, so the comparison cannot be
            # case-sensitive -- the check is about the cell existing, not
            # about how it is typeset.
            labels = [t.strip().lower() for t in
                      first.locator(".design-desc-cell span").all_inner_texts()]
            check("card shows synthetic accessibility and QED",
                  "sa" in labels and "qed" in labels, str(labels))

            # the level of theory alone is not reproducible; the geometry
            # source travels with it as A//B
            heads = page.locator(".design-dft-head").all_inner_texts()
            check("a measured card states the geometry provenance",
                  any("//" in h for h in heads) or not heads,
                  str(heads[:2]))

        # a demanded substructure is stated up front, and every returned
        # molecule really carries it -- "design 4 amines" used to return
        # molecules without an amine
        req = page.locator("#designTargets .design-target", has_text="must contain")
        check("the demanded substructure is stated", req.count() >= 1,
              str(req.all_inner_texts()[:2]))
        check("no returned card is flagged as missing it",
              page.locator(".design-badges .badge.warn",
                           has_text="missing a required group").count() == 0)

        # A score of zero has to be explained on the card -- either by naming
        # the criteria the molecule failed, or by saying it cannot be judged
        # until the calculation exists.  A bare 0 is the bug this watches for.
        unexplained = page.evaluate(
            "() => Array.from(document.querySelectorAll('.design-card'))"
            ".filter((c) => { const em = c.querySelector('.design-score em');"
            " return em && (em.textContent || '').trim() === '0'; })"
            ".filter((c) => !c.querySelectorAll('.design-badges .badge.warn')"
            ".length)"
            ".map((c) => { const s = c.querySelector('code');"
            " return s ? s.textContent : '?'; })")
        check("a zero score is never shown without a reason",
              not unexplained, str(unexplained[:3]))

        summary = page.locator("#designSummary").inner_text()
        check("the quality filters that ran are stated",
              "quality filters" in summary, summary[:120])

        # nothing on a card may be the string "undefined" or "NaN"
        body = page.locator("#designList").inner_text()
        check("no card shows undefined or NaN",
              "undefined" not in body and "NaN" not in body)

        goal = page.locator("#designGoal").inner_text().strip()
        check("goal is shown", len(goal) > 8, goal[:70])

        # CSV download
        try:
            with page.expect_download(timeout=15000) as dl:
                page.click("#designCsv")
            path = dl.value.path()
            rows = open(path, encoding="utf-8", errors="replace").read().splitlines()
            check("CSV download works", len(rows) >= 2, f"{len(rows)} rows")
            check("CSV has a header", rows[0].startswith("rank,name,smiles"),
                  rows[0][:60])
            check("CSV carries the quality columns",
                  "SA_score" in rows[0] and "QED" in rows[0]
                  and "geometry" in rows[0], rows[0][:110])
            # an unscored candidate must be exported as "no", never as 0:
            # a zero in a spreadsheet is a value
            check("CSV says whether a candidate was scored at all",
                  ",scored,score," in rows[0], rows[0][:80])
        except Exception as exc:
            check("CSV download works", False, str(exc)[:120])

        # "Show structure" must load the viewer
        if n:
            page.locator(".design-actions button", has_text="Show structure").first.click()
            page.wait_for_timeout(2500)
            meta = page.locator("#molMeta").inner_text()
            check("Show structure loads the molecule", len(meta.strip()) > 3, meta)

        # snapshot the Design panel itself before any user click hides it
        page.evaluate(
            "() => document.querySelector('#resultTabs .tab[data-tab=\"design\"]').click()")
        page.wait_for_timeout(400)

        shot = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "shot_design.png")
        page.screenshot(path=shot, full_page=False)
        print(f"  screenshot -> {shot}")

        real_errors = [e for e in errors if "favicon" not in e.lower()]
        check("no console errors", not real_errors, "; ".join(real_errors[:3]))

        browser.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: " + ", ".join(FAILURES))
        return 1
    print("browser design check: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
