"""Static check: every DOM selector used in app.js must exist in index.html.

Catches a whole class of UI-breaking bugs (null dereference on $('#missing'))
that API-level tests cannot see.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = ROOT / "frontend" / "static" / "index.html"
JS = ROOT / "frontend" / "static" / "js" / "app.js"

html = HTML.read_text(encoding="utf-8")
js = JS.read_text(encoding="utf-8")

# ---- what the HTML actually provides --------------------------------
html_ids = set(re.findall(r'id="([^"]+)"', html))
html_classes: set[str] = set()
for m in re.findall(r'class="([^"]+)"', html):
    html_classes.update(m.split())
html_data = set(re.findall(r"data-([a-z-]+)=", html))

# ---- plus anything the JS creates at runtime ------------------------
# el('div', 'lib-item'), class="foo bar", classList.add('x'),
# dataset.thinking = '1'
js_classes: set[str] = set()
for m in re.findall(r"""el\(\s*['"][a-z0-9]+['"]\s*,\s*['"]([^'"]+)['"]""", js):
    js_classes.update(m.split())
for m in re.findall(r'class="([^"]+)"', js):
    js_classes.update(m.split())
for m in re.findall(r"""classList\.(?:add|toggle|remove)\(\s*['"]([A-Za-z0-9_-]+)""", js):
    js_classes.add(m)
html_classes |= js_classes

js_data: set[str] = set(re.findall(r"dataset\.([A-Za-z]+)\s*=", js))
# dataset.thinking -> data-thinking
js_data = {re.sub(r"(?<!^)(?=[A-Z])", "-", d).lower() for d in js_data}
html_data |= js_data

# ids the JS builds at runtime, the same way classes are collected above.
# A panel that renders a button into a template literal creates a real
# element with a real id; without this the checker reports it as a dangling
# selector, and the honest fix is to teach the checker, not to add a
# placeholder div to the HTML that would then be a lie.
js_ids = set(re.findall(r'id="([^"]+)"', js))
js_ids |= set(re.findall(r"""setAttribute\(\s*['"]id['"]\s*,\s*['"]([^'"]+)""", js))
js_ids |= set(re.findall(r"""\.id\s*=\s*['"]([A-Za-z0-9_-]+)['"]""", js))
html_ids |= js_ids

# ---- what the JS asks for -------------------------------------------
# $('#x'), $$('#x'), $('#x')?.foo, document.querySelector('#x')
sel_ids = set(re.findall(r"""[$]{1,2}\(\s*['"]#([A-Za-z0-9_-]+)""", js))
sel_ids |= set(re.findall(r"""querySelector(?:All)?\(\s*['"]#([A-Za-z0-9_-]+)""", js))

sel_cls = set(re.findall(r"""[$]{1,2}\(\s*['"]\.([A-Za-z0-9_-]+)""", js))
sel_cls |= set(re.findall(r"""querySelector(?:All)?\(\s*['"]\.([A-Za-z0-9_-]+)""", js))

sel_data = set(re.findall(r"""[$]{1,2}\(\s*['"]\[data-([a-z-]+)""", js))
sel_data |= set(re.findall(r"""querySelector(?:All)?\(\s*['"]\[data-([a-z-]+)""", js))

print(f"index.html: {len(html_ids)} ids, {len(html_classes)} classes, "
      f"{len(html_data)} data-* attributes")
print(f"app.js:     {len(sel_ids)} #id, {len(sel_cls)} .class, {len(sel_data)} [data-*] selectors")
print()

failures: list[str] = []

missing_ids = sorted(sel_ids - html_ids)
if missing_ids:
    failures.append(f"ids referenced but absent from HTML: {missing_ids}")
    print("MISSING ids:", missing_ids)
else:
    print("MISSING ids: none")

missing_cls = sorted(sel_cls - html_classes)
if missing_cls:
    failures.append(f"classes referenced but absent from HTML: {missing_cls}")
    print("MISSING classes:", missing_cls)
else:
    print("MISSING classes: none")

missing_data = sorted(sel_data - html_data)
if missing_data:
    failures.append(f"data-* referenced but absent from HTML: {missing_data}")
    print("MISSING data-* attrs:", missing_data)
else:
    print("MISSING data-* attrs: none")

# ---- ids must be unique ---------------------------------------------
# A duplicated id is worse than a missing one: querySelector silently
# returns the first, so the second element looks present but is never
# wired.  The chat panel was once copied wholesale and its duplicate
# composer accepted typing and then did nothing at all.
body = re.sub(r"<!--.*?-->", "", html, flags=re.S)
all_ids = re.findall(r'id="([^"]+)"', body)
repeated = sorted({i for i in all_ids if all_ids.count(i) > 1})
if repeated:
    failures.append(f"duplicate ids in index.html: {repeated}")
    print("DUPLICATE ids:", repeated)
else:
    print("DUPLICATE ids: none")

# ---- no asset may be loaded twice ------------------------------------
# index.html loaded 3Dmol-min.js and chart.umd.min.js once in <head> (the
# offline-vendoring fix) and again just before app.js.  Nothing breaks
# visibly -- both are evaluated after the first and the second window.Chart
# replaces the first -- but it is ~700 KB fetched, parsed and evaluated
# twice, and the two registries are separate: a chart built against the
# first Chart class is invisible to the second one's Chart.getChart().  A
# duplicate <script src> is exactly the kind of thing a diff review skips.
srcs = re.findall(r'<script[^>]*\bsrc="([^"]+)"', body)
dupe_srcs = sorted({s for s in srcs if srcs.count(s) > 1})
if dupe_srcs:
    failures.append(f"script src loaded more than once: {dupe_srcs}")
    print("DUPLICATE script src:", dupe_srcs)
else:
    print("DUPLICATE script src: none")

# ---- every tab needs a pane and every pane needs a tab ----------------
# A pane with no button can never be reached by clicking, and a button with
# no pane switches to nothing.  Both were true of the Design panel at one
# point: the pane was added and the button silently was not.
tabs = set(re.findall(r'data-tab="([^"]+)"', body))
panes = set(re.findall(r'data-pane="([^"]+)"', body))
if tabs != panes:
    only_tab = sorted(tabs - panes)
    only_pane = sorted(panes - tabs)
    failures.append(f"tab/pane mismatch: buttons only {only_tab}, "
                    f"panes only {only_pane}")
    print("TAB/PANE mismatch:", {"button only": only_tab,
                                 "pane only": only_pane})
else:
    print(f"TAB/PANE match: {len(tabs)} tabs <-> {len(panes)} panes")

# ---- template literals build some ids dynamically (e.g. data-pane) ---
print()
print("data-* used in HTML:", sorted(html_data))
print("classes used in HTML:", len(html_classes))

if failures:
    print("\n=== FAIL ===")
    for f in failures:
        print(" -", f)
    sys.exit(1)

print("\n=== PASS: every JS selector resolves against index.html ===")
