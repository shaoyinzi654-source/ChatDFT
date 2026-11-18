"""Probe -- is the audit report well formed, and do its numbers agree?

The report is a deliverable, so "it was written" is not the same as "it
renders".  This checks the tag nesting with the standard parser, that every
table row has the same number of cells as its header, and that the figures
quoted in the prose match the ones the probes actually produced -- a report
that misquotes its own evidence is worse than no report.

Changes nothing.

Run:  python probes/probe_html1.py
"""
import io
import os
import re
import sys
from html.parser import HTMLParser

VOID = {"br", "hr", "img", "meta", "link", "input"}


class Check(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.problems = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.problems.append(f"line {self.getpos()[0]}: </{tag}> with "
                                 "nothing open")
            return
        open_tag, line = self.stack.pop()
        if open_tag != tag:
            self.problems.append(
                f"line {self.getpos()[0]}: </{tag}> closes <{open_tag}> "
                f"opened at line {line}")


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "reports", "round-18-audit.html")
    text = io.open(path, encoding="utf-8").read()
    print(f"=== {os.path.basename(path)}: {len(text)} chars ===\n")

    p = Check()
    p.feed(text)
    if p.stack:
        p.problems.append(f"unclosed at end: {p.stack}")
    if p.problems:
        print("STRUCTURE PROBLEMS:")
        for x in p.problems:
            print("  -", x)
    else:
        print("  structure: all tags balanced")

    # Table cells per row, against the row above it.
    bad = 0
    for m in re.finditer(r"<table>(.*?)</table>", text, re.S):
        rows = re.findall(r"<tr>(.*?)</tr>", m.group(1), re.S)
        widths = [len(re.findall(r"<t[hd][ >]", r)) for r in rows]
        if len(set(widths)) > 1:
            bad += 1
            print(f"  - a table has rows of widths {widths}")
    if not bad:
        print("  tables: every row matches its header width")

    # The numbers the prose quotes, checked against what the probes printed.
    want = {
        "5165": "the counted TMS estimate",
        "6346": "the measured TMS estimate at 68.8",
        "6364": "the measured TMS estimate at 69",
        "68.8": "the measured slope",
        "0.99997": "the fit R^2",
        "584.3": "ethanol peak",
        "72.2": "formaldehyde peak",
        "380.2": "the 15N conversion",
        "30.6": "BF3.OEt2 at the estimate",
        "964": "the 33S range",
    }
    missing = [k for k in want if k not in text]
    print(f"  figures present: {len(want) - len(missing)}/{len(want)}")
    for k in missing:
        print(f"  - MISSING {k} ({want[k]})")
    return 1 if (p.problems or bad or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
