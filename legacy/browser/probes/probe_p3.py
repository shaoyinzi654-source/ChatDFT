"""Probe -- does the library cover every nucleus that has a reference?

The phosphorus gap was invisible because nothing stated the coverage rule.
A rule that can be checked is worth more than a resolution to remember, and
this one is cheap: every nucleus the program can produce a shift for ought to
be reachable by name from the library, or the feature exists only for people
who already know the SMILES.

Reports which referenced nuclei have a library molecule carrying them, and
which do not.

Changes nothing.

Run:  python probes/probe_p3.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "data", "molecules", "library.json")


def main() -> int:
    lib = json.loads(open(LIB, encoding="utf-8").read())
    print(f"=== {len(lib)} library entries ===\n")

    # Element symbols from the formula, which is what the entry declares.
    pat = re.compile(r"([A-Z][a-z]?)")
    have = {}
    for key, e in lib.items():
        for sym in set(pat.findall(e.get("formula", ""))):
            if sym in nmr.NUCLEI:
                have.setdefault(sym, []).append(key)

    referenced = sorted(nmr.REFERENCE_FOR)
    print(f"  nuclei with a reference compound: {referenced}\n")
    missing = []
    for sym in referenced:
        got = have.get(sym, [])
        mark = "ok  " if got else "MISS"
        if not got:
            missing.append(sym)
        print(f"  {mark} {sym:3s} {len(got):3d} entries"
              + (f"   e.g. {', '.join(got[:4])}" if got else ""))

    print()
    if missing:
        print(f"  {len(missing)} referenced nucleus/nuclei unreachable by "
              f"name: {missing}")
    else:
        print("  every referenced nucleus is reachable from the library")
    print()
    # Not a failure either way -- this is a coverage report.  Whether the
    # library *must* cover every reference is a decision, and it is worth
    # knowing which side of it the current data sits on.
    print(f"  active nuclei overall: {len(nmr.NUCLEI)}; "
          f"with a reference: {len(referenced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
