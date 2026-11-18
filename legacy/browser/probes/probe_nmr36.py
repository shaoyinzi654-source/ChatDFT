"""Probe -- does the scale note actually reach the payload?

contract can assert that ``scale_note("N")`` returns the right sentence all
day, and the last round showed exactly why that is not enough: ``atom_extremes``
was computed correctly, asserted correctly, and still missing from the API
payload because ``surfaces()`` copies the mep fields one at a time and nobody
added the new one.  A correct function nobody calls is not a feature.

So this probe runs a real job on a molecule with nitrogen and one with
phosphorus -- the two nuclei whose scale is not the IUPAC primary one -- and
looks for the note in two places: per nucleus, next to the number it is about,
and in the warnings the panel shows.  Ammonia and phosphine are both in the
library and both small enough to run in seconds.

Changes nothing.

Run:  python probes/probe_nmr36.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr
from backend.engine.molecule import from_smiles, resolve


def look(name: str, xyz: str) -> None:
    print(f"=== {name} ===")
    try:
        res = nmr.compute(xyz, basis="6-31g*", use_cache=True)
    except Exception as exc:                              # noqa: BLE001
        print(f"  job failed: {type(exc).__name__}: {str(exc)[:110]}")
        return
    notes = [w for w in (res.get("warnings") or [])
             if "IUPAC primary reference" in w]
    print(f"  warnings carrying the scale: {len(notes)}")
    for w in notes:
        print(f"    - {w}")
    # "nuclei", not "entries": that is what compute() calls the list in the
    # payload, and the first version of this probe looked for "entries" and
    # found nothing, which read like the note was missing when it was the key
    # that was wrong.
    nuclei = res.get("nuclei") or []
    per = [(e["symbol"], e.get("delta_ppm"), e["reference_scale_note"])
           for e in nuclei if e.get("reference_scale_note")]
    if not per:
        print("  NO nucleus carries reference_scale_note")
    for sym, d, note in per:
        dstr = "n/a" if d is None else f"{d:8.3f} ppm"
        print(f"    {sym:3s} delta {dstr}  note: {note[:96]}")
    # The field has to exist and be None for the nuclei that are on the
    # primary standard, not be missing -- otherwise a caller cannot tell
    # "on the IUPAC scale" from "nobody thought about it".
    keys = {e["symbol"] for e in nuclei}
    have = {e["symbol"] for e in nuclei if "reference_scale_note" in e}
    print(f"  elements {sorted(keys)}; field present on {sorted(have)}")
    missing = sorted(keys - have)
    if missing:
        print(f"  !! field MISSING on {missing}")
    print()


def main() -> int:
    print("=== does the scale note reach the payload? ===\n")
    print("scale_note directly: ")
    for el in ("H", "C", "N", "O", "F", "Si", "P"):
        print(f"  {el:2s}: {nmr.scale_note(el)}")
    print()
    look("ammonia", resolve("ammonia", kind="name").to_xyz())
    # No molecule in the library contains phosphorus at all -- 18 entries
    # carry nitrogen and not one carries P -- so the 31P path has never been
    # exercised by a real job in this build.  Building phosphine from SMILES
    # is the only way to see the note land next to an actual shift.
    look("phosphine (from SMILES 'P')", from_smiles("P").to_xyz())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
