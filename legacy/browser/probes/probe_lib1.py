"""Probe -- which library molecules carry nitrogen or phosphorus?

probe_nmr36 needs one of each: 15N and 31P are the two nuclei this build
quotes against something other than the IUPAC primary reference, so they are
the only ones whose scale note is not None.  The library has no 'phosphine'
entry (that stopped the first run), so this probe asks what it does have.

Changes nothing.

Run:  python probes/probe_lib1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine.molecule import _load_library


def main() -> int:
    lib = _load_library()
    print("=== library molecules containing N or P ===\n")
    for key in sorted(lib):
        entry = lib[key]
        xyz = entry.get("xyz") or ""
        syms = set()
        for line in xyz.strip().splitlines():
            parts = line.split()
            if parts:
                syms.add(parts[0])
        if "N" in syms or "P" in syms:
            n = sum(1 for line in xyz.strip().splitlines() if line.split())
            print(f"  {key:16s} natm={n:3d}  {sorted(syms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
