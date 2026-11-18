"""Probe 34 -- are the strangest library geometries chemically possible?

probe 33 broke each suspect geometry down per element pair and turned up
distances that cannot exist:

* methanol and methanethiol both contain an H...H distance of 1.0938 A.  Two
  hydrogens on the same carbon sit 1.78 A apart (H-C-H is 109.5 deg, not
  60 deg), and no through-space H...H contact is that short either;
* tfa puts its fluorines 1.68 A apart, which with a tetrahedral carbon implies
  C-F = 1.03 A instead of 1.35 A.

Those are not "a different conformer" or "a different level of theory".  They
are geometries that no molecule has, and every number computed on them --
energies, gaps, spectra, NMR shieldings -- describes something that does not
exist.  That is a far worse defect than a provenance string that is wrong.

This probe prints the short interatomic distances of the worst offenders
next to a fresh embedding, so the two can be read side by side, and flags
any distance that is shorter than the sum of the two covalent radii allows.

Changes nothing.

Run:  python probes/probe_nmr34.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend.engine.elements import get as element_get
from backend.engine.molecule import _load_library, from_smiles, parse_xyz

WORST = ["methanol", "methanethiol", "tfa", "toluene", "glycine", "urea",
         "acetic_acid", "styrene", "ozone"]


def short_distances(atoms, cutoff: float = 2.1):
    pos = np.array([[a.x, a.y, a.z] for a in atoms])
    syms = [a.symbol for a in atoms]
    out = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d <= cutoff:
                out.append((d, syms[i], i, syms[j], j))
    return sorted(out)


def implausible(atoms) -> list:
    """Contacts shorter than 0.75 x the sum of covalent radii (allowing for bonds)."""
    pos = np.array([[a.x, a.y, a.z] for a in atoms])
    syms = [a.symbol for a in atoms]
    bad = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            try:
                rc = element_get(syms[i]).covalent_radius + element_get(syms[j]).covalent_radius
            except Exception:
                continue
            if rc and d < 0.75 * rc:
                bad.append((d, syms[i], i, syms[j], j, rc))
    return sorted(bad)


def main() -> int:
    lib = _load_library()
    print("=== probe 34: chemically impossible contacts in the library ===\n")
    flagged = []
    for key, e in sorted(lib.items()):
        try:
            atoms = parse_xyz(e["xyz"], name=key).atoms
        except Exception as exc:
            print(f"{key}: unreadable xyz ({exc})")
            continue
        bad = implausible(atoms)
        if bad:
            flagged.append(key)
            print(f"{key}: {len(bad)} contact(s) shorter than 0.75 x (r1 + r2)")
            for d, s1, i1, s2, i2, rc in bad[:6]:
                print(f"   {s1}{i1}...{s2}{i2}  {d:7.4f} A   (sum of radii {rc:6.4f} A)")
    print(f"\nentries with impossible contacts: {len(flagged)}")
    if flagged:
        print("   " + ", ".join(flagged))

    print("\n--- side by side: library vs fresh embedding (distances < 2.1 A) ---")
    for key in WORST:
        e = lib.get(key)
        if e is None:
            continue
        lib_atoms = parse_xyz(e["xyz"], name=key).atoms
        try:
            fresh = from_smiles(e["smiles"]).atoms
        except Exception as exc:
            print(f"\n{key}: fresh embedding failed ({exc})")
            continue
        print(f"\n{key}: library")
        for d, s1, i1, s2, i2 in short_distances(lib_atoms):
            print(f"   {s1}{i1}-{s2}{i2}  {d:7.4f}")
        print(f"{key}: fresh")
        for d, s1, i1, s2, i2 in short_distances(fresh):
            print(f"   {s1}{i1}-{s2}{i2}  {d:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
