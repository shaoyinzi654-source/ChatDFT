"""Probe 27 -- is the reference compound the same molecule as the library one?

The contract asserts "water is its own 17O reference, so its shift must be
0.00", and that assertion had never actually executed: the payload needs a
TMS cache entry for 1H, and no machine here could produce one until the swap
opt-in existed.  Now that it runs, it fails:

    nmr/water: water is its own 17O reference, so its shift must be 0.00;
    got 0.374

A self-reference that is not zero can only mean one thing: the geometry used
for the reference is not the geometry of the molecule that shares its name.
This probe measures that, for every reference compound that is also a library
molecule, so the size of the problem is known before anything is changed.

It does NOT recompute shieldings -- it compares geometries, which is the part
that can be checked for free, and reports the bond lengths and angles that
differ.
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend.engine import nmr
from backend.engine.molecule import resolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "data", "molecules", "library.json")

# reference key -> the library name a user would type to get that molecule
LIB_NAME = {
    "water": "water",
    "ammonia": "ammonia",
    "methane": "methane",
    "phosphine": "phosphine",
    "chloroform_f": "chloroform",
    "tms": "tetramethylsilane",
}


def coords(atoms):
    return np.array([[a[1], a[2], a[3]] for a in atoms], dtype=float)


def symbols(atoms):
    return [a[0] for a in atoms]


def canon(atoms):
    """Atoms as a sorted list of (symbol, x, y, z), order-independent."""
    return sorted((a[0], round(a[1], 8), round(a[2], 8), round(a[3], 8))
                  for a in atoms)


def bonds(atoms):
    """All bonded pairs as (symbols, distance), for a geometry comparison.

    A bonded pair is any pair closer than 1.9 A (every reference compound here
    is a hydride or a halide, so that threshold separates bonds from contacts).
    """
    c = coords(atoms)
    s = symbols(atoms)
    out = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            d = float(np.linalg.norm(c[i] - c[j]))
            if d < 1.9:
                out.append((tuple(sorted((s[i], s[j]))), d))
    return sorted(out)


def angles(atoms):
    """All X-A-Y angles, as (X, A, Y, degrees)."""
    c = coords(atoms)
    s = symbols(atoms)
    out = []
    n = len(s)
    for a in range(n):
        nb = [k for k in range(n)
              if k != a and np.linalg.norm(c[k] - c[a]) < 1.9]
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                v1 = c[nb[i]] - c[a]
                v2 = c[nb[j]] - c[a]
                cosang = float(np.dot(v1, v2)
                               / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                out.append((tuple(sorted((s[nb[i]], s[nb[j]]))), s[a],
                            round(ang, 4)))
    return sorted(out)


def main() -> int:
    lib = json.load(open(LIB, encoding="utf-8"))

    print("=== probe 27: reference compound vs library molecule ===\n")

    n_diff = 0
    for key, name in sorted(LIB_NAME.items()):
        ref = nmr.reference_geometry(key)
        entry = lib.get(name)
        print(f"--- {key}  (library name '{name}') ---")
        if entry is None:
            print(f"    not in the library; nothing to compare")
            print()
            continue
        xyz = entry["xyz"].splitlines()
        natm = int(xyz[0])
        mol_atoms = []
        for line in xyz[2:2 + natm]:
            p = line.split()
            mol_atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))

        same_formula = sorted(symbols(ref)) == sorted(symbols(mol_atoms))
        print(f"    reference formula {''.join(sorted(symbols(ref)))}"
              f"   library formula {''.join(sorted(symbols(mol_atoms)))}"
              f"   {'match' if same_formula else 'DIFFER'}")

        if not same_formula:
            print()
            continue

        identical = canon(ref) == canon(mol_atoms)
        print(f"    coordinates identical: {identical}")

        rb, mb = bonds(ref), bonds(mol_atoms)
        if [b[0] for b in rb] != [b[0] for b in mb]:
            print(f"    bond TOPOLOGY differs: {[b[0] for b in rb]} vs "
                  f"{[b[0] for b in mb]}")
            n_diff += 1
        else:
            worst = 0.0
            for (bs, dr), (_, dm) in zip(rb, mb):
                worst = max(worst, abs(dr - dm))
            if worst > 1e-6:
                n_diff += 1
            print(f"    bond lengths: worst difference {worst:.6f} A")
            for (bs, dr), (_, dm) in zip(rb, mb):
                if abs(dr - dm) > 1e-6:
                    print(f"        {bs[0]}-{bs[1]:2s}  ref {dr:.5f}"
                          f"   lib {dm:.5f}   d {dm - dr:+.5f}")

        ra, ma = angles(ref), angles(mol_atoms)
        if [a[:2] for a in ra] == [a[:2] for a in ma]:
            worst = 0.0
            for x, y in zip(ra, ma):
                worst = max(worst, abs(x[2] - y[2]))
            if worst > 1e-4:
                n_diff += 1
            print(f"    angles: worst difference {worst:.4f} deg")
            for x, y in zip(ra, ma):
                if abs(x[2] - y[2]) > 1e-4:
                    print(f"        {x[0][0]}-{x[1]}-{x[0][1]}  ref {x[2]:.4f}"
                          f"   lib {y[2]:.4f}   d {y[2] - x[2]:+.4f}")
        print()

    print(f"=== {n_diff} reference/library pair(s) differ in geometry ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
