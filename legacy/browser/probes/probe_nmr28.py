"""Probe 28 -- how much does the 17O shielding actually move with geometry?

The contract asserts that water's own 17O shift is exactly 0.00, and the
mutation harness reported it at 0.374 ppm.  That number has to be explained
before it is fixed, because there are two candidate explanations and they
imply different fixes:

  1. the reference geometry is not the geometry of the water the job computed
     (a geometry mismatch), or
  2. the shielding is so sensitive to the O-H length that even a matching
     geometry drifts by 0.4 ppm.

The three geometries in play:

  library    O-H 0.96857 A, H-O-H 104.00 deg   <- what "water" resolves to
  harness    O-H 0.95792 A, H-O-H 104.42 deg   <- WATER_XYZ in mutate_nmr.py
  reference  O-H 0.95720 A, H-O-H 104.52 deg   <- the module's built-in

If (1) holds, the three shieldings order by bond length and the harness's own
water is within ~0.02 ppm of the reference.  If (2) holds, they scatter
without a pattern.

Run:  python probes/probe_nmr28.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend.engine import nmr
from backend.engine.molecule import resolve


def geom(atoms):
    """(O-H, H-O-H) for a three-atom water, from the coordinates."""
    c = np.array([a[1:] for a in atoms], dtype=float)
    o = [i for i, a in enumerate(atoms) if a[0] == "O"][0]
    hs = [i for i, a in enumerate(atoms) if a[0] == "H"]
    r = float(np.linalg.norm(c[hs[0]] - c[o]))
    v1, v2 = c[hs[0]] - c[o], c[hs[1]] - c[o]
    ang = np.degrees(np.arccos(float(np.dot(v1, v2))
                               / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    return r, float(ang)


def sigma_17o(xyz, label):
    t0 = time.time()
    mol = nmr._build_mol(xyz, "6-31g*", 0, 1)
    ops = nmr.build_operators(mol)
    sigma, _ = nmr.shielding_tensor(mol, ops)
    o = [a for a in range(mol.natm) if mol.atom_symbol(a) == "O"][0]
    s = float(np.trace(sigma[o]) / 3.0 * nmr.PPM)
    print(f"  {label:11s} {s:12.6f} ppm   ({time.time() - t0:.1f}s)")
    return s


def main() -> int:
    lib = resolve("water", kind="name")
    lib_atoms = [(a.symbol, a.x, a.y, a.z) for a in lib.atoms]

    harness_atoms = [("O", 0.0, 0.0, 0.0),
                     ("H", 0.0, -0.757, 0.587),
                     ("H", 0.0, 0.757, 0.587)]
    ref_atoms = nmr._builtin_reference_geometry("water")

    print("=== probe 28: 17O shielding vs water geometry (RHF/6-31G*) ===\n")
    print("geometry        O-H / H-O-H")
    for label, atoms in (("library", lib_atoms), ("harness", harness_atoms),
                         ("reference", ref_atoms)):
        r, a = geom(atoms)
        print(f"  {label:11s} {r:.5f} A / {a:.2f} deg")

    print("\nabsolute 17O shielding")
    s_lib = sigma_17o(lib.to_xyz(), "library")
    s_har = sigma_17o(
        "3\nharness\n" + "\n".join(f"{s} {x} {y} {z}"
                                   for s, x, y, z in harness_atoms) + "\n",
        "harness")
    s_ref = sigma_17o(
        "3\nreference\n" + "\n".join(f"{s} {x} {y} {z}"
                                     for s, x, y, z in ref_atoms) + "\n",
        "reference")

    print("\nshifts against each candidate reference (sigma_ref - sigma_mol)")
    for rlabel, sr in (("library", s_lib), ("harness", s_har),
                       ("reference", s_ref)):
        row = []
        for mlabel, sm in (("library", s_lib), ("harness", s_har),
                           ("reference", s_ref)):
            row.append(f"{mlabel[:4]} {sr - sm:+.4f}")
        print(f"  ref = {rlabel:10s} " + "   ".join(row))

    print("\n  the diagonal is the self-shift: it is 0.0000 by construction")
    print("  for every choice -- which is the point.  The off-diagonal terms")
    print("  are what a user sees when the two geometries disagree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
