"""Probe -- is the 0.169 ppm geometry effect above the noise, or is it noise?

probe_se3 measured sigma_iso(77Se) of the Me2Se reference at two geometries and
found they differ by 0.169 ppm.  That is the number that decides whether the
reference geometry matters, so it has to be defensible before it is written
down.

It may not be.  The shielding is obtained by a finite-difference field scan,
and the module's own record (probe_nmr16/19/20, quoted at the top of nmr.py)
says the 17O shielding of water is "flat to 0.13 ppm from 3e-3 to 3e-5" in dB
at conv_tol = 1e-13.  A 0.169 ppm effect is one and a bit times a 0.13 ppm
floor, which is not a measurement -- it is a measurement plus an unknown.

So the floor gets measured for this nucleus and this molecule: the same
geometry at three field steps.  Whatever spread comes out is the resolution,
and the geometry effect is only real if it stands clear of it.

Changes nothing.

Run:  python -u probes/probe_se4.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr

# Geometry A from probe_se1 (this build's own B3LYP/6-31G* stationary point).
A = [("C", -1.385276, -0.102226, -0.201221),
     ("Se", 0.175710, -1.111778, -0.781079),
     ("C", 1.390408, 0.069750, 0.178405),
     ("H", -1.448782, -0.109843, 0.906585),
     ("H", -1.313065, 0.944225, -0.563228),
     ("H", -2.299096, -0.570218, -0.621219),
     ("H", 1.291410, 1.105592, -0.207023),
     ("H", 1.155690, 0.051528, 1.262794),
     ("H", 2.433000, -0.277030, 0.025986)]


def iso(atoms, dB, label):
    mol = nmr._build_mol(nmr._xyz(atoms), "6-31g*", 0, 1)
    ops = nmr.build_operators(mol)
    t0 = time.time()
    sigma, _ = nmr.shielding_tensor(mol, ops, dB=dB)
    se = [a[0] for a in atoms].index("Se")
    v = float(np.trace(sigma[se]) / 3.0)
    print(f"  dB={dB:<8g} sigma_iso(77Se) = {v:10.4f} ppm   "
          f"({time.time() - t0:.1f} s)  {label}")
    return v


def main() -> int:
    print("=== how repeatable is sigma_iso(77Se) of Me2Se? ===\n")
    vals = []
    for dB in (3.0e-4, 1.0e-4, 6.0e-5):
        vals.append(iso(A, dB, ""))
    spread = max(vals) - min(vals)
    print(f"\n  spread over three field steps: {spread:.4f} ppm")
    print(f"  the geometry effect measured in probe_se3 was 0.1690 ppm")
    print()
    if spread < 0.05:
        print("  the field-step spread is well under the geometry effect, so "
              "0.169 ppm is a measurement")
    elif spread < 0.169:
        print("  the spread is smaller than the effect but not by much; the "
              "geometry effect is real and its uncertainty is about "
              f"+/-{spread:.3f} ppm")
    else:
        print("  the spread is as large as the effect: the geometry choice "
              "for the 77Se reference is NOT resolvable at this field step, "
              "and probe_se3's 0.169 ppm must not be quoted as a measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
