"""Probe -- is 77Se shiftable at all at RHF/6-31G*, or is the number garbage?

probe_se5 gave sigma_iso(77Se) of the Me2Se reference as 38.5-39.8 ppm depending
on the geometry.  That is a suspicious number.  The absolute 77Se shielding of
Me2Se is around 1900 ppm on the established absolute shielding scale -- the
selenium valence shell is 4s4p, the paramagnetic term is enormous, and a
non-relativistic RHF/6-31G* treatment is not obviously entitled to any of it.

An absolute shielding that is wrong by 1850 ppm is not automatically a problem:
delta = sigma_ref - sigma_mol and a constant error cancels exactly.  What is a
problem is a *relative* error -- if the method compresses the 77Se scale, every
shift this program prints is wrong by the compression factor, and no reference
geometry fixes that.

So the method is tested on a shift whose experimental value is known.  H2Se is
616 ppm upfield of Me2Se (delta(77Se) = -616, Me2Se = 0 by definition), which
is the largest single-compound move in the whole selenium range and therefore
the hardest test available cheaply.  Three atoms, nao = 36, about 25 s.

    sigma(Me2Se) - sigma(H2Se)  vs  -616 ppm

If it comes out near -616 the scale is real and the reference is worth wiring
up.  If it comes out near zero, 6-31G* has nothing to say about 77Se, and the
honest response is to leave selenium unreferenced and say why with the number
attached -- not to ship a scale that compresses the literature by 10x.

H2Se is computed at the experimental r_e geometry (Se-H 1.4599 A, H-Se-H 90.6
deg), which is what the module's other references use, and the sensitivity to
that geometry is measured on either side of it because a 25 s calculation is
cheaper than an argument.

Changes nothing.

Run:  python -u probes/probe_se6.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr

SE_H = 1.4599          # experimental r_e
HSE_H = 90.6           # experimental r_e

# probe_se5, geometry C -- the reference candidate.
SIGMA_ME2SE = 39.8330

# The literature shift this has to reproduce.  Quoted relative to Me2Se.
LIT_H2SE = -616.0


def h2se(se_h=SE_H, angle=HSE_H):
    """H2Se with the C2 axis along +z, in the style of _c3v."""
    half = math.radians(angle / 2.0)
    return [("Se", 0.0, 0.0, 0.0),
            ("H", se_h * math.sin(half), 0.0, se_h * math.cos(half)),
            ("H", -se_h * math.sin(half), 0.0, se_h * math.cos(half))]


def se_shielding(atoms, label):
    xyz = nmr._xyz(atoms)
    mol = nmr._build_mol(xyz, "6-31g*", 0, 1)
    ops = nmr.build_operators(mol)
    t0 = time.time()
    sigma, _diag = nmr.shielding_tensor(mol, ops)
    dt = time.time() - t0
    se = [a[0] for a in atoms].index("Se")
    iso = float(np.trace(sigma[se]) / 3.0)
    print(f"  {label:22s} nao={mol.nao_nr():3d}  sigma_iso(77Se) = "
          f"{iso:10.4f} ppm   ({dt:.1f} s)", flush=True)
    return iso


def main() -> int:
    print("=== does RHF/6-31G* reproduce a known 77Se shift? ===\n")
    print(f"  the reference, Me2Se at geometry C, is {SIGMA_ME2SE:.4f} ppm "
          "(probe_se5)")
    print()

    a = se_shielding(h2se(), "H2Se r_e")
    delta = SIGMA_ME2SE - a
    print(f"\n  delta(77Se) of H2Se against Me2Se: {delta:+.1f} ppm")
    print(f"  experimental value:                {LIT_H2SE:+.1f} ppm")
    if delta == 0.0:
        ratio = float("inf")
    else:
        ratio = LIT_H2SE / delta
    print(f"  ratio experimental / computed:     {ratio:+.2f}")
    print()

    print("  sensitivity of the H2Se geometry:")
    short = se_shielding(h2se(se_h=SE_H - 0.02), "Se-H -0.02 A")
    long_ = se_shielding(h2se(se_h=SE_H + 0.02), "Se-H +0.02 A")
    print(f"    d(sigma)/d(Se-H) = {(long_ - short) / 0.04:+.1f} ppm/A")
    narrow = se_shielding(h2se(angle=HSE_H - 2.0), "angle -2 deg")
    wide = se_shielding(h2se(angle=HSE_H + 2.0), "angle +2 deg")
    print(f"    d(sigma)/d(H-Se-H) = {(wide - narrow) / 4.0:+.1f} ppm/deg")
    print()

    if abs(ratio) < 2.0 and abs(ratio) > 0.5:
        print("  the scale is reproduced to within a factor of two -- the "
              "reference is worth wiring up")
    else:
        print("  the computed scale is off by a factor of "
              f"{abs(ratio):.1f} from experiment.  A 77Se shift printed by "
              "this build")
        print("  would not be a 77Se shift, and no choice of reference "
              "geometry can fix that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
