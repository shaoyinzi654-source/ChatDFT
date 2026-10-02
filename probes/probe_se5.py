"""Probe -- what geometry should the 77Se reference be built at?

probe_se4 settled whether the question is answerable: the field-step spread of
sigma_iso(77Se) in Me2Se is 0.0005 ppm, so the 0.169 ppm difference between
geometry A (this program's B3LYP/6-31G* stationary point) and geometry B (the
microwave heavy-atom frame) is a measurement, not noise.  It is now a choice,
and a choice needs a reason.

The reason is in nmr.py itself: all six existing references are *experimental*
structures -- TMS from electron diffraction, NH3/PH3/H2O from experimental r_e,
CFCl3 and CH4 constructed at the experimental bond lengths.  Not one of them is
a stationary point of the same functional that computes the shielding.  Using
geometry A for selenium alone would make 77Se the only nucleus in the product
whose scale is set by an optimised geometry, and the one thing that is certain
about a B3LYP/6-31G* angle at a heavy atom with lone pairs is that it is the
weakest number in the molecule (92.22 deg against a measured 96.2).

So the reference gets built the way the other five are: from experimental
parameters, with the parts the experiment does not determine -- the methyl
groups -- constructed rather than borrowed from a DFT optimisation.

Geometry C below is that construction: Se-C 1.943 A and C-Se-C 96.2 deg from
the microwave structure (Beecher, J. Mol. Spectrosc. 21, 414 (1966); CRC
Handbook 97th ed. 9-41), C-H 1.090 A and H-C-H 109.47 deg as the idealised
tetrahedral methyl used for TMS.

Geometry C differs from geometry B only in the methyls: B carries each methyl
over rigidly from the DFT geometry, C builds it.  The difference is measured
here too, because "it cannot matter, the hydrogens are 2.5 A from the selenium"
is exactly the kind of reasoning this project keeps finding to be false.

And the reference geometry is not the only geometry that matters.  Every
selenium molecule in the library is UFF-optimised -- MMFF94 has no selenium
parameters at all -- so the geometry of the *molecule* is the larger
uncertainty.  The sensitivities are measured here rather than asserted:
d(sigma)/d(Se-C) and d(sigma)/d(C-Se-C) for the reference itself.  They are
also, to first order, the sensitivities of every 77Se shift the program prints,
because delta = sigma_ref - sigma_mol.

Changes nothing.

Run:  python -u probes/probe_se5.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr

SE_C = 1.943            # microwave
CSE_C = 96.2            # microwave
C_H = 1.090             # the idealised methyl used for TMS
HCH = 109.47


def _perp_frame(axis):
    """Two unit vectors perpendicular to ``axis``, right-handed."""
    a = np.array(axis, dtype=float)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    w1 = np.cross(a, ref)
    w1 = w1 / np.linalg.norm(w1)
    w2 = np.cross(a, w1)
    return a, w1, w2


def dimethyl_selenide(se_c=SE_C, cse_c=CSE_C, twist=0.0):
    """Me2Se built from experimental parameters, in the style of _c3v/_tms.

    Se at the origin, the C-Se-C bisector along +z, the two carbons in the xz
    plane, so the molecule has a mirror plane (one C-H of each methyl lies in
    it).  ``twist`` rotates the second methyl about its own C-Se axis, which is
    the only conformational degree of freedom this frame has.
    """
    half = math.radians(cse_c / 2.0)
    out = [("Se", 0.0, 0.0, 0.0)]
    for k, sgn in enumerate((1.0, -1.0)):
        u = np.array([sgn * math.sin(half), 0.0, math.cos(half)])
        cpos = se_c * u
        out.append(("C", *cpos))
        # H's tetrahedral about the carbon, with Se anti to one of them:
        # the H-C-Se angle is 109.47 deg, so a = (Se - C) normalised.
        a, w1, w2 = _perp_frame(-u)
        for j in range(3):
            phi = 2.0 * math.pi * j / 3.0 + (twist if k == 1 else 0.0)
            v = (math.cos(math.radians(HCH)) * a
                 + math.sin(math.radians(HCH)) * (math.cos(phi) * w1
                                                  + math.sin(phi) * w2))
            out.append(("H", *(cpos + C_H * v)))
    return out


def se_shielding(atoms, label):
    xyz = nmr._xyz(atoms)
    mol = nmr._build_mol(xyz, "6-31g*", 0, 1)
    ops = nmr.build_operators(mol)
    t0 = time.time()
    sigma, _diag = nmr.shielding_tensor(mol, ops)
    dt = time.time() - t0
    se = [a[0] for a in atoms].index("Se")
    iso = float(np.trace(sigma[se]) / 3.0)
    print(f"  {label:26s} nao={mol.nao_nr():3d}  sigma_iso(77Se) = "
          f"{iso:10.4f} ppm   ({dt:.1f} s)", flush=True)
    return iso


def geometry_report(atoms, label):
    syms = [a[0] for a in atoms]
    c = np.array([a[1:] for a in atoms], dtype=float)
    se = syms.index("Se")
    cs = [i for i, s in enumerate(syms) if s == "C"]
    hs = [i for i, s in enumerate(syms) if s == "H"]
    d1 = float(np.linalg.norm(c[cs[0]] - c[se]))
    d2 = float(np.linalg.norm(c[cs[1]] - c[se]))
    v1, v2 = c[cs[0]] - c[se], c[cs[1]] - c[se]
    ang = math.degrees(math.acos(
        float(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    ch = [float(np.linalg.norm(c[h] - c[cs[0]])) for h in hs[:3]]
    print(f"  {label:9s} n={len(atoms)}  Se-C {d1:.4f}/{d2:.4f} A   "
          f"C-Se-C {ang:.2f} deg   C-H {sum(ch) / 3:.4f} A")


def main() -> int:
    print("=== the 77Se reference geometry, chosen rather than assumed ===\n")

    c0 = dimethyl_selenide()
    geometry_report(c0, "C built")
    print()

    iso_c = se_shielding(c0, "C built")

    # (1) does building the methyl instead of borrowing it change anything?
    iso_tw = se_shielding(dimethyl_selenide(twist=math.pi / 3.0),
                          "C twisted 60 deg")
    print(f"\n  methyl conformation: {iso_tw - iso_c:+.4f} ppm\n")

    # (2) the reference's own geometry sensitivity
    short = se_shielding(dimethyl_selenide(se_c=SE_C - 0.02), "C Se-C -0.02 A")
    long_ = se_shielding(dimethyl_selenide(se_c=SE_C + 0.02), "C Se-C +0.02 A")
    d_r = (long_ - short) / 0.04
    print(f"  d(sigma)/d(Se-C)   = {d_r:+9.1f} ppm/A\n")

    narrow = se_shielding(dimethyl_selenide(cse_c=CSE_C - 2.0), "C angle -2 deg")
    wide = se_shielding(dimethyl_selenide(cse_c=CSE_C + 2.0), "C angle +2 deg")
    d_a = (wide - narrow) / 4.0
    print(f"  d(sigma)/d(C-Se-C) = {d_a:+9.1f} ppm/deg\n")

    print("  what this says about the choice:")
    print(f"    A (this build's stationary point) vs C (experimental, built): "
          f"38.4790 -> {iso_c:.4f} ppm")
    print(f"    the offset A -> C is {iso_c - 38.4790:+.3f} ppm on every 77Se "
          "shift, and it is")
    print("    quoted against a measured noise floor of 0.0005 ppm "
          "(probe_se4)")
    print(f"    B (experimental frame, DFT methyls) vs C (built methyls): "
          f"{iso_c - 38.6485:+.3f} ppm")
    print("\n  geometry C, for the module:")
    for s, x, y, z in c0:
        print(f'        ("{s}", {x:.6f}, {y:.6f}, {z:.6f}),')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
