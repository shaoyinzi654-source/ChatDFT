"""Probe -- how much does the 77Se reference geometry choice move the scale?

Me2Se is the 77Se reference.  Two geometries are available:

  A  this program's own B3LYP/6-31G* stationary point
     Se-C 1.9473 A, C-Se-C 92.22 deg  (probe_se1, converged)
  B  the experimental microwave structure
     Se-C 1.943 A, C-Se-C 96.2 deg    (Beecher, J. Mol. Spectrosc. 21, 414
     (1966); also CRC Handbook 97th ed. 9-41).  Gas electron diffraction
     gives 1.98 A / 98 deg (Goldish et al., JACS 77, 2948 (1955)).

A reference geometry sets the scale for every 77Se shift the program prints,
so the 4.0 deg difference is not a detail to be waved through -- B3LYP/6-31G*
is known to be weak on angles at heavy atoms with lone pairs, and this project
has already measured 31P shielding moving 2.819 ppm between two geometries of
PH3.

So it gets measured rather than argued about.  Geometry B keeps the
experimental heavy-atom frame and carries each methyl group over rigidly from
geometry A, so the C-H lengths and H-C-H angles are this program's own and the
only thing taken from experiment is the Se-C distance and the C-Se-C angle.

Changes nothing.

Run:  python -u probes/probe_se3.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr

# Geometry A, exactly as probe_se1 printed it.
A = [("C", -1.385276, -0.102226, -0.201221),
     ("Se", 0.175710, -1.111778, -0.781079),
     ("C", 1.390408, 0.069750, 0.178405),
     ("H", -1.448782, -0.109843, 0.906585),
     ("H", -1.313065, 0.944225, -0.563228),
     ("H", -2.299096, -0.570218, -0.621219),
     ("H", 1.291410, 1.105592, -0.207023),
     ("H", 1.155690, 0.051528, 1.262794),
     ("H", 2.433000, -0.277030, 0.025986)]

SE_C = 1.943            # microwave
CSE_C = 96.2            # microwave


def experimental_frame(atoms):
    """Geometry B: experimental heavy-atom frame, methyls carried over rigidly.

    The methyl groups are rotated as rigid bodies about their carbon, so the
    only information taken from the microwave structure is the Se-C distance
    and the C-Se-C angle -- both of which the microwave study determines far
    better than a 6-31G* gradient does.
    """
    syms = [a[0] for a in atoms]
    xyz = np.array([a[1:] for a in atoms], dtype=float)
    se = syms.index("Se")
    cs = [i for i, s in enumerate(syms) if s == "H" or s == "Se"]
    carbons = [i for i, s in enumerate(syms) if s == "C"]
    assert len(carbons) == 2, "expected two methyl carbons"

    # Bisector of the two Se-C bonds, and the plane they span.
    u1 = xyz[carbons[0]] - xyz[se]
    u2 = xyz[carbons[1]] - xyz[se]
    bis = u1 / np.linalg.norm(u1) + u2 / np.linalg.norm(u2)
    bis /= np.linalg.norm(bis)
    perp = u1 / np.linalg.norm(u1) - u2 / np.linalg.norm(u2)
    perp /= np.linalg.norm(perp)

    out = xyz.copy()
    half = math.radians(CSE_C / 2.0)
    for k, c in enumerate(carbons):
        sgn = 1.0 if k == 0 else -1.0
        new_axis = (SE_C * (math.cos(half) * bis
                            + sgn * math.sin(half) * perp))
        old_axis = xyz[c] - xyz[se]
        old_axis = old_axis / np.linalg.norm(old_axis)
        new_dir = new_axis / np.linalg.norm(new_axis)
        # Rotate the methyl rigidly from old_axis onto new_dir.
        v = np.cross(old_axis, new_dir)
        s = np.linalg.norm(v)
        cn = float(np.dot(old_axis, new_dir))
        if s < 1e-12:
            R = np.eye(3) if cn > 0 else -np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - cn) / (s * s))
        out[se] = np.zeros(3)
        out[c] = new_axis
        for i, sy in enumerate(syms):
            if sy == "H" and np.linalg.norm(xyz[i] - xyz[c]) < 1.3:
                out[i] = out[c] + R @ (xyz[i] - xyz[c])
    return [(syms[i], *out[i]) for i in range(len(syms))]


def se_shielding(atoms, label):
    xyz = nmr._xyz(atoms)
    mol = nmr._build_mol(xyz, "6-31g*", 0, 1)
    ops = nmr.build_operators(mol)
    t0 = time.time()
    sigma, diag = nmr.shielding_tensor(mol, ops)
    dt = time.time() - t0
    syms = [a[0] for a in atoms]
    se = syms.index("Se")
    iso = float(np.trace(sigma[se]) / 3.0)
    print(f"  {label:12s} nao={mol.nao_nr():3d}  sigma_iso(77Se) = "
          f"{iso:10.4f} ppm   ({dt:.1f} s)")
    return iso


def main() -> int:
    B = experimental_frame(A)
    print("=== 77Se reference geometry: A (this build) vs B (microwave) ===\n")
    for tag, atoms in (("A dft", A), ("B exptl", B)):
        syms = [a[0] for a in atoms]
        c = np.array([a[1:] for a in atoms], dtype=float)
        se = syms.index("Se")
        cs = [i for i, s in enumerate(syms) if s == "C"]
        d1 = float(np.linalg.norm(c[cs[0]] - c[se]))
        d2 = float(np.linalg.norm(c[cs[1]] - c[se]))
        v1, v2 = c[cs[0]] - c[se], c[cs[1]] - c[se]
        ang = math.degrees(math.acos(
            float(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        print(f"  {tag:9s} Se-C {d1:.4f} / {d2:.4f} A   C-Se-C {ang:.2f} deg")
    print()

    a = se_shielding(A, "A dft")
    b = se_shielding(B, "B exptl")
    print(f"\n  difference in the reference shielding: {b - a:+.3f} ppm")
    print("  (this is a rigid offset on EVERY 77Se shift the program prints)")
    print("\n  geometry B, for the module:")
    for s, x, y, z in B:
        print(f'        ("{s}", {x:.6f}, {y:.6f}, {z:.6f}),')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
