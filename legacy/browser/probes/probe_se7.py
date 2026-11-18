"""Probe -- do the *shipped* nuclei reproduce their own scales?

probe_se6 settled 77Se: RHF/6-31G* gives -1.9 ppm for the Me2Se/H2Se pair where
experiment gives -616, a factor of 324.  Selenium has no usable shift at this
level, which is a reason not to ship it -- and a reason that was only found by
measuring it.

That raises the obvious next question, and it is about nuclei this program
already prints numbers for.  A reference compound makes a shift *definable*.
It does not make it *correct*, and nothing in this build has ever checked that
the method reproduces the scale it claims to be on.  Two of the claims are
written down as facts in the output:

  15N   the scale note tells the user to subtract 380.2 ppm to move an
        ammonia-referenced shift onto the nitromethane scale.  That number is
        a claim about THIS METHOD: compute NH3 and CH3NO2 at the same level and
        the difference has to come out at 380.2 ppm.  If it comes out at 38,
        the program is printing a conversion its own numbers do not obey.
  19F   the reference is CFCl3, the IUPAC primary standard, so the shifts are
        on the literature scale by construction -- but only if the method
        reproduces the distance between CFCl3 and a simple fluoride.  CH3F is
        271 ppm upfield of CFCl3, which is the largest move in the fluorine
        range for a molecule small enough to compute here.

Both tests are cheap: CH3NO2 is nao = 66 (1.3 GB) and CH3F is nao = 24.  The
references are taken from the module's own reference_geometry(), so what is
measured is the scale the product actually prints and not a private one.

Geometries are experimental r_e, constructed, in the style of the module's own
reference geometries:
    CH3F     C-F 1.382 A, C-H 1.095 A, H-C-F 108.5 deg
    CH3NO2   C-N 1.489 A, N-O 1.224 A, C-N-O 117.3 deg, C-H 1.094 A

Changes nothing.

Run:  python -u probes/probe_se7.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr

# (element, reference key, sample name, experimental shift in ppm)
# 19F:  CH3F is -271 ppm against CFCl3 (CFCl3 = 0 by definition)
# 15N:  the IUPAC conversion from ammonia to neat nitromethane
TESTS = [
    ("F", "chloroform_f", "CH3F", -271.0),
    ("N", "ammonia", "CH3NO2", 380.2),
]


def _perp(axis):
    a = np.array(axis, dtype=float)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    w1 = np.cross(a, ref)
    w1 = w1 / np.linalg.norm(w1)
    return a, w1, np.cross(a, w1)


def ch3f(c_f=1.382, c_h=1.095, hch=110.6):
    """CH3F: F up +z, three H at H-C-H, built the C3v way.

    This function used to combine a = (0,0,1) with the frame _perp([1,0,0])
    returns, and that frame is not perpendicular to a: w2 comes out (0,0,-1),
    parallel to a.  The hydrogens were therefore placed at the wrong radius
    (1.2656 A for one of them) and sigma(F) came out at -680 ppm instead of
    +485, which reads exactly like the method having the fluorine scale
    backwards.  The correct construction uses the C3v relation
    cos^2(polar) = (2 cos(angle) + 1) / 3.
    """
    c = math.cos(math.radians(hch))
    cz = math.sqrt(max(0.0, (2.0 * c + 1.0) / 3.0))
    sz = math.sqrt(max(0.0, 1.0 - cz * cz))
    out = [("C", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, c_f)]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        out.append(("H", c_h * sz * math.cos(phi), c_h * sz * math.sin(phi),
                    -c_h * cz))
    return out


def ch3no2(c_n=1.489, n_o=1.224, cno=117.3, c_h=1.094, hch=107.3):
    """CH3NO2: the nitro group is planar; N at the origin, C along +x."""
    ang = math.radians(cno)
    out = [("N", 0.0, 0.0, 0.0),
           ("C", c_n, 0.0, 0.0),
           ("O", n_o * math.cos(ang), n_o * math.sin(ang), 0.0),
           ("O", n_o * math.cos(ang), -n_o * math.sin(ang), 0.0)]
    # three H on the carbon, tetrahedral about the C->N axis
    cpos = np.array([c_n, 0.0, 0.0])
    a, w1, w2 = _perp([-1.0, 0.0, 0.0])
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        v = (math.cos(math.radians(hch)) * a
             + math.sin(math.radians(hch)) * (math.cos(phi) * w1
                                              + math.sin(phi) * w2))
        out.append(("H", *(cpos + c_h * v)))
    return out


def shielding(atoms, label):
    xyz = nmr._xyz(atoms)
    mol = nmr._build_mol(xyz, "6-31g*", 0, 1)
    ops = nmr.build_operators(mol)
    t0 = time.time()
    sigma, _diag = nmr.shielding_tensor(mol, ops)
    dt = time.time() - t0
    out = {}
    syms = [a[0] for a in atoms]
    for el in sorted(set(syms)):
        idx = [i for i, s in enumerate(syms) if s == el]
        out[el] = float(np.mean([np.trace(sigma[i]) / 3.0 for i in idx])) * nmr.PPM
    txt = "  ".join(f"sigma({el})={v:9.4f}" for el, v in sorted(out.items()))
    print(f"  {label:14s} nao={mol.nao_nr():3d}  {txt}   ({dt:.1f} s)", flush=True)
    return out


def main() -> int:
    print("=== does the method reproduce the scale each nucleus claims? ===\n")

    results = {}
    for el, refkey, sample, lit in TESTS:
        print(f"-- {el} --")
        ref_atoms = nmr.reference_geometry(refkey)
        print(f"  reference {nmr.REFERENCES[refkey]['label']} from "
              f"{nmr.reference_geometry_source(refkey)}: "
              f"{nmr.reference_geometry_note(refkey)}")
        sref = shielding(ref_atoms, nmr.REFERENCES[refkey]["label"])
        atoms = ch3f() if sample == "CH3F" else ch3no2()
        ssam = shielding(atoms, sample)
        delta = sref[el] - ssam[el]
        results[el] = (delta, lit, sample, nmr.REFERENCES[refkey]["label"])
        print(f"  delta({el}) of {sample} against "
              f"{nmr.REFERENCES[refkey]['label']}: {delta:+.1f} ppm")
        print(f"  experimental value:                    {lit:+.1f} ppm")
        if delta == 0.0:
            print("  ratio experimental / computed:         undefined (0.0)")
        else:
            print(f"  ratio experimental / computed:         {lit / delta:+.2f}")
        print()

    print("=== verdict ===")
    for el, (delta, lit, sample, reflab) in sorted(results.items()):
        ratio = float("inf") if delta == 0 else lit / delta
        if abs(ratio - 1.0) < 0.35:
            print(f"  {el}: the scale is reproduced "
                  f"({delta:+.1f} vs {lit:+.1f} ppm, ratio {ratio:.2f})")
        else:
            print(f"  {el}: the scale is NOT reproduced "
                  f"({delta:+.1f} vs {lit:+.1f} ppm, ratio {ratio:.2f}) -- "
                  f"{sample} against {reflab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
