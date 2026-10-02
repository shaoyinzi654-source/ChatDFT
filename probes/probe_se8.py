"""Probe -- is the 19F failure the method or the reference compound?

probe_se7 measured the 19F scale against its own IUPAC primary reference and got
the wrong answer with the wrong sign:

    CH3F against CFCl3   computed +17.9 ppm   experimental -271.9 ppm

That is a shipped nucleus.  Before anything is said about it, the cause has to
be separated, because the two causes have different fixes:

  (a) the reference compound CFCl3 is the problem -- three chlorines at 6-31G*
      is a poor description and its sigma(F) may be the outlier;
  (b) the method has no fluorine shift information at all.

The way to tell them apart is to leave the reference out of it.  The four
fluoromethanes have a known, monotonic experimental series (Bruker Almanac
1991, via the Indiana University 19F shift table; CFCl3 = 0 by definition):

    CF4    -62.3
    CHF3   -78.6
    CH2F2 -143.6
    CH3F  -271.9

Monotonic, and a spread of 209.6 ppm that contains no chlorine.  If the computed
sigma(F) follows that order -- increasing from CF4 to CH3F, because a more
negative delta is more shielded -- then the method does carry fluorine shift
information and the reference compound is the thing to look at.  If the computed
values are flat, or ordered the other way, then fluorine has to be treated the
way selenium is.

Geometries are experimental r_e, constructed:
    CH3F   C-F 1.382, C-H 1.095, H-C-H 110.6
    CH2F2  C-F 1.358, C-H 1.093, F-C-F 108.3, H-C-H 113.7
    CHF3   C-F 1.332, C-H 1.098, F-C-F 108.8
    CF4    C-F 1.317, tetrahedral

Changes nothing.

Run:  python -u probes/probe_se8.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr

# (label, experimental shift against CFCl3)
SERIES = [("CF4", -62.3), ("CHF3", -78.6), ("CH2F2", -143.6), ("CH3F", -271.9)]


def _c3v_axis(angle_deg):
    """cos^2(polar) = (2 cos(angle) + 1) / 3, for three substituents."""
    c = math.cos(math.radians(angle_deg))
    cz = math.sqrt(max(0.0, (2.0 * c + 1.0) / 3.0))
    return cz, math.sqrt(max(0.0, 1.0 - cz * cz))


def _spokes(n, cz, sz, up=True, phase=0.0):
    out = []
    for k in range(n):
        phi = 2.0 * math.pi * k / n + phase
        z = cz if up else -cz
        out.append((sz * math.cos(phi), sz * math.sin(phi), z))
    return out


def cf4(c_f=1.317):
    t = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                  [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    t /= np.linalg.norm(t, axis=1)[:, None]
    return [("C", 0.0, 0.0, 0.0)] + [("F", *(c_f * v)) for v in t]


def chf3(c_f=1.332, c_h=1.098, fcf=108.8):
    cz, sz = _c3v_axis(fcf)
    out = [("C", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, -c_h)]
    out += [("F", *(c_f * x for x in v)) for v in _spokes(3, cz, sz, up=True)]
    return out


def ch2f2(c_f=1.358, c_h=1.093, fcf=108.3, hch=113.7):
    """F's in the xz plane, H's in the perpendicular yz plane."""
    half_f = math.radians(fcf / 2.0)
    half_h = math.radians(hch / 2.0)
    out = [("C", 0.0, 0.0, 0.0),
           ("F", c_f * math.sin(half_f), 0.0, c_f * math.cos(half_f)),
           ("F", -c_f * math.sin(half_f), 0.0, c_f * math.cos(half_f)),
           ("H", 0.0, c_h * math.sin(half_h), -c_h * math.cos(half_h)),
           ("H", 0.0, -c_h * math.sin(half_h), -c_h * math.cos(half_h))]
    return out


def ch3f(c_f=1.382, c_h=1.095, hch=110.6):
    cz, sz = _c3v_axis(hch)
    out = [("C", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, c_f)]
    out += [("H", *(c_h * x for x in v)) for v in _spokes(3, cz, sz, up=False)]
    return out


BUILDERS = {"CF4": cf4, "CHF3": chf3, "CH2F2": ch2f2, "CH3F": ch3f}


def sigma_f(atoms, label):
    xyz = nmr._xyz(atoms)
    mol = nmr._build_mol(xyz, "6-31g*", 0, 1)
    ops = nmr.build_operators(mol)
    t0 = time.time()
    sigma, _diag = nmr.shielding_tensor(mol, ops)
    dt = time.time() - t0
    idx = [i for i, a in enumerate(atoms) if a[0] == "F"]
    iso = float(np.mean([np.trace(sigma[i]) / 3.0 for i in idx])) * nmr.PPM
    print(f"  {label:8s} nao={mol.nao_nr():3d}  sigma(F) = {iso:9.4f} ppm  "
          f"({dt:.1f} s)", flush=True)
    return iso


def main() -> int:
    print("=== the fluoromethane series, no chlorine involved ===\n")
    got = {}
    for label, lit in SERIES:
        got[label] = sigma_f(BUILDERS[label](), label)

    print("\n  label    experimental d    computed sigma(F)   computed d vs CF4")
    for label, lit in SERIES:
        rel = got[label] - got["CF4"]
        print(f"  {label:8s} {lit:9.1f}          {got[label]:9.4f}      "
              f"{rel:+9.4f}")

    # ascending delta == most shielded first (a negative delta is upfield)
    print("\n  experimental order (most shielded first):")
    print("    " + " < ".join(f"{l}({v:.1f})" for l, v in
                              sorted(SERIES, key=lambda kv: kv[1])))
    print("  computed order (smallest sigma first):")
    print("    " + " < ".join(f"{l}({got[l]:.4f})" for l in
                              sorted(got, key=lambda k: got[k])))
    print()

    # most shielded first on both sides: ascending delta vs LARGEST sigma
    exp_order = [l for l, _ in sorted(SERIES, key=lambda kv: kv[1])]
    comp_order = sorted(got, key=lambda k: -got[k])
    print(f"  experimental order, most shielded first: {exp_order}")
    print(f"  computed order,     largest sigma first: {comp_order}")
    if exp_order == comp_order:
        print("\n  the computed ordering FOLLOWS experiment, so the method does")
        print("  carry fluorine shift information and the fault is elsewhere")
        print("  (the reference compound, or the magnitude only)")
    else:
        print("\n  the computed ordering does NOT follow experiment.  Fluorine")
        print("  shifts from this method are not fluorine shifts, and no choice")
        print("  of reference compound repairs that.")

    span_exp = max(v for _, v in SERIES) - min(v for _, v in SERIES)
    span_comp = max(got.values()) - min(got.values())
    print(f"\n  experimental span over the series: {span_exp:.1f} ppm")
    print(f"  computed span over the series:     {span_comp:.4f} ppm")
    if span_comp != 0.0:
        print(f"  ratio: {span_exp / span_comp:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
