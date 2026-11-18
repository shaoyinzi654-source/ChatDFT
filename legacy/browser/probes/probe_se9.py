"""Probe -- the accuracy ledger: does the method reproduce each nucleus's scale?

probe_se6/7 measured three nuclei and found the method is not equally good for
all of them.  This is the same measurement done properly, for every nucleus
this build prints a shift for, with the units right.

Units first, because the first version of this got them wrong: ``shielding_
tensor`` returns the tensor in **atomic units** (its own docstring says so), and
``reference_shieldings`` is what multiplies by PPM.  probe_se5/6/7 labelled au
as ppm, which inflated every geometry sensitivity by 53.25x.  The *ratios*
below were never affected -- a common factor cancels -- but every absolute
number was, and the first draft of this round's conclusions was wrong by that
factor.  Hence ``* nmr.PPM`` on every number printed here.

For each nucleus: the reference the product actually uses (``reference_
geometry``, so library geometries win exactly as they do in production), and one
molecule whose shift against it is known.  The computed shift is
``sigma(ref) - sigma(sample)`` in ppm, and it is compared with experiment.

    nucleus  reference   sample     experimental delta   source
    1H       TMS         CH4              0.23           (module's own ledger)
    13C      TMS         CH4             -2.30          (module's own ledger)
    14N      NH3         CH3NO2        +380.2           ammonia -> nitromethane
    19F      CFCl3       CH3F          -271.9           Bruker Almanac 1991
    31P      PH3         CH3PH2         +75.6           see below
    29Si     TMS         SiH4           -92.5           see below
    17O      H2O         H2CO          +600            see below
    77Se     Me2Se       H2Se          -616.0          Lardon 1970

The 1H/13C rows are the module's own twelve-shift ledger, which until now lived
only in a docstring that no gate recomputed.

Changes nothing.

Run:  python -u probes/probe_se9.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr


def _perp(axis):
    a = np.array(axis, dtype=float)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    w1 = np.cross(a, ref)
    w1 = w1 / np.linalg.norm(w1)
    return a, w1, np.cross(a, w1)


def _c3v_axis(angle_deg):
    c = math.cos(math.radians(angle_deg))
    cz = math.sqrt(max(0.0, (2.0 * c + 1.0) / 3.0))
    return cz, math.sqrt(max(0.0, 1.0 - cz * cz))


def _spokes(n, cz, sz, up=True):
    out = []
    for k in range(n):
        phi = 2.0 * math.pi * k / n
        z = cz if up else -cz
        out.append((sz * math.cos(phi), sz * math.sin(phi), z))
    return out


def ch3f(c_f=1.382, c_h=1.095, hch=110.6):
    cz, sz = _c3v_axis(hch)
    out = [("C", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, c_f)]
    out += [("H", *(c_h * x for x in v)) for v in _spokes(3, cz, sz, up=False)]
    return out


def ch3no2(c_n=1.489, n_o=1.224, cno=117.3, c_h=1.094, hch=107.3):
    ang = math.radians(cno)
    out = [("N", 0.0, 0.0, 0.0),
           ("C", c_n, 0.0, 0.0),
           ("O", n_o * math.cos(ang), n_o * math.sin(ang), 0.0),
           ("O", n_o * math.cos(ang), -n_o * math.sin(ang), 0.0)]
    cpos = np.array([c_n, 0.0, 0.0])
    a, w1, w2 = _perp([-1.0, 0.0, 0.0])
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        v = (math.cos(math.radians(hch)) * a
             + math.sin(math.radians(hch)) * (math.cos(phi) * w1
                                              + math.sin(phi) * w2))
        out.append(("H", *(cpos + c_h * v)))
    return out


def h2se(se_h=1.4599, angle=90.6):
    """H2Se, experimental r_e."""
    half = math.radians(angle / 2.0)
    return [("Se", 0.0, 0.0, 0.0),
            ("H", se_h * math.sin(half), 0.0, se_h * math.cos(half)),
            ("H", -se_h * math.sin(half), 0.0, se_h * math.cos(half))]


def dimethyl_selenide(se_c=1.943, cse_c=96.2, c_h=1.090):
    """Me2Se built from experimental parameters (probe_se5, geometry C)."""
    half = math.radians(cse_c / 2.0)
    out = [("Se", 0.0, 0.0, 0.0)]
    for sgn in (1.0, -1.0):
        u = np.array([sgn * math.sin(half), 0.0, math.cos(half)])
        cpos = se_c * u
        out.append(("C", *cpos))
        a, w1, w2 = _perp(-u)
        for j in range(3):
            phi = 2.0 * math.pi * j / 3.0
            v = (math.cos(math.radians(109.47)) * a
                 + math.sin(math.radians(109.47)) * (math.cos(phi) * w1
                                                     + math.sin(phi) * w2))
            out.append(("H", *(cpos + c_h * v)))
    return out


def sih4(si_h=1.480):
    t = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                  [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    t /= np.linalg.norm(t, axis=1)[:, None]
    return [("Si", 0.0, 0.0, 0.0)] + [("H", *(si_h * v)) for v in t]


def h2co(c_o=1.208, c_h=1.111, hch=116.5):
    """Formaldehyde, experimental r_e; the C2 axis along +z."""
    half = math.radians(hch / 2.0)
    return [("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, c_o),
            ("H", c_h * math.sin(half), 0.0, -c_h * math.cos(half)),
            ("H", -c_h * math.sin(half), 0.0, -c_h * math.cos(half))]


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
    txt = "  ".join(f"sigma({el})={v:9.2f}" for el, v in sorted(out.items()))
    print(f"  {label:12s} nao={mol.nao_nr():3d}  {txt}   ({dt:.1f} s)", flush=True)
    return out


# (nucleus, reference key, sample label, sample atoms, experimental delta)
LEDGER = [
    ("H", "tms", "CH4", lambda: None, 0.23),
    ("C", "tms", "CH4", lambda: None, -2.30),
    ("N", "ammonia", "CH3NO2", ch3no2, 380.2),
    ("F", "chloroform_f", "CH3F", ch3f, -271.9),
    ("P", "phosphine", "CH3PH2", lambda: None, 75.6),
    ("Si", "tms", "SiH4", sih4, -92.5),
    ("O", "water", "H2CO", h2co, 600.0),
    ("Se", None, "H2Se", h2se, -616.0),
]


def main() -> int:
    from backend.engine.molecule import resolve

    print("=== the accuracy ledger, units right this time ===\n")

    # CH4 and CH3PH2 come from the library, so the sample is the molecule the
    # product ships and not a private geometry.
    cache = {}
    for lab, smiles in (("CH4", "C"), ("CH3PH2", "CP")):
        m = resolve(smiles, kind="smiles")
        cache[lab] = [(a.symbol, a.x, a.y, a.z) for a in m.atoms]
        print(f"  sample {lab} from the library: {m.formula}")

    refs = {}
    for key in sorted({k for _, k, *_ in LEDGER if k}):
        # The product's own path, so the on-disk cache is used and the number
        # is the one a job would get.
        res = nmr.reference_shieldings(key, "6-31g*", use_cache=True)
        vals = {el: v["sigma_iso_ppm"]
                for el, v in res["element_shielding_ppm"].items()}
        print(f"  ref {res['label']:12s} nao={res['nao']:3d}  "
              + "  ".join(f"sigma({el})={v:9.2f}" for el, v in sorted(vals.items()))
              + f"   (cached={res.get('cached')})")
        refs[key] = vals

    print()
    rows = []
    for nuc, refkey, sample, builder, lit in LEDGER:
        atoms = cache.get(sample) or builder()
        s = shielding(atoms, sample)
        if refkey is None:
            ref = refs.setdefault("dimethyl_selenide",
                                  shielding(dimethyl_selenide(), "(CH3)2Se"))
        else:
            ref = refs[refkey]
        delta = ref[nuc] - s[nuc]
        ratio = float("inf") if delta == 0 else lit / delta
        rows.append((nuc, refkey or "Me2Se", sample, delta, lit, ratio))
        print()

    print(f"  {'nuc':4s} {'ref':10s} {'sample':8s} {'computed':>10s} "
          f"{'exptl':>9s} {'ratio':>8s}")
    for nuc, refkey, sample, delta, lit, ratio in rows:
        print(f"  {nuc:4s} {refkey:10s} {sample:8s} {delta:10.2f} "
              f"{lit:9.1f} {ratio:8.2f}")
    print()
    # The ratio is useless for 1H and 13C, where the shifts are near zero: a
    # 0.39 ppm error on a 0.23 ppm shift is a ratio of 0.37 and the largest
    # relative error in the table while being the smallest absolute one.  What
    # decides whether a scale is usable is the error in ppm against the range
    # of that nucleus.
    for nuc, refkey, sample, delta, lit, ratio in rows:
        err = abs(delta - lit)
        if abs(lit) < 20.0:
            verdict = "reproduced" if err < 2.5 else "off by more than 2.5 ppm"
        elif err < 0.35 * abs(lit):
            verdict = "reproduced"
        elif err < 1.0 * abs(lit):
            verdict = "same scale, wrong size"
        else:
            verdict = "NOT on this scale"
        print(f"  {nuc:4s} {verdict:26s} (error {err:7.2f} ppm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
