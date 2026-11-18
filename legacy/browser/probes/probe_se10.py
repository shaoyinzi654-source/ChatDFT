"""Probe -- is the twelve-shift ledger in the module docstring still true?

The docstring of engine/nmr.py states, as an accuracy claim:

    chemical shifts vs TMS, delta = sigma(TMS) - sigma(molecule)
        acetylene 1H   1.833  vs  1.80   (+0.03)
        methane   1H   0.442  vs  0.23   (+0.21)
        ...
        methanol  13C 29.488  vs 49.50   (-20.01)

    Ten of the twelve shifts are inside 2.5 ppm.

and attributes it to probe_nmr20.  No gate recomputes any of it.  That is the
same shape as the unreferenced-element list that had no reasons and the memory
constant that was never measured: a claim in prose cannot be wrong, so it
cannot be trusted.

It is now testable, because probe_se9 measured two of the same numbers against
the *current* molecule library and got different answers:

    methane 1H    docstring 0.442    measured now 0.62
    methane 13C   docstring -1.487   measured now -0.44

Both are still inside 2.5 ppm, so the headline claim survives for methane --
but the numbers have moved, and the reason is that the library's geometries
were re-derived after that ledger was written (round 17: all 62 entries were
stamped "MMFF94 (RDKit ETKDGv3)" while re-embedding them from their own SMILES
reproduced only 25).  A ledger measured on geometries the library no longer
ships describes a product that no longer exists.

So the twelve are re-measured here, on the library as it is, with the reference
shieldings taken from the product's own cache path.

Changes nothing.

Run:  python -u probes/probe_se10.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr
from backend.engine.molecule import resolve

# (name in the library, nucleus, docstring value, experimental value)
LEDGER = [
    ("acetylene", "H", 1.833, 1.80),
    ("methane", "H", 0.442, 0.23),
    ("ethane", "H", 1.030, 0.86),
    ("ethylene", "H", 5.732, 5.40),
    ("benzene", "H", 7.627, 7.26),
    ("methane", "C", -1.487, -2.30),
    ("ethane", "C", 7.597, 5.70),
    ("benzene", "C", 126.309, 128.50),
    ("ethylene", "C", 121.091, 123.50),
    ("acetylene", "C", 65.785, 71.90),
    ("methanol", "H", 0.340, 3.35),
    ("methanol", "C", 29.488, 49.50),
]


def shielding_of(name):
    m = resolve(name, kind="name")
    atoms = [(a.symbol, a.x, a.y, a.z) for a in m.atoms]
    xyz = nmr._xyz(atoms)
    mol = nmr._build_mol(xyz, "6-31g*", m.charge, m.multiplicity)
    ops = nmr.build_operators(mol)
    sigma, _diag = nmr.shielding_tensor(mol, ops)
    syms = [a[0] for a in atoms]
    out = {}
    for el in sorted(set(syms)):
        idx = [i for i, s in enumerate(syms) if s == el]
        out[el] = float(sum(float(sigma[i][0, 0] + sigma[i][1, 1]
                                   + sigma[i][2, 2]) / 3.0 for i in idx)
                         / len(idx)) * nmr.PPM
    return out, mol.nao_nr()


def main() -> int:
    ref = nmr.reference_shieldings("tms", "6-31g*", use_cache=True)
    s_tms = {el: v["sigma_iso_ppm"]
             for el, v in ref["element_shielding_ppm"].items()}
    print(f"TMS reference: sigma(1H)={s_tms['H']:.4f} "
          f"sigma(13C)={s_tms['C']:.4f}  (cached={ref.get('cached')})\n")

    cache = {}
    rows = []
    for name, nuc, doc, exp in LEDGER:
        if name not in cache:
            cache[name] = shielding_of(name)
        sig, nao = cache[name]
        delta = s_tms[nuc] - sig[nuc]
        rows.append((name, nuc, doc, delta, exp, nao))

    print(f"  {'molecule':10s} {'nuc':4s} {'docstring':>10s} {'measured':>10s} "
          f"{'drift':>8s} {'exptl':>8s} {'err now':>8s}")
    worst = 0.0
    inside = 0
    for name, nuc, doc, delta, exp, nao in rows:
        drift = delta - doc
        err = delta - exp
        worst = max(worst, abs(drift))
        if abs(err) <= 2.5:
            inside += 1
        print(f"  {name:10s} {nuc:4s} {doc:10.3f} {delta:10.3f} "
              f"{drift:+8.3f} {exp:8.2f} {err:+8.3f}")
    print(f"\n  largest drift from the docstring: {worst:.3f} ppm")
    print(f"  shifts inside 2.5 ppm of experiment: {inside} of {len(rows)}")
    if inside >= 10:
        print("  the docstring's headline claim still holds")
    else:
        print("  the docstring's headline claim NO LONGER holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
