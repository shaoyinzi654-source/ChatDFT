"""probe_nmr20.py -- re-validate the reference table through backend/engine/nmr.py.

probe_nmr19 found that the finite-difference derivative breaks down at small
dB when conv_tol = 1e-11: the 17O shielding of water jumps from 329.74 to
345.05 between dB = 1e-4 and 4e-5, and then sits bit-identical for three more
decades.  At conv_tol = 1e-13 the same scan is flat to 0.07 ppm from dB = 3e-3
down to 3e-5.  The jump was an under-converged SCF, and 345.046 -- the value
quoted as H2O 17O in earlier rounds -- sits on the broken side of it.

That matters beyond water: every 17O chemical shift is delta = sigma(H2O) -
sigma(molecule), so an 15 ppm error in the reference is a 15 ppm error in every
17O shift the product prints.

This probe therefore
  1. scans dB at conv_tol = 1e-13 over the whole usable range, for three
     molecules of different electronic character, to pick a step with real
     margin on both sides;
  2. runs the actual product entry point (nmr.compute) on the reference
     compounds and on a set of molecules with known gas-phase shifts;
  3. prints the accuracy table that the product's documentation will quote.

Run:  python probes/probe_nmr20.py
"""
import time

import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto

from backend.engine import nmr

np.set_printoptions(precision=6, suppress=True, linewidth=200)

WATER = [("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587),
         ("H", 0.0, 0.757, 0.587)]
AMMONIA = [("N", 0.0, 0.0, 0.0), ("H", 0.0, 0.9391, -0.3816),
           ("H", 0.8133, -0.4696, -0.3816), ("H", -0.8133, -0.4696, -0.3816)]
METHANE = [("C", 0.0, 0.0, 0.0),
           ("H", 0.6291, 0.6291, 0.6291), ("H", -0.6291, -0.6291, 0.6291),
           ("H", -0.6291, 0.6291, -0.6291), ("H", 0.6291, -0.6291, -0.6291)]

# molecule -> geometry, and the gas-phase shifts a paper would compare against
SHIFT_CASES = [
    ("methane", [("C", 0.0, 0.0, 0.0)] + [
        ("H", *(1.087 * v)) for v in
        (np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
         / np.sqrt(3))], {"C": -2.30, "H": 0.23}),
    ("ethane", [("C", 0.0, 0.0, 0.764), ("C", 0.0, 0.0, -0.764),
                ("H", 1.017, 0.0, 1.163), ("H", -0.509, 0.881, 1.163),
                ("H", -0.509, -0.881, 1.163),
                ("H", -1.017, 0.0, -1.163), ("H", 0.509, 0.881, -1.163),
                ("H", 0.509, -0.881, -1.163)], {"C": 5.70, "H": 0.86}),
    ("ethylene", [("C", 0.0, 0.0, 0.667), ("C", 0.0, 0.0, -0.667),
                  ("H", 0.0, 0.923, 1.238), ("H", 0.0, -0.923, 1.238),
                  ("H", 0.0, 0.923, -1.238), ("H", 0.0, -0.923, -1.238)],
     {"C": 123.50, "H": 5.40}),
    ("acetylene", [("C", 0.0, 0.0, 0.601), ("C", 0.0, 0.0, -0.601),
                   ("H", 0.0, 0.0, 1.663), ("H", 0.0, 0.0, -1.663)],
     {"C": 71.90, "H": 1.80}),
    ("benzene", [("C", *(1.397 * np.array([np.cos(np.pi / 3 * k),
                                           np.sin(np.pi / 3 * k), 0.0])))
                 for k in range(6)]
                + [("H", *(2.481 * np.array([np.cos(np.pi / 3 * k),
                                             np.sin(np.pi / 3 * k), 0.0])))
                   for k in range(6)], {"C": 128.50, "H": 7.26}),
    ("methanol", [("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.43),
                  ("H", 0.895, 0.0, -0.44), ("H", -0.448, 0.775, -0.44),
                  ("H", -0.448, -0.775, -0.44), ("H", 0.885, 0.0, 1.85)],
     {"C": 49.50, "H": 3.35}),
]

ABS_CASES = [
    ("H2O", WATER, {"O": 344.0, "H": 30.7}),
    ("NH3", AMMONIA, {"N": 264.0, "H": 30.8}),
    ("CH4", METHANE, {"C": 195.0, "H": 30.6}),
]


def as_xyz(atoms):
    return nmr._xyz(atoms)


def scan_db():
    print("=" * 88)
    print(" 1. dB scan at conv_tol = 1e-13 (the tolerance the product will use)")
    print("=" * 88)
    for name, atoms, nuc in (("H2O", WATER, [(0, "O"), (1, "H")]),
                             ("CH4", METHANE, [(0, "C"), (1, "H")])):
        mol = gto.M(atom=atoms, basis="6-31g*", verbose=0)
        ops = nmr.build_operators(mol)
        m0 = nmr._scf_at(mol, ops, [0.0] * 3, None, 1e-13)
        dm0 = m0.make_rdm1()
        print(f"\n  {name}")
        print(f"    {'dB':>9s} {'iso(heavy)':>12s} {'iso(H)':>10s} "
              f"{'asym(H)':>9s}  conv")
        for dB in (3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
            sig, diag = nmr.shielding_tensor(mol, ops, dB=dB, conv_tol=1e-13)
            conv = "ok" if all(r["converged"] for r in diag["scf_runs"]) else "NO"
            print(f"    {dB:9.0e} "
                  f"{np.trace(sig[nuc[0][0]]) / 3 * nmr.PPM:12.4f} "
                  f"{np.trace(sig[nuc[1][0]]) / 3 * nmr.PPM:10.4f} "
                  f"{np.abs(sig[nuc[1][0]] - sig[nuc[1][0]].T).max() * nmr.PPM:9.5f}"
                  f"  {conv}")


def main():
    import sys
    which = sys.argv[1:] or ["scan", "refs", "abs", "shifts"]
    if "scan" in which:
        scan_db()
    if "refs" not in which:
        return

    print()
    print("=" * 88)
    print(" 2. reference compounds through nmr.reference_shieldings")
    print("=" * 88)
    refs = {}
    for key in ("tms", "ammonia", "water", "phosphine", "chloroform_f"):
        t0 = time.time()
        r = nmr.reference_shieldings(key, "6-31g*", use_cache=True)
        refs[key] = r
        els = ", ".join(
            f"{el} {v['sigma_iso_ppm']:.3f} (spread {v['spread_ppm']:.4f}, "
            f"n={v['n_equivalent']})"
            for el, v in r["element_shielding_ppm"].items())
        print(f"  {r['label']:14s} natm={r['natm']:2d} nao={r['nao']:3d}  "
              f"dE/dBmax={r['dE_dB_max']:.1e}  asym={r['tensor_asymmetry_max_ppm']:.4f}")
        print(f"      {els}")
        print(f"      [{time.time() - t0:.1f}s]")

    print()
    print("=" * 88)
    print(" 3. absolute shieldings vs gas-phase experiment")
    print("=" * 88)
    print(f"  {'molecule':10s} {'nuc':>4s} {'sigma calc':>11s} {'sigma exp':>10s} "
          f"{'error':>8s}")
    for name, atoms, exp in ABS_CASES:
        res = nmr.compute(as_xyz(atoms), basis="6-31g*", use_cache=True)
        for el, target in exp.items():
            vals = [n["sigma_iso_ppm"] for n in res["nuclei"]
                    if n["symbol"] == el]
            s = float(np.mean(vals))
            print(f"  {name:10s} {el:>4s} {s:11.3f} {target:10.2f} "
                  f"{s - target:+8.2f}")

    print()
    print("=" * 88)
    print(" 4. chemical shifts vs TMS   delta = sigma(TMS) - sigma(molecule)")
    print("=" * 88)
    sC = refs["tms"]["element_shielding_ppm"]["C"]["sigma_iso_ppm"]
    sH = refs["tms"]["element_shielding_ppm"]["H"]["sigma_iso_ppm"]
    print(f"  reference: TMS  13C {sC:.3f}   1H {sH:.3f}   "
          f"29Si {refs['tms']['element_shielding_ppm']['Si']['sigma_iso_ppm']:.3f}")
    print()
    print(f"  {'molecule':11s} {'nuc':>4s} {'sigma':>10s} {'delta calc':>11s} "
          f"{'delta exp':>10s} {'error':>8s}")
    errs = []
    for name, atoms, exp in SHIFT_CASES:
        res = nmr.compute(as_xyz(atoms), basis="6-31g*", use_cache=True)
        for el, target in exp.items():
            vals = [n["sigma_iso_ppm"] for n in res["nuclei"]
                    if n["symbol"] == el]
            s = float(np.mean(vals))
            ref = sC if el == "C" else sH
            d = ref - s
            errs.append(abs(d - target))
            print(f"  {name:11s} {el:>4s} {s:10.3f} {d:11.3f} {target:10.2f} "
                  f"{d - target:+8.2f}")
    hydro = [e for e, (n, _, x) in zip(errs, SHIFT_CASES) if True]
    print(f"\n  mean |error| over all {len(errs)} shifts: "
          f"{np.mean(errs):.3f} ppm")
    print()


if __name__ == "__main__":
    main()
