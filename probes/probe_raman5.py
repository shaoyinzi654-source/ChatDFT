"""Round-15 probe 5: does the Raman spectrum come out physical?

The polarizability kernel is settled (probe 4: CPHF matches finite field to
9e-6 relative).  This probe drives the whole chain -- Hessian, 6N displaced
SCFs, dalpha/dx, projection onto the normal modes -- and checks it against
things that are true of water no matter what code produced the numbers.

The checks, and why each one can actually fail:

1. **The symmetric O-H stretch is the strongest Raman line but the weakest IR
   line.**  Measured IR intensities for water are bend 79.6, symmetric stretch
   1.8, asymmetric stretch 20.1 km/mol, so the symmetric stretch is ~44x weaker
   than the bend in IR.  In Raman it is the strongest.  Any implementation that
   reuses the IR intensities, or that mis-projects the tensor onto the modes,
   gets this exactly backwards -- and it is the single most robust statement
   about water's vibrational spectra.

2. **rho <= 0.75 for every mode**, with equality exactly when the isotropic
   derivative vanishes.  The bound is a theorem, not a fit.

3. **The asymmetric stretch sits at rho = 0.750 to within numerical noise.**
   It is B2 in C2v, so its isotropic derivative vanishes by symmetry.  This is
   a parameter-free symmetry statement: it tests the projection and the tensor
   simultaneously, and a sign error or a transposed index will move it.

4. **The Raman sum rule.**  sum_i ||dalpha/dQ_i||_F^2 must equal the full
   mass-weighted sum of ||dalpha/dx||_F^2 minus the rotational term, with
   translation contributing exactly zero.  No experimental number involved.

5. **The Frobenius identity** sum_ij a_ij^2 = 3 a_bar^2 + (2/3) gamma^2, which
   ties the activity convention to the sum rule's convention.

6. **alpha_iso at the equilibrium geometry.**  This one is expected to FAIL at
   6-31G* and to pass at a diffuse basis -- that is the whole finding of probe
   3, and reporting it either way is the point.

Run:  python probes/probe_raman5.py
"""
from __future__ import annotations

import time

import numpy as np

import backend.bootstrap as bootstrap

bootstrap.setup()

from backend.engine import analysis
from backend.engine.dft import DFTEngine

WATER_XYZ = "3\nwater\nO 0.000000 0.000000 0.119262\nH 0.000000 0.763239 -0.477047\nH 0.000000 -0.763239 -0.477047\n"

# Experimental / high-level reference values.
ALPHA_ISO_EXPT = (9.6, 9.9)
IR_REF = {"bend": 79.6, "sym": 1.8, "asym": 20.1}


def classify(freqs):
    """Label water's three modes by frequency, the way a spectroscopist would."""
    out = {}
    for i, f in enumerate(freqs):
        if f < 2000.0:
            out["bend"] = i
        elif i == max(range(len(freqs)), key=lambda j: freqs[j]):
            out["asym"] = i
        else:
            out["sym"] = i
    return out


def run(basis, optimize=True):
    print("=" * 78)
    print(f"  water, B3LYP/{basis}")
    print("=" * 78)
    eng = DFTEngine(atom_xyz=WATER_XYZ, charge=0, multiplicity=1,
                    functional="b3lyp", basis=basis)
    t0 = time.time()
    res = analysis.vibrations(eng, optimize_first=optimize, with_raman=True)
    dt = time.time() - t0
    freq = res["frequencies_cm1"]
    ir = res["ir_intensities_km_mol"]
    ram = res.get("raman") or {}
    print(f"  {dt:.1f} s, {len(freq)} modes, imaginary "
          f"{res['n_imaginary']}, opt_converged {res.get('opt_converged')}")
    print()
    if not ram or ram.get("available") is False:
        print("  RAMAN NOT AVAILABLE:", ram.get("reason"))
        return
    act = ram["activities_a4_amu"]
    rho = ram["depolarization"]
    lab = classify(freq)
    print(f"  {'mode':<6} {'freq':>8} {'IR km/mol':>10} {'Raman A^4/amu':>14} "
          f"{'rho':>7}")
    for name in ("bend", "sym", "asym"):
        i = lab.get(name)
        if i is None:
            continue
        print(f"  {name:<6} {freq[i]:>8.1f} {ir[i]:>10.2f} "
              f"{act[i]:>14.3f} {rho[i]:>7.4f}")

    # --- 1. the IR/Raman intensity inversion ---------------------------
    sym, bend, asym = lab.get("sym"), lab.get("bend"), lab.get("asym")
    ok = True
    if None not in (sym, bend, asym):
        strongest_raman = max(range(len(act)), key=lambda i: act[i])
        weakest_ir = min(range(len(ir)), key=lambda i: ir[i])
        print()
        print(f"  strongest Raman mode : {strongest_raman} "
              f"({freq[strongest_raman]:.0f} cm-1) -> "
              f"{'sym stretch' if strongest_raman == sym else 'NOT sym stretch'}")
        print(f"  weakest IR mode      : {weakest_ir} "
              f"({freq[weakest_ir]:.0f} cm-1) -> "
              f"{'sym stretch' if weakest_ir == sym else 'NOT sym stretch'}")
        print(f"  IR ratios vs sym stretch: bend {ir[bend] / ir[sym]:.1f}x, "
              f"asym {ir[asym] / ir[sym]:.1f}x   (measured 44x, 11x)")
        print(f"  Raman sym/bend = {act[sym] / act[bend]:.2f}, "
              f"sym/asym = {act[sym] / act[asym]:.2f}")
        if strongest_raman != sym:
            print("  *** FAIL: the symmetric stretch is not the strongest "
                  "Raman line")
            ok = False
        if weakest_ir != sym:
            print("  *** FAIL: the symmetric stretch is not the weakest IR band")
            ok = False

    # --- 2/3. depolarization ------------------------------------------
    print()
    worst_rho = max(rho)
    print(f"  max rho = {worst_rho:.4f}  (theorem: <= 0.7500)")
    if worst_rho > 0.7501:
        print("  *** FAIL: rho exceeds the 3/4 ceiling")
        ok = False
    if asym is not None:
        print(f"  rho(asym stretch) = {rho[asym]:.4f}, expected 0.7500 "
              f"(B2, isotropic derivative vanishes by symmetry)")
        if abs(rho[asym] - 0.75) > 0.005:
            print("  *** FAIL: the B2 mode is not at the depolarized limit")
            ok = False
    if sym is not None and rho[sym] > 0.70:
        print(f"  note: rho(sym stretch) = {rho[sym]:.4f}; the A1 stretch "
              f"should be strongly polarised (rho << 0.75)")

    # --- 4/5. the two invariants --------------------------------------
    sr = ram["sum_rule"]
    print()
    print(f"  sum rule: vibrational {sr.get('vibrational_frob2_per_amu')} vs "
          f"expected {sr.get('expected_frob2_per_amu')} "
          f"(rotation {sr.get('rotational_frob2_per_amu')})")
    print(f"            residual {sr.get('residual_pct')}%")
    if abs(float(sr.get("residual_pct") or 0.0)) > 0.5:
        print("  *** FAIL: the Raman sum rule is not satisfied")
        ok = False
    print(f"  Frobenius vs 3 a_bar^2 + 2/3 gamma^2: worst "
          f"{ram['frobenius_vs_invariants_pct']}%")
    if float(ram["frobenius_vs_invariants_pct"]) > 0.01:
        print("  *** FAIL: the activity convention does not match the sum rule")
        ok = False

    # --- 6. absolute polarizability -----------------------------------
    iso = ram["alpha_iso_au"]
    lo, hi = ALPHA_ISO_EXPT
    print()
    print(f"  alpha_iso = {iso:.4f} a.u. ({ram['alpha_iso_a3']:.4f} A^3); "
          f"experiment {lo}-{hi}")
    if not (lo - 0.3 <= iso <= hi + 0.3):
        print(f"  *** alpha is {100 * (iso / ((lo + hi) / 2) - 1):+.0f}% off the "
              f"measured value -- a basis without diffuse functions cannot "
              f"carry this property")
    else:
        print("  alpha agrees with experiment")
    print(f"  tensor (a.u.):")
    for row in ram["alpha_tensor_au"]:
        print("   ", " ".join(f"{v:9.4f}" for v in row))
    print()
    print(f"  OVERALL: {'all checks passed' if ok else 'FAILURES above'}")


if __name__ == "__main__":
    import sys

    for basis in (sys.argv[1:] or ["6-31g*"]):
        run(basis)
        print()
